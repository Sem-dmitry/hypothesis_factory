# -*- coding: utf-8 -*-

"""
FastAPI backend for the Фабрика гипотез web UI.

Wraps the existing ``StudioEngine`` (our project's "начинка"): projects, runs in a
background thread (always the full deep tournament), a live Server-Sent-Events
stream of agent activity, live steering, per-agent settings, key status, corpus
upload, and a full request/response log per run.

Run:  python -m uvicorn webapp.server:app --port 8800
  or: python webapp/server.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

# --- Bootstrap env (same as the Streamlit app) BEFORE importing the engine. ---
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coscientist.env_utils import (  # noqa: E402
    apply_llm_runtime_defaults,
    bridge_provider_aliases,
    load_env_file,
    silence_optional_warnings,
)

load_env_file()
bridge_provider_aliases()
apply_llm_runtime_defaults()
silence_optional_warnings()

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import (  # noqa: E402
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles  # noqa: E402

from coscientist.studio import (  # noqa: E402
    DEFAULT_AGENT_MODELS,
    AgentLLMSettings,
    RunSettings,
    StudioEngine,
    StudioStore,
    format_full_log,
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

MODEL_OPTIONS = [
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "anthropic/claude-sonnet-4",
]
THINKING_OPTIONS = ["default", "off", "low", "medium", "high"]

AGENT_LABELS = {
    "literature": "Literature — обзор литературы",
    "generation": "Generation — генерация гипотез",
    "reflection": "Reflection — deep verification",
    "ranking": "Ranking — Elo-турнир",
    "evolution": "Evolution — улучшение гипотез",
    "meta_review": "Meta-review — синтез + фидбэк",
    "supervisor": "Supervisor — оркестрация",
    "final_report": "Final report — итоговый отчёт",
    "assessment": "Assessor — оценка гипотез",
}
# Pipeline phases (by transcript agent label) for the Tasks/progress panel.
PHASES = [
    ("Constraints", "Доуточнение ограничений"),
    ("Literature", "Обзор литературы"),
    ("Generation", "Генерация гипотез"),
    ("Reflection", "Рефлексия и верификация"),
    ("Ranking", "Турнир (ранжирование)"),
    ("Evolution", "Эволюция гипотез"),
    ("Meta-review", "Мета-ревью"),
    ("Supervisor", "Оркестрация"),
    ("Final report", "Финальный отчёт"),
    ("Assessor", "Оценка гипотез"),
]

# ---------------------------------------------------------------------------
# Storage: projects + runs
# ---------------------------------------------------------------------------


def _webapp_dir() -> str:
    base = os.environ.get("COSCIENTIST_DIR", os.path.expanduser("~/.coscientist"))
    d = os.path.join(base, "webapp")
    os.makedirs(d, exist_ok=True)
    return d


class ProjectStore:
    """Named projects that group runs (persisted as a small JSON file)."""

    def __init__(self):
        self.path = os.path.join(_webapp_dir(), "projects.json")
        self._lock = threading.Lock()

    def _load(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return []

    def _save(self, projects: list[dict]) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(projects, fh, ensure_ascii=False, indent=2)

    def list(self) -> list[dict]:
        return self._load()

    def create(self, name: str) -> dict:
        name = (name or "").strip() or "Новый проект"
        with self._lock:
            projects = self._load()
            proj = {
                "id": uuid.uuid4().hex[:12],
                "name": name,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            projects.insert(0, proj)
            self._save(projects)
        return proj

    def get(self, project_id: str) -> Optional[dict]:
        return next((p for p in self._load() if p["id"] == project_id), None)

    def ensure_default(self) -> dict:
        projects = self._load()
        if projects:
            return projects[0]
        return self.create("Хвосты флотации")


projects = ProjectStore()
store = StudioStore()
engine = StudioEngine(store)


# ---------------------------------------------------------------------------
# Run manager: background runs + live event queues
# ---------------------------------------------------------------------------


class _Run:
    def __init__(self, run_id: str, project: str):
        self.id = run_id
        self.project = project
        self.q: "queue.Queue[dict]" = queue.Queue()
        self.status = "running"
        self.record_id: Optional[str] = None
        self.error: Optional[str] = None
        self.transcript: list[dict] = []
        self.console: list[str] = []
        # Live steering: messages the user sends while the run executes. The
        # framework drains this at each supervisor-loop boundary (chat-to-agents).
        self._inbox: list[str] = []
        self._inbox_lock = threading.Lock()

    def add_message(self, text: str) -> None:
        with self._inbox_lock:
            self._inbox.append(text)

    def drain_inbox(self) -> list[str]:
        with self._inbox_lock:
            pending, self._inbox = self._inbox, []
        return pending


class _ConsoleCapture:
    """Line-buffered stdout/stderr tee used while a background run executes."""

    def __init__(self, stream: str, original, emit):
        self.stream = stream
        self.original = original
        self.emit = emit
        self._buf = ""

    def write(self, data):
        if self.original is not None:
            self.original.write(data)
        self._buf += str(data)
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.emit(line.rstrip(), self.stream)
        return len(data)

    def flush(self):
        if self._buf.strip():
            self.emit(self._buf.rstrip(), self.stream)
        self._buf = ""
        if self.original is not None:
            self.original.flush()

    def isatty(self):
        return bool(getattr(self.original, "isatty", lambda: False)())

    def __getattr__(self, name):
        if self.original is None:
            raise AttributeError(name)
        return getattr(self.original, name)


class _RunLogHandler(logging.Handler):
    def __init__(self, emit):
        super().__init__()
        self.emit_line = emit
        self.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))

    def emit(self, record):
        try:
            self.emit_line(self.format(record), "logging")
        except Exception:
            pass


class RunManager:
    def __init__(self):
        self.runs: dict[str, _Run] = {}
        self._lock = threading.Lock()

    def start(self, *, project: str, goal: str, constraints: str,
              settings: RunSettings, corpus_files: Optional[list[str]],
              use_web: bool) -> str:
        run_id = uuid.uuid4().hex[:12]
        run = _Run(run_id, project)
        with self._lock:
            self.runs[run_id] = run

        t = threading.Thread(
            target=self._real_worker,
            kwargs=dict(run=run, goal=goal, constraints=constraints,
                        settings=settings, corpus_files=corpus_files, use_web=use_web),
            daemon=True,
        )
        t.start()
        return run_id

    def get(self, run_id: str) -> Optional[_Run]:
        return self.runs.get(run_id)

    # -- event helpers --

    def _console_event(self, run: _Run, message: str, stream: str = "log") -> None:
        line = (message or "").rstrip()
        if not line:
            return
        ev = {
            "type": "log",
            "stream": stream,
            "message": line,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        run.console.append(f"[{ev['ts']}] {stream}: {line}")
        run.q.put(ev)

    def _capture_console(self, run: _Run):
        manager = self

        class _CaptureContext:
            def __enter__(self):
                self.old_stdout = sys.stdout
                self.old_stderr = sys.stderr
                self.out = _ConsoleCapture("stdout", self.old_stdout, manager._console_event_for(run))
                self.err = _ConsoleCapture("stderr", self.old_stderr, manager._console_event_for(run))
                self.handler = _RunLogHandler(manager._console_event_for(run))
                self.root = logging.getLogger()
                self.root.addHandler(self.handler)
                sys.stdout = self.out
                sys.stderr = self.err
                return self

            def __exit__(self, exc_type, exc, tb):
                self.out.flush()
                self.err.flush()
                sys.stdout = self.old_stdout
                sys.stderr = self.old_stderr
                self.root.removeHandler(self.handler)
                return False

        return _CaptureContext()

    def _console_event_for(self, run: _Run):
        return lambda message, stream="log": self._console_event(run, message, stream)

    @staticmethod
    def _agent_event(msg) -> dict:
        return {
            "type": "agent",
            "agent": msg.agent,
            "model": getattr(msg, "model", ""),
            "content": msg.content,
            "prompt": getattr(msg, "prompt", ""),
            "tokens_in": msg.tokens_in,
            "tokens_out": msg.tokens_out,
            "cost_usd": msg.cost_usd,
            "seconds": msg.seconds,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def _real_worker(self, *, run: _Run, goal, constraints, settings, corpus_files, use_web):
        def on_event(msg):
            ev = self._agent_event(msg)
            run.transcript.append(ev)
            run.q.put(ev)

        try:
            self._console_event(run, f"Run {run.id} started (real mode), project={run.project}", "system")
            self._console_event(run, f"Goal: {goal}", "system")
            self._console_event(run, f"Corpus files: {len(corpus_files or [])}", "system")
            if corpus_files:
                for path in corpus_files:
                    self._console_event(run, f"Uploaded: {os.path.basename(path)}", "system")
            with self._capture_console(run):
                record = engine.run_deep(
                    goal=goal, constraints=constraints, settings=settings,
                    corpus_files=corpus_files, use_web=use_web, on_event=on_event,
                    pull_messages=run.drain_inbox,
                )
            run.record_id = record.id
            self._console_event(run, f"Run completed, record_id={record.id}", "system")
            self._write_log(record, run)
            run.status = "completed"
            run.q.put({"type": "done", "run_id": record.id,
                       "summary": _run_summary(record)})
        except Exception as exc:  # surface to the UI
            run.status = "error"
            run.error = str(exc)
            self._console_event(run, str(exc), "error")
            run.q.put({"type": "error", "message": str(exc)})

    def _write_log(self, record, run: Optional[_Run] = None) -> None:
        try:
            d = store.run_dir(record.id)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "full_log.txt"), "w", encoding="utf-8") as fh:
                fh.write(format_full_log(record))
            if run is not None:
                with open(os.path.join(d, "console_log.txt"), "w", encoding="utf-8") as fh:
                    fh.write("\n".join(run.console))
        except Exception:
            pass


runs = RunManager()


def _run_summary(record) -> dict:
    return {
        "id": record.id,
        "hypotheses": len(record.assessments),
        "messages": record.metrics.get("messages", 0),
        "cost_usd": record.metrics.get("cost_usd", 0),
        "seconds": record.metrics.get("seconds_wall", record.metrics.get("seconds", 0)),
        "mode": record.mode,
    }


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------


def _settings_from_payload(payload: dict) -> RunSettings:
    agents = {}
    for key, (dm, dt) in DEFAULT_AGENT_MODELS.items():
        cfg = (payload.get("agents") or {}).get(key, {})
        agents[key] = AgentLLMSettings(
            model=cfg.get("model", dm),
            temperature=float(cfg.get("temperature", dt)),
            thinking=cfg.get("thinking", "default"),
        )
    w = payload.get("weights") or {}
    from coscientist.hypothesis_assessment import AssessmentWeights

    weights = AssessmentWeights(
        novelty=float(w.get("novelty", 0.25)), feasibility=float(w.get("feasibility", 0.25)),
        impact=float(w.get("impact", 0.30)), risk=float(w.get("risk", 0.20)),
    )
    return RunSettings(
        project=payload.get("project", "Новый проект"),
        num_hypotheses=int(payload.get("num_hypotheses", 4)),
        max_hypotheses=int(payload.get("max_hypotheses", 0)),
        retrieval_k=int(payload.get("retrieval_k", 6)),
        weights=weights,
        agents=agents,
        web_retriever="routerai",
        web_search_model=payload.get("web_search_model", "google/gemini-2.5-flash"),
        auto_elicit_constraints=bool(payload.get("auto_elicit_constraints", True)),
        lite=bool(payload.get("lite", True)),
    )


def _default_settings_payload() -> dict:
    return {
        "num_hypotheses": 4,
        "max_hypotheses": 12,
        "retrieval_k": 6,
        "weights": {"novelty": 0.25, "feasibility": 0.25, "impact": 0.30, "risk": 0.20},
        "agents": {
            key: {"model": m, "temperature": t, "thinking": "default"}
            for key, (m, t) in DEFAULT_AGENT_MODELS.items()
        },
        "web_retriever": "routerai",
        "web_search_model": "google/gemini-2.5-flash",
        "auto_elicit_constraints": True,
        "lite": True,
        "model_options": MODEL_OPTIONS,
        "thinking_options": THINKING_OPTIONS,
        "agent_labels": AGENT_LABELS,
        "phases": [{"key": k, "label": lbl} for k, lbl in PHASES],
    }


# ---------------------------------------------------------------------------
# App + routes
# ---------------------------------------------------------------------------

app = FastAPI(title="Фабрика гипотез")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


@app.get("/api/health")
def health():
    key = os.environ.get
    return {
        "routerai": bool(key("ROUTER_AI_API_KEY")),
        "embeddings": bool(
            key("ROUTER_AI_API_KEY")
            or key("ROUTER_AI_EMBEDDING_API_KEY")
            or key("COSCIENTIST_EMBEDDING_API_KEY")
        ),
    }


def _is_upload_part(value: Any) -> bool:
    """True for FastAPI/Starlette multipart file parts."""
    return bool(getattr(value, "filename", None)) and callable(
        getattr(value, "read", None)
    )


@app.get("/api/settings")
def get_settings():
    return _default_settings_payload()


@app.get("/api/projects")
def list_projects():
    projs = projects.list() or [projects.ensure_default()]
    all_runs = store.list_runs()
    counts: dict[str, int] = {}
    for r in all_runs:
        counts[r.get("project", "")] = counts.get(r.get("project", ""), 0) + 1
    for p in projs:
        p["run_count"] = counts.get(p["name"], 0)
    return projs


@app.post("/api/projects")
async def create_project(request: Request):
    body = await request.json()
    return projects.create(body.get("name", ""))


@app.get("/api/projects/{project_id}/runs")
def project_runs(project_id: str):
    proj = projects.get(project_id)
    if not proj:
        raise HTTPException(404, "project not found")
    return [r for r in store.list_runs() if r.get("project") == proj["name"]]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    try:
        record = store.load_run(run_id)
    except (FileNotFoundError, OSError):
        raise HTTPException(404, "run not found")
    return JSONResponse(json.loads(record.model_dump_json()))


@app.get("/api/runs/{run_id}/log", response_class=PlainTextResponse)
def get_run_log(run_id: str):
    try:
        record = store.load_run(run_id)
    except (FileNotFoundError, OSError):
        raise HTTPException(404, "run not found")
    return format_full_log(record)


@app.get("/api/runs/{run_id}/console-log", response_class=PlainTextResponse)
def get_run_console_log(run_id: str):
    path = os.path.join(store.run_dir(run_id), "console_log.txt")
    if not os.path.exists(path):
        raise HTTPException(404, "console log not found")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Expert refinement: feedback / chat / branch (feature 3)
# ---------------------------------------------------------------------------


def _load_record(run_id: str):
    try:
        return store.load_run(run_id)
    except (FileNotFoundError, OSError):
        raise HTTPException(404, "run not found")


def _run_retriever(run_id: str):
    """Reload the run's persisted corpus index for grounded refinement (or None)."""
    base = os.path.join(store.run_dir(run_id), "corpus")
    if os.path.exists(base + ".json"):
        try:
            from coscientist.corpus.retrieval import CorpusRetriever

            return CorpusRetriever.from_path(base)
        except Exception:
            return None
    return None


def _source_chunk(run_id: str, chunk_id: str) -> dict:
    """Return a persisted corpus chunk for source-evidence drill-down."""
    base = os.path.join(store.run_dir(run_id), "corpus")
    if not os.path.exists(base + ".json"):
        raise HTTPException(404, "run corpus not found")
    try:
        from coscientist.corpus import citations
        from coscientist.corpus.store import CorpusIndex, RetrievedChunk

        index = CorpusIndex.load(base)
    except Exception:
        raise HTTPException(404, "run corpus not readable")

    for chunk in index.chunks:
        if chunk.chunk_id == chunk_id:
            hit = RetrievedChunk(chunk=chunk, score=0.0, rank=1)
            return {
                "chunk_id": chunk.chunk_id,
                "source_name": chunk.source_name,
                "locator": chunk.locator,
                "modality": chunk.modality,
                "citation": citations.format_reference(hit, 1),
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
    raise HTTPException(404, "source chunk not found")


def _refine_llm():
    from coscientist.model_factory import get_chat_model

    return get_chat_model("google/gemini-2.5-pro", temperature=0.3)


def _feedback_path(run_id: str) -> str:
    return os.path.join(store.run_dir(run_id), "feedback.json")


@app.get("/api/runs/{run_id}/sources/{chunk_id}")
def get_run_source_chunk(run_id: str, chunk_id: str):
    return _source_chunk(run_id, chunk_id)


@app.get("/api/runs/{run_id}/feedback")
def get_feedback(run_id: str):
    from coscientist.feedback import FeedbackStore

    path = _feedback_path(run_id)
    if not os.path.exists(path):
        return {"records": [], "counts": {"confirmed": 0, "refuted": 0, "inconclusive": 0}}
    fb = FeedbackStore.load(path)
    return {"records": [r.model_dump() for r in fb.records], "counts": fb.counts()}


@app.post("/api/runs/{run_id}/feedback")
async def add_feedback(run_id: str, request: Request):
    from coscientist.feedback import FeedbackStore

    body = await request.json()
    outcome = body.get("outcome", "inconclusive")
    if outcome not in ("confirmed", "refuted", "inconclusive"):
        raise HTTPException(400, "invalid outcome")
    path = _feedback_path(run_id)
    fb = FeedbackStore.load(path) if os.path.exists(path) else FeedbackStore()
    fb.add(body.get("hypothesis", ""), outcome, note=body.get("note", ""))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fb.save(path)
    return {"records": [r.model_dump() for r in fb.records], "counts": fb.counts()}


@app.post("/api/runs/{run_id}/chat")
async def chat(run_id: str, request: Request):
    from coscientist.refinement import chat_about_hypothesis

    body = await request.json()
    rec = _load_record(run_id)
    assessment = _find_assessment(rec, body.get("hypothesis", ""))
    reply = chat_about_hypothesis(
        hypothesis=body.get("hypothesis", ""),
        assessment=assessment, goal=rec.goal, message=body.get("message", ""),
        history=body.get("history", []), retriever=_run_retriever(run_id), llm=_refine_llm(),
    )
    return {"reply": reply}


@app.post("/api/runs/{run_id}/branch")
async def branch(run_id: str, request: Request):
    from coscientist.refinement import branch_hypotheses

    body = await request.json()
    rec = _load_record(run_id)
    news = branch_hypotheses(
        hypothesis=body.get("hypothesis", ""), direction=body.get("direction", ""),
        goal=rec.goal, constraints=rec.effective_constraints or rec.constraints,
        retriever=_run_retriever(run_id), llm=_refine_llm(),
        n=int(body.get("n", 2)),
    )
    return {"hypotheses": [a.model_dump() for a in news]}


def _find_assessment(rec, hypothesis: str) -> dict:
    for a in rec.assessments:
        if a.get("hypothesis") == hypothesis:
            return a
    return {}


@app.post("/api/runs")
async def start_run(request: Request):
    form = await request.form()
    payload = json.loads(form.get("settings", "{}"))
    goal = form.get("goal", "").strip()
    constraints = form.get("constraints", "").strip()
    project = form.get("project", "").strip() or "Новый проект"
    use_web = str(form.get("use_web", "true")).lower() in ("1", "true", "yes")
    if not goal:
        raise HTTPException(400, "goal is required")

    payload["project"] = project
    settings = _settings_from_payload(payload)

    # Optional corpus uploads.
    corpus_files = None
    uploads = [v for v in form.multi_items() if _is_upload_part(v[1])]
    if uploads:
        import tempfile

        tmp = tempfile.mkdtemp(prefix="webapp_corpus_")
        corpus_files = []
        for _name, uf in uploads:
            safe_name = os.path.basename(uf.filename or "upload.bin")
            dest = os.path.join(tmp, safe_name)
            with open(dest, "wb") as fh:
                fh.write(await uf.read())
            corpus_files.append(dest)

    run_id = runs.start(project=project, goal=goal, constraints=constraints,
                        settings=settings, corpus_files=corpus_files,
                        use_web=use_web)
    return {"run_id": run_id}


@app.post("/api/runs/{run_id}/message")
async def send_run_message(run_id: str, request: Request):
    """
    Live steering: queue a message for a RUNNING run's agents. It is folded into
    the agents' context at the next supervisor-loop boundary (no restart), and
    echoed into the SSE stream so it shows in the feed immediately.
    """
    run = runs.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    if run.status != "running":
        raise HTTPException(409, "run is not active")
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text is required")
    run.add_message(text)
    ev = {
        "type": "user_message",
        "text": text,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    run.transcript.append(ev)
    run.q.put(ev)
    run.console.append(f"[{ev['ts']}] user_message: {text}")
    return {"ok": True}


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str):
    run = runs.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")

    async def gen():
        yield _sse({"type": "hello", "run_id": run_id})
        while True:
            try:
                ev = run.q.get_nowait()
            except queue.Empty:
                if run.status in ("completed", "error") and run.q.empty():
                    break
                await asyncio.sleep(0.15)
                continue
            yield _sse(ev)
            if ev.get("type") in ("done", "error"):
                break

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# Static assets (js/css/fonts) served under /static.
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main():
    import uvicorn

    port = int(os.environ.get("PORT", "8800"))
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
