# -*- coding: utf-8 -*-

"""
Studio service layer — the engine behind the unified Streamlit interface.

Runs a self-contained hypothesis pipeline (generate → assess → rank → export),
records the inter-agent transcript with token/cost/time metrics, and persists
each run as a browsable "project". No Streamlit import here, and every model is
injectable (``framework_factory`` / ``llm_factory``), so the whole thing is
unit-testable offline with fakes and no API keys.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from coscientist.hypothesis_assessment import (
    AssessmentWeights,
    HypothesisAssessment,
)

# ---------------------------------------------------------------------------
# Cost / token estimation
# ---------------------------------------------------------------------------

# Rough USD per 1K tokens (input, output). Estimates only — for a live "cost
# awareness" readout, not billing. Unknown models fall back to DEFAULT_PRICE.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "google/gemini-2.5-pro": (0.00125, 0.010),
    "google/gemini-2.5-flash": (0.0003, 0.0025),
    "openai/o3": (0.010, 0.040),
    "openai/o4-mini": (0.0011, 0.0044),
    "anthropic/claude-sonnet-4": (0.003, 0.015),
    "claude-sonnet-4-20250514": (0.003, 0.015),
}
DEFAULT_PRICE = (0.002, 0.008)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token)."""
    return max(1, len(text or "") // 4)


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = PRICE_TABLE.get(model, DEFAULT_PRICE)
    return round(tokens_in / 1000 * price_in + tokens_out / 1000 * price_out, 6)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class TranscriptMessage(BaseModel):
    agent: str
    role: str
    content: str
    prompt: str = ""  # full request sent to the model (for the complete log)
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0


class AgentLLMSettings(BaseModel):
    """Per-agent model + temperature + thinking level (deep mode)."""

    model: str = "google/gemini-2.5-pro"
    temperature: float = 0.4
    # Thinking/reasoning depth: "default" (provider default), "off", "low",
    # "medium", "high". Safely ignored by RouterAI if the model can't honor it.
    thinking: str = "default"


# LITE mode knobs (fast, cheap test runs — the default).
LITE_MODEL = "google/gemini-2.5-flash"
LITE_NUM_HYPOTHESES = 4
LITE_MAX_HYPOTHESES = 8
LITE_LITERATURE_SUBTOPICS = 2   # vs 5 in full mode
LITE_MAX_ASSUMPTIONS = 3        # research only the top-K assumptions per hypothesis
FULL_LITERATURE_SUBTOPICS = 5

# Deep-mode agent keys -> (default model, default temperature). "assessment" is
# deliberately distinct from the fast-mode "assessor" role to avoid collision.
DEFAULT_AGENT_MODELS: dict[str, tuple[str, float]] = {
    "literature": ("google/gemini-2.5-pro", 0.4),
    "generation": ("google/gemini-2.5-pro", 1.0),
    "reflection": ("google/gemini-2.5-pro", 0.3),
    "ranking": ("google/gemini-2.5-flash", 0.3),
    "evolution": ("google/gemini-2.5-pro", 1.0),
    "meta_review": ("google/gemini-2.5-flash", 0.4),
    "supervisor": ("google/gemini-2.5-pro", 0.2),
    "final_report": ("google/gemini-2.5-pro", 0.3),
    "assessment": ("google/gemini-2.5-pro", 0.3),
}

# Human-readable labels for the UI, in display order.
AGENT_LABELS: dict[str, str] = {
    "literature": "📚 Literature (обзор литературы)",
    "generation": "🧠 Generation (генерация гипотез)",
    "reflection": "🔎 Reflection (deep verification)",
    "ranking": "🎯 Ranking (Elo-турнир)",
    "evolution": "🧬 Evolution (улучшение гипотез)",
    "meta_review": "🧭 Meta-review (синтез + фидбэк)",
    "supervisor": "🎛 Supervisor (оркестрация)",
    "final_report": "📄 Final report (итоговый отчёт)",
    "assessment": "🔬 Assessor (оценка гипотез)",
}


def _default_agents() -> dict[str, AgentLLMSettings]:
    return {
        key: AgentLLMSettings(model=model, temperature=temp)
        for key, (model, temp) in DEFAULT_AGENT_MODELS.items()
    }


class RunSettings(BaseModel):
    project: str = "Новый проект"
    num_hypotheses: int = 4
    # Hard cap on total hypotheses the run may accumulate (0 = unlimited). Once
    # reached, the supervisor can no longer generate/evolve — the run converges.
    max_hypotheses: int = 0
    generator_model: str = "google/gemini-2.5-pro"
    generator_temperature: float = 1.0
    assessor_model: str = "google/gemini-2.5-pro"
    assessor_temperature: float = 0.3
    retrieval_k: int = 6
    weights: AssessmentWeights = Field(default_factory=AssessmentWeights)
    # Per-agent models for deep mode (keys from DEFAULT_AGENT_MODELS).
    agents: dict[str, AgentLLMSettings] = Field(default_factory=_default_agents)
    # Web-search backend for deep-mode research. Only RouterAI web-search is
    # supported (it tolerates the long deep-verification queries and needs no
    # extra key); the field is kept for backward-compatible deserialization of
    # older runs and is otherwise ignored.
    web_retriever: str = "routerai"
    web_search_model: str = "google/gemini-2.5-flash"
    # Auto-elicit missing-but-relevant constraints before generation (killer feature).
    auto_elicit_constraints: bool = True
    # LITE mode (default): fast, cheap test run — every agent on gemini-2.5-flash
    # with reasoning off, reduced web-search constants, top-3 assumptions, fixed
    # 4 hypotheses / 8 max. Full mode (lite=False) keeps the configured models,
    # reasoning, hypotheses/max and full web-search fan-out.
    lite: bool = True

    def lite_overrides(self) -> "RunSettings":
        """Return an effective copy with LITE-mode overrides applied."""
        s = self.model_copy(deep=True)
        s.num_hypotheses = LITE_NUM_HYPOTHESES
        s.max_hypotheses = LITE_MAX_HYPOTHESES
        s.generator_model = LITE_MODEL
        s.assessor_model = LITE_MODEL
        s.web_search_model = LITE_MODEL
        s.agents = {
            key: AgentLLMSettings(model=LITE_MODEL, temperature=temp, thinking="off")
            for key, (_, temp) in DEFAULT_AGENT_MODELS.items()
        }
        return s

    def agent_model_temp(self, key: str) -> tuple[str, float]:
        """Resolve (model, temperature) for an agent/role key."""
        if key in self.agents:
            a = self.agents[key]
            return a.model, a.temperature
        if key == "generator":
            return self.generator_model, self.generator_temperature
        return self.assessor_model, self.assessor_temperature

    def agent_thinking(self, key: str) -> str:
        """Resolve the thinking/reasoning level for an agent/role key."""
        if key in self.agents:
            return self.agents[key].thinking
        return "default"


class RunRecord(BaseModel):
    id: str
    project: str
    goal: str
    constraints: str = ""
    created_at: str
    settings: RunSettings
    hypotheses: list[str] = Field(default_factory=list)
    assessments: list[dict] = Field(default_factory=list)
    transcript: list[TranscriptMessage] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    status: str = "completed"
    # Auto-elicited constraints (killer feature) + the merged effective constraints.
    elicited_constraints: list[dict] = Field(default_factory=list)
    effective_constraints: str = ""
    # Live steering messages the expert sent during the run (chat-to-agents).
    user_guidance: list[str] = Field(default_factory=list)
    # Deep-mode extras (fast runs leave these empty). Backward compatible.
    mode: str = "fast"  # "fast" | "deep"
    final_report: str = ""
    meta_review: str = ""
    tournament_summary: dict = Field(default_factory=dict)

    def assessment_objects(self) -> list[HypothesisAssessment]:
        return [HypothesisAssessment(**a) for a in self.assessments]


# ---------------------------------------------------------------------------
# Recording proxy: wraps any chat model to log each .invoke into a transcript
# ---------------------------------------------------------------------------


class _RecordingLLM:
    """
    Transparent proxy that records each ``.invoke`` into a shared transcript.

    Any other attribute/method is delegated to the wrapped model via
    ``__getattr__`` so the proxy can stand in for a full chat model wherever a
    framework agent needs one (the agents call ``.invoke`` synchronously).
    """

    def __init__(self, llm: Any, agent: str, model_name: str, sink: list[TranscriptMessage],
                 on_message: Optional[Callable[[TranscriptMessage], None]] = None):
        self._llm = llm
        self._agent = agent
        self._model = model_name
        self._sink = sink
        self._on_message = on_message

    def invoke(self, prompt: Any) -> Any:
        started = time.perf_counter()
        response = self._llm.invoke(prompt)
        elapsed = time.perf_counter() - started
        content = _content_str(response)
        prompt_str = _content_str(prompt)
        t_in = estimate_tokens(prompt_str)
        t_out = estimate_tokens(content)
        msg = TranscriptMessage(
            agent=self._agent,
            role=self._agent,
            content=content,
            prompt=prompt_str,
            model=self._model,
            tokens_in=t_in,
            tokens_out=t_out,
            cost_usd=estimate_cost(self._model, t_in, t_out),
            seconds=round(elapsed, 3),
        )
        self._sink.append(msg)
        if self._on_message is not None:
            try:
                self._on_message(msg)
            except Exception:
                pass  # never let a UI callback break the run
        return response

    def __getattr__(self, name: str) -> Any:
        # Delegate everything we do not explicitly wrap to the real model.
        # Guard the proxy's own attributes to avoid recursion before init.
        if name in ("_llm", "_agent", "_model", "_sink", "_on_message"):
            raise AttributeError(name)
        return getattr(self._llm, name)


def _content_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    content = getattr(x, "content", x)
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _record_manual_event(
    *,
    transcript: list[TranscriptMessage],
    agent: str,
    content: str,
    prompt: str = "",
    model: str = "system",
    on_event: Optional[Callable[[TranscriptMessage], None]] = None,
) -> TranscriptMessage:
    """Record a non-LLM pipeline event in the same stream as agent calls."""
    msg = TranscriptMessage(
        agent=agent,
        role=agent,
        content=content or "",
        prompt=prompt or "",
        model=model,
        tokens_in=estimate_tokens(prompt or ""),
        tokens_out=estimate_tokens(content or ""),
        cost_usd=0.0,
        seconds=0.0,
    )
    transcript.append(msg)
    if on_event is not None:
        try:
            on_event(msg)
        except Exception:
            pass
    return msg


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _studio_dir() -> str:
    base = os.environ.get("COSCIENTIST_DIR", os.path.expanduser("~/.coscientist"))
    return os.path.join(base, "studio")


class StudioStore:
    """Persists runs under ``COSCIENTIST_DIR/studio/<id>/`` and lists history."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or _studio_dir()
        os.makedirs(self.base_dir, exist_ok=True)

    def run_dir(self, run_id: str) -> str:
        return os.path.join(self.base_dir, run_id)

    def save(self, record: RunRecord) -> str:
        d = self.run_dir(record.id)
        os.makedirs(os.path.join(d, "exports"), exist_ok=True)
        with open(os.path.join(d, "record.json"), "w", encoding="utf-8") as fh:
            fh.write(record.model_dump_json(indent=2))
        meta = {
            "id": record.id,
            "project": record.project,
            "goal": record.goal,
            "created_at": record.created_at,
            "num_hypotheses": len(record.assessments),
            "status": record.status,
        }
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        return d

    def list_runs(self) -> list[dict]:
        runs = []
        if not os.path.isdir(self.base_dir):
            return runs
        for name in os.listdir(self.base_dir):
            meta_path = os.path.join(self.base_dir, name, "meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, encoding="utf-8") as fh:
                        runs.append(json.load(fh))
                except (OSError, json.JSONDecodeError):
                    continue
        runs.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return runs

    def load_run(self, run_id: str) -> RunRecord:
        with open(os.path.join(self.run_dir(run_id), "record.json"), encoding="utf-8") as fh:
            return RunRecord.model_validate_json(fh.read())


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

LLMFactory = Callable[[str, RunSettings], Any]

class StudioEngine:
    """Runs and persists Studio hypothesis-generation runs."""

    def __init__(self, store: Optional[StudioStore] = None):
        self.store = store or StudioStore()

    def _resolve_llm(self, role: str, settings: RunSettings, llm_factory: Optional[LLMFactory]) -> Any:
        if llm_factory is not None:
            return llm_factory(role, settings)
        from coscientist.model_factory import get_chat_model

        model, temp = settings.agent_model_temp(role)
        return get_chat_model(model, temperature=temp, reasoning=settings.agent_thinking(role))

    def _maybe_build_retriever(self, settings, retriever, corpus_files, embed_fn):
        if retriever is not None or not corpus_files:
            return retriever
        from coscientist.corpus.build import build_corpus_index
        from coscientist.corpus.retrieval import CorpusRetriever

        index = build_corpus_index(
            data_dir="",
            index_path="",
            include_images=True,
            embed_fn=embed_fn,
            files=list(corpus_files),
        )
        return CorpusRetriever(index, default_k=settings.retrieval_k)


    def run_deep(
        self,
        goal: str,
        constraints: str = "",
        settings: Optional[RunSettings] = None,
        *,
        retriever: Any = None,
        corpus_files: Optional[list[str]] = None,
        embed_fn: Optional[Callable] = None,
        llm_factory: Optional[LLMFactory] = None,
        framework_factory: Optional[Callable] = None,
        use_web: bool = True,
        on_event: Optional[Callable[[TranscriptMessage], None]] = None,
        pull_messages: Optional[Callable[[], list[str]]] = None,
    ) -> RunRecord:
        """
        Run the FULL multi-agent tournament (CoscientistFramework) and record it
        as a Studio run. Grounds every agent in the private corpus; optionally
        uses RouterAI web research (``use_web``). ``framework_factory`` is
        injectable so this is unit-testable offline without real LLMs/web search.

        ``on_event`` is called with each :class:`TranscriptMessage` as agents run,
        enabling live progress display in the UI.
        """
        import asyncio

        from coscientist.framework import CoscientistConfig, CoscientistFramework
        from coscientist.global_state import (
            CoscientistState,
            CoscientistStateManager,
        )

        settings = settings or RunSettings()
        # LITE (default): fast/cheap run — flash everywhere, reasoning off, fixed
        # 4/8, reduced web fan-out. Full mode keeps the configured settings.
        if settings.lite:
            settings = settings.lite_overrides()
        # Point gpt_researcher at the reduced (lite) or full researcher config.
        os.environ["COSCIENTIST_RESEARCHER_CONFIG"] = _researcher_config_file(settings.lite)

        transcript: list[TranscriptMessage] = []
        retriever = self._maybe_build_retriever(settings, retriever, corpus_files, embed_fn)

        # Point gpt_researcher (literature review + deep verification) at
        # RouterAI web search — it tolerates the long deep-verification queries
        # and needs no separate search key.
        if use_web:
            _configure_web_retriever(settings)

        # Wrap every agent LLM (by its own per-agent key) so all inter-agent
        # calls are recorded and each agent can run its own model/temperature.
        def _wrap(key: str, agent_label: str) -> Any:
            base = self._resolve_llm(key, settings, llm_factory)
            model, _ = settings.agent_model_temp(key)
            return _RecordingLLM(base, agent=agent_label, model_name=model,
                                 sink=transcript, on_message=on_event)

        # Killer feature: infer missing-but-relevant constraints BEFORE generation,
        # fold the assumed ones into the effective constraints, and stream it.
        elicited: list = []
        effective_constraints = constraints
        if settings.auto_elicit_constraints:
            from coscientist.constraint_elicitation import (
                elicit_constraints,
                merge_constraints,
            )

            elicited = elicit_constraints(
                goal, constraints, retriever,
                _wrap("assessment", "Constraints"),
            )
            effective_constraints = merge_constraints(constraints, elicited)

        config = CoscientistConfig(
            literature_review_agent_llm=_wrap("literature", "Literature"),
            generation_agent_llms={"studio": _wrap("generation", "Generation")},
            reflection_agent_llms={"studio": _wrap("reflection", "Reflection")},
            evolution_agent_llms={"studio": _wrap("evolution", "Evolution")},
            meta_review_agent_llm=_wrap("meta_review", "Meta-review"),
            ranking_agent_llm=_wrap("ranking", "Ranking"),
            supervisor_agent_llm=_wrap("supervisor", "Supervisor"),
            final_report_agent_llm=_wrap("final_report", "Final report"),
            # Inert: ProximityGraph calls model_factory.get_embeddings() directly,
            # so this placeholder just avoids building a client at construction.
            proximity_agent_embedding_model=object(),
            retriever=retriever,
            assessment_llm=_wrap("assessment", "Assessor"),
            assessment_weights=settings.weights,
            # Clamp the cap to be at least the seed count so start() never
            # exceeds it before the loop even begins (0 stays unlimited).
            max_total_hypotheses=(
                max(settings.max_hypotheses, settings.num_hypotheses)
                if settings.max_hypotheses and settings.max_hypotheses > 0
                else 0
            ),
            web_research_enabled=use_web,
            pull_messages=pull_messages,
            # LITE trims literature breadth and researches only the top-K
            # assumptions per hypothesis; full mode keeps the wider fan-out.
            literature_subtopics=(
                LITE_LITERATURE_SUBTOPICS if settings.lite else FULL_LITERATURE_SUBTOPICS
            ),
            max_assumptions_researched=(LITE_MAX_ASSUMPTIONS if settings.lite else 0),
        )

        # Fresh state for this goal (clear any stale directory).
        try:
            CoscientistState.clear_goal_directory(goal)
        except Exception:
            pass
        state = CoscientistState(goal=goal, constraints=effective_constraints)
        manager = CoscientistStateManager(state)

        # Corpus-only grounding when web research is disabled: seed the
        # literature review from the private corpus so start() skips the web.
        if not use_web:
            from coscientist.corpus.literature import build_corpus_literature_review

            literature_state = build_corpus_literature_review(
                goal=goal, retriever=retriever, k=settings.retrieval_k
            )
            manager.update_literature_review(literature_state)
            _record_manual_event(
                transcript=transcript,
                agent="Literature",
                model="corpus-only",
                prompt=(
                    "Build a literature/context review from the private corpus "
                    "because web research is disabled."
                ),
                content=literature_state["subtopic_reports"][0],
                on_event=on_event,
            )

        if framework_factory is not None:
            framework = framework_factory(config, manager)
        else:
            framework = CoscientistFramework(config, manager)

        started = time.perf_counter()
        final_report, meta_review = asyncio.run(
            framework.run(n_hypotheses=settings.num_hypotheses)
        )
        wall = round(time.perf_counter() - started, 3)

        assessments = manager.assessments  # HypothesisAssessment objects (Phase 5)
        tournament_summary = _safe_tournament_summary(manager)

        metrics = _totals(transcript)
        metrics["seconds_wall"] = wall
        metrics.update(tournament_summary)

        record = RunRecord(
            id=_new_run_id(),
            project=settings.project,
            goal=goal,
            constraints=constraints,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            settings=settings,
            hypotheses=[a.hypothesis for a in assessments],
            assessments=[a.model_dump() for a in assessments],
            transcript=transcript,
            metrics=metrics,
            status="completed",
            mode="deep",
            elicited_constraints=[c.model_dump() for c in elicited],
            effective_constraints=effective_constraints,
            final_report=final_report or "",
            meta_review=meta_review or "",
            tournament_summary=tournament_summary,
            user_guidance=list(getattr(state, "user_guidance", []) or []),
        )
        self.store.save(record)
        # Persist the corpus index so expert refinement (chat / branch) can
        # reload it later and stay grounded in the same evidence.
        if retriever is not None:
            try:
                retriever.index.save(os.path.join(self.store.run_dir(record.id), "corpus"))
            except Exception:
                pass
        return record

    def export_run(self, record: RunRecord, out_dir: str) -> dict[str, str]:
        """Write report + tasks + graph for a run's assessments."""
        from coscientist import viz
        from coscientist.export import (
            assessments_to_csv,
            assessments_to_jira,
            assessments_to_json,
            write_report,
        )

        assessments = record.assessment_objects()
        os.makedirs(out_dir, exist_ok=True)
        paths = write_report(assessments, out_dir, goal=record.goal)

        def _w(name: str, text: str) -> str:
            p = os.path.join(out_dir, name)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text)
            return p

        paths["csv"] = _w("tasks.csv", assessments_to_csv(assessments))
        paths["json"] = _w("tasks.json", assessments_to_json(assessments))
        paths["jira"] = _w(
            "jira_tasks.json",
            json.dumps(assessments_to_jira(assessments), ensure_ascii=False, indent=2),
        )
        paths["graph"] = _w("graph.html", viz.to_html(assessments))
        return paths


def _researcher_config_file(lite: bool) -> str:
    """Absolute path to the GPTResearcher config for the run's mode."""
    name = "researcher_config_lite.json" if lite else "researcher_config.json"
    return os.path.join(os.path.dirname(__file__), name)


def _configure_web_retriever(settings: RunSettings) -> None:
    """
    Point GPT Researcher at RouterAI web search via env + factory hooks.

    RouterAI web search is the only supported backend: it tolerates the long,
    multi-sentence deep-verification queries (which a dedicated search API like
    Tavily rejects) and needs no separate key. Any legacy ``web_retriever`` value
    on an old run is ignored.
    """
    os.environ["RETRIEVER"] = "routerai"
    os.environ["COSCIENTIST_WEBSEARCH_MODEL"] = (
        settings.web_search_model or "google/gemini-2.5-flash"
    )
    try:
        from coscientist.web_search import register_routerai_gpt_researcher

        register_routerai_gpt_researcher()
    except Exception:
        pass


def _safe_tournament_summary(manager: Any) -> dict:
    """Best-effort tournament stats from the state (empty on any failure)."""
    try:
        stats = manager.summarize_tournament_trajectory()
        return {
            "hypotheses": manager.num_tournament_hypotheses,
            "matches_played": stats.get("total_matches_played", 0),
            "rounds_played": stats.get("total_rounds_played", 0),
            "max_elo": stats.get("max_elo_rating", ""),
        }
    except Exception:
        return {}


def format_full_log(record: "RunRecord") -> str:
    """
    Render the complete request/response log of a run as readable text — every
    model call with its agent, model, full prompt and full response, plus
    tokens/cost/time. Suitable to study or hand to another agent.
    """
    lines = [
        "=" * 100,
        f"RUN {record.id}  ·  project: {record.project}  ·  mode: {record.mode}",
        f"goal: {record.goal}",
        f"constraints: {record.constraints}",
        f"created: {record.created_at}",
        f"metrics: {record.metrics}",
        "=" * 100,
        "",
    ]
    for i, m in enumerate(record.transcript, start=1):
        lines += [
            f"\n{'─' * 100}",
            f"[{i}] AGENT: {m.agent}   MODEL: {m.model or '—'}   "
            f"tokens {m.tokens_in}->{m.tokens_out}   ${m.cost_usd}   {m.seconds}s",
            f"{'─' * 100}",
            "── PROMPT ──",
            m.prompt or "(not captured)",
            "",
            "── RESPONSE ──",
            m.content or "(empty)",
        ]
    if record.final_report:
        lines += ["\n" + "=" * 100, "FINAL REPORT", "=" * 100, record.final_report]
    if record.meta_review:
        lines += ["\n" + "=" * 100, "META-REVIEW", "=" * 100, record.meta_review]
    return "\n".join(lines)


def _totals(transcript: list[TranscriptMessage]) -> dict:
    return {
        "messages": len(transcript),
        "tokens_in": sum(m.tokens_in for m in transcript),
        "tokens_out": sum(m.tokens_out for m in transcript),
        "cost_usd": round(sum(m.cost_usd for m in transcript), 6),
        "seconds": round(sum(m.seconds for m in transcript), 3),
    }


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"
