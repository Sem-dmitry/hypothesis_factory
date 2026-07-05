"""
Фабрика гипотез — unified Studio.

A single Streamlit interface. Launch with:

    streamlit run app/studio.py

Everything is here: create projects/runs and browse their history, enter the
goal + constraints and upload knowledge files, configure each agent's model and
temperature, launch generation, and see the ranked hypotheses, the inter-agent
transcript, the messages/cost/time summary, and the knowledge graph — no
separate scripts. Every run is the full multi-agent tournament (requires API keys).
"""

import os
import sys
import tempfile

# Make the repo root importable so `streamlit run app/studio.py` works from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Auto-load credentials from a .env / env file so users don't have to export
# variables by hand (real environment variables still take precedence).
from coscientist.env_utils import (
    apply_llm_runtime_defaults,
    bridge_provider_aliases,
    load_env_file,
    silence_optional_warnings,
)

_LOADED_ENV_VARS = load_env_file()
# Deep-mode web research (gpt_researcher) reads OPENAI_API_KEY for its embeddings;
# alias our COSCIENTIST_EMBEDDING_* onto the OpenAI-named vars so it just works.
bridge_provider_aliases()
# Give long research LLM calls room (avoid the aggressive 120s stream timeout).
apply_llm_runtime_defaults()
# Quiet the harmless "Failed to import MCPRetriever" warning (we don't use MCP).
silence_optional_warnings()

import threading

import streamlit as st
import streamlit.components.v1 as components

# Used to attach the Streamlit script context to worker threads (gpt_researcher's
# thread pool) so live-progress updates from those threads don't emit
# "missing ScriptRunContext!" warnings. Guarded: internal API across versions.
try:
    from streamlit.runtime.scriptrunner import (
        add_script_run_ctx as _add_ctx,
        get_script_run_ctx as _get_ctx,
    )
except Exception:  # pragma: no cover - streamlit internals moved
    _add_ctx = _get_ctx = None

from coscientist import viz
from coscientist.export import (
    assessments_to_csv,
    assessments_to_jira,
    assessments_to_json,
    render_html,
    render_markdown,
)
from coscientist.hypothesis_assessment import AssessmentWeights
from coscientist.studio import (
    AGENT_LABELS,
    DEFAULT_AGENT_MODELS,
    AgentLLMSettings,
    RunSettings,
    StudioEngine,
    StudioStore,
)

st.set_page_config(page_title="Фабрика гипотез — Studio", page_icon="🧪", layout="wide")

MODEL_OPTIONS = [
    "google/gemini-2.5-pro", "google/gemini-2.5-flash",
    "anthropic/claude-sonnet-4",
]

# Thinking/reasoning depth (RouterAI). Label -> value passed to get_chat_model.
THINKING_OPTIONS = {
    "🧠 по умолчанию": "default",
    "⛔ выкл": "off",
    "🟢 low": "low",
    "🟡 medium": "medium",
    "🔴 high": "high",
}
_THINKING_LABELS = list(THINKING_OPTIONS.keys())


def _model_index(model: str) -> int:
    return MODEL_OPTIONS.index(model) if model in MODEL_OPTIONS else 0


def _thinking_index(value: str) -> int:
    values = list(THINKING_OPTIONS.values())
    return values.index(value) if value in values else 0


store = StudioStore()
engine = StudioEngine(store)

if "current_run_id" not in st.session_state:
    st.session_state.current_run_id = None


# ---------------------------------------------------------------------------
# Sidebar: settings + run history
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🧪 Фабрика гипотез")

    # Show which credentials are visible (auto-loaded from .env/env if present).
    _keys = {
        "RouterAI": os.environ.get("ROUTER_AI_API_KEY"),
        "Embeddings": os.environ.get("COSCIENTIST_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    }
    _status = " ".join(f"{'🟢' if v else '⚪'} {k}" for k, v in _keys.items())
    st.caption(f"Ключи: {_status}")
    if _LOADED_ENV_VARS:
        st.caption(f"↳ загружено из .env: {', '.join(_LOADED_ENV_VARS)}")

    st.caption("Модели и температуры агентов — на вкладке «🛠 Агенты».")
    n_hyp = st.slider("Число гипотез", 2, 8, 4, 1)

    st.subheader("🎚 Веса ранжирования")
    w_nov = st.slider("Новизна", 0.0, 1.0, 0.25, 0.05)
    w_fea = st.slider("Реализуемость", 0.0, 1.0, 0.25, 0.05)
    w_imp = st.slider("Эффект", 0.0, 1.0, 0.30, 0.05)
    w_risk = st.slider("Учёт риска", 0.0, 1.0, 0.20, 0.05)

    st.divider()
    st.subheader("🗂 История запусков")
    runs = store.list_runs()
    if runs:
        labels = {
            f"{r['created_at'][:16]} · {r['project']} · {r['goal'][:32]}": r["id"]
            for r in runs
        }
        chosen = st.radio("Проекты", list(labels.keys()), index=0, label_visibility="collapsed")
        if st.button("Открыть выбранный", use_container_width=True):
            st.session_state.current_run_id = labels[chosen]
    else:
        st.caption("Пока нет запусков — создайте первый справа.")


def _current_record():
    if st.session_state.current_run_id is None:
        return None
    try:
        return store.load_run(st.session_state.current_run_id)
    except (FileNotFoundError, OSError):
        return None


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab_new, tab_agentcfg, tab_hyp, tab_report, tab_agents, tab_metrics, tab_graph = st.tabs(
    ["🚀 Новый запуск", "🛠 Агенты", "🧪 Гипотезы", "📄 Отчёт",
     "💬 Коммуникация", "📊 Метрики", "🕸 Граф"]
)

# --- Agent model/temperature config (rendered before the run button uses it) ---
with tab_agentcfg:
    st.header("Модели и температуры агентов")
    st.caption(
        "Полный турнир использует всех агентов ниже — у каждого своя модель, температура "
        "и режим мышления (thinking). Если модель не поддерживает выбранный режим "
        "(не умеет думать или, наоборот, думает всегда) — ошибки не будет, модель отработает "
        "в своём штатном режиме."
    )
    deep_agents: dict[str, AgentLLMSettings] = {}
    for _key, _label in AGENT_LABELS.items():
        _dm, _dt = DEFAULT_AGENT_MODELS[_key]
        ac1, ac2, ac3 = st.columns([3, 2, 2])
        with ac1:
            _m = st.selectbox(_label, MODEL_OPTIONS, index=_model_index(_dm), key=f"m_{_key}")
        with ac2:
            _t = st.slider("темп.", 0.0, 2.0, _dt, 0.1, key=f"t_{_key}")
        with ac3:
            _think_label = st.selectbox("мышление", _THINKING_LABELS, index=0, key=f"th_{_key}")
        deep_agents[_key] = AgentLLMSettings(
            model=_m, temperature=_t, thinking=THINKING_OPTIONS[_think_label]
        )

    st.divider()
    st.subheader("🌐 Веб-поиск")
    st.caption(
        "Веб-поиск для обзора литературы и deep-верификации — через RouterAI "
        "web-search (переваривает длинные запросы deep-верификации и не требует "
        "отдельного ключа). Применяется, только если на вкладке «🚀 Новый запуск» "
        "включено «Веб-исследование»."
    )
    web_retriever = "routerai"
    web_search_model = st.selectbox(
        "Модель для RouterAI web-search", MODEL_OPTIONS,
        index=_model_index("google/gemini-2.5-flash"), key="web_model",
    )

with tab_new:
    st.header("Новый запуск генерации гипотез")
    project = st.text_input("Название проекта", value="Хвосты флотации")
    goal = st.text_area(
        "Цель / технологическая проблема",
        value="Снизить потери никеля с хвостами флотации без потери качества концентрата",
        height=80,
    )
    constraints = st.text_area(
        "Ограничения (сырьё, оборудование, бюджет, нормативы)",
        value="Действующая схема флотации; без замены основного оборудования",
        height=80,
    )
    uploaded = st.file_uploader(
        "База знаний: PDF / DOCX / XLSX / изображения (опционально)",
        type=["pdf", "docx", "xlsx", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    st.divider()
    use_web = st.checkbox(
        "🌐 Веб-исследование", value=True,
        help="Обзор литературы и верификация через RouterAI web-search; модель "
             "выбирается на вкладке «🛠 Агенты». Без веб-исследования — заземление "
             "только на загруженную базу знаний.",
    )

    if st.button("▶️ Запустить генерацию", type="primary"):
        weights = AssessmentWeights(novelty=w_nov, feasibility=w_fea, impact=w_imp, risk=w_risk)
        settings = RunSettings(
            project=project, num_hypotheses=n_hyp,
            weights=weights, agents=deep_agents,
            web_retriever=web_retriever, web_search_model=web_search_model,
        )
        corpus_files = None
        tmpdir = None
        if uploaded:
            tmpdir = tempfile.mkdtemp(prefix="studio_corpus_")
            corpus_files = []
            for uf in uploaded:
                p = os.path.join(tmpdir, uf.name)
                with open(p, "wb") as fh:
                    fh.write(uf.getbuffer())
                corpus_files.append(p)
        import time as _t

        # Pipeline phases (by transcript agent label) with a friendly title.
        PHASES = [
            ("Literature", "📚 Обзор литературы"),
            ("Generation", "🧠 Генерация гипотез"),
            ("Reflection", "🔎 Рефлексия и верификация"),
            ("Ranking", "🎯 Турнир (ранжирование)"),
            ("Evolution", "🧬 Эволюция гипотез"),
            ("Meta-review", "🧭 Мета-ревью"),
            ("Supervisor", "🎛 Оркестрация"),
            ("Final report", "📄 Финальный отчёт"),
            ("Assessor", "🔬 Оценка гипотез"),
        ]
        _AICON = {k: t.split()[0] for k, t in PHASES}

        prog = st.container(border=True)
        with prog:
            st.markdown("#### ⚙️ Прогресс турнира")
            phase_ph = st.empty()
            metrics_ph = st.empty()
            st.caption("Лента активности агентов")
            feed_ph = st.empty()

        pstate = {"start": _t.perf_counter(), "n": 0, "tok": 0, "cost": 0.0,
                  "seen": set(), "current": None, "feed": []}
        # Capture the script context so agent calls on gpt_researcher worker
        # threads can update the UI without "missing ScriptRunContext!" warnings.
        _ui_ctx = _get_ctx() if _get_ctx else None
        _ui_lock = threading.Lock()

        def _render():
            rows = []
            for label, title in PHASES:
                mark = "🔄" if pstate["current"] == label else (
                    "✅" if label in pstate["seen"] else "⚪")
                rows.append(f"{mark} {title}")
            phase_ph.markdown("  \n".join(rows))
            el = _t.perf_counter() - pstate["start"]
            metrics_ph.markdown(
                f"**Вызовов агентов:** {pstate['n']} &nbsp;·&nbsp; "
                f"**токенов:** ~{pstate['tok']:,} &nbsp;·&nbsp; "
                f"**стоимость:** ~${pstate['cost']:.4f} &nbsp;·&nbsp; "
                f"**время:** {el:.0f}s"
            )
            feed_ph.markdown("\n".join(pstate["feed"][-8:]) or "_ожидание…_")

        def on_event(msg):
            # Agents may run on gpt_researcher worker threads; attach the script
            # context so st.* updates are valid (silences ScriptRunContext warnings).
            if _add_ctx and _ui_ctx is not None:
                try:
                    _add_ctx(threading.current_thread(), _ui_ctx)
                except Exception:
                    pass
            with _ui_lock:
                pstate["n"] += 1
                pstate["tok"] += msg.tokens_in + msg.tokens_out
                pstate["cost"] += msg.cost_usd
                pstate["seen"].add(msg.agent)
                pstate["current"] = msg.agent
                snippet = " ".join((msg.content or "").split())[:90]
                pstate["feed"].append(
                    f"{_AICON.get(msg.agent, '🤖')} **{msg.agent}** · {msg.seconds}s · "
                    f"{msg.tokens_out} ток. — {snippet}…"
                )
                _render()

        _render()
        try:
            record = engine.run_deep(
                goal=goal, constraints=constraints, settings=settings,
                corpus_files=corpus_files, use_web=use_web, on_event=on_event,
            )
            pstate["current"] = None
            _render()
            st.session_state.current_run_id = record.id
            st.success(
                f"✅ Готово! Гипотез: {len(record.assessments)} · "
                f"агентов сработало: {record.metrics.get('messages', 0)} · "
                f"оценка стоимости: ${record.metrics.get('cost_usd', 0)} · "
                f"время: {record.metrics.get('seconds_wall', 0)}s"
            )
        except Exception as exc:  # surface API/key errors in the UI
            st.error(f"Ошибка запуска: {exc}")

    st.caption(
        "Реальный прогон — всегда полный турнир агентов. Нужен ROUTER_AI_API_KEY "
        "(chat, vision, embeddings и веб-исследование идут через RouterAI, "
        "отдельный ключ не нужен)."
    )

record = _current_record()

with tab_hyp:
    st.header("Ранжированные гипотезы")
    if record is None:
        st.info("Запустите генерацию или выберите проект в истории слева.")
    else:
        assessments = record.assessment_objects()
        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇️ Отчёт (MD)", render_markdown(assessments, goal=record.goal),
                           file_name="hypotheses_report.md")
        c2.download_button("⬇️ Отчёт (HTML)", render_html(assessments, goal=record.goal),
                           file_name="hypotheses_report.html")
        c3.download_button("⬇️ Задачи (CSV)", assessments_to_csv(assessments),
                           file_name="hypotheses.csv")
        c1.download_button("⬇️ JSON", assessments_to_json(assessments),
                           file_name="hypotheses.json")
        import json as _json
        c2.download_button("⬇️ Jira", _json.dumps(assessments_to_jira(assessments),
                           ensure_ascii=False, indent=2), file_name="jira_tasks.json")

        for i, a in enumerate(assessments, start=1):
            with st.expander(f"{i}. {a.hypothesis}  —  оценка {a.overall_score}", expanded=(i == 1)):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Новизна", a.novelty_score)
                m2.metric("Реализуемость", a.feasibility_score)
                m3.metric("Эффект", a.impact_score)
                m4.metric("Риск", a.risk_level)
                st.markdown(f"**Обоснование:** {a.justification}")
                st.markdown(f"**Механизм влияния:** {a.mechanism_of_influence}")
                st.markdown(f"**Ожидаемая ценность:** {a.expected_value}")
                st.markdown(f"**Влияние на KPI:** {a.target_kpi_impact}")
                if a.technical_risks:
                    st.markdown("**Технические риски:** " + "; ".join(a.technical_risks))
                if a.economic_risks:
                    st.markdown("**Экономические риски:** " + "; ".join(a.economic_risks))
                if a.verification_plan:
                    st.markdown("**Дорожная карта проверки:**")
                    for s in a.verification_plan:
                        st.markdown(f"- {s}")
                if a.citations:
                    st.markdown("**Источники:**")
                    for c in a.citations:
                        st.markdown(f"- {c}")

with tab_report:
    st.header("Финальный отчёт и мета-ревью")
    if record is None:
        st.info("Нет данных. Запустите генерацию.")
    elif record.mode != "deep":
        st.info("У этого запуска нет финального отчёта (старый запуск облегчённого движка).")
    else:
        if record.final_report:
            st.subheader("📄 Финальный отчёт")
            st.markdown(record.final_report)
        if record.meta_review:
            st.subheader("🧭 Мета-ревью")
            st.markdown(record.meta_review)
        if not record.final_report and not record.meta_review:
            st.warning("Отчёт пуст (турнир не дошёл до финализации).")

with tab_agents:
    st.header("Коммуникация агентов")
    if record is None:
        st.info("Нет данных. Запустите генерацию.")
    else:
        _icons = {
            "Generator": "🧠", "Assessor": "🔬", "Literature": "📚",
            "Reflection": "🔎", "Evolution": "🧬", "Meta-review": "🧭",
            "Supervisor": "🎯", "Final report": "📄",
        }
        for msg in record.transcript:
            icon = _icons.get(msg.agent, "🤖")
            with st.chat_message("assistant"):
                st.markdown(f"{icon} **{msg.agent}** · "
                            f"{msg.tokens_in}→{msg.tokens_out} ток. · "
                            f"${msg.cost_usd} · {msg.seconds}s")
                st.markdown(msg.content[:4000])

with tab_metrics:
    st.header("Сводка по запуску")
    if record is None:
        st.info("Нет данных. Запустите генерацию.")
    else:
        m = record.metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Сообщений", m.get("messages", 0))
        c2.metric("Токенов (вх/вых)", f"{m.get('tokens_in', 0)}/{m.get('tokens_out', 0)}")
        c3.metric("Оценка стоимости", f"${m.get('cost_usd', 0)}")
        c4.metric("Время обработки", f"{m.get('seconds_wall', m.get('seconds', 0))} s")
        if record.mode == "deep" and record.tournament_summary:
            ts = record.tournament_summary
            st.subheader("🏆 Турнир")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Гипотез в турнире", ts.get("hypotheses", 0))
            t2.metric("Матчей", ts.get("matches_played", 0))
            t3.metric("Раундов", ts.get("rounds_played", 0))
            t4.metric("Макс. Elo", ts.get("max_elo", "—"))
        st.subheader("Сообщения")
        st.dataframe(
            [
                {"агент": t.agent, "ток_вх": t.tokens_in, "ток_вых": t.tokens_out,
                 "стоимость$": t.cost_usd, "сек": t.seconds}
                for t in record.transcript
            ],
            use_container_width=True,
        )

with tab_graph:
    st.header("Граф гипотез и источников")
    if record is None:
        st.info("Нет данных. Запустите генерацию.")
    else:
        components.html(viz.to_html(record.assessment_objects()), height=660, scrolling=True)
