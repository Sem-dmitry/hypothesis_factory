#!/usr/bin/env python3
"""
One-command "Фабрика гипотез" pipeline: private corpus -> grounded assessment
-> ranked report + tasks + graph.

Real run (needs API keys):
    export ROUTER_AI_API_KEY=...           # chat + vision + embeddings
    python scripts/run_pipeline.py --data-dir data --goal "Снизить потери никеля с хвостами" \
        --hypotheses "Доизмельчение хвостов повысит извлечение никеля" --out out

Offline demo (no keys, deterministic fakes — for checking the wiring end to end):
    python scripts/run_pipeline.py --offline-demo --out out

Outputs land in <out>/: hypotheses_report.{md,html,docx,pdf}, tasks.csv, tasks.json,
jira_tasks.json, graph.html
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure the repo root is importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coscientist.corpus.build import build_corpus_index  # noqa: E402
from coscientist.corpus.retrieval import CorpusRetriever  # noqa: E402
from coscientist.export import (  # noqa: E402
    assessments_to_csv,
    assessments_to_jira,
    assessments_to_json,
    write_report,
)
from coscientist.hypothesis_assessment import (  # noqa: E402
    AssessmentWeights,
    assess_hypothesis,
    rank_assessments,
)
from coscientist import viz  # noqa: E402

DEMO_GOAL = "Снизить потери никеля с хвостами флотации без потери качества концентрата"
DEMO_HYPOTHESES = [
    "Доизмельчение хвостов до более тонкого класса повысит извлечение никеля за счёт раскрытия пентландита",
    "Замена собирателя повысит селективность флотации халькопирита и качество медного концентрата",
    "Оптимизация pH пульпы снизит потери никеля с пирротиновой фракцией",
]

# A generic, defensible stub assessment used only in --offline-demo (no LLM key).
_DEMO_JSON = json.dumps(
    {
        "justification": "Согласуется с практикой обогащения и приведёнными данными по хвостам.",
        "mechanism_of_influence": "Изменение раскрытия/поверхностных свойств минералов влияет на флотационное закрепление.",
        "novelty": "Адресная настройка режима под конкретные классы крупности.",
        "novelty_score": 6,
        "feasibility_score": 8,
        "impact_score": 7,
        "risk_level": 3,
        "technical_risks": ["переизмельчение/ошламование", "снижение селективности"],
        "economic_risks": ["рост энергозатрат", "дополнительный расход реагентов"],
        "expected_value": "Снижение потерь ценных металлов с хвостами.",
        "target_kpi_impact": "+1–2% к извлечению целевого металла",
        "verification_plan": ["лабораторная флотация при 2 режимах", "баланс по классам крупности"],
        "confidence": 0.6,
    },
    ensure_ascii=False,
)


class _StubLLM:
    """Offline stand-in that returns a fixed structured assessment."""

    def invoke(self, prompt):
        class _Msg:
            content = _DEMO_JSON

        return _Msg()


def _demo_embed(texts):
    vocab = ["хвост", "флотац", "измельч", "никел", "реагент", "крупност", "потер", "извлеч"]
    return [[float(t.lower().count(w)) for w in vocab] + [1.0] for t in texts]


def run(args) -> int:
    goal = args.goal or (DEMO_GOAL if args.offline_demo else "")
    if not goal:
        print("error: --goal is required (or use --offline-demo)", file=sys.stderr)
        return 2

    hypotheses = args.hypotheses or (DEMO_HYPOTHESES if args.offline_demo else [])
    if args.hypotheses_file:
        with open(args.hypotheses_file, encoding="utf-8") as fh:
            hypotheses += [ln.strip() for ln in fh if ln.strip()]
    if not hypotheses:
        print("error: provide --hypotheses / --hypotheses-file (or --offline-demo)", file=sys.stderr)
        return 2

    embed_fn = _demo_embed if args.offline_demo else None
    llm = _StubLLM() if args.offline_demo else None

    # 1) Build (or reuse) the private-corpus index.
    print(f"[1/4] Indexing corpus from {args.data_dir} ...")
    index = build_corpus_index(
        data_dir=args.data_dir,
        index_path=args.index_path,
        include_images=not args.no_images and not args.offline_demo,
        embed_fn=embed_fn,
        verbose=True,
    )
    retriever = CorpusRetriever(index, default_k=args.k)
    print(f"      corpus chunks: {len(index)}")

    # 2) Assess each hypothesis grounded in the corpus.
    if llm is None:
        from coscientist.model_factory import get_chat_model

        llm = get_chat_model(args.model)
    weights = AssessmentWeights(
        novelty=args.w_novelty, feasibility=args.w_feasibility,
        impact=args.w_impact, risk=args.w_risk,
    )
    print(f"[2/4] Assessing {len(hypotheses)} hypotheses (model={args.model if not args.offline_demo else 'stub'}) ...")
    assessments = [
        assess_hypothesis(h, goal=goal, retriever=retriever, llm=llm, weights=weights)
        for h in hypotheses
    ]
    ranked = rank_assessments(assessments)

    # 3) Export deliverables.
    print(f"[3/4] Writing report + tasks + graph to {args.out} ...")
    os.makedirs(args.out, exist_ok=True)
    paths = write_report(ranked, args.out, goal=goal)
    _write(os.path.join(args.out, "tasks.csv"), assessments_to_csv(ranked))
    _write(os.path.join(args.out, "tasks.json"), assessments_to_json(ranked))
    _write(os.path.join(args.out, "jira_tasks.json"),
           json.dumps(assessments_to_jira(ranked), ensure_ascii=False, indent=2))
    _write(os.path.join(args.out, "graph.html"), viz.to_html(ranked))

    # 4) Summary.
    print("[4/4] Done. Ranked hypotheses:")
    for i, a in enumerate(ranked, start=1):
        print(f"      {i}. [{a.overall_score}] {a.hypothesis[:70]}")
    print("Files:", ", ".join([*paths.values(), os.path.join(args.out, 'graph.html')]))
    return 0


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Фабрика гипотез end-to-end pipeline.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--index-path",
                   default=os.path.join(os.path.expanduser(os.environ.get("COSCIENTIST_DIR", "~/.coscientist")), "corpus", "index"))
    p.add_argument("--goal", default="")
    p.add_argument("--hypotheses", nargs="*", default=None, help="Candidate hypotheses to assess.")
    p.add_argument("--hypotheses-file", default=None, help="Text file, one hypothesis per line.")
    p.add_argument("--out", default="out")
    p.add_argument("--model", default="google/gemini-2.5-pro", help="RouterAI model spec.")
    p.add_argument("--k", type=int, default=6, help="Top-k corpus chunks per hypothesis.")
    p.add_argument("--no-images", action="store_true", help="Skip VLM image parsing.")
    p.add_argument("--offline-demo", action="store_true",
                   help="Run with deterministic fakes (no API keys needed).")
    p.add_argument("--w-novelty", type=float, default=0.25)
    p.add_argument("--w-feasibility", type=float, default=0.25)
    p.add_argument("--w-impact", type=float, default=0.30)
    p.add_argument("--w-risk", type=float, default=0.20)
    return p


def main(argv=None) -> int:
    # Make Cyrillic console output readable on Windows terminals.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
