"""
Knowledge / influence graph over hypotheses and their sources.

Builds a networkx graph where each hypothesis is a node linked to the corpus
sources that ground it (from an assessment's structural citations). Exports to
JSON (nodes/edges) and a self-contained interactive HTML (inline SVG + JS, no
external hosts) so it can be embedded anywhere or printed.
"""

from __future__ import annotations

import html
import json
import math
import re
from typing import Any, Sequence

import networkx as nx

from coscientist.hypothesis_assessment import HypothesisAssessment, rank_assessments


def _source_key(citation: str) -> str:
    """Reduce a citation line to a stable source label (strip the [n] prefix)."""
    return re.sub(r"^\[\d+\]\s*", "", citation).strip()


def build_graph(assessments: Sequence[HypothesisAssessment]) -> nx.Graph:
    """Build an undirected hypothesis<->source graph."""
    g = nx.Graph()
    ranked = rank_assessments(list(assessments))
    for i, a in enumerate(ranked, start=1):
        h_node = f"H{i}"
        g.add_node(
            h_node,
            kind="hypothesis",
            label=a.hypothesis,
            score=a.overall_score,
            kpi=a.target_kpi_impact,
        )
        for citation in a.citations:
            src = _source_key(citation)
            if not src:
                continue
            if not g.has_node(src):
                g.add_node(src, kind="source", label=src)
            g.add_edge(h_node, src, kind="cites")
    return g


def to_json(assessments: Sequence[HypothesisAssessment]) -> str:
    """Node-link JSON representation of the graph."""
    g = build_graph(assessments)
    data = {
        "nodes": [{"id": n, **g.nodes[n]} for n in g.nodes],
        "edges": [{"source": u, "target": v, **g.edges[u, v]} for u, v in g.edges],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def _esc(text: Any) -> str:
    return html.escape(str(text or ""))


def to_html(
    assessments: Sequence[HypothesisAssessment],
    *,
    title: str = "Граф гипотез и источников",
) -> str:
    """
    Self-contained interactive HTML graph (inline SVG + a tiny force layout in
    inline JS). No external hosts/CDNs are referenced.
    """
    g = build_graph(assessments)
    nodes = list(g.nodes)
    # Deterministic circular initial layout; JS relaxes it slightly.
    n = max(1, len(nodes))
    positions = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        positions[node] = (400 + 300 * math.cos(angle), 300 + 220 * math.sin(angle))

    node_payload = [
        {
            "id": node,
            "x": positions[node][0],
            "y": positions[node][1],
            "kind": g.nodes[node].get("kind", "source"),
            "label": g.nodes[node].get("label", node),
            "score": g.nodes[node].get("score", ""),
        }
        for node in nodes
    ]
    edge_payload = [{"source": u, "target": v} for u, v in g.edges]
    data_json = json.dumps({"nodes": node_payload, "edges": edge_payload}, ensure_ascii=False)

    # NOTE: the JS is inline and references no external resources.
    return f"""<!doctype html><html lang='ru'><head><meta charset='utf-8'>
<title>{_esc(title)}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Arial,sans-serif;margin:0;background:#f7fafc}}
h1{{font-size:1.1rem;padding:.6rem 1rem;margin:0;background:#2b6cb0;color:#fff}}
#legend{{padding:.4rem 1rem;font-size:.85rem;color:#4a5568}}
.hyp{{fill:#2b6cb0}} .src{{fill:#dd6b20}} text{{font-size:11px;fill:#1a202c}}
line{{stroke:#cbd5e0;stroke-width:1.5}}
</style></head><body>
<h1>{_esc(title)}</h1>
<div id='legend'>● гипотеза &nbsp; ● источник &nbsp; (перетаскивайте узлы)</div>
<svg id='g' width='820' height='620'></svg>
<script>
const DATA = {data_json};
const svg = document.getElementById('g');
const NS = 'http://www.w3.org/2000/svg';
const pos = {{}};
DATA.nodes.forEach(n => pos[n.id] = {{x:n.x, y:n.y}});
function draw() {{
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  DATA.edges.forEach(e => {{
    const l = document.createElementNS(NS,'line');
    l.setAttribute('x1',pos[e.source].x); l.setAttribute('y1',pos[e.source].y);
    l.setAttribute('x2',pos[e.target].x); l.setAttribute('y2',pos[e.target].y);
    svg.appendChild(l);
  }});
  DATA.nodes.forEach(n => {{
    const c = document.createElementNS(NS,'circle');
    c.setAttribute('cx',pos[n.id].x); c.setAttribute('cy',pos[n.id].y);
    c.setAttribute('r', n.kind==='hypothesis'?12:8);
    c.setAttribute('class', n.kind==='hypothesis'?'hyp':'src');
    c.style.cursor='grab';
    c.addEventListener('mousedown', ev => drag(ev, n.id));
    svg.appendChild(c);
    const t = document.createElementNS(NS,'text');
    t.setAttribute('x',pos[n.id].x+14); t.setAttribute('y',pos[n.id].y+4);
    t.textContent = (n.label||n.id).slice(0,42);
    svg.appendChild(t);
  }});
}}
let dragId=null;
function drag(ev,id){{dragId=id;}}
svg.addEventListener('mousemove', ev => {{
  if(!dragId) return;
  const r = svg.getBoundingClientRect();
  pos[dragId]={{x:ev.clientX-r.left, y:ev.clientY-r.top}}; draw();
}});
window.addEventListener('mouseup', ()=>dragId=null);
draw();
</script>
</body></html>"""
