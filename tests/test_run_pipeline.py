"""Offline smoke test for the one-command pipeline runner."""

import glob
import os
import shutil

import pytest

from scripts import run_pipeline


def _small_data_dir(tmp_path):
    """Copy one docx + one xlsx from data/ into a tiny dir for a fast build."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    picks = []
    for pat in ("*.docx", "*.xlsx"):
        matches = [p for p in glob.glob(os.path.join(repo, "data", "**", pat), recursive=True)
                   if os.path.isfile(p)]
        if matches:
            picks.append(min(matches, key=os.path.getsize))
    if not picks:
        pytest.skip("no data/ files available")
    d = tmp_path / "mini_data"
    d.mkdir()
    for p in picks:
        shutil.copy(p, d / os.path.basename(p))
    return str(d)


def test_offline_demo_pipeline(tmp_path):
    data_dir = _small_data_dir(tmp_path)
    out = tmp_path / "out"
    index = tmp_path / "idx"
    rc = run_pipeline.main([
        "--offline-demo",
        "--data-dir", data_dir,
        "--index-path", str(index),
        "--out", str(out),
    ])
    assert rc == 0
    for name in ("hypotheses_report.md", "hypotheses_report.html",
                 "hypotheses_report.docx", "hypotheses_report.pdf",
                 "tasks.csv", "tasks.json", "jira_tasks.json", "graph.html"):
        f = out / name
        assert f.exists() and f.stat().st_size > 0, name


def test_requires_goal_or_demo():
    # No goal and not demo -> usage error exit code 2.
    rc = run_pipeline.main(["--hypotheses", "h1"])
    assert rc == 2
