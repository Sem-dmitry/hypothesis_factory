# -*- coding: utf-8 -*-
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "webapp" / "static"


def read_static(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def test_current_spa_uses_premium_light_shell():
    html = read_static("index.html")
    styles = read_static("styles.css")

    assert '<html lang="ru" class="light">' in html
    assert 'class="icon-sprite"' in html
    assert 'id="i-ai-mark"' in html
    assert 'id="i-folder"' in html
    assert '<svg class="i"><use href="#i-play"></use></svg>' in html

    assert "--background: #f7fffc" in styles
    assert "--primary: #5a00e8" in styles
    assert "--teal: #10e6bf" in styles
    assert "background: rgba(255,255,255,.92)" in styles
    assert "rgba(255,255,255,.98)" in styles


def test_dynamic_webapp_ui_uses_svg_icons_not_emoji():
    app_js = read_static("app.js")
    html = read_static("index.html")

    assert "const icon = " in app_js
    assert "const agentIcon = (a) => icon(" in app_js
    assert "${icon('file')}" in app_js
    assert "${icon('folder')}" in app_js
    assert "${icon('globe')}" in app_js

    static_ui = "\n".join([html, app_js])
    assert not re.search(r"[\U0001F300-\U0001FAFF]", static_ui)


def test_source_links_remain_supported_in_premium_ui():
    app_js = read_static("app.js")
    styles = read_static("styles.css")

    assert 'class="source-link"' in app_js
    assert 'class="source-open"' in app_js
    assert "wireSourceButtons" in app_js
    assert ".source-open, .source-link" in styles
