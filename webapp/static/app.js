/* Фабрика гипотез — SPA logic (vanilla JS, no build step). */
'use strict';

const $ = (s) => document.querySelector(s);
const el = (t, cls, html) => { const e = document.createElement(t); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const icon = (name, cls = 'i') => `<svg class="${cls}" aria-hidden="true"><use href="#i-${name}"></use></svg>`;

const AGENT_ICON = {
  'Constraints': 'puzzle', 'Literature': 'scroll', 'Generation': 'flask', 'Reflection': 'analysis', 'Ranking': 'chart',
  'Evolution': 'branch', 'Meta-review': 'network', 'Supervisor': 'settings', 'Final report': 'file', 'Assessor': 'flask',
};
const agentIcon = (a) => icon(AGENT_ICON[a] || 'bot');

const state = {
  settings: null,          // /api/settings payload (editable)
  projects: [],
  currentProject: null,
  currentRunId: null,
  runViewId: null,         // record id being viewed
  es: null,                // EventSource
  agents: {},              // label -> {calls, tokens, cost, status}
  phasesSeen: new Set(),
  phaseCurrent: null,
  totals: { messages: 0, tokens_in: 0, tokens_out: 0, cost: 0, start: 0 },
  activeView: 'analysis',
  fullMode: false,          // LITE by default; FullMode gated behind a warning
  selectedFiles: [],
  live: {
    hypotheses: [],
    final_report: '',
    meta_review: '',
    metrics: {},
    graph: {},
  },
};

/* ---------------- tiny markdown ---------------- */
function esc(s) { return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function escAttr(s) { return esc(s).replace(/"/g, '&quot;').replace(/'/g, '&#39;'); }
function mdInline(s) {
  return esc(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}
function renderMarkdown(src) {
  const lines = (src || '').split('\n');
  let html = '', i = 0, inCode = false, codeBuf = [], listType = null, listBuf = [];
  const flushList = () => { if (listType) { html += `<${listType}>` + listBuf.join('') + `</${listType}>`; listType = null; listBuf = []; } };
  while (i < lines.length) {
    let ln = lines[i];
    if (ln.trim().startsWith('```')) {
      if (!inCode) { flushList(); inCode = true; codeBuf = []; }
      else { html += '<pre><code>' + esc(codeBuf.join('\n')) + '</code></pre>'; inCode = false; }
      i++; continue;
    }
    if (inCode) { codeBuf.push(ln); i++; continue; }
    const h = ln.match(/^(#{1,4})\s+(.*)$/);
    if (h) { flushList(); html += `<h${h[1].length}>${mdInline(h[2])}</h${h[1].length}>`; i++; continue; }
    const ol = ln.match(/^\s*\d+[.)]\s+(.*)$/);
    const ul = ln.match(/^\s*[-*•]\s+(.*)$/);
    if (ol) { if (listType !== 'ol') { flushList(); listType = 'ol'; } listBuf.push('<li>' + mdInline(ol[1]) + '</li>'); i++; continue; }
    if (ul) { if (listType !== 'ul') { flushList(); listType = 'ul'; } listBuf.push('<li>' + mdInline(ul[1]) + '</li>'); i++; continue; }
    if (ln.trim().startsWith('>')) { flushList(); html += '<blockquote>' + mdInline(ln.replace(/^\s*>\s?/, '')) + '</blockquote>'; i++; continue; }
    if (ln.trim() === '') { flushList(); i++; continue; }
    flushList(); html += '<p>' + mdInline(ln) + '</p>'; i++;
  }
  flushList();
  if (inCode) html += '<pre><code>' + esc(codeBuf.join('\n')) + '</code></pre>';
  return html;
}

/* ---------------- toast ---------------- */
let toastT;
function toast(msg) { const t = $('#toast'); t.textContent = msg; t.classList.add('show'); clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove('show'), 3200); }

/* ---------------- live results helpers ---------------- */
function fmtBytes(bytes) {
  const n = Number(bytes || 0);
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
function normalizeHypothesis(s) { return (s || '').replace(/\s+/g, ' ').trim().toLowerCase(); }
function compactText(s) { return (s || '').replace(/\r/g, '').replace(/[ \t]+/g, ' ').trim(); }
function parseJsonArrayFromText(text) {
  const start = (text || '').indexOf('[');
  const end = (text || '').lastIndexOf(']');
  if (start === -1 || end <= start) return [];
  try {
    const arr = JSON.parse(text.slice(start, end + 1));
    return Array.isArray(arr) ? arr.map(x => compactText(String(x))).filter(Boolean) : [];
  } catch { return []; }
}
function parseJsonObjectFromText(text) {
  const start = (text || '').indexOf('{');
  const end = (text || '').lastIndexOf('}');
  if (start === -1 || end <= start) return {};
  try {
    const obj = JSON.parse(text.slice(start, end + 1));
    return obj && typeof obj === 'object' && !Array.isArray(obj) ? obj : {};
  } catch { return {}; }
}
// Pull the body of a markdown heading (up to the next `#` heading or the end).
function extractHeadingSection(text, heading) {
  const re = new RegExp(`#\\s*(?:${heading})\\s*\\n+([\\s\\S]*?)(?=\\n#\\s|$)`, 'i');
  const m = (text || '').match(re);
  return m ? compactText(m[1].split(/\n\s*\n/)[0]) : '';
}
// The REAL hypotheses a Generation event introduces — never its predictions or
// assumptions (those are sub-parts of one hypothesis, not new hypotheses).
// Light mode emits a JSON array of hypotheses; deep mode emits one `# Hypothesis`.
function generatedHypothesesFrom(text) {
  const arr = parseJsonArrayFromText(text);
  if (arr.length) return arr.filter(h => h.length >= 12);
  const h = extractHeadingSection(text, 'Hypothesis|Гипотеза');
  return h && h.length >= 12 ? [h] : [];
}
// The hypothesis a prompt is ABOUT (assessment / reflection / evolution), so we
// can attribute that agent's work to an existing hypothesis card.
function extractHypothesisFromPrompt(prompt) {
  const p = prompt || '';
  const assess = p.match(/# Hypothesis to assess\s+([\s\S]*?)\n\s*#\s/i);
  if (assess) return compactText(assess[1]);
  return extractHeadingSection(p, 'Original Hypothesis|Hypothesis to (?:evolve|develop)|Hypothesis|Гипотеза');
}
function resetLiveResults() {
  state.live = { hypotheses: [], final_report: '', meta_review: '', metrics: {}, graph: {} };
  renderLivePanels();
}
function findHypothesis(text) {
  const key = normalizeHypothesis(compactText(text));
  return key ? state.live.hypotheses.find(h => normalizeHypothesis(h.hypothesis) === key) : null;
}
function upsertHypothesis(text, patch = {}) {
  const hypothesis = compactText(text);
  if (!hypothesis) return null;
  let item = findHypothesis(hypothesis);
  if (!item) {
    item = { hypothesis, status: 'generated', created_at: Date.now(), trajectory: [] };
    state.live.hypotheses.push(item);
    item._isNew = true;
  } else {
    item._isNew = false;
  }
  Object.assign(item, patch, { hypothesis });
  return item;
}
// Append a stage to a hypothesis' formation trajectory (collapsing repeats).
function pushTrajectory(item, agent, label) {
  if (!item) return;
  if (!item.trajectory) item.trajectory = [];
  const last = item.trajectory[item.trajectory.length - 1];
  if (last && last.agent === agent && last.label === label) {
    last.count = (last.count || 1) + 1;
    return;
  }
  item.trajectory.push({ agent, label, count: 1 });
}
function nextPendingHypothesis() {
  return state.live.hypotheses.find(h => h.status !== 'assessed') || state.live.hypotheses[state.live.hypotheses.length - 1];
}
function updateLiveResultsFromAgent(ev) {
  const agent = ev.agent;
  if (agent === 'Generation') {
    // Only real hypotheses become cards — not their predictions/assumptions.
    generatedHypothesesFrom(ev.content).forEach((h) => {
      const item = upsertHypothesis(h, { status: 'generated' });
      if (item && item._isNew) pushTrajectory(item, 'Generation', 'Сформирована');
    });
  } else if (agent === 'Reflection') {
    const item = findHypothesis(extractHypothesisFromPrompt(ev.prompt));
    if (item) pushTrajectory(item, 'Reflection', 'Проверена рефлексией');
  } else if (agent === 'Evolution') {
    const item = findHypothesis(extractHypothesisFromPrompt(ev.prompt));
    if (item) pushTrajectory(item, 'Evolution', 'Улучшена');
  } else if (agent === 'Ranking') {
    // Turnament touches every current contender once (relative signal, not per-card).
    state.live.hypotheses.forEach(h => { if (h.status !== 'assessed') pushTrajectory(h, 'Ranking', 'В турнире'); });
  } else if (agent === 'Assessor') {
    const data = parseJsonObjectFromText(ev.content);
    if (Object.keys(data).length) {
      const promptHyp = extractHypothesisFromPrompt(ev.prompt);
      const target = promptHyp || nextPendingHypothesis()?.hypothesis || `Гипотеза ${state.live.hypotheses.length + 1}`;
      const item = upsertHypothesis(target, { ...data, status: 'assessed' });
      pushTrajectory(item, 'Assessor', `Оценена: ${data.overall_score ?? data.impact_score ?? '—'}`);
    }
  } else if (agent === 'Final report') {
    state.live.final_report = ev.content || state.live.final_report;
  } else if (agent === 'Meta-review') {
    state.live.meta_review = ev.content || state.live.meta_review;
  }
  renderLivePanels();
}
function setFinalRecordResults(rec) {
  // Merge the authoritative, ranked assessments onto any trajectory built from
  // the replayed transcript (so formation history survives), rather than wiping.
  (rec.assessments || []).forEach((a, i) => {
    const item = upsertHypothesis(a.hypothesis, { ...a, status: 'assessed', rank: i + 1 });
    if (item && !(item.trajectory || []).length) {
      pushTrajectory(item, 'Assessor', `Оценена: ${a.overall_score ?? '—'}`);
    }
  });
  state.live.hasAuthoritative = (rec.assessments || []).length > 0;
  state.live.final_report = rec.final_report || state.live.final_report || '';
  state.live.meta_review = rec.meta_review || state.live.meta_review || '';
  state.live.metrics = rec.metrics || {};
  state.live.graph = rec.tournament_summary || {};
  renderLivePanels();
}
function renderLivePanels() {
  renderLiveHypotheses();
  renderLiveReport();
  renderLiveMetricsPanel();
  renderLiveGraph();
}
function isAssessed(a) { return a.status === 'assessed' || a.overall_score !== undefined; }
function scoreOf(a) { return Number(a.overall_score ?? a.impact_score ?? -1); }

// Compact "formation trajectory" — how the hypothesis took shape across agents.
function trajectoryHtml(a) {
  const steps = a.trajectory || [];
  if (!steps.length) return '';
  const chip = (s) => `<span class="traj-step">${agentIcon(s.agent)} ${esc(s.label)}${s.count > 1 ? ` ×${s.count}` : ''}</span>`;
  return `<div class="traj" title="Траектория формирования гипотезы">${steps.map(chip).join('<span class="traj-arr">→</span>')}</div>`;
}

function assessmentBodyHtml(a) {
  return renderMarkdown(
    (a.mechanism_of_influence ? `**Механизм:** ${a.mechanism_of_influence}\n\n` : '') +
    (a.causal_chain ? `**Причина (why):** ${a.causal_chain}\n\n` : '') +
    (a.world_practice ? `**Мировая практика:** ${a.world_practice}\n\n` : '') +
    (a.expected_value ? `**Ценность:** ${a.expected_value}\n\n` : '') +
    (a.target_kpi_impact ? `**KPI:** ${a.target_kpi_impact}\n\n` : '') +
    (a.novelty_vs_input ? `**Новизна vs вход:** ${a.novelty_vs_input}\n\n` : '') +
    (a.economic_estimate ? `**Экономика (прикидка):** ${a.economic_estimate}\n\n` : '') +
    (a.kinetics_note ? `**Кинетика:** ${a.kinetics_note}\n\n` : '') +
    (a.constraint_adherence ? `**Ограничения:** ${a.constraint_adherence}\n\n` : '') +
    ((a.constraint_violations || []).length ? `**Нарушения ограничений:** ${a.constraint_violations.join('; ')}\n\n` : '') +
    ((a.technical_risks || []).length ? `**Тех. риски:** ${a.technical_risks.join('; ')}\n\n` : '') +
    ((a.verification_plan || []).length ? `**Проверка:**\n${(a.verification_plan || []).map(s => '- ' + s).join('\n')}\n` : ''));
}

function sourceEvidenceHtml(a) {
  const sources = a.source_evidence || [];
  if (sources.length) {
    return `<div class="sources">
      <div class="sources-head">Источники</div>
      ${sources.map((s, i) => `<div class="source-item">
        ${s.url
          ? `<a class="source-link" href="${escAttr(s.url)}" target="_blank" rel="noopener noreferrer">
              ${esc(s.source_name || `Источник ${i + 1}`)}
              ${s.locator ? `<span>${esc(s.locator)}</span>` : ''}
            </a>`
          : `<button type="button" class="source-open" data-chunk="${escAttr(s.chunk_id || '')}">
              ${esc(s.source_name || `Источник ${i + 1}`)}
              ${s.locator ? `<span>${esc(s.locator)}</span>` : ''}
            </button>`}
        ${s.quote ? `<blockquote>${esc(s.quote)}</blockquote>` : ''}
      </div>`).join('')}
    </div>`;
  }
  if (Object.prototype.hasOwnProperty.call(a, 'source_evidence') ||
      Object.prototype.hasOwnProperty.call(a, 'evidence_refs')) {
    return '';
  }
  return (a.citations || []).length
    ? `<div class="cite">${icon('paperclip')} ${a.citations.map(esc).join('<br/>')}</div>`
    : '';
}

function buildAssessedCard(a, rank) {
  const h = el('div', 'hyp');
  h.innerHTML = `
    <div class="h-top"><span class="h-rank">#${rank}</span><span class="h-title">${esc(a.hypothesis)}</span><span class="h-score">${esc(String(a.overall_score ?? '—'))}</span></div>
    <div class="h-grid">
      <div class="h-metric"><b>${esc(String(a.novelty_score ?? '—'))}</b><span>новизна</span></div>
      <div class="h-metric"><b>${esc(String(a.feasibility_score ?? '—'))}</b><span>реализуемость</span></div>
      <div class="h-metric"><b>${esc(String(a.impact_score ?? '—'))}</b><span>эффект</span></div>
      <div class="h-metric"><b>${esc(String(a.risk_level ?? '—'))}</b><span>риск</span></div>
    </div>
    ${trajectoryHtml(a)}
    <div class="md">${assessmentBodyHtml(a)}</div>
    ${sourceEvidenceHtml(a)}
    ${state.runViewId ? expertBarHtml() : ''}`;
  wireSourceButtons(h);
  if (state.runViewId) wireExpertActions(h, a.hypothesis);
  return h;
}

function wireSourceButtons(root) {
  root.querySelectorAll('.source-open').forEach((btn) => {
    btn.onclick = () => openSourceChunk(btn.dataset.chunk || '');
  });
}

async function openSourceChunk(chunkId) {
  if (!chunkId) return;
  if (!state.runViewId) { toast('Источник доступен после сохранения запуска'); return; }
  const title = $('#sourceTitle');
  const meta = $('#sourceMeta');
  const text = $('#sourceText');
  title.textContent = 'Источник';
  meta.textContent = 'Загрузка...';
  text.textContent = '';
  $('#sourceOverlay').classList.add('open');
  try {
    const resp = await fetch(`/api/runs/${state.runViewId}/sources/${encodeURIComponent(chunkId)}`);
    if (!resp.ok) throw new Error(String(resp.status));
    const s = await resp.json();
    title.textContent = s.source_name || 'Источник';
    meta.textContent = [s.locator, s.modality].filter(Boolean).join(' · ');
    text.textContent = s.text || '';
  } catch (e) {
    meta.textContent = 'Не удалось открыть источник';
  }
}

function renderLiveHypotheses() {
  const all = state.live.hypotheses;
  const assessed = all.filter(isAssessed).sort((x, y) => (x.rank ?? 1e9) - (y.rank ?? 1e9) || scoreOf(y) - scoreOf(x));
  const drafts = all.filter(a => !isAssessed(a));
  $('#hypCount').textContent = String(assessed.length || all.length);

  const box = $('#liveHypotheses');
  if (!box) return;
  if (!all.length) {
    box.innerHTML = '<div class="result-empty"><div><b>Гипотезы ещё формируются</b><div class="small">Оценённые гипотезы появятся здесь по мере готовности, ранжированные по совокупной оценке.</div></div></div>';
    return;
  }

  const sub = assessed.length
    ? `${assessed.length} оценено${drafts.length ? ` · ${drafts.length} в проработке` : ''}`
    : `${drafts.length} формируется`;
  box.innerHTML = `<div class="result-head"><h2>Гипотезы и оценки</h2><span class="small">${sub}</span></div><div class="result-stack"></div>`;
  const stack = box.querySelector('.result-stack');

  assessed.forEach((a, i) => stack.appendChild(buildAssessedCard(a, i + 1)));

  if (drafts.length) {
    const det = el('details', 'drafts');
    // Keep drafts open only while nothing is assessed yet (early in a run).
    if (!assessed.length) det.open = true;
    det.innerHTML = `<summary>${icon('flask')} В проработке — ${drafts.length} ${assessed.length ? '(ещё не прошли оценку)' : ''}</summary>`;
    drafts.forEach((a) => {
      const d = el('div', 'hyp pending');
      d.innerHTML = `
        <div class="h-top"><span class="h-rank">•</span><span class="h-title">${esc(a.hypothesis)}</span><span class="h-score">формируется</span></div>
        ${trajectoryHtml(a)}`;
      det.appendChild(d);
    });
    stack.appendChild(det);
  }
}

function expertBarHtml() {
  return `<div class="expert">
    <div class="expert-row">
      <span class="small">Эксперт:</span>
      <button class="btn sm eb-fb" data-o="confirmed">${icon('check')} подтвердить</button>
      <button class="btn sm eb-fb" data-o="refuted">${icon('x')} опровергнуть</button>
      <button class="btn sm eb-fb" data-o="inconclusive">${icon('minus')} неясно</button>
      <button class="btn ghost sm eb-chat">${icon('chat')} Обсудить</button>
      <button class="btn ghost sm eb-branch">${icon('branch')} Развить направление</button>
    </div>
    <div class="expert-row"><input class="field eb-note" placeholder="Комментарий эксперта (опционально)…"/></div>
    <div class="eb-status small"></div>
    <div class="eb-chat" style="display:none"></div>
    <div class="eb-branch" style="display:none"></div>
  </div>`;
}

function wireExpertActions(card, hypothesis) {
  const runId = state.runViewId;
  const status = card.querySelector('.eb-status');
  const note = card.querySelector('.eb-note');
  card.querySelectorAll('.eb-fb').forEach(btn => btn.onclick = async () => {
    status.textContent = 'Сохранение…';
    try {
      const r = await (await fetch(`/api/runs/${runId}/feedback`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hypothesis, outcome: btn.dataset.o, note: note.value || '' }),
      })).json();
      const c = r.counts || {};
      status.textContent = `Оценка сохранена. Подтверждено: ${c.confirmed||0}; опровергнуто: ${c.refuted||0}; неясно: ${c.inconclusive||0}`;
    } catch (e) { status.textContent = 'Ошибка сохранения'; }
  });
  // chat
  const chatBox = card.querySelector('.eb-chat');
  const chatHistory = [];
  const chatToggle = card.querySelectorAll('.expert-row button')[3];
  chatToggle.onclick = () => {
    const open = chatBox.style.display === 'none';
    chatBox.style.display = open ? 'block' : 'none';
    if (open && !chatBox.dataset.init) {
      chatBox.dataset.init = '1';
      chatBox.innerHTML = `<div class="eb-log"></div><div class="expert-row"><input class="field eb-msg" placeholder="Спросите о гипотезе…"/><button class="btn sm eb-send">Отпр.</button></div>`;
      const log = chatBox.querySelector('.eb-log'), msg = chatBox.querySelector('.eb-msg');
      chatBox.querySelector('.eb-send').onclick = async () => {
        const text = msg.value.trim(); if (!text) return;
        log.appendChild(el('div', 'md', renderMarkdown('**Вы:** ' + text))); msg.value = '';
        const wait = el('div', 'small', 'Думаю…'); log.appendChild(wait); log.scrollTop = log.scrollHeight;
        try {
          const r = await (await fetch(`/api/runs/${runId}/chat`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ hypothesis, message: text, history: chatHistory }),
          })).json();
          wait.remove();
          chatHistory.push({ role: 'user', content: text }, { role: 'assistant', content: r.reply });
          log.appendChild(el('div', 'md', renderMarkdown('**Ассистент:** ' + (r.reply || '—'))));
          log.scrollTop = log.scrollHeight;
        } catch (e) { wait.textContent = 'Ошибка'; }
      };
    }
  };
  // branch
  const branchBox = card.querySelector('.eb-branch');
  const branchToggle = card.querySelectorAll('.expert-row button')[4];
  branchToggle.onclick = () => {
    const open = branchBox.style.display === 'none';
    branchBox.style.display = open ? 'block' : 'none';
    if (open && !branchBox.dataset.init) {
      branchBox.dataset.init = '1';
      branchBox.innerHTML = `<div class="expert-row"><input class="field eb-dir" placeholder="В каком направлении развить? (напр.: другой собиратель, иная стадия)"/><button class="btn sm eb-go">Развить</button></div><div class="eb-out"></div>`;
      const out = branchBox.querySelector('.eb-out');
      branchBox.querySelector('.eb-go').onclick = async () => {
        const dir = branchBox.querySelector('.eb-dir').value.trim(); if (!dir) return;
        out.innerHTML = '<div class="small">Генерирую и оцениваю новые гипотезы…</div>';
        try {
          const r = await (await fetch(`/api/runs/${runId}/branch`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ hypothesis, direction: dir, n: 2 }),
          })).json();
          out.innerHTML = '';
          (r.hypotheses || []).forEach((a, i) => {
            out.appendChild(el('div', 'hyp', `<div class="h-top"><span class="h-rank">${icon('branch')} ${i+1}</span><span class="h-title">${esc(a.hypothesis)}</span><span class="h-score">${esc(String(a.overall_score ?? '—'))}</span></div><div class="md">${renderMarkdown((a.mechanism_of_influence?`**Механизм:** ${a.mechanism_of_influence}\n\n`:'')+(a.target_kpi_impact?`**KPI:** ${a.target_kpi_impact}`:''))}</div>`));
          });
          if (!(r.hypotheses||[]).length) out.innerHTML = '<div class="small">Не удалось сгенерировать — попробуйте иначе сформулировать направление.</div>';
        } catch (e) { out.innerHTML = '<div class="small">Ошибка</div>'; }
      };
    }
  };
}
function renderLiveReport() {
  const box = $('#liveReport');
  if (!box) return;
  const parts = [];
  if (state.live.final_report) parts.push(`<div class="report-card"><h2>Финальный отчёт</h2><div class="md">${renderMarkdown(state.live.final_report)}</div></div>`);
  if (state.live.meta_review) parts.push(`<div class="report-card"><h2>Мета-ревью</h2><div class="md">${renderMarkdown(state.live.meta_review)}</div></div>`);
  box.innerHTML = parts.length ? parts.join('') : '<div class="result-empty"><div><b>Отчёт ещё не готов</b><div class="small">Мета-ревью и финальный отчёт появятся здесь по мере генерации.</div></div></div>';
}
function renderLiveMetricsPanel() {
  const box = $('#liveMetrics');
  if (!box) return;
  const m = { ...state.live.metrics };
  if (state.totals.messages) {
    m.messages = state.totals.messages;
    m.tokens_in = state.totals.tokens_in;
    m.tokens_out = state.totals.tokens_out;
    m.cost_usd = state.totals.cost.toFixed(4);
    m.seconds = Math.round((Date.now() - state.totals.start) / 1000);
  }
  const entries = [
    ['Гипотез', state.live.hypotheses.length || m.hypotheses || 0],
    ['Вызовов', m.messages ?? '—'],
    ['Токены in', m.tokens_in ?? '—'],
    ['Токены out', m.tokens_out ?? '—'],
    ['Стоимость', m.cost_usd !== undefined ? `$${m.cost_usd}` : '—'],
    ['Время', (m.seconds_wall ?? m.seconds ?? '—') + (m.seconds_wall || m.seconds ? 's' : '')],
    ['Матчей', m.matches_played ?? '—'],
    ['Раундов', m.rounds_played ?? '—'],
  ];
  box.innerHTML = `<div class="result-head"><h2>Метрики запуска</h2></div><div class="metric-grid">${entries.map(([k, v]) => `<div class="metric-card"><b>${esc(String(v))}</b><span>${esc(k)}</span></div>`).join('')}</div>`;
}
/* ---------------- hypothesis ↔ source graph ---------------- */
const GRAPH_W = 960, GRAPH_H = 600;
let _graph = null;         // { nodes, links, adj, svg, raf }

// Parse a citation string into a stable source identity.
//   corpus: "[1] source_name — locator (текст)"  → key = source_name
//   web:    "[W1] title — https://… (web)"        → key = url
function parseCitation(cit) {
  const s = (cit || '').trim();
  if (!s) return null;
  const isWeb = /\(web\)\s*$/i.test(s);
  const urlm = s.match(/https?:\/\/[^\s)]+/);
  const url = urlm ? urlm[0].replace(/[.,;]+$/, '') : '';
  const body = s.replace(/^\s*\[[^\]]+\]\s*/, '');       // drop leading [n]/[Wn]
  let name = body.split(' — ')[0].trim() || body.trim();
  name = name.replace(/\s*\((?:web|текст|таблица|изображение|image|table|text)\)\s*$/i, '').trim();
  const key = (isWeb && url) ? 'url:' + url.toLowerCase() : 'doc:' + name.toLowerCase();
  return { key, title: name || url, url, isWeb };
}

// Sources a hypothesis ACTUALLY cites — the same set the Hypotheses tab shows.
// Prefer the LLM-selected `source_evidence`; fall back to parsing citation
// strings only for legacy records that never had that field. This keeps the
// graph consistent with each card's "Источники" (not the whole retrieved pool).
function hypothesisSources(h) {
  const hasEv = Object.prototype.hasOwnProperty.call(h, 'source_evidence') ||
                Object.prototype.hasOwnProperty.call(h, 'evidence_refs');
  if (hasEv) {
    return (h.source_evidence || []).map((s) => {
      const isWeb = !!s.url || s.modality === 'web';
      const name = (s.source_name || (isWeb ? s.url : '') || '').trim();
      const key = (isWeb && s.url) ? 'url:' + s.url.toLowerCase() : 'doc:' + name.toLowerCase();
      return { key, title: name || s.url || 'Источник', url: s.url || '', isWeb };
    }).filter((p) => p.key !== 'doc:');
  }
  return (h.citations || []).map(parseCitation).filter(Boolean);
}

function buildGraphModel(hyps) {
  const nodes = [], links = [];
  const sources = new Map();
  hyps.forEach((h, i) => {
    const hn = { id: 'h' + i, type: 'hyp', label: h.hypothesis, rank: i + 1,
                 score: h.overall_score, risk: h.risk_level, deg: 0 };
    nodes.push(hn);
    hypothesisSources(h).forEach((p) => {
      let sn = sources.get(p.key);
      if (!sn) { sn = { id: 's' + sources.size, type: 'src', label: p.title, url: p.url, isWeb: p.isWeb, deg: 0 }; sources.set(p.key, sn); nodes.push(sn); }
      sn.deg++; hn.deg++;
      links.push({ s: hn, t: sn });
    });
  });
  nodes.forEach((n) => {
    n.r = n.type === 'hyp'
      ? 15 + Math.min(7, Math.max(0, (Number(n.score) || 0)) * 0.7)   // ~15–22
      : 5 + Math.min(6, (n.deg - 1) * 2.5);                            // ~5–11
    n.shared = n.type === 'src' && n.deg > 1;
  });
  const shared = [...sources.values()].filter((s) => s.deg > 1).length;
  return { nodes, links, hyps: hyps.length, srcs: sources.size, shared };
}

// Ideal edge length — scales with the drawing area so the graph fills the panel.
function idealDist(n, W, H) { return Math.max(80, Math.sqrt((W * H) / Math.max(1, n)) * 0.52); }

// One Fruchterman–Reingold iteration (repulsion + edge attraction + mild gravity),
// with a temperature cap on displacement so it settles smoothly.
function forceStep(nodes, links, W, H, k, temp) {
  const cx = W / 2, cy = H / 2;
  nodes.forEach((v) => { v.dx = 0; v.dy = 0; });
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      let dx = a.x - b.x, dy = a.y - b.y, d = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const f = (k * k) / d;
      const ux = dx / d * f, uy = dy / d * f;
      a.dx += ux; a.dy += uy; b.dx -= ux; b.dy -= uy;
    }
  }
  links.forEach((l) => {
    let dx = l.t.x - l.s.x, dy = l.t.y - l.s.y, d = Math.sqrt(dx * dx + dy * dy) || 0.01;
    const f = (d * d) / k;
    const ux = dx / d * f, uy = dy / d * f;
    l.s.dx += ux; l.s.dy += uy; l.t.dx -= ux; l.t.dy -= uy;
  });
  nodes.forEach((v) => { v.dx += (cx - v.x) * 0.016; v.dy += (cy - v.y) * 0.02; });
  nodes.forEach((v) => {
    if (v.fixed) return;
    const dl = Math.sqrt(v.dx * v.dx + v.dy * v.dy) || 0.01;
    const lim = Math.min(dl, temp) / dl;
    v.x += v.dx * lim; v.y += v.dy * lim;
    const m = v.r + 10;
    v.x = Math.max(m, Math.min(W - m, v.x)); v.y = Math.max(m, Math.min(H - m, v.y));
  });
}

function layoutGraph(nodes, links, W, H, iters) {
  const n = nodes.length;
  if (!n) return;
  const k = idealDist(n, W, H), cx = W / 2, cy = H / 2;
  nodes.forEach((v, i) => {
    if (v.x == null) {
      const a = 2 * Math.PI * i / n, rad = Math.min(W, H) * 0.34;
      v.x = cx + Math.cos(a) * rad * (0.7 + 0.3 * ((i * 13 % 7) / 7));
      v.y = cy + Math.sin(a) * rad * (0.7 + 0.3 * ((i * 7 % 5) / 5));
    }
  });
  let temp = Math.min(W, H) * 0.16;
  for (let it = 0; it < iters; it++) {
    forceStep(nodes, links, W, H, k, temp);
    temp = Math.max(Math.min(W, H) * 0.006, temp * 0.965);
  }
}

function truncate(s, n) { s = s || ''; return s.length > n ? s.slice(0, n - 1) + '…' : s; }
function clampNum(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function renderLiveGraph() {
  const box = $('#liveGraph');
  if (!box) return;
  if (_graph) { if (_graph.raf) cancelAnimationFrame(_graph.raf); if (_graph.ro) _graph.ro.disconnect(); }
  const hyps = state.live.hypotheses.filter(isAssessed)
    .sort((x, y) => (x.rank ?? 1e9) - (y.rank ?? 1e9) || scoreOf(y) - scoreOf(x));
  if (!hyps.length) {
    box.innerHTML = '<div class="result-empty"><div><b>Граф появится после оценки</b><div class="small">Узлы-гипотезы связываются через общие источники: если на один источник ссылаются несколько гипотез, вы увидите связь между ними.</div></div></div>';
    _graph = null;
    return;
  }
  const model = buildGraphModel(hyps);

  const SVGNS = 'http://www.w3.org/2000/svg';
  box.innerHTML = `<div class="result-head"><h2>Граф гипотез и источников</h2>
    <span class="small">${model.hyps} гипотез · ${model.srcs} источников · ${model.shared} общих</span></div>
    <div class="graph-legend">
      <span class="lg"><i class="dot hyp"></i>гипотеза</span>
      <span class="lg"><i class="dot src"></i>источник</span>
      <span class="lg"><i class="dot shared"></i>общий источник (связывает гипотезы)</span>
      <span class="lg-hint">потяните узлы · наведите для связей</span>
    </div>
    <div class="graph-wrap"></div>`;
  const wrap = box.querySelector('.graph-wrap');

  // Size the drawing to the panel so the graph fills it (no letterboxing).
  const rect = wrap.getBoundingClientRect();
  let W = clampNum(Math.round(rect.width) || GRAPH_W, 560, 3200);
  let H = clampNum(Math.round(rect.height) || GRAPH_H, 360, 1800);
  layoutGraph(model.nodes, model.links, W, H, 340);

  const svg = document.createElementNS(SVGNS, 'svg');
  svg.setAttribute('class', 'graph-svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  svg.innerHTML = `<defs>
    <radialGradient id="hypGrad" cx="35%" cy="28%" r="80%">
      <stop offset="0%" stop-color="#10e6bf"/>
      <stop offset="100%" stop-color="#5a00e8"/>
    </radialGradient>
  </defs>`;
  const gLinks = document.createElementNS(SVGNS, 'g');
  const gNodes = document.createElementNS(SVGNS, 'g');
  svg.appendChild(gLinks); svg.appendChild(gNodes);

  const adj = new Map(model.nodes.map((n) => [n.id, new Set([n.id])]));
  model.links.forEach((l) => { adj.get(l.s.id).add(l.t.id); adj.get(l.t.id).add(l.s.id); });

  model.links.forEach((l) => {
    const line = document.createElementNS(SVGNS, 'line');
    line.setAttribute('class', 'g-link' + (l.t.shared ? ' shared' : ''));
    l.el = line; gLinks.appendChild(line);
  });

  model.nodes.forEach((n) => {
    const g = document.createElementNS(SVGNS, 'g');
    g.setAttribute('class', 'g-node ' + (n.type === 'hyp' ? 'g-hyp' : 'g-src' + (n.shared ? ' shared' : '') + (n.isWeb ? ' web' : '')));
    const c = document.createElementNS(SVGNS, 'circle');
    c.setAttribute('r', n.r);
    g.appendChild(c);
    if (n.type === 'hyp') {
      const t = document.createElementNS(SVGNS, 'text');
      t.setAttribute('class', 'g-rank'); t.setAttribute('dy', '0.34em'); t.textContent = '#' + n.rank;
      g.appendChild(t);
    }
    // Readable label: a rounded chip behind the text (approx-sized), below the node.
    const fs = n.type === 'hyp' ? 12 : 10.5;
    const text = truncate(n.label, n.type === 'hyp' ? 30 : 22);
    const cw = text.length * fs * 0.56 + 14;      // approximate chip width
    const ly = n.r + 8;                            // chip top offset below node
    const lg = document.createElementNS(SVGNS, 'g');
    lg.setAttribute('class', 'g-lbl');
    const chip = document.createElementNS(SVGNS, 'rect');
    chip.setAttribute('class', 'g-chip'); chip.setAttribute('x', -cw / 2); chip.setAttribute('y', ly);
    chip.setAttribute('width', cw); chip.setAttribute('height', fs + 8); chip.setAttribute('rx', (fs + 8) / 2);
    const lbl = document.createElementNS(SVGNS, 'text');
    lbl.setAttribute('class', 'g-label'); lbl.setAttribute('text-anchor', 'middle');
    lbl.setAttribute('y', ly + (fs + 8) / 2); lbl.setAttribute('font-size', fs); lbl.textContent = text;
    lg.appendChild(chip); lg.appendChild(lbl); g.appendChild(lg);

    const title = document.createElementNS(SVGNS, 'title');
    title.textContent = (n.type === 'hyp' ? `#${n.rank} · оценка ${n.score ?? '—'}\n` : (n.shared ? `Общий источник (${n.deg} гипотез)\n` : '')) + n.label + (n.url ? '\n' + n.url : '');
    g.appendChild(title);
    n.g = g; n.circle = c; gNodes.appendChild(g);

    g.addEventListener('mouseenter', () => { g.parentNode.appendChild(g); highlight(n.id); });
    g.addEventListener('mouseleave', () => highlight(null));
    if (n.type === 'src' && n.isWeb && n.url) {
      g.classList.add('clickable');
      g.addEventListener('click', () => { if (!n._dragged) window.open(n.url, '_blank', 'noopener'); });
    }
    attachDrag(g, n, svg);
  });

  function tick() {
    model.links.forEach((l) => {
      l.el.setAttribute('x1', l.s.x); l.el.setAttribute('y1', l.s.y);
      l.el.setAttribute('x2', l.t.x); l.el.setAttribute('y2', l.t.y);
    });
    model.nodes.forEach((n) => n.g.setAttribute('transform', `translate(${n.x} ${n.y})`));
  }
  function highlight(id) {
    if (!id) { model.nodes.forEach((n) => n.g.classList.remove('faded', 'hot')); model.links.forEach((l) => l.el.classList.remove('faded', 'lit')); return; }
    const near = adj.get(id);
    model.nodes.forEach((n) => { n.g.classList.toggle('faded', !near.has(n.id)); n.g.classList.toggle('hot', n.id === id); });
    model.links.forEach((l) => { const on = l.s.id === id || l.t.id === id; l.el.classList.toggle('faded', !on); l.el.classList.toggle('lit', on); });
  }
  function attachDrag(g, n, svgEl) {
    g.addEventListener('pointerdown', (e) => {
      e.preventDefault(); n.fixed = true; n._dragged = false; g.setPointerCapture(e.pointerId);
      const move = (ev) => {
        n._dragged = true;
        const p = svgEl.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY;
        const loc = p.matrixTransform(svgEl.getScreenCTM().inverse());
        n.x = loc.x; n.y = loc.y; tick(); relax();
      };
      const up = () => { n.fixed = false; g.releasePointerCapture(e.pointerId); g.removeEventListener('pointermove', move); g.removeEventListener('pointerup', up); };
      g.addEventListener('pointermove', move); g.addEventListener('pointerup', up);
    });
  }
  let cooling = 0;
  function relax() {
    cooling = 20;
    if (_graph.raf) return;
    const step = () => {
      forceStep(model.nodes, model.links, _graph.W, _graph.H, idealDist(model.nodes.length, _graph.W, _graph.H), Math.min(_graph.W, _graph.H) * 0.02);
      tick();
      if (--cooling > 0) { _graph.raf = requestAnimationFrame(step); } else { _graph.raf = 0; }
    };
    _graph.raf = requestAnimationFrame(step);
  }
  function fit() {
    const r = wrap.getBoundingClientRect();
    const nW = clampNum(Math.round(r.width), 560, 3200), nH = clampNum(Math.round(r.height), 360, 1800);
    if (!r.width || !r.height || (Math.abs(nW - _graph.W) < 2 && Math.abs(nH - _graph.H) < 2)) return;
    const sx = nW / _graph.W, sy = nH / _graph.H;
    model.nodes.forEach((n) => { n.x *= sx; n.y *= sy; });
    _graph.W = nW; _graph.H = nH; svg.setAttribute('viewBox', `0 0 ${nW} ${nH}`);
    const k = idealDist(model.nodes.length, nW, nH);
    for (let i = 0; i < 50; i++) forceStep(model.nodes, model.links, nW, nH, k, Math.min(nW, nH) * 0.05 * (1 - i / 50));
    tick();
  }

  wrap.appendChild(svg);
  const ro = new ResizeObserver(() => { if (_graph) fit(); });
  ro.observe(wrap);
  _graph = { model, svg, wrap, W, H, ro, raf: 0, tick, fit };
  tick();
  requestAnimationFrame(() => svg.classList.add('ready'));
}
function switchView(view) {
  state.activeView = view;
  document.querySelectorAll('.vtab').forEach(t => t.classList.toggle('active', t.dataset.view === view));
  const map = { analysis: '#viewAnalysis', feed: '#viewFeed', hypotheses: '#viewHypotheses', report: '#viewReport', metrics: '#viewMetrics', graph: '#viewGraph' };
  Object.entries(map).forEach(([k, id]) => $(id).classList.toggle('active', k === view));
  if (view === 'graph' && _graph) requestAnimationFrame(() => _graph.fit());  // fit once the panel is visible
}

/* ---------------- init ---------------- */
async function init() {
  state.settings = await (await fetch('/api/settings')).json();
  syncSettingsControls();
  await loadHealth();
  await loadProjects();
  bindUI();
}

function syncSettingsControls() {
  if (!state.settings) return;
  if ($('#numHyp')) $('#numHyp').value = state.settings.num_hypotheses ?? 4;
  if ($('#maxHyp')) $('#maxHyp').value = state.settings.max_hypotheses ?? 0;
  if ($('#autoConstraints')) {
    $('#autoConstraints').checked = state.settings.auto_elicit_constraints ?? true;
  }
  applyModeLock();
}

// LITE (default): fixed 4 hypotheses / 8 max, models+reasoning locked (backend
// forces gemini-2.5-flash / reasoning off). FullMode unlocks everything.
const LITE_NUM = 4, LITE_MAX = 8;
function applyModeLock() {
  const lite = !state.fullMode;
  const num = $('#numHyp'), max = $('#maxHyp'), gear = $('#settingsBtn');
  if (lite) {
    if (num) num.value = LITE_NUM;
    if (max) max.value = LITE_MAX;
  }
  [num, max].forEach((el) => { if (el) { el.disabled = lite; el.classList.toggle('locked', lite); } });
  if (gear) { gear.disabled = lite; gear.classList.toggle('locked', lite);
    gear.title = lite ? 'Доступно в FullMode' : 'Настройки агентов'; }
  const badge = $('#modeBadge');
  if (badge) badge.textContent = lite ? 'LiteMode' : 'FullMode';
  const cb = $('#fullMode'); if (cb) cb.checked = state.fullMode;
}

// FullMode is entered only after confirming the warning modal.
function requestFullMode(on) {
  if (on) {
    $('#fullModeOverlay').classList.add('open');   // ask for confirmation
  } else {
    state.fullMode = false;
    applyModeLock();
  }
}
function confirmFullMode() {
  state.fullMode = true;
  $('#fullModeOverlay').classList.remove('open');
  applyModeLock();
  toast('FullMode включён — доступны модели, мышление и параметры');
}
function stayLite() {
  state.fullMode = false;
  $('#fullModeOverlay').classList.remove('open');
  applyModeLock();
}

async function loadHealth() {
  try {
    const h = await (await fetch('/api/health')).json();
    const dot = (on) => `<span class="keydot ${on ? 'on' : 'off'}"></span>`;
    $('#keyStatus').innerHTML = `${dot(h.routerai)}AI ${dot(h.embeddings)}Emb`;
    if (!h.routerai || !h.embeddings) $('#composerHint').textContent = 'Нет ключей → включите демо-режим';
  } catch (e) { /* ignore */ }
}

async function loadProjects() {
  state.projects = await (await fetch('/api/projects')).json();
  if (!state.currentProject && state.projects.length) state.currentProject = state.projects[0];
  renderProjects();
  if (state.currentProject) await loadRuns();
}

function renderProjects() {
  const list = $('#projectList'); list.innerHTML = '';
  state.projects.forEach((p) => {
    const item = el('div', 'proj-item' + (state.currentProject && p.id === state.currentProject.id ? ' active' : ''));
    item.innerHTML = `<span class="ico">${icon('folder')}</span><span class="name">${esc(p.name)}</span><span class="badge-count">${p.run_count || 0}</span>`;
    item.onclick = async () => { state.currentProject = p; renderProjects(); await loadRuns(); };
    list.appendChild(item);
  });
}

async function loadRuns() {
  const runs = await (await fetch(`/api/projects/${state.currentProject.id}/runs`)).json();
  const list = $('#runList'); list.innerHTML = '';
  if (!runs.length) { list.innerHTML = '<div class="small" style="padding:6px 8px">Пока нет запусков</div>'; return; }
  runs.forEach((r) => {
    const item = el('div', 'run-item' + (r.id === state.runViewId ? ' active' : ''));
    item.innerHTML = `<span class="dot ${r.status}"></span><span class="rgoal">${esc(r.goal)}</span>`;
    item.title = `${r.goal}\n${r.created_at}`;
    item.onclick = () => viewRun(r.id);
    list.appendChild(item);
  });
}

/* ---------------- run history view ---------------- */
async function viewRun(id) {
  try {
    const rec = await (await fetch(`/api/runs/${id}`)).json();
    state.runViewId = id; state.currentRunId = null;
    renderRecord(rec);
    loadRuns();
  } catch (e) { toast('Не удалось загрузить запуск'); }
}

function renderRecord(rec) {
  $('#centerTitle').textContent = rec.goal;
  switchView('feed');
  const feed = $('#feed'); feed.innerHTML = '';
  state.totals = { messages: 0, tokens_in: 0, tokens_out: 0, cost: 0, start: Date.now() };
  resetPanels();
  // Replay transcript into feed + right panels
  (rec.transcript || []).forEach((m) => { onAgentEvent(m, false); });
  setFinalRecordResults(rec);
  $('#downloadLog').href = `/api/runs/${rec.id}/log`;
  $('#downloadConsoleLog').href = `/api/runs/${rec.id}/console-log`;
  updateMetrics(rec.metrics || {});
  feed.scrollTop = feed.scrollHeight;
}

/* ---------------- launching a run ---------------- */
async function startRun() {
  const goal = $('#goalInput').value.trim();
  if (!goal) { toast('Введите цель исследования'); return; }
  resetPanels(); $('#feed').innerHTML = ''; $('#emptyState')?.remove();
  state.totals = { messages: 0, tokens_in: 0, tokens_out: 0, cost: 0, start: Date.now() };
  $('#centerTitle').textContent = goal;
  switchView('feed');
  $('#runBtn').disabled = true; $('#runBtn').innerHTML = '<span class="spinner"></span> Идёт…';
  if ($('#steerBar')) $('#steerBar').style.display = 'flex';
  $('#downloadLog').href = '#';
  $('#downloadConsoleLog').href = '#';

  const fd = new FormData();
  fd.append('goal', goal);
  fd.append('constraints', $('#constraintsInput').value.trim());
  fd.append('project', state.currentProject ? state.currentProject.name : 'Новый проект');
  fd.append('use_web', $('#useWeb').checked ? 'true' : 'false');
  const settings = buildSettingsPayload();
  if (state.fullMode) {
    settings.num_hypotheses = parseInt($('#numHyp').value || '4', 10);
    settings.max_hypotheses = parseInt($('#maxHyp').value || '0', 10);
  } else {
    // LITE: fixed and non-editable (backend re-applies these anyway).
    settings.num_hypotheses = LITE_NUM;
    settings.max_hypotheses = LITE_MAX;
  }
  fd.append('settings', JSON.stringify(settings));
  for (const f of state.selectedFiles) fd.append('files', f);

  try {
    const { run_id } = await (await fetch('/api/runs', { method: 'POST', body: fd })).json();
    state.currentRunId = run_id;
    openStream(run_id);
  } catch (e) { finishRun(); toast('Ошибка запуска: ' + e); }
}

function openStream(runId) {
  if (state.es) state.es.close();
  const es = new EventSource(`/api/runs/${runId}/events`);
  state.es = es;
  es.onmessage = (e) => {
    let ev; try { ev = JSON.parse(e.data); } catch { return; }
    if (ev.type === 'agent') onAgentEvent(ev, true);
    else if (ev.type === 'user_message') appendUserMessage(ev);
    else if (ev.type === 'log') appendConsoleLog(ev);
    else if (ev.type === 'done') { onDone(ev); es.close(); }
    else if (ev.type === 'error') { toast('Ошибка: ' + ev.message); appendSystem('Ошибка: ' + ev.message); appendConsoleLog({ stream: 'error', message: ev.message, ts: new Date().toISOString() }); finishRun(); es.close(); }
  };
  es.onerror = () => { /* stream ended */ };
}

function finishRun() {
  $('#runBtn').disabled = false; $('#runBtn').innerHTML = `${icon('play')} Запустить`;
  if ($('#steerBar')) $('#steerBar').style.display = 'none';
}

async function sendSteer() {
  const input = $('#steerInput');
  const text = (input.value || '').trim();
  if (!text) return;
  if (!state.currentRunId) { toast('Нет активного запуска'); return; }
  input.value = '';
  try {
    const r = await fetch(`/api/runs/${state.currentRunId}/message`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!r.ok) { toast(r.status === 409 ? 'Запуск уже завершён' : 'Не удалось отправить'); }
    // The message is echoed back via the SSE stream (type "user_message").
  } catch (e) { toast('Ошибка отправки'); }
}

function appendUserMessage(ev) {
  const feed = $('#feed');
  const card = el('div', 'msg user');
  card.innerHTML = `
    <div class="msg-head">
      <div class="msg-ava">${icon('user')}</div>
      <div><div class="msg-who">Вы — эксперт (steering)</div>
      <div class="msg-meta">${esc(ev.ts || '')}</div></div>
    </div>
    <div class="msg-body"><div class="md">${renderMarkdown(ev.text || '')}</div></div>`;
  feed.appendChild(card);
  feed.scrollTop = feed.scrollHeight;
}

async function onDone(ev) {
  finishRun();
  markAllDone();
  try {
    const rec = await (await fetch(`/api/runs/${ev.run_id}`)).json();
    state.runViewId = ev.run_id;
    setFinalRecordResults(rec);
    $('#downloadLog').href = `/api/runs/${rec.id}/log`;
    $('#downloadConsoleLog').href = `/api/runs/${rec.id}/console-log`;
    updateMetrics(rec.metrics || {});
  } catch (e) { /* ignore */ }
  toast('Готово: гипотез ' + (ev.summary ? ev.summary.hypotheses : '') );
  await loadProjects();
}

/* ---------------- event -> UI ---------------- */
function onAgentEvent(ev, animate) {
  // center feed
  appendMessage(ev);
  // metrics
  state.totals.messages++;
  state.totals.tokens_in += ev.tokens_in || 0;
  state.totals.tokens_out += ev.tokens_out || 0;
  state.totals.cost += ev.cost_usd || 0;
  updateLiveMetrics(ev.agent);
  // agents panel
  updateAgentCard(ev);
  // phases
  updatePhases(ev.agent);
  // live result tabs
  updateLiveResultsFromAgent(ev);
}

function renderConstraintsMarkdown(content) {
  // The Constraints agent streams a JSON array of {text, rationale, kind}.
  let items = [];
  try {
    const s = content.indexOf('['), e = content.lastIndexOf(']');
    if (s !== -1 && e > s) items = JSON.parse(content.slice(s, e + 1));
  } catch { /* fall back */ }
  if (!Array.isArray(items) || !items.length) return renderMarkdown(content || '');
  const line = (c) => {
    const kind = (c.kind || 'assumption') === 'clarification' ? 'уточнить' : 'допущение';
    const rat = c.rationale ? ` — _${c.rationale}_` : '';
    return `- **${kind}:** ${c.text}${rat}`;
  };
  return renderMarkdown('Уточнённые ограничения (система вывела недостающие, но важные для гипотез):\n\n' +
    items.map(line).join('\n'));
}

function appendMessage(ev) {
  const feed = $('#feed');
  const card = el('div', 'msg');
  const long = (ev.content || '').length > 1600;
  const bodyHtml = ev.agent === 'Constraints' ? renderConstraintsMarkdown(ev.content || '') : renderMarkdown(ev.content || '');
  card.innerHTML = `
    <div class="msg-head">
      <div class="msg-ava">${agentIcon(ev.agent)}</div>
      <div><div class="msg-who">${esc(ev.agent === 'Constraints' ? 'Constraints — доуточнение ограничений' : ev.agent)}</div>
      <div class="msg-meta"><span class="chip">${esc(ev.model || '')}</span> ${ev.tokens_in}→${ev.tokens_out} ток · $${(ev.cost_usd||0).toFixed(4)} · ${ev.seconds}s</div></div>
      <div class="spacer"></div>
      <div class="msg-actions"><button class="btn ghost sm copyBtn" title="Копировать">${icon('copy')}</button></div>
    </div>
    <div class="msg-body"><div class="md">${bodyHtml}</div>
      ${ev.prompt ? `<div class="collapser togglePrompt">Показать полный запрос</div><div class="prompt-box" style="display:none"></div>` : ''}
    </div>`;
  const copyBtn = card.querySelector('.copyBtn');
  copyBtn.onclick = () => { navigator.clipboard.writeText(ev.content || ''); toast('Скопировано'); };
  const tp = card.querySelector('.togglePrompt');
  if (tp) {
    const box = card.querySelector('.prompt-box');
    box.textContent = ev.prompt;
    tp.onclick = () => { const open = box.style.display === 'none'; box.style.display = open ? 'block' : 'none'; tp.textContent = (open ? 'Скрыть' : 'Показать') + ' полный запрос'; };
  }
  feed.appendChild(card);
  feed.scrollTop = feed.scrollHeight;
}

function appendSystem(text) { const f = $('#feed'); const d = el('div', 'small', text); d.style.textAlign = 'center'; f.appendChild(d); }
function appendReport(title, body) {
  const feed = $('#feed');
  const card = el('div', 'msg');
  card.innerHTML = `<div class="msg-head"><div class="msg-ava">${icon('file')}</div><div class="msg-who">${title}</div></div><div class="msg-body"><div class="md">${renderMarkdown(body)}</div></div>`;
  feed.appendChild(card); feed.scrollTop = feed.scrollHeight;
}

function renderHypotheses(assessments) {
  const feed = $('#feed');
  const wrap = el('div', 'msg');
  wrap.innerHTML = `<div class="msg-head"><div class="msg-ava">${icon('chart')}</div><div class="msg-who">Ранжированные гипотезы (${assessments.length})</div></div>`;
  const body = el('div', 'msg-body');
  assessments.forEach((a, i) => {
    const h = el('div', 'hyp'); h.style.marginBottom = '10px';
    h.innerHTML = `
      <div class="h-top"><span class="h-rank">#${i + 1}</span><span class="h-title">${esc(a.hypothesis)}</span><span class="h-score">${a.overall_score}</span></div>
      <div class="h-grid">
        <div class="h-metric"><b>${a.novelty_score}</b><span>новизна</span></div>
        <div class="h-metric"><b>${a.feasibility_score}</b><span>реализуемость</span></div>
        <div class="h-metric"><b>${a.impact_score}</b><span>эффект</span></div>
        <div class="h-metric"><b>${a.risk_level}</b><span>риск</span></div>
      </div>
      <div class="md">${renderMarkdown(
        (a.mechanism_of_influence ? `**Механизм:** ${a.mechanism_of_influence}\n\n` : '') +
        (a.causal_chain ? `**Причина (why):** ${a.causal_chain}\n\n` : '') +
        (a.world_practice ? `**Мировая практика:** ${a.world_practice}\n\n` : '') +
        (a.expected_value ? `**Ценность:** ${a.expected_value}\n\n` : '') +
        (a.target_kpi_impact ? `**KPI:** ${a.target_kpi_impact}\n\n` : '') +
        (a.novelty_vs_input ? `**Новизна vs вход:** ${a.novelty_vs_input}\n\n` : '') +
        (a.economic_estimate ? `**Экономика (прикидка):** ${a.economic_estimate}\n\n` : '') +
        (a.kinetics_note ? `**Кинетика:** ${a.kinetics_note}\n\n` : '') +
        (a.constraint_adherence ? `**Ограничения:** ${a.constraint_adherence}\n\n` : '') +
        ((a.constraint_violations||[]).length ? `**Нарушения ограничений:** ${a.constraint_violations.join('; ')}\n\n` : '') +
        ((a.technical_risks||[]).length ? `**Тех. риски:** ${a.technical_risks.join('; ')}\n\n` : '') +
        ((a.verification_plan||[]).length ? `**Проверка:**\n${(a.verification_plan||[]).map(s=>'- '+s).join('\n')}\n` : ''))}</div>
      ${sourceEvidenceHtml(a)}`;
    wireSourceButtons(h);
    body.appendChild(h);
  });
  wrap.appendChild(body); feed.appendChild(wrap); feed.scrollTop = feed.scrollHeight;
}

/* ---------------- right panels ---------------- */
function resetPanels() {
  state.agents = {}; state.phasesSeen = new Set(); state.phaseCurrent = null;
  $('#rbodyAgents').innerHTML = '<div class="small">Агенты появятся по мере работы…</div>';
  renderPhases();
  $('#term').innerHTML = '<div class="small">Консольные события запуска появятся здесь. Ответы агентов остаются в центральной ленте.</div>';
  resetLiveResults();
}

function updateAgentCard(ev) {
  const a = ev.agent;
  if (!state.agents[a]) state.agents[a] = { calls: 0, tin: 0, tout: 0, cost: 0, model: ev.model };
  const s = state.agents[a];
  s.calls++; s.tin += ev.tokens_in || 0; s.tout += ev.tokens_out || 0; s.cost += ev.cost_usd || 0; s.model = ev.model || s.model;
  state.phaseCurrent = a;
  renderAgents();
}
function renderAgents() {
  const box = $('#rbodyAgents');
  const labels = (state.settings.phases || []).map(p => p.key);
  const known = labels.length ? labels : Object.keys(state.agents);
  const order = [...new Set([...known, ...Object.keys(state.agents)])];
  box.innerHTML = '';
  order.forEach((a) => {
    const s = state.agents[a];
    if (!s && !AGENT_ICON[a]) return;
    const status = !s ? 'idle' : (state.phaseCurrent === a ? 'active' : 'done');
    const card = el('div', 'agent-card');
    card.innerHTML = `<div class="a-top"><div class="a-ico">${agentIcon(a)}</div>
      <div style="flex:1"><div class="a-name">${esc(a)}</div><div class="a-model">${esc(s ? s.model || '' : '')}</div></div>
      <span class="status-pill ${status}">${status === 'active' ? 'работает' : status === 'done' ? 'готово' : 'ожидание'}</span></div>
      ${s ? `<div class="a-stats"><span>вызовов: ${s.calls}</span><span>ток: ${s.tin}→${s.tout}</span><span>$${s.cost.toFixed(4)}</span></div>` : ''}`;
    box.appendChild(card);
  });
}
function updatePhases(agent) { state.phasesSeen.add(agent); state.phaseCurrent = agent; renderPhases(); }
function renderPhases() {
  const box = $('#rbodyTasks'); box.innerHTML = '';
  (state.settings?.phases || []).forEach((p) => {
    const status = state.phaseCurrent === p.key ? 'active' : (state.phasesSeen.has(p.key) ? 'done' : 'pending');
    const ico = status === 'active' ? icon('refresh') : status === 'done' ? icon('check') : icon('circle');
    const row = el('div', 'phase ' + status);
    row.innerHTML = `<div class="p-ico">${ico}</div><div class="p-label">${agentIcon(p.key)} ${esc(p.label)}</div>`;
    box.appendChild(row);
  });
}
function markAllDone() { state.phaseCurrent = null; renderAgents(); renderPhases(); }

function appendConsoleLog(ev) {
  const term = $('#term');
  if (term.querySelector('.small')) term.innerHTML = '';
  const ts = ev.ts ? new Date(ev.ts).toLocaleTimeString() : new Date().toLocaleTimeString();
  const stream = ev.stream || 'log';
  const msg = ev.message || '';
  const block = el('div');
  block.innerHTML = `<span class="t-meta">${esc(ts)} · ${esc(stream)}</span>\n${esc(msg)}\n`;
  term.appendChild(block); term.scrollTop = term.scrollHeight;
}

/* ---------------- metrics ---------------- */
function updateLiveMetrics(currentAgent) {
  const secs = Math.round((Date.now() - state.totals.start) / 1000);
  $('#metrics').innerHTML = metricHtml('этап', agentIcon(currentAgent) + ' ' + currentAgent)
    + metricHtml('вызовов', state.totals.messages)
    + metricHtml('токены', `${state.totals.tokens_in}→${state.totals.tokens_out}`)
    + metricHtml('стоимость', '$' + state.totals.cost.toFixed(4))
    + metricHtml('время', secs + 's');
}
function updateMetrics(m) {
  $('#metrics').innerHTML = metricHtml('гипотез', m.hypotheses ?? '—')
    + metricHtml('вызовов', m.messages ?? '—')
    + metricHtml('стоимость', '$' + (m.cost_usd ?? 0))
    + metricHtml('время', (m.seconds_wall ?? m.seconds ?? 0) + 's');
}
function metricHtml(label, val) { return `<div class="metric"><b>${val}</b><span>${label}</span></div>`; }

/* ---------------- settings ---------------- */
function buildSettingsPayload() {
  // read from state.settings (edited in modal), fall back to defaults
  return JSON.parse(JSON.stringify({
    num_hypotheses: state.settings.num_hypotheses,
    max_hypotheses: state.settings.max_hypotheses ?? 0,
    retrieval_k: state.settings.retrieval_k,
    weights: state.settings.weights,
    agents: state.settings.agents,
    web_retriever: state.settings.web_retriever,
    web_search_model: state.settings.web_search_model,
    auto_elicit_constraints: $('#autoConstraints') ? $('#autoConstraints').checked : true,
    lite: !state.fullMode,
  }));
}
function openSettings() {
  const s = state.settings;
  const body = $('#settingsBody');
  const maxHypotheses = s.max_hypotheses ?? 0;
  const modelOpts = (sel) => s.model_options.map(m => `<option ${m === sel ? 'selected' : ''}>${m}</option>`).join('');
  const thinkOpts = (sel) => s.thinking_options.map(t => `<option ${t === sel ? 'selected' : ''}>${t}</option>`).join('');
  let html = `<div class="agent-row" style="grid-template-columns:1fr 1fr">
    <div class="field-group"><label class="lbl">Число гипотез: <b id="nhLbl">${s.num_hypotheses}</b></label>
      <div class="slider-row"><input type="range" id="setNumHyp" min="2" max="8" value="${s.num_hypotheses}"/></div></div>
    <div class="field-group"><label class="lbl">Максимум гипотез: <b id="mhLbl">${maxHypotheses}</b></label>
      <div class="slider-row"><input type="range" id="setMaxHyp" min="0" max="40" value="${maxHypotheses}"/></div>
      <div class="small">0 — без лимита</div></div>
    </div>
    <div class="field-group"><label class="lbl">Веса ранжирования</label><div class="agent-row" style="grid-template-columns:repeat(4,1fr)">
      ${['novelty','feasibility','impact','risk'].map(k => `<div><label class="lbl">${k}</label><input class="field wset" data-k="${k}" type="number" step="0.05" value="${s.weights[k]}"/></div>`).join('')}
    </div></div>
    <div class="hr"></div><label class="lbl" style="font-size:13px">Модель · температура · мышление — по агентам</label>`;
  Object.keys(s.agent_labels).forEach((key) => {
    const a = s.agents[key];
    html += `<div class="agent-row" data-agent="${key}">
      <div><label class="lbl">${esc(s.agent_labels[key])}</label><select class="field am">${modelOpts(a.model)}</select></div>
      <div><label class="lbl">темп. <b class="atLbl">${a.temperature}</b></label><input class="field at" type="range" min="0" max="2" step="0.1" value="${a.temperature}"/></div>
      <div><label class="lbl">мышление</label><select class="field ath">${thinkOpts(a.thinking)}</select></div>
    </div>`;
  });
  html += `<div class="hr"></div><div class="field-group"><label class="lbl">${icon('globe')} Веб-поиск <span class="small">(RouterAI web-search)</span></label>
    <div class="agent-row" style="grid-template-columns:1fr">
      <div><label class="lbl">Модель web-search</label><select class="field" id="setWebModel">${modelOpts(s.web_search_model)}</select></div>
    </div></div>
    <div style="display:flex;justify-content:flex-end;gap:8px;margin-top:8px"><button class="btn primary" id="saveSettings">Сохранить</button></div>`;
  body.innerHTML = html;

  body.querySelector('#setNumHyp').oninput = (e) => { s.num_hypotheses = +e.target.value; body.querySelector('#nhLbl').textContent = e.target.value; };
  body.querySelector('#setMaxHyp').oninput = (e) => { s.max_hypotheses = +e.target.value; body.querySelector('#mhLbl').textContent = e.target.value; };
  body.querySelectorAll('.at').forEach(r => r.oninput = (e) => { e.target.closest('.agent-row').querySelector('.atLbl').textContent = e.target.value; });
  body.querySelector('#saveSettings').onclick = () => {
    s.num_hypotheses = +body.querySelector('#setNumHyp').value;
    s.max_hypotheses = +body.querySelector('#setMaxHyp').value;
    body.querySelectorAll('.wset').forEach(w => s.weights[w.dataset.k] = +w.value);
    body.querySelectorAll('.agent-row[data-agent]').forEach(row => {
      const k = row.dataset.agent;
      s.agents[k] = { model: row.querySelector('.am').value, temperature: +row.querySelector('.at').value, thinking: row.querySelector('.ath').value };
    });
    s.web_retriever = 'routerai';
    s.web_search_model = body.querySelector('#setWebModel').value;
    syncSettingsControls();
    $('#settingsOverlay').classList.remove('open'); toast('Настройки сохранены');
  };
  $('#settingsOverlay').classList.add('open');
}

/* ---------------- file selection ---------------- */
function fileKey(f) { return `${f.name}:${f.size}:${f.lastModified}`; }
function syncFileInput() {
  const input = $('#corpusFiles');
  try {
    const dt = new DataTransfer();
    state.selectedFiles.forEach(f => dt.items.add(f));
    input.files = dt.files;
  } catch {
    // Older browsers may not allow assigning FileList; FormData uses state.selectedFiles.
  }
}
function renderFileSelection() {
  const summary = $('#fileSummary');
  const list = $('#fileList');
  const clear = $('#clearFilesBtn');
  if (!state.selectedFiles.length) {
    summary.textContent = 'Файлы не выбраны';
    list.innerHTML = '';
    clear.style.display = 'none';
    return;
  }
  const total = state.selectedFiles.reduce((sum, f) => sum + f.size, 0);
  summary.textContent = `${state.selectedFiles.length} файл(ов), ${fmtBytes(total)}`;
  clear.style.display = 'inline-flex';
  list.innerHTML = '';
  state.selectedFiles.forEach((f, idx) => {
    const pill = el('div', 'file-pill');
    pill.title = `${f.name} · ${fmtBytes(f.size)}`;
    pill.innerHTML = `<span>${icon('file')}</span><span class="fname">${esc(f.name)}</span><span class="fsize">${fmtBytes(f.size)}</span><button type="button" title="Убрать файл">${icon('x')}</button>`;
    pill.querySelector('button').onclick = () => {
      state.selectedFiles.splice(idx, 1);
      syncFileInput();
      renderFileSelection();
    };
    list.appendChild(pill);
  });
}
function addSelectedFiles(files) {
  const seen = new Set(state.selectedFiles.map(fileKey));
  Array.from(files || []).forEach((f) => {
    const key = fileKey(f);
    if (!seen.has(key)) {
      state.selectedFiles.push(f);
      seen.add(key);
    }
  });
  syncFileInput();
  renderFileSelection();
}

/* ---------------- UI bindings ---------------- */
function bindUI() {
  $('#runBtn').onclick = startRun;
  $('#goalInput').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) startRun(); });
  $('#steerSend').onclick = sendSteer;
  $('#steerInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') sendSteer(); });
  $('#newProjectBtn').onclick = async () => {
    const name = prompt('Название проекта:'); if (!name) return;
    const p = await (await fetch('/api/projects', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) })).json();
    state.currentProject = p; await loadProjects();
  };
  $('#settingsBtn').onclick = () => { if (!state.fullMode) { toast('Настройки моделей доступны в FullMode'); return; } openSettings(); };
  $('#fullMode').onchange = (e) => requestFullMode(e.target.checked);
  $('#runFull').onclick = confirmFullMode;
  $('#stayLite').onclick = stayLite;
  $('#fullModeOverlay').onclick = (e) => { if (e.target.id === 'fullModeOverlay') stayLite(); };
  $('#closeSettings').onclick = () => $('#settingsOverlay').classList.remove('open');
  $('#settingsOverlay').onclick = (e) => { if (e.target.id === 'settingsOverlay') $('#settingsOverlay').classList.remove('open'); };
  $('#closeSource').onclick = () => $('#sourceOverlay').classList.remove('open');
  $('#sourceOverlay').onclick = (e) => { if (e.target.id === 'sourceOverlay') $('#sourceOverlay').classList.remove('open'); };
  $('#chooseFilesBtn').onclick = () => $('#corpusFiles').click();
  $('#corpusFiles').onchange = (e) => { addSelectedFiles(e.target.files); e.target.value = ''; };
  $('#clearFilesBtn').onclick = () => { state.selectedFiles = []; syncFileInput(); renderFileSelection(); };
  renderFileSelection();
  document.querySelectorAll('.vtab').forEach(tab => tab.onclick = () => switchView(tab.dataset.view));
  document.querySelectorAll('.rtab').forEach(tab => tab.onclick = () => {
    document.querySelectorAll('.rtab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const map = { agents: '#rbodyAgents', tasks: '#rbodyTasks', log: '#rbodyLog' };
    Object.values(map).forEach(id => $(id).style.display = 'none');
    $(map[tab.dataset.tab]).style.display = 'block';
  });
  renderPhases();
  renderLivePanels();
}

init();
