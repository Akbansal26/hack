import json

with open("dashboard_data.json") as f:
    data_json = f.read()

with open("model_comparison_report.md") as f:
    report_md = f.read()

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Machine Failure Early-Warning System</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #f5f6f8;
    --panel: #ffffff;
    --text: #1a1d23;
    --text-muted: #5b6270;
    --border: #e2e5ea;
    --accent: #2563eb;
    --accent-2: #7c3aed;
    --good: #16a34a;
    --warn: #d97706;
    --bad: #dc2626;
    --track: #eef0f4;
    --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #0f1115;
      --panel: #171a21;
      --text: #e8eaed;
      --text-muted: #9aa1ac;
      --border: #2a2e37;
      --accent: #5b8def;
      --accent-2: #a78bfa;
      --good: #4ade80;
      --warn: #fbbf24;
      --bad: #f87171;
      --track: #1f232c;
      --shadow: 0 1px 3px rgba(0,0,0,0.4);
    }
  }
  :root[data-theme="dark"] {
    --bg: #0f1115;
    --panel: #171a21;
    --text: #e8eaed;
    --text-muted: #9aa1ac;
    --border: #2a2e37;
    --accent: #5b8def;
    --accent-2: #a78bfa;
    --good: #4ade80;
    --warn: #fbbf24;
    --bad: #f87171;
    --track: #1f232c;
    --shadow: 0 1px 3px rgba(0,0,0,0.4);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.45;
  }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 28px 20px 64px; }
  header { margin-bottom: 24px; }
  header h1 { font-size: 1.5rem; margin: 0 0 4px; letter-spacing: -0.01em; }
  header p { color: var(--text-muted); margin: 0; font-size: 0.92rem; }
  .stat-row { display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap; }
  .stat {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 16px; box-shadow: var(--shadow); min-width: 130px;
  }
  .stat .num { font-size: 1.35rem; font-weight: 650; }
  .stat .lbl { font-size: 0.76rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; }

  .panel {
    background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
    padding: 20px; margin-bottom: 20px; box-shadow: var(--shadow);
  }
  .panel h2 { font-size: 1.05rem; margin: 0 0 4px; }
  .panel .sub { color: var(--text-muted); font-size: 0.85rem; margin: 0 0 16px; }
  .grid-2 { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 20px; }
  @media (max-width: 880px) { .grid-2 { grid-template-columns: 1fr; } }

  table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
  th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
  th { color: var(--text-muted); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.03em; }
  tbody tr:hover { background: var(--track); cursor: pointer; }
  .pill {
    display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.74rem; font-weight: 600;
  }
  .pill.high { background: color-mix(in srgb, var(--bad) 18%, transparent); color: var(--bad); }
  .pill.med { background: color-mix(in srgb, var(--warn) 18%, transparent); color: var(--warn); }
  .pill.low { background: color-mix(in srgb, var(--good) 18%, transparent); color: var(--good); }
  .bar-track { background: var(--track); border-radius: 6px; height: 8px; width: 90px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 6px; }

  select {
    background: var(--panel); color: var(--text); border: 1px solid var(--border); border-radius: 8px;
    padding: 7px 10px; font-size: 0.88rem; font-family: inherit;
  }
  .chart-box { position: relative; height: 220px; margin-bottom: 6px; }
  .chart-box.tall { height: 300px; }
  .legend-note { color: var(--text-muted); font-size: 0.78rem; margin-top: 4px; }

  .tablewrap { max-height: 420px; overflow: auto; border: 1px solid var(--border); border-radius: 10px; }
  .tablewrap table { min-width: 480px; }
  .tablewrap thead th { position: sticky; top: 0; background: var(--panel); }

  .report { white-space: pre-wrap; font-size: 0.86rem; color: var(--text); }
  .report h1 { font-size: 1.05rem; }
  .report h2 { font-size: 0.95rem; margin-top: 18px; }
  details summary { cursor: pointer; font-weight: 600; padding: 4px 0; }
  footer { color: var(--text-muted); font-size: 0.78rem; text-align: center; margin-top: 28px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>&#9881;&#65039; Machine Failure Early-Warning System</h1>
    <p>Rolling-window feature model vs. sequence model, evaluated on held-out machines never seen in training.</p>
    <div class="stat-row" id="statRow"></div>
  </header>

  <div class="panel">
    <h2>Model comparison</h2>
    <p class="sub">Precision, recall, F1 and false-alarm rate on the held-out test machines (0.5 alert threshold).</p>
    <div class="chart-box tall"><canvas id="comparisonChart"></canvas></div>
    <div class="tablewrap" style="margin-top:14px;">
      <table id="leadTimeTable"></table>
    </div>
  </div>

  <div class="grid-2">
    <div class="panel">
      <h2>Maintenance priority ranking</h2>
      <p class="sub">Machines ranked by current early-warning risk score (Random Forest, engineered features).</p>
      <div class="tablewrap">
        <table>
          <thead><tr><th>Machine</th><th>Risk</th><th></th><th>Status</th></tr></thead>
          <tbody id="rankingBody"></tbody>
        </table>
      </div>
    </div>
    <div class="panel">
      <h2>Signals that changed before failure</h2>
      <p class="sub">Top features by importance in the Random Forest model.</p>
      <div class="chart-box tall"><canvas id="featureChart"></canvas></div>
    </div>
  </div>

  <div class="panel">
    <h2>Machine detail &mdash; sensor trend &amp; risk score over time</h2>
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
      <label class="sub" style="margin:0;">Machine</label>
      <select id="machineSelect"></select>
      <span class="legend-note" id="machineStatusNote"></span>
    </div>
    <div class="chart-box"><canvas id="riskChart"></canvas></div>
    <div class="chart-box"><canvas id="sensorChart"></canvas></div>
    <p class="legend-note">Dashed vertical line marks the actual failure event (for machines that failed). Shaded band on the risk chart shows the 48-hour early-warning horizon before failure.</p>
  </div>

  <div class="panel">
    <details>
      <summary>Full model comparison report</summary>
      <div class="report" id="reportText"></div>
    </details>
  </div>

  <footer>Synthetic predictive-maintenance dataset &middot; Random Forest on rolling-window features vs. a from-scratch NumPy LSTM &middot; built for the hackathon challenge brief.</footer>
</div>

<script id="dashboard-data" type="application/json">__DATA_JSON__</script>
<script id="report-md" type="application/text">__REPORT_MD__</script>

<script>
const DATA = JSON.parse(document.getElementById('dashboard-data').textContent);
document.getElementById('reportText').textContent = document.getElementById('report-md').textContent;

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// ---- Stat row ----
const statRow = document.getElementById('statRow');
const meta = DATA.meta;
const stats = [
  [meta.n_machines, 'Machines monitored'],
  [meta.n_failing, 'Failures in horizon'],
  [meta.horizon_hours + 'h', 'Warning horizon'],
  [(DATA.comparison.random_forest.f1).toFixed(2), 'Best model F1 (RF)'],
];
statRow.innerHTML = stats.map(s => `<div class="stat"><div class="num">${s[0]}</div><div class="lbl">${s[1]}</div></div>`).join('');

// ---- Model comparison chart ----
const models = ['naive_threshold', 'random_forest', 'lstm'];
const modelLabels = {naive_threshold: 'Naive threshold', random_forest: 'Random Forest', lstm: 'LSTM (numpy)'};
const metricsKeys = ['precision', 'recall', 'f1', 'false_alarm_rate'];
const metricLabels = {precision: 'Precision', recall: 'Recall', f1: 'F1', false_alarm_rate: 'False-alarm rate'};
const colors = [cssVar('--text-muted'), cssVar('--accent'), cssVar('--accent-2')];

new Chart(document.getElementById('comparisonChart'), {
  type: 'bar',
  data: {
    labels: metricsKeys.map(k => metricLabels[k]),
    datasets: models.map((m, idx) => ({
      label: modelLabels[m],
      data: metricsKeys.map(k => DATA.comparison[m][k]),
      backgroundColor: colors[idx],
      borderRadius: 4,
    }))
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    scales: {
      y: { beginAtZero: true, max: 1, ticks: { color: cssVar('--text-muted') }, grid: { color: cssVar('--track') } },
      x: { ticks: { color: cssVar('--text') }, grid: { display: false } }
    },
    plugins: { legend: { labels: { color: cssVar('--text') } } }
  }
});

// ---- Lead time table ----
const leadTable = document.getElementById('leadTimeTable');
let rows = '<thead><tr><th>Model</th><th>Mean lead time</th><th>Median lead time</th><th>Failures missed</th><th>ROC AUC</th></tr></thead><tbody>';
models.forEach(m => {
  const c = DATA.comparison[m];
  const lt = c.lead_time;
  rows += `<tr><td>${modelLabels[m]}</td><td>${lt.mean_lead_hours !== null ? lt.mean_lead_hours.toFixed(0)+'h' : '—'}</td>` +
          `<td>${lt.median_lead_hours !== null ? lt.median_lead_hours.toFixed(0)+'h' : '—'}</td>` +
          `<td>${lt.n_missed} / ${lt.n_failing_machines}</td>` +
          `<td>${c.roc_auc !== undefined && c.roc_auc !== null ? c.roc_auc.toFixed(3) : '—'}</td></tr>`;
});
rows += '</tbody>';
leadTable.innerHTML = rows;

// ---- Feature importance chart ----
const feats = Object.entries(DATA.top_features).sort((a,b) => b[1]-a[1]).slice(0, 12).reverse();
new Chart(document.getElementById('featureChart'), {
  type: 'bar',
  data: {
    labels: feats.map(f => f[0]),
    datasets: [{ data: feats.map(f => f[1]), backgroundColor: cssVar('--accent'), borderRadius: 4 }]
  },
  options: {
    indexAxis: 'y', responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: cssVar('--text-muted') }, grid: { color: cssVar('--track') } },
      y: { ticks: { color: cssVar('--text'), font: { size: 10.5 } }, grid: { display: false } }
    }
  }
});

// ---- Ranking table ----
const rankingBody = document.getElementById('rankingBody');
const sortedRanking = [...DATA.ranking].sort((a,b) => b.risk_score - a.risk_score);
function riskPill(score) {
  if (score >= 0.5) return ['high', 'High'];
  if (score >= 0.15) return ['med', 'Watch'];
  return ['low', 'Healthy'];
}
rankingBody.innerHTML = sortedRanking.map(r => {
  const [cls, label] = riskPill(r.risk_score);
  const pct = Math.round(r.risk_score * 100);
  const barColor = cls === 'high' ? cssVar('--bad') : cls === 'med' ? cssVar('--warn') : cssVar('--good');
  return `<tr data-machine="${r.machine_id}">
    <td><strong>${r.machine_id}</strong></td>
    <td>${(r.risk_score*100).toFixed(1)}%</td>
    <td><div class="bar-track"><div class="bar-fill" style="width:${pct}%; background:${barColor};"></div></div></td>
    <td><span class="pill ${cls}">${label}</span></td>
  </tr>`;
}).join('');

// ---- Machine detail view ----
const machineSelect = document.getElementById('machineSelect');
const machineIds = Object.keys(DATA.machines).sort();
machineSelect.innerHTML = machineIds.map(m => `<option value="${m}">${m}${DATA.machines[m].will_fail ? ' (failed)' : ''}</option>`).join('');

let riskChart, sensorChart;
function renderMachine(mid) {
  const m = DATA.machines[mid];
  document.getElementById('machineStatusNote').textContent = m.will_fail
    ? `Failure occurred at hour ${m.failure_step}.`
    : 'No failure observed in this window (healthy machine).';

  const horizonStart = m.will_fail ? Math.max(0, m.failure_step - 48) : null;

  if (riskChart) riskChart.destroy();
  riskChart = new Chart(document.getElementById('riskChart'), {
    type: 'line',
    data: {
      labels: m.steps,
      datasets: [{
        label: 'Early-warning risk score',
        data: m.risk,
        borderColor: cssVar('--accent'),
        backgroundColor: 'transparent',
        pointRadius: 0,
        borderWidth: 2,
        tension: 0.15,
        spanGaps: true,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: cssVar('--text') } },
        annotation: undefined,
      },
      scales: {
        y: { min: 0, max: 1, title: { display: true, text: 'Risk', color: cssVar('--text-muted') }, ticks: { color: cssVar('--text-muted') }, grid: { color: cssVar('--track') } },
        x: { title: { display: true, text: 'Hour', color: cssVar('--text-muted') }, ticks: { color: cssVar('--text-muted'), maxTicksLimit: 10 }, grid: { display: false } }
      }
    }
  });

  const sensorDatasets = [
    { key: 'vibration', color: cssVar('--accent'), label: 'Vibration (g)' },
    { key: 'temperature', color: cssVar('--bad'), label: 'Temperature (°C)' },
    { key: 'current', color: cssVar('--warn'), label: 'Current (A)' },
    { key: 'pressure', color: cssVar('--good'), label: 'Pressure (PSI)' },
  ];
  if (sensorChart) sensorChart.destroy();
  sensorChart = new Chart(document.getElementById('sensorChart'), {
    type: 'line',
    data: {
      labels: m.steps,
      datasets: sensorDatasets.map(s => ({
        label: s.label,
        data: m[s.key],
        borderColor: s.color,
        backgroundColor: 'transparent',
        pointRadius: 0,
        borderWidth: 1.5,
        tension: 0.1,
        yAxisID: s.key === 'temperature' ? 'y1' : 'y',
      }))
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: cssVar('--text'), boxWidth: 12, font: { size: 10.5 } } } },
      scales: {
        y: { title: { display: true, text: 'vib / current / pressure', color: cssVar('--text-muted') }, ticks: { color: cssVar('--text-muted') }, grid: { color: cssVar('--track') } },
        y1: { position: 'right', title: { display: true, text: 'temp (°C)', color: cssVar('--text-muted') }, ticks: { color: cssVar('--text-muted') }, grid: { display: false } },
        x: { title: { display: true, text: 'Hour', color: cssVar('--text-muted') }, ticks: { color: cssVar('--text-muted'), maxTicksLimit: 10 }, grid: { display: false } }
      }
    }
  });
}

machineSelect.addEventListener('change', e => renderMachine(e.target.value));
document.getElementById('rankingBody').addEventListener('click', e => {
  const tr = e.target.closest('tr');
  if (!tr) return;
  const mid = tr.getAttribute('data-machine');
  machineSelect.value = mid;
  renderMachine(mid);
  document.querySelector('.panel h2 + p + div canvas#riskChart')?.scrollIntoView({behavior:'smooth', block:'center'});
});

// initial: highest-risk failing machine
renderMachine(sortedRanking[0].machine_id);
machineSelect.value = sortedRanking[0].machine_id;
</script>
</body>
</html>
"""

html = html.replace("__DATA_JSON__", data_json).replace("__REPORT_MD__", report_md.replace("</script>", "<\\/script>"))

with open("/mnt/user-data/outputs/machine_failure_dashboard.html", "w") as f:
    f.write(html)

print("wrote dashboard, size MB:", len(html) / 1e6)
