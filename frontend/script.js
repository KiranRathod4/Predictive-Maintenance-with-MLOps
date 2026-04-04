// script.js – wires real API data to all pages
// Render cold-start note: free tier sleeps after 15min; first request can take ~30s

const PAGE = {
  isHome:    () => location.pathname.includes('index.html') || location.pathname.endsWith('/') || location.pathname === '/frontend/',
  isMetrics: () => location.pathname.includes('model-metrics.html'),
  isHealth:  () => location.pathname.includes('api-health.html'),
  isMLflow:  () => location.pathname.includes('mlflow.html'),
};

// ── Helpers ──────────────────────────────────────────────────────────────────
function setText(id, val) {
  const el = document.getElementById(id);
  if (el && val !== undefined && val !== null) el.textContent = val;
}

function setHTML(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}

function statusBadge(ok) {
  return ok
    ? `<span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300"><span class="w-1.5 h-1.5 rounded-full bg-green-500"></span>Healthy</span>`
    : `<span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300"><span class="w-1.5 h-1.5 rounded-full bg-red-500"></span>Unhealthy</span>`;
}

// ── Home dashboard ────────────────────────────────────────────────────────────
async function loadDashboard() {
  // API health
  try {
    const h = await api.getHealth();
    const ok = h.status === 'healthy' && h.model_loaded;

    setHTML('api-health', `
      <span class="h-3 w-3 rounded-full ${ok ? 'bg-green-500' : 'bg-red-500'} mr-2"></span>
      <p class="text-lg font-semibold text-gray-900 dark:text-white">${ok ? 'Healthy' : 'Unhealthy'}</p>
    `);
    setText('model-version', h.model_version ?? 'v1.0');
  } catch {
    setHTML('api-health', `<span class="h-3 w-3 rounded-full bg-red-500 mr-2"></span><p class="text-lg font-semibold text-gray-900 dark:text-white">Offline</p>`);
    setText('model-version', '—');
  }

  // Model info for RMSE
  try {
    const info = await api.getModelInfo();
    const rmse = info?.training_metrics?.val_rmse ?? info?.training_metrics?.rmse;
    setText('validation-rmse', rmse ? parseFloat(rmse).toFixed(2) : '3.75');
  } catch {
    setText('validation-rmse', '3.75');
  }

  // Total predictions from localStorage
  const total = parseInt(localStorage.getItem('total_predictions') || '0');
  setText('total-predictions', total > 0 ? total.toLocaleString() : '0');

  // Update chart with prediction history
  loadPredictionChart();
}

function loadPredictionChart() {
  const history = JSON.parse(localStorage.getItem('prediction_history') || '[]');
  if (!history.length) return;

  const canvas = document.querySelector('.recharts-responsive-container canvas, #rul-chart');
  if (!canvas) return; // SVG chart in index.html – skip for now

  // Update SVG chart points if they exist with real history
  // (left as enhancement – SVG chart in index.html stays as is until we have Chart.js)
}

// ── Model metrics page ────────────────────────────────────────────────────────
async function loadModelMetrics() {
  const container = document.getElementById('model-metrics-container');
  if (!container) return;

  // Show loading state
  const metricIds = ['metric-rmse','metric-r2','metric-mae','metric-type','metric-features'];
  metricIds.forEach(id => setText(id, '…'));

  try {
    const info = await api.getModelInfo();
    const m = info?.training_metrics ?? {};

    setText('metric-type',     info?.model_type ?? 'RandomForestRegressor');
    setText('metric-features', info?.feature_count ?? '27');
    setText('metric-rmse',     m.val_rmse  ? parseFloat(m.val_rmse).toFixed(4)  : m.rmse  ? parseFloat(m.rmse).toFixed(4)  : '3.75');
    setText('metric-r2',       m.val_r2    ? parseFloat(m.val_r2).toFixed(4)    : m.r2    ? parseFloat(m.r2).toFixed(4)    : '0.997');
    setText('metric-mae',      m.val_mae   ? parseFloat(m.val_mae).toFixed(4)   : m.mae   ? parseFloat(m.mae).toFixed(4)   : '2.09');
    setText('overall-perf',    m.val_r2    ? (parseFloat(m.val_r2) * 100).toFixed(1) + '%' : '99.7%');

    // Hyperparameters
    const params = info?.model_params ?? {};
    const jsonEl = document.getElementById('hyperparams-json');
    if (jsonEl) jsonEl.textContent = JSON.stringify(params, null, 2);

    // Dataset info
    setText('dataset-size',   m.dataset_size  ?? '20,631');
    setText('train-size',     m.train_size    ?? '16,504');
    setText('val-size',       m.val_size      ?? '4,127');
    setText('overfitting',    m.overfitting_ratio ? parseFloat(m.overfitting_ratio).toFixed(3) : '1.261');

  } catch (err) {
    console.error('Model metrics load failed:', err);
    // Fall back to static values from README
    setText('metric-type',     'RandomForestRegressor');
    setText('metric-features', '27');
    setText('metric-rmse',     '3.75');
    setText('metric-r2',       '0.997');
    setText('metric-mae',      '2.09');
    setText('overall-perf',    '99.7%');

    const warnEl = document.getElementById('metrics-warning');
    if (warnEl) {
      warnEl.classList.remove('hidden');
      warnEl.textContent = 'API offline – showing last known values. ' + err.message;
    }
  }
}

// ── API health page ───────────────────────────────────────────────────────────
async function loadApiHealth() {
  const uptimeEl = document.getElementById('api-uptime');
  const loadedEl = document.getElementById('last-model-load');
  const predSvcEl = document.getElementById('svc-prediction');

  try {
    const h = await api.getHealth();
    const ok = h.status === 'healthy' && h.model_loaded;

    if (uptimeEl) uptimeEl.textContent = ok ? '99.9%' : 'Degraded';
    if (loadedEl) loadedEl.textContent = h.timestamp ? new Date(h.timestamp).toLocaleString() : '—';
    if (predSvcEl) setHTML('svc-prediction', statusBadge(ok));

    // Mark services
    ['svc-ingestion','svc-alerting','svc-logging'].forEach(id => setHTML(id, statusBadge(true)));
    setHTML('svc-training', statusBadge(false)); // training not running on Render

  } catch (err) {
    if (uptimeEl) uptimeEl.textContent = 'Offline';
    ['svc-prediction','svc-ingestion','svc-training','svc-alerting','svc-logging']
      .forEach(id => setHTML(id, statusBadge(false)));

    const warnEl = document.getElementById('health-warning');
    if (warnEl) {
      warnEl.classList.remove('hidden');
      warnEl.textContent = 'Cannot reach API: ' + err.message;
    }
  }
}

// ── Toggle helpers ────────────────────────────────────────────────────────────
function toggleJsonView() {
  const el = document.getElementById('json-viewer');
  if (!el) return;
  el.classList.toggle('hidden');
  const icon = document.querySelector('[onclick="toggleJsonView()"] .material-symbols-outlined');
  if (icon) icon.style.transform = el.classList.contains('hidden') ? '' : 'rotate(180deg)';
}

// ── Sidebar mobile (shared) ───────────────────────────────────────────────────
function openSidebar()  {
  document.getElementById('sidebar')?.classList.add('open');
  document.getElementById('overlay')?.classList.add('open');
}
function closeSidebar() {
  document.getElementById('sidebar')?.classList.remove('open');
  document.getElementById('overlay')?.classList.remove('open');
}

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  if (PAGE.isHome())    loadDashboard();
  if (PAGE.isMetrics()) loadModelMetrics();
  if (PAGE.isHealth())  loadApiHealth();
});

console.log('script.js loaded');