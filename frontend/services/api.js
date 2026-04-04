const API_BASE_URL = 'https://predictive-maintenance-with-mlops-1.onrender.com';

class APIService {
  constructor(baseURL = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  // ── Core request with proper FastAPI error parsing ────────────────────────
  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = { ...options };

    if (!(options.body instanceof FormData)) {
      config.headers = { 'Content-Type': 'application/json', ...options.headers };
    }

    let response;
    try {
      response = await fetch(url, config);
    } catch (networkErr) {
      throw new Error(
        'Cannot reach the API. Render free tier takes ~30s to wake up after inactivity — wait and retry.'
      );
    }

    if (!response.ok) {
      let errMsg = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        // FastAPI validation errors: { detail: [{loc, msg, type}, ...] }
        if (Array.isArray(body.detail)) {
          errMsg = body.detail
            .map(e => {
              const field = Array.isArray(e.loc) ? e.loc.slice(1).join('.') : 'field';
              return `${field}: ${e.msg}`;
            })
            .join(' | ');
        } else if (typeof body.detail === 'string') {
          errMsg = body.detail;
        } else if (typeof body.message === 'string') {
          errMsg = body.message;
        }
      } catch (_) { /* response body was not JSON */ }
      throw new Error(errMsg);
    }

    return response.json();
  }

  // ── Endpoints ─────────────────────────────────────────────────────────────
  getHealth()    { return this.request('/health'); }
  getModelInfo() { return this.request('/model/info'); }
  reloadModel()  { return this.request('/model/reload', { method: 'POST' }); }

  predictRUL(data) {
    return this.request('/predict', { method: 'POST', body: JSON.stringify(data) });
  }

  // ── Batch: client-side CSV parse → POST JSON to /predict/batch ────────────
  async batchPredictCSV(file) {
    const text = await this._readFile(file);
    const records = this._parseCSV(text);

    if (!records.length) {
      throw new Error('CSV is empty or could not be parsed. Download the template to see the correct format.');
    }
    if (records.length > 100) {
      throw new Error('Batch limit is 100 rows. Please split your file.');
    }

    // Validate required columns exist
    const first = records[0];
    const required = ['engine_id', 'cycle', 'max_cycle'];
    const missing = required.filter(k => !(k in first));
    if (missing.length) {
      throw new Error(
        `CSV is missing required columns: ${missing.join(', ')}. ` +
        'Use "Download Template" to get the correct headers.'
      );
    }

    // Normalise every row — fill missing sensor fields with 0
    const normalised = records.map((row, idx) => {
      const r = {
        engine_id: parseInt(row.engine_id) || idx + 1,
        cycle:     parseInt(row.cycle)     || 1,
        max_cycle: parseInt(row.max_cycle) || 200,
      };
      for (let i = 2; i <= 25; i++) {
        const key = `sensor_${i}`;
        // Accept both "sensor_2" and "sensor2" column names
        r[key] = parseFloat(row[key] ?? row[`sensor${i}`] ?? 0) || 0;
      }
      return r;
    });

    return this.request('/predict/batch', {
      method: 'POST',
      body: JSON.stringify(normalised),
    });
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  _readFile(file) {
    return new Promise((res, rej) => {
      const r = new FileReader();
      r.onload  = e => res(e.target.result);
      r.onerror = () => rej(new Error('Could not read file.'));
      r.readAsText(file);
    });
  }

  _parseCSV(text) {
    const lines = text.trim().split(/\r?\n/).filter(Boolean);
    if (lines.length < 2) return [];

    // Normalise header names: lowercase, underscored
    const headers = lines[0]
      .split(',')
      .map(h => h.trim().toLowerCase().replace(/[\s-]+/g, '_'));

    const intFields = new Set(['engine_id', 'cycle', 'max_cycle']);

    return lines.slice(1).map(line => {
      // Simple quoted CSV parser
      const vals = [];
      let cur = '', inQ = false;
      for (const ch of line + ',') {
        if (ch === '"')               { inQ = !inQ; continue; }
        if (ch === ',' && !inQ)       { vals.push(cur.trim()); cur = ''; continue; }
        cur += ch;
      }

      const rec = {};
      headers.forEach((h, i) => {
        const raw = vals[i] ?? '';
        rec[h] = intFields.has(h) ? (parseInt(raw) || 0) : (parseFloat(raw) || 0);
      });
      return rec;
    });
  }

  downloadCSVTemplate() {
    const sensorCols = Array.from({ length: 24 }, (_, i) => `sensor_${i + 2}`).join(',');
    const header = `engine_id,cycle,max_cycle,${sensorCols}`;
    const row1   = `1,150,192,-0.0007,-0.0004,100.0,518.67,641.82,1589.7,1400.6,14.62,21.61,554.36,2388.06,9046.19,1.3,47.47,521.66,2388.02,8138.62,8.4195,0.03,392.0,2388.0,100.0,39.06,23.419`;
    const row2   = `2,80,180,-0.0009,-0.0003,100.0,518.67,641.82,1589.7,1400.6,14.62,21.61,554.36,2388.06,9046.19,1.3,47.47,521.66,2388.02,8138.62,8.4195,0.03,392.0,2388.0,100.0,39.06,23.419`;
    const csv    = [header, row1, row2].join('\n');
    const blob   = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'batch_template.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
  }
}

window.api = new APIService();
console.log('APIService ready →', API_BASE_URL);