const API_BASE_URL = 'https://predictive-maintenance-with-mlops-1.onrender.com';

class APIService {
  constructor(baseURL = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    };
    if (options.body instanceof FormData) delete config.headers['Content-Type'];

    try {
      const response = await fetch(url, config);
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      if (error.name === 'TypeError' || error.message.includes('fetch')) {
        throw new Error('Cannot reach API. Render free tier may be waking up — wait 30s and retry.');
      }
      throw error;
    }
  }

  getHealth()      { return this.request('/health'); }
  getModelInfo()   { return this.request('/model/info'); }
  reloadModel()    { return this.request('/model/reload', { method: 'POST' }); }

  predictRUL(data) {
    return this.request('/predict', { method: 'POST', body: JSON.stringify(data) });
  }

  // Parse CSV on the client, then POST as JSON to /predict/batch
  async batchPredictCSV(file) {
    const text = await this._readFile(file);
    const records = this._parseCSV(text);
    if (!records.length) throw new Error('CSV is empty or could not be parsed.');
    return this.request('/predict/batch', {
      method: 'POST',
      body: JSON.stringify(records),
    });
  }

  _readFile(file) {
    return new Promise((res, rej) => {
      const r = new FileReader();
      r.onload = e => res(e.target.result);
      r.onerror = () => rej(new Error('Failed to read file'));
      r.readAsText(file);
    });
  }

  _parseCSV(text) {
    const lines = text.trim().split('\n').filter(Boolean);
    if (lines.length < 2) return [];
    const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
    const intFields = new Set(['engine_id', 'cycle', 'max_cycle']);
    return lines.slice(1).map(line => {
      const vals = line.split(',').map(v => v.trim());
      const rec = {};
      headers.forEach((h, i) => {
        rec[h] = intFields.has(h) ? (parseInt(vals[i]) || 0) : (parseFloat(vals[i]) || 0);
      });
      return rec;
    });
  }

  // Generate and download a CSV template
  downloadCSVTemplate() {
    const sensorHeaders = Array.from({ length: 24 }, (_, i) => `sensor_${i + 2}`).join(',');
    const header = `engine_id,cycle,max_cycle,${sensorHeaders}`;
    const sampleRow = `1,150,192,-0.0007,-0.0004,100.0,518.67,641.82,1589.7,1400.6,14.62,21.61,554.36,2388.06,9046.19,1.3,47.47,521.66,2388.02,8138.62,8.4195,0.03,392.0,2388.0,100.0,39.06,23.419`;
    const blob = new Blob([header + '\n' + sampleRow], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'sample_batch_input.csv';
    a.click();
    URL.revokeObjectURL(a.href);
  }
}

window.api = new APIService();
console.log('API initialised →', API_BASE_URL);