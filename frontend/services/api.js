class APIService {
    constructor(baseURL = 'http://localhost:8000') {
        this.baseURL = baseURL;
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
        };

        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API Request failed:', error);
            throw error;
        }
    }

    async getHealth() { return await this.request('/health'); }
    async getModelInfo() { return await this.request('/model/info'); }
    async predictRUL(data) { return await this.request('/predict', { method: 'POST', body: JSON.stringify(data) }); }
    async batchPredict(file) {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch(`${this.baseURL}/predict/batch`, { method: 'POST', body: formData });
        if (!response.ok) throw new Error('Batch prediction failed');
        return await response.json();
    }
    async reloadModel() { return await this.request('/model/reload', { method: 'POST' }); }
    async getMetrics() { return await this.request('/metrics'); }
}

// Create global API instance
window.api = new APIService(); // <== Attach directly to window as "api"
console.log('API Service initialized');
