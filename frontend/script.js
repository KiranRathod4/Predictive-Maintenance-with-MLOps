// Utility functions
function showLoading(element) {
    element.innerHTML = '<div class="flex items-center justify-center p-8"><div class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div></div>';
}

function hideLoading(element) {
    // Loading is cleared when we set new content
}

function showError(message, container) {
    container.innerHTML = `
        <div class="p-6 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <p class="text-red-800 dark:text-red-200">${message}</p>
        </div>
    `;
}

// Track predictions count globally
let totalPredictions = 0;

// Form submission handler
document.addEventListener('DOMContentLoaded', function() {
    console.log('Page loaded, initializing...');
    
    // Generate sensor inputs on predict page
    generateSensorInputs();
    
    // Initialize prediction form
    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            console.log('Form submitted!');
            
            const resultContainer = document.getElementById('prediction-result');
            showLoading(resultContainer);
            
            try {
                // Collect form data
                const formData = new FormData(form);
                const data = {
                    engine_id: parseInt(formData.get('engine_id')),
                    cycle: parseInt(formData.get('cycle')),
                    max_cycle: parseInt(formData.get('max_cycle'))
                };
                
                // Add all sensor values with correct field names
                for (let i = 2; i <= 25; i++) {
                    const sensorValue = formData.get(`sensor_${i}`);
                    data[`sensor_${i}`] = sensorValue ? parseFloat(sensorValue) : 0;
                }
                
                console.log('Sending data to API:', data);
                
                // Call API
                const result = await api.predictRUL(data);
                console.log('API Response:', result);
                
                // Increment prediction counter
                totalPredictions++;
                
                // Display result
                displayPredictionResult(result);
                
            } catch (error) {
                console.error('Prediction error:', error);
               showError(`Prediction failed: ${error.message}. Make sure the API is running at http://65.0.135.39:8000`, resultContainer);

            }
        });
        console.log('Form handler attached successfully');
    }
    
    // Initialize batch upload
    setupFileUpload();
    
    // Initialize dashboard if on dashboard page
    if (window.location.pathname.includes('index.html') || window.location.pathname.endsWith('/') || window.location.pathname === '/frontend/') {
        loadDashboardData();
    }
    
    // Initialize model metrics if on that page
    if (window.location.pathname.includes('model-metrics.html')) {
        loadModelMetrics();
    }
    
    // Initialize API health if on that page
    if (window.location.pathname.includes('api-health.html')) {
        loadApiHealth();
    }
});

// Generate sensor inputs dynamically
function generateSensorInputs() {
    const container = document.getElementById('sensor-inputs-container');
    if (!container) return;
    
    // Sample values from your correct_sample_data.json
    const sampleValues = {
        2: -0.0007, 3: -0.0004, 4: 100.0, 5: 518.67, 6: 641.82,
        7: 1589.7, 8: 1400.6, 9: 14.62, 10: 21.61, 11: 554.36,
        12: 2388.06, 13: 9046.19, 14: 1.3, 15: 47.47, 16: 521.66,
        17: 2388.02, 18: 8138.62, 19: 8.4195, 20: 0.03, 21: 392.0,
        22: 2388.0, 23: 100.0, 24: 39.06, 25: 23.419
    };
    
    for (let i = 2; i <= 25; i++) {
        const div = document.createElement('div');
        div.innerHTML = `
            <label for="sensor_${i}" class="block text-sm font-medium text-gray-700 dark:text-gray-300">
                Sensor ${i}
            </label>
            <input 
                type="number" 
                name="sensor_${i}" 
                id="sensor_${i}" 
                step="any" 
                value="${sampleValues[i]}"
                class="form-input mt-1 block w-full rounded-lg border-gray-300 bg-white/5 shadow-sm focus:border-primary focus:ring focus:ring-primary/50 dark:border-gray-700 dark:bg-background-dark/50"
                required
            >
        `;
        container.appendChild(div);
    }
}

function displayPredictionResult(result) {
    const container = document.getElementById('prediction-result');
    
    // Extract the actual RUL value
    const rulValue = result.rul_prediction || result.predicted_rul || 'N/A';
    
    container.innerHTML = `
        <div class="flex flex-col items-center justify-center p-8 text-center">
            <p class="text-lg font-medium text-gray-500 dark:text-gray-400">Predicted RUL</p>
            <p class="mt-1 text-7xl font-bold tracking-tighter text-primary">${Math.round(rulValue)}</p>
            <p class="mt-1 text-gray-500 dark:text-gray-400">cycles</p>
        </div>
        <div class="border-t border-gray-200/50 p-6 dark:border-gray-700/50">
            <dl class="grid grid-cols-1 gap-x-4 gap-y-6 sm:grid-cols-2">
                <div class="sm:col-span-1">
                    <dt class="text-sm font-medium text-gray-500 dark:text-gray-400">Model Version</dt>
                    <dd class="mt-1 text-sm text-gray-900 dark:text-white">${result.model_version || 'v1.0'}</dd>
                </div>
                <div class="sm:col-span-1">
                    <dt class="text-sm font-medium text-gray-500 dark:text-gray-400">Timestamp</dt>
                    <dd class="mt-1 text-sm text-gray-900 dark:text-white">${result.prediction_timestamp || new Date().toLocaleString()}</dd>
                </div>
                <div class="sm:col-span-2">
                    <dt class="text-sm font-medium text-gray-500 dark:text-gray-400">Engine ID</dt>
                    <dd class="mt-1 text-sm text-gray-900 dark:text-white">${result.engine_id || 'N/A'}</dd>
                </div>
                <div class="sm:col-span-2">
                    <dt class="text-sm font-medium text-gray-500 dark:text-gray-400">Status</dt>
                    <dd class="mt-1 text-sm text-green-600 dark:text-green-400 font-semibold">${result.status || 'Success'}</dd>
                </div>
            </dl>
        </div>
        <div class="flex items-center justify-center bg-gray-100 dark:bg-gray-800 rounded-b-xl p-8">
            <span class="material-symbols-outlined text-9xl text-primary opacity-50">precision_manufacturing</span>
        </div>
    `;
}

// File upload functions
function handleBatchUpload() {
    const fileInput = document.getElementById('batch-upload-input');
    if (fileInput) {
        fileInput.click();
    }
}

function setupFileUpload() {
    const dropZone = document.getElementById('batch-upload-dropzone');
    const fileInput = document.getElementById('batch-upload-input');
    
    if (!dropZone || !fileInput) return;

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-primary', 'bg-primary/10');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('border-primary', 'bg-primary/10');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-primary', 'bg-primary/10');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
}

async function handleFileSelect(file) {
    if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
        alert('Please upload a CSV file');
        return;
    }

    console.log('File selected:', file.name);
    const resultContainer = document.getElementById('batch-prediction-result');
    if (resultContainer) showLoading(resultContainer);

    try {
        // Call batch prediction API
        const response = await api.batchPredict(file);
        console.log('Batch API Response:', response);

        if (resultContainer) {
            hideLoading(resultContainer);
            resultContainer.innerHTML = `
                <h3 class="text-lg font-medium mb-2">Batch Predictions (${response.length})</h3>
                <table class="table-auto w-full text-left border-collapse border border-gray-300">
                    <thead>
                        <tr>
                            ${Object.keys(response[0] || {}).map(k => `<th class="border px-2 py-1">${k}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        ${response.map(row => `
                            <tr>
                                ${Object.values(row).map(v => `<td class="border px-2 py-1">${v}</td>`).join('')}
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        }
    } catch (error) {
        console.error('Batch prediction error:', error);
        if (resultContainer) {
            showError(`Batch prediction failed: ${error.message}`, resultContainer);
        }
    }
}

// --- Rest of your dashboard, metrics, API health functions remain unchanged ---

// Store prediction count
window.addEventListener('beforeunload', () => {
    localStorage.setItem('total_predictions', totalPredictions.toString());
});

console.log('script.js loaded successfully');
