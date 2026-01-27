// Global variables
let currentFile = null;
let modelReady = false;
let isProcessing = false;

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const predictBtn = document.getElementById('predictBtn');
const explainBtn = document.getElementById('explainBtn');
const clearBtn = document.getElementById('clearBtn');
const predictionResults = document.getElementById('predictionCard');
const explanationResults = document.getElementById('explanationCard');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingMessage = document.getElementById('loadingMessage');
const modelStatus = document.getElementById('modelStatus');
const aboutModal = document.getElementById('aboutModal');
const closeModalBtn = document.querySelector('.modal-close');
const aboutLink = document.querySelector('[href="#about"]');

// API Configuration
const API_BASE_URL = 'http://localhost:8000';
const SUPPORTED_FORMATS = ['image/jpeg', 'image/png', 'image/bmp'];
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    checkModelStatus();
});

// Initialize application
function initializeApp() {
    console.log('🚀 AI Action Recognition App initialized');
    hideElements([previewContainer, predictionResults, explanationResults, loadingOverlay]);
    hideElement(document.getElementById('resultsSection'));
}

// Setup event listeners
function setupEventListeners() {
    // File upload handlers
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    
    // Button handlers
    predictBtn.addEventListener('click', handlePredict);
    explainBtn.addEventListener('click', handleExplain);
    clearBtn.addEventListener('click', handleClear);
    
    // Modal handlers
    aboutLink.addEventListener('click', (e) => {
        e.preventDefault();
        openModal();
    });
    closeModalBtn.addEventListener('click', closeModal);
    aboutModal.addEventListener('click', (e) => {
        if (e.target === aboutModal) closeModal();
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyPress);
}

// Check model status
async function checkModelStatus() {
    try {
        updateStatus('loading', 'Checking model status...');
        
        const response = await fetch(`${API_BASE_URL}/health`);
        
        if (response.ok) {
            const data = await response.json();
            modelReady = data.model_loaded;
            
            if (modelReady) {
                updateStatus('ready', 'Model ready for predictions');
                updateModelDetails(data);
            } else {
                updateStatus('error', 'Model not loaded');
            }
        } else {
            throw new Error(`HTTP ${response.status}`);
        }
    } catch (error) {
        console.error('❌ Model status check failed:', error);
        updateStatus('error', 'API connection failed');
        modelReady = false;
    }
}

// Update model status display
function updateStatus(type, message) {
    const statusElement = document.getElementById('modelStatus');
    const statusMessages = {
        loading: { icon: 'fas fa-spinner fa-spin', class: 'status-loading' },
        ready: { icon: 'fas fa-check-circle', class: 'status-ready' },
        error: { icon: 'fas fa-exclamation-triangle', class: 'status-error' }
    };
    
    const status = statusMessages[type];
    statusElement.className = status.class;
    statusElement.innerHTML = `
        <i class="${status.icon}"></i>
        <span>${message}</span>
    `;
}

// Update model details
function updateModelDetails(data) {
    const detailsContainer = document.getElementById('statusDetails');
    if (detailsContainer && data.model_info) {
        detailsContainer.innerHTML = `
            <div class="status-item">
                <div class="status-value">${data.model_info.accuracy || 'N/A'}</div>
                <div class="status-label">Accuracy</div>
            </div>
            <div class="status-item">
                <div class="status-value">${data.model_info.classes || 10}</div>
                <div class="status-label">Action Classes</div>
            </div>
            <div class="status-item">
                <div class="status-value">${data.model_info.version || '1.0'}</div>
                <div class="status-label">Model Version</div>
            </div>
        `;
        detailsContainer.style.display = 'grid';
    }
}

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    if (!uploadArea.contains(e.relatedTarget)) {
        uploadArea.classList.remove('dragover');
    }
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// File selection handler
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// File validation and preview
function handleFile(file) {
    if (!validateFile(file)) return;
    
    currentFile = file;
    showFilePreview(file);
    hideElements([predictionResults, explanationResults]);
    hideElement(document.getElementById('resultsSection'));
}

// Validate selected file
function validateFile(file) {
    if (!SUPPORTED_FORMATS.includes(file.type)) {
        showError('Please select a valid image file (JPEG, PNG, or BMP).');
        return false;
    }
    
    if (file.size > MAX_FILE_SIZE) {
        showError('File size must be less than 10MB.');
        return false;
    }
    
    return true;
}

// Show file preview
function showFilePreview(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.alt = file.name;
        showElement(previewContainer);
        
        // Update UI state
        predictBtn.disabled = !modelReady;
        explainBtn.disabled = !modelReady;
        clearBtn.disabled = false;
        
        console.log(`📷 Image loaded: ${file.name} (${formatFileSize(file.size)})`);
    };
    
    reader.onerror = () => {
        showError('Error reading the selected file.');
    };
    
    reader.readAsDataURL(file);
}

// Handle prediction request
async function handlePredict() {
    if (!currentFile || !modelReady || isProcessing) return;
    
    try {
        isProcessing = true;
        showLoading('Analyzing image...', 'Our AI model is identifying the action in your image');
        disableButtons();
        
        const formData = new FormData();
        formData.append('file', currentFile);
        
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        displayPredictionResults(result);
        
        console.log('✅ Prediction successful:', result);
        
    } catch (error) {
        console.error('❌ Prediction failed:', error);
        showError(`Prediction failed: ${error.message}`);
    } finally {
        hideLoading();
        enableButtons();
        isProcessing = false;
    }
}

// Handle explanation request
async function handleExplain() {
    if (!currentFile || !modelReady || isProcessing) return;
    
    try {
        isProcessing = true;
        showLoading('Generating explanation...', 'LIME is analyzing which parts of the image influenced the prediction');
        disableButtons();
        
        const formData = new FormData();
        formData.append('file', currentFile);
        
        const response = await fetch(`${API_BASE_URL}/explain`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        displayExplanationResults(result);
        
        console.log('✅ Explanation successful:', result);
        
    } catch (error) {
        console.error('❌ Explanation failed:', error);
        showError(`Explanation failed: ${error.message}`);
    } finally {
        hideLoading();
        enableButtons();
        isProcessing = false;
    }
}

// Display prediction results
function displayPredictionResults(result) {
    const container = document.getElementById('predictionCard');
    
    container.innerHTML = `
        <div class="card-header">
            <h3><i class="fas fa-brain"></i>Prediction Results</h3>
            <span class="confidence-score">${(result.confidence * 100).toFixed(1)}% confidence</span>
        </div>
        <div class="card-content">
            <div class="main-prediction">
                <div class="predicted-action">${result.predicted_action}</div>
                <div class="confidence-score">Primary prediction</div>
            </div>
            <div class="confidence-chart">
                ${generateConfidenceChart(result.top_predictions)}
            </div>
            <div class="model-info">
                <small>Model Accuracy: ${result.model_accuracy}</small>
            </div>
        </div>
    `;
    
    
    showElement(predictionResults);
    showElement(document.getElementById('resultsSection')); // Show the results section
    
    // Animate confidence bars
    setTimeout(() => {
        document.querySelectorAll('.confidence-fill').forEach(bar => {
            const width = bar.getAttribute('data-width');
            bar.style.width = width;
        });
    }, 100);
}

// Generate confidence chart HTML
function generateConfidenceChart(topPredictions) {
    if (!topPredictions || !Array.isArray(topPredictions)) return '';
    
    // Take up to 5 predictions
    const predictions = topPredictions.slice(0, 5);
    
    return predictions.map(prediction => {
        const percentage = (prediction.confidence * 100);
        return `
            <div class="confidence-item">
                <div class="confidence-label">${prediction.action}</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" data-width="${percentage}%" style="width: 0%"></div>
                </div>
                <div class="confidence-value">${percentage.toFixed(1)}%</div>
            </div>
        `;
    }).join('');
}

// Display explanation results
function displayExplanationResults(result) {
    const container = document.getElementById('explanationCard');
    
    container.innerHTML = `
        <div class="card-header">
            <h3><i class="fas fa-eye"></i>Visual Explanation</h3>
            <div class="explanation-info">
                <i class="fas fa-info-circle"></i>
                <div class="tooltip">Green areas support the prediction, red areas contradict it</div>
            </div>
        </div>
        <div class="card-content">
            <div class="explanation-content">
                <img src="data:image/png;base64,${result.explanation_image}" 
                     alt="LIME Explanation" 
                     class="lime-visualization">
                <div class="explanation-text">
                    <p><strong>How to interpret:</strong> The colored regions show which parts of the image most influenced the AI's decision. Green areas support the predicted action "${result.predicted_action}", while red areas suggest alternative interpretations.</p>
                </div>
            </div>
        </div>
    `;
    
    showElement(explanationResults);
    showElement(document.getElementById('resultsSection')); // Show the results section
}

// Clear current session
function handleClear() {
    currentFile = null;
    fileInput.value = '';
    
    hideElements([previewContainer, predictionResults, explanationResults]);
    hideElement(document.getElementById('resultsSection'));
    uploadArea.classList.remove('dragover');
    
    // Reset button states
    predictBtn.disabled = true;
    explainBtn.disabled = true;
    clearBtn.disabled = true;
    
    console.log('🧹 Session cleared');
}

// Keyboard shortcuts
function handleKeyPress(e) {
    if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
            case 'o': // Ctrl/Cmd + O: Open file
                e.preventDefault();
                fileInput.click();
                break;
            case 'Enter': // Ctrl/Cmd + Enter: Predict
                e.preventDefault();
                if (!predictBtn.disabled) handlePredict();
                break;
            case 'e': // Ctrl/Cmd + E: Explain
                e.preventDefault();
                if (!explainBtn.disabled) handleExplain();
                break;
        }
    }
    
    if (e.key === 'Escape') {
        if (aboutModal.classList.contains('active')) {
            closeModal();
        }
    }
}

// Loading overlay functions
function showLoading(title, message) {
    document.getElementById('loadingTitle').textContent = title;
    document.getElementById('loadingMessage').textContent = message;
    showElement(loadingOverlay);
}

function hideLoading() {
    hideElement(loadingOverlay);
}

// Button state management
function disableButtons() {
    [predictBtn, explainBtn, clearBtn].forEach(btn => btn.disabled = true);
}

function enableButtons() {
    predictBtn.disabled = !modelReady || !currentFile;
    explainBtn.disabled = !modelReady || !currentFile;
    clearBtn.disabled = !currentFile;
}

// Modal functions
function openModal() {
    aboutModal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    aboutModal.classList.remove('active');
    document.body.style.overflow = '';
}

// Utility functions
function showElement(element) {
    if (element) {
        element.style.display = 'block';
    }
}

function hideElement(element) {
    if (element) {
        element.style.display = 'none';
    }
}

function hideElements(elements) {
    elements.forEach(hideElement);
}

function showError(message) {
    // Create or update error toast
    let errorToast = document.getElementById('errorToast');
    
    if (!errorToast) {
        errorToast = document.createElement('div');
        errorToast.id = 'errorToast';
        errorToast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #f56565;
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(245, 101, 101, 0.3);
            z-index: 4000;
            max-width: 400px;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        `;
        document.body.appendChild(errorToast);
    }
    
    errorToast.innerHTML = `
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <i class="fas fa-exclamation-circle"></i>
            <span>${message}</span>
        </div>
    `;
    
    // Show toast
    setTimeout(() => {
        errorToast.style.transform = 'translateX(0)';
    }, 10);
    
    // Auto hide after 5 seconds
    setTimeout(() => {
        if (errorToast) {
            errorToast.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (errorToast && errorToast.parentNode) {
                    errorToast.parentNode.removeChild(errorToast);
                }
            }, 300);
        }
    }, 5000);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Health check interval
setInterval(() => {
    if (!isProcessing) {
        checkModelStatus();
    }
}, 30000); // Check every 30 seconds

// Console welcome message
console.log(`
🎯 AI Action Recognition Platform
Version: 1.0.0
API Endpoint: ${API_BASE_URL}
Supported Formats: ${SUPPORTED_FORMATS.join(', ')}

Keyboard Shortcuts:
• Ctrl/Cmd + O: Open file
• Ctrl/Cmd + Enter: Predict action
• Ctrl/Cmd + E: Generate explanation
• Escape: Close modal

Ready for action recognition! 🚀
`);