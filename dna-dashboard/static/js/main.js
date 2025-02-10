let isTraining = false;
let socket;
let lossChart = null;
let accuracyChart = null;
let trainingHistory = {
    loss: [],
    val_loss: [],
    accuracy: [],
    val_accuracy: []
};

async function loadReferenceData() {
    try {
        const response = await fetch('/api/list-reference-data');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        const select = document.getElementById('referenceDataSelect'); // Changed from 'referenceData'
        select.innerHTML = '<option value="">Select reference data...</option>';
        
        console.log('Reference data response:', data); // Debug logging
        
        if (data.files && data.files.length > 0) {
            data.files.forEach(file => {
                const option = document.createElement('option');
                option.value = file.path;
                option.textContent = file.name;
                select.appendChild(option);
            });
            select.disabled = false; // Enable select when data is available
        } else {
            console.warn('No reference files found.', {
                dataDirectoryExists: data.data_dir_exists,
                dataDirectoryPath: data.data_dir_path
            });
            const option = document.createElement('option');
            option.value = "";
            option.textContent = "No reference files found";
            option.disabled = true;
            select.appendChild(option);
        }

        // Show download status if needed
        const statusDiv = document.getElementById('downloadStatus');
        if (data.download_status) {
            statusDiv.textContent = data.download_status.message;
            statusDiv.style.display = 'block';
        } else {
            statusDiv.style.display = 'none';
        }

    } catch (error) {
        console.error('Error loading reference data:', error);
        const select = document.getElementById('referenceDataSelect');
        select.innerHTML = '<option value="">Error loading reference data</option>';
    }
}

async function checkDownloadStatus() {
    try {
        const response = await fetch('/api/download-status');
        const status = await response.json();
        
        const select = document.getElementById('referenceDataSelect'); // Changed from 'referenceData'
        
        if (status.status === 'completed') {
            await loadReferenceData();
        } else {
            select.innerHTML = `<option value="">${status.message}</option>`;
            select.disabled = true;
            
            // Show status message
            const statusDiv = document.getElementById('downloadStatus');
            statusDiv.textContent = status.message;
            statusDiv.style.display = 'block';
            
            if (status.status !== 'error') {
                // Check again in 5 seconds
                setTimeout(checkDownloadStatus, 5000);
            }
        }
    } catch (error) {
        console.error('Error checking download status:', error);
    }
}

function updateProgressBar(progress) {
    const progressContainer = document.getElementById('trainingProgress');
    const progressBar = progressContainer.querySelector('.progress-bar');
    
    // Show training progress section
    progressContainer.style.display = 'block';
    
    // Update progress bar
    progressBar.style.width = `${progress}%`;
    progressBar.setAttribute('aria-valuenow', progress);
}

async function checkModelStatus() {
    try {
        const response = await fetch('/api/model-status');
        const status = await response.json();
        
        const predictButton = document.getElementById('predictButton');
        const modelStatus = document.createElement('div');
        
        // Add warning if new data is selected
        if (status.trained && (
            document.getElementById('referenceDataSelect').value ||
            document.getElementById('trainingFile').files.length
        )) {
            status.message += ' (Model will need retraining with new data)';
            modelStatus.className = 'alert alert-warning mt-2';
        } else {
            modelStatus.className = `alert alert-${status.trained ? 'success' : 'warning'} mt-2`;
        }
        
        modelStatus.textContent = status.message;
        
        const predictionSection = predictButton.parentElement;
        const existingStatus = predictionSection.querySelector('.alert');
        if (existingStatus) {
            existingStatus.remove();
        }
        predictionSection.insertBefore(modelStatus, predictButton);
        
        return status.trained;
    } catch (error) {
        console.error('Error checking model status:', error);
        return false;
    }
}

function updateChart(newData) {
    // Remove old createTrainingChart call and just use updateMetrics
    updateMetrics({
        ...newData,
        total_epochs: window.last_training_epochs || 50
    });
}

function initializeCharts() {
    const lossCanvas = document.getElementById('lossChart');
    const accCanvas = document.getElementById('accuracyChart');

    if (!lossCanvas || !accCanvas) {
        console.error('Chart canvases not found');
        return;
    }

    // Destroy existing charts if they exist
    if (lossChart) {
        lossChart.destroy();
        lossChart = null;
    }
    if (accuracyChart) {
        accuracyChart.destroy();
        accuracyChart = null;
    }
    
    const lossCtx = lossCanvas.getContext('2d');
    const accCtx = accCanvas.getContext('2d');
    
    // Create new charts with error handling
    try {
        lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Loss',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1
                }, {
                    label: 'Validation Loss',
                    data: [],
                    borderColor: 'rgb(54, 162, 235)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                animation: {
                    duration: 0
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
        
        accuracyChart = new Chart(accCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Accuracy',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }, {
                    label: 'Validation Accuracy',
                    data: [],
                    borderColor: 'rgb(153, 102, 255)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                animation: {
                    duration: 0
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error initializing charts:', error);
    }
}

// Update metrics function with error handling
function updateMetrics(data) {
    try {
        const elements = {
            epoch: document.getElementById('currentEpoch'),
            loss: document.getElementById('currentLoss'),
            accuracy: document.getElementById('currentAccuracy'),
            lr: document.getElementById('currentLR')
        };

        // Check if all elements exist
        if (!Object.values(elements).every(el => el)) {
            console.error('One or more metric elements not found');
            return;
        }

        elements.epoch.textContent = `${data.epoch + 1}/${data.total_epochs}`;
        elements.loss.textContent = data.loss.toFixed(4);
        elements.accuracy.textContent = (data.accuracy * 100).toFixed(2) + '%';
        elements.lr.textContent = data.learning_rate?.toExponential(2) || '-';
        
        // Update history safely
        if (!trainingHistory) {
            trainingHistory = {
                loss: [], val_loss: [],
                accuracy: [], val_accuracy: []
            };
        }
        
        // Update history
        trainingHistory.loss.push(data.loss);
        trainingHistory.val_loss.push(data.val_loss);
        trainingHistory.accuracy.push(data.accuracy);
        trainingHistory.val_accuracy.push(data.val_accuracy);
        
        // Update charts
        const epochs = Array.from({length: data.epoch + 1}, (_, i) => i + 1);
        
        lossChart.data.labels = epochs;
        lossChart.data.datasets[0].data = trainingHistory.loss;
        lossChart.data.datasets[1].data = trainingHistory.val_loss;
        lossChart.update('none');
        
        accuracyChart.data.labels = epochs;
        accuracyChart.data.datasets[0].data = trainingHistory.accuracy;
        accuracyChart.data.datasets[1].data = trainingHistory.val_accuracy;
        accuracyChart.update('none');

    } catch (error) {
        console.error('Error updating metrics:', error);
    }
}

function initializeSocket() {
    socket = io();
    
    socket.on('connect', () => {
        console.log('Connected to server');
    });
    
    socket.on('training_progress', (data) => {
        console.log('Training progress:', data);
        
        // Show training progress section
        document.getElementById('trainingProgress').style.display = 'block';
        
        // Update metrics and charts
        updateMetrics({
            epoch: data.epoch,
            loss: data.loss,
            accuracy: data.accuracy,
            val_loss: data.val_loss,
            val_accuracy: data.val_accuracy,
            learning_rate: data.learning_rate,
            total_epochs: window.last_training_epochs || 50
        });
        
        // Update progress bar
        const progress = ((data.epoch + 1) / (window.last_training_epochs || 50)) * 100;
        updateProgressBar(progress);
    });
    
    socket.on('download_progress', (data) => {
        const container = document.getElementById('downloadProgressContainer');
        const progressBar = document.getElementById('downloadProgressBar');
        const progressText = document.getElementById('downloadProgressText');
        
        if (data.progress < 0) {
            // Error state
            container.style.display = 'none';
            alert('Download error: ' + data.message);
            return;
        }
        
        container.style.display = 'block';
        progressBar.style.width = `${data.progress}%`;
        progressText.textContent = data.message;
        
        if (data.progress === 100) {
            // Download complete, refresh reference data
            setTimeout(() => {
                container.style.display = 'none';
                loadReferenceData();
            }, 1000);
        }
    });
}

function initializeEventListeners() {
    const referenceSelect = document.getElementById('referenceDataSelect');
    const trainReferenceButton = document.getElementById('trainReferenceButton');
    const trainingFile = document.getElementById('trainingFile');
    const labelFile = document.getElementById('labelFile');
    const trainCustomButton = document.getElementById('trainCustomButton');
    const predictionFile = document.getElementById('predictionFile');
    const predictButton = document.getElementById('predictButton');

    // Reference data training - update model status when reference data changes
    referenceSelect.addEventListener('change', async () => {
        trainReferenceButton.disabled = !referenceSelect.value;
        if (referenceSelect.value) {
            await checkModelStatus(); // Check if model needs retraining with new data
        }
    });

    // Custom data training - update model status when files change
    const checkCustomFiles = async () => {
        const hasFiles = trainingFile.files.length && labelFile.files.length;
        trainCustomButton.disabled = !hasFiles;
        if (hasFiles) {
            await checkModelStatus(); // Check if model needs retraining with new data
        }
    };
    
    trainingFile.addEventListener('change', checkCustomFiles);
    labelFile.addEventListener('change', checkCustomFiles);

    // Prediction
    predictionFile.addEventListener('change', () => {
        predictButton.disabled = !predictionFile.files.length;
    });

    trainReferenceButton.addEventListener('click', () => handleTraining('reference'));
    trainCustomButton.addEventListener('click', () => handleTraining('custom'));
    predictButton.addEventListener('click', handlePrediction);

    checkDownloadStatus();
    
    // Add model status check
    checkModelStatus();
    
    // Update handleTraining to check model status after training
    const originalHandleTraining = handleTraining;
    handleTraining = async (mode) => {
        await originalHandleTraining(mode);
        await checkModelStatus();
    };

    const sequenceText = document.getElementById('sequenceText');
    const predictTextButton = document.getElementById('predictTextButton');

    // Add text input validation and button enable/disable
    sequenceText.addEventListener('input', () => {
        // Clean up the sequence: remove spaces, newlines, and convert to uppercase
        const sequence = sequenceText.value.replace(/[\s\r\n]+/g, '').toUpperCase();
        sequenceText.value = sequence; // Update the input value with cleaned text
        
        const isValidSequence = /^[ATGC]+$/.test(sequence);
        predictTextButton.disabled = !isValidSequence;
        
        if (!isValidSequence && sequence.length > 0) {
            sequenceText.classList.add('is-invalid');
        } else {
            sequenceText.classList.remove('is-invalid');
        }
    });

    predictTextButton.addEventListener('click', handleTextPrediction);

    initializeSocket();
}

function updateTrainingState(training) {
    isTraining = training;
    const buttons = [
        document.getElementById('trainReferenceButton'),
        document.getElementById('trainCustomButton'),
        document.getElementById('predictButton'),
        document.getElementById('predictTextButton')  // Add text prediction button
    ];
    
    buttons.forEach(btn => {
        if (btn) {
            btn.disabled = training;
            if (training) {
                btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Training...';
            } else {
                btn.innerHTML = btn.id.includes('predict') ? 'Predict' : 'Train';
            }
        }
    });
    
    // Disable inputs during training
    const inputs = [
        'trainingFile', 
        'labelFile', 
        'predictionFile', 
        'referenceDataSelect',
        'sequenceText'  // Add sequence text input
    ];
    inputs.forEach(id => {
        const input = document.getElementById(id);
        if (input) input.disabled = training;
    });
}

async function handleTraining(mode) {
    if (isTraining) {
        alert('Training is already in progress');
        return;
    }

    try {
        updateTrainingState(true);
        updateProgressBar(0);
        
        // Reset training history and charts
        trainingHistory = {
            loss: [],
            val_loss: [],
            accuracy: [],
            val_accuracy: []
        };
        
        // Initialize new charts
        initializeCharts();
        
        let formData = new FormData();
        let endpoint = '/api/train';

        if (mode === 'reference') {
            const referenceFile = document.getElementById('referenceDataSelect').value;
            formData.append('reference', referenceFile);
            formData.append('mode', 'reference');
        } else {
            const seqFile = document.getElementById('trainingFile').files[0];
            const labelFile = document.getElementById('labelFile').files[0];
            formData.append('file', seqFile);
            formData.append('labels', labelFile);
            formData.append('mode', 'custom');
        }

        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (!response.ok || result.error) {
            throw new Error(result.error || 'Training failed');
        }
        
        // Update charts with final training history
        if (result.training_history && result.training_history.length > 0) {
            result.training_history.forEach(data => updateMetrics({
                ...data,
                total_epochs: result.total_epochs
            }));
        }
        
        updateProgressBar(100);
        await checkModelStatus();
        
    } catch (error) {
        console.error('Error:', error);
        alert('Training failed: ' + error.message);
    } finally {
        updateTrainingState(false);
    }
}

async function handlePrediction() {
    // Check model status before prediction
    const modelReady = await checkModelStatus();
    if (!modelReady) {
        alert('Please train the model first');
        return;
    }
    
    const file = document.getElementById('predictionFile').files[0];
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Prediction failed');
        
        const result = await response.json();
        displayResults(result.predictions);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Prediction failed: ' + error.message);
    }
}

async function handleTextPrediction() {
    const predictTextButton = document.getElementById('predictTextButton');
    const sequenceText = document.getElementById('sequenceText');

    try {
        // Check model status
        const modelReady = await checkModelStatus();
        if (!modelReady) {
            alert('Please train the model first');
            return;
        }

        // Clean up the sequence before sending
        const sequence = sequenceText.value.replace(/[\s\r\n]+/g, '').toUpperCase();
        if (!sequence || !/^[ATGC]+$/.test(sequence)) {
            alert('Please enter a valid DNA sequence containing only A, T, G, C');
            return;
        }

        // Disable button and show loading state
        predictTextButton.disabled = true;
        predictTextButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Predicting...';

        // Send as JSON instead of FormData
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sequence: sequence,
                type: 'text'
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText || 'Prediction failed');
        }

        const result = await response.json();
        displayResults([{
            header: 'Input Sequence',
            sequence: sequence,
            prediction: result.prediction,
            confidence: result.confidence
        }]);

    } catch (error) {
        console.error('Error:', error);
        alert('Prediction failed: ' + error.message);
    } finally {
        // Reset button state
        predictTextButton.disabled = false;
        predictTextButton.innerHTML = 'Predict from Text';
    }
}

function displayResults(predictions) {
    const container = document.getElementById('resultsContainer');
    const table = document.createElement('table');
    table.className = 'table mt-4 results-table';
    
    table.innerHTML = `
        <thead>
            <tr>
                <th class="header-col">Header</th>
                <th class="sequence-col">Sequence</th>
                <th class="prediction-col">Prediction</th>
                <th class="confidence-col">Confidence</th>
            </tr>
        </thead>
        <tbody>
            ${predictions.map(p => `
                <tr>
                    <td class="text-truncate" title="${p.header}">${p.header}</td>
                    <td class="sequence-cell" title="${p.sequence}">${p.sequence}</td>
                    <td class="text-${p.prediction === 1 ? 'success' : 'danger'}">
                        ${p.prediction === 1 ? 'Resistant' : 'Sensitive'}
                    </td>
                    <td>${parseFloat(p.confidence).toFixed(2)}%</td>
                </tr>
            `).join('')}
        </tbody>
    `;
    
    container.innerHTML = '';
    container.appendChild(table);
}

document.addEventListener('DOMContentLoaded', initializeEventListeners);
