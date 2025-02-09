// This file contains JavaScript functions for handling file uploads and updating the UI dynamically based on user interactions.

document.addEventListener('DOMContentLoaded', function() {
    const imageUploadForm = document.getElementById('image-upload-form');
    const videoUploadForm = document.getElementById('video-upload-form');
    const batchUploadForm = document.getElementById('batch-upload-form');

    if (imageUploadForm) {
        imageUploadForm.addEventListener('submit', function(event) {
            event.preventDefault();
            const formData = new FormData(imageUploadForm);
            fetch('/upload-image', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Update UI with prediction results
                displayImageResults(data);
            })
            .catch(error => console.error('Error:', error));
        });
    }

    if (videoUploadForm) {
        videoUploadForm.addEventListener('submit', function(event) {
            event.preventDefault();
            const formData = new FormData(videoUploadForm);
            fetch('/upload-video', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Update UI with video results
                displayVideoResults(data);
            })
            .catch(error => console.error('Error:', error));
        });
    }

    if (batchUploadForm) {
        batchUploadForm.addEventListener('submit', function(event) {
            event.preventDefault();
            const formData = new FormData(batchUploadForm);
            fetch('/upload-batch', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Update UI with batch results
                displayBatchResults(data);
            })
            .catch(error => console.error('Error:', error));
        });
    }

    function displayImageResults(data) {
        // Logic to display image results with bounding boxes and labels
    }

    function displayVideoResults(data) {
        // Logic to display video results with detected objects
    }

    function displayBatchResults(data) {
        // Logic to display batch processing results
    }
});

function loadModel() {
    const modelFolder = document.getElementById('modelFolder').value;
    fetch('/load_model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_folder: modelFolder })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Model loaded successfully');
        } else {
            alert('Error loading model: ' + data.error);
        }
    });
}

function selectFile(filename) {
    // Show loading state
    document.querySelector('.preview-container').innerHTML = '<div class="text-center"><div class="spinner-border"></div><p>Loading...</p></div>';
    document.getElementById('detectionResults').innerHTML = '<div class="text-center"><div class="spinner-border"></div></div>';

    fetch(`/predict/${filename}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update preview
                document.querySelector('.preview-container').innerHTML = data.preview_html;
                
                // Update results with cache status
                const cacheStatus = data.cached ? 
                    '<div class="text-muted small mb-2">(Loaded from cache)</div>' : 
                    '<div class="text-muted small mb-2">(Newly predicted)</div>';
                document.getElementById('detectionResults').innerHTML = cacheStatus + data.results_html;
                
                // Handle image-specific functionality
                if (data.predictions) {
                    const img = document.querySelector('.preview-container img');
                    if (img) {
                        img.onload = function() {
                            const canvas = document.getElementById('detectionCanvas');
                            if (canvas) {
                                canvas.width = img.width;
                                canvas.height = img.height;
                                drawDetections(data.predictions);
                            }
                        };
                    }
                }
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error processing file');
        });
}

function drawDetections(detections) {
    const canvas = document.getElementById('detectionCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    ctx.font = '12px Arial';
    ctx.fillStyle = '#00ff00';

    detections.forEach(detection => {
        const [x1, y1, x2, y2] = detection.bbox;
        ctx.strokeRect(x1, y1, x2-x1, y2-y1);
        ctx.fillText(
            `${detection.label} ${detection.confidence}%`,
            x1, y1 > 20 ? y1 - 5 : y1 + 20
        );
    });
}

function showUploadModal() {
    const modal = new bootstrap.Modal(document.getElementById('uploadModal'));
    modal.show();
}

function uploadFile() {
    const formData = new FormData(document.getElementById('uploadForm'));
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a file first');
        return;
    }

    // Show loading state
    const uploadButton = document.querySelector('#uploadModal .btn-primary');
    const originalText = uploadButton.textContent;
    uploadButton.textContent = 'Uploading...';
    uploadButton.disabled = true;

    // Show progress bar
    const progressBar = document.querySelector('#uploadProgress');
    const progressBarInner = progressBar.querySelector('.progress-bar');
    progressBar.classList.remove('d-none');
    progressBarInner.style.width = '0%';

    fetch('/upload_file', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            // Hide modal
            const modalElement = document.getElementById('uploadModal');
            const modal = bootstrap.Modal.getInstance(modalElement);
            if (modal) {
                modal.hide();
            }
            
            // Show success message
            alert(data.message || 'File uploaded successfully');
            
            // Refresh the page to show new file
            location.reload();
        } else {
            throw new Error(data.error || 'Upload failed');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Upload failed: ' + error.message);
    })
    .finally(() => {
        // Reset button state and hide progress bar
        uploadButton.textContent = originalText;
        uploadButton.disabled = false;
        progressBar.classList.add('d-none');
    });
}

function clearAllFiles() {
    if (confirm('Are you sure you want to delete all uploaded files?')) {
        fetch('/clear_files', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    location.reload();
                }
            });
    }
}