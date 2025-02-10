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
                
                // Update results
                const cacheStatus = data.cached ? 
                    '<div class="text-muted small mb-2">(Using processed version)</div>' : 
                    '<div class="text-muted small mb-2">(Newly processed)</div>';
                document.getElementById('detectionResults').innerHTML = cacheStatus + data.results_html;
                
                // Handle video initialization if needed
                const video = document.querySelector('.preview-container video');
                if (video) {
                    video.addEventListener('loadeddata', () => {
                        console.log('Video data loaded');
                        // Play video after loading
                        video.play().catch(e => {
                            console.log('Auto-play prevented:', e);
                            // Browser might prevent autoplay, show play button prominently
                            const playButton = document.createElement('button');
                            playButton.className = 'btn btn-primary position-absolute top-50 start-50 translate-middle';
                            playButton.innerHTML = '<i class="fas fa-play"></i> Play Video';
                            playButton.onclick = () => video.play();
                            video.parentElement.appendChild(playButton);
                        });
                    });
                }
                
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
    uploadButton.textContent = 'Uploading and Processing...';
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
            
            // If there's a redirect URL (for videos), use it
            if (data.redirect_url) {
                window.location.href = data.redirect_url;
            } else {
                // Otherwise just reload the page
                location.reload();
            }
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

function regenerateVideo(filename, event) {
    if (!confirm('Are you sure you want to regenerate this video? The current processed version will be removed and generated again.')) {
        return;
    }

    // Show loading state
    const btn = event.currentTarget;
    const originalContent = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';

    // Add processing class to the file item
    const fileItem = btn.closest('.file-item');
    fileItem.classList.add('processing');

    // Add global loading indicator
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'position-fixed top-50 start-50 translate-middle bg-white p-3 rounded shadow';
    loadingDiv.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary mb-2"></div>
            <div>Processing video...</div>
        </div>
    `;
    document.body.appendChild(loadingDiv);

    fetch(`/regenerate_video/${encodeURIComponent(filename)}`, {
        method: 'POST',
        headers: {
            'Accept': 'application/json'
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            // Show success message
            const toast = new bootstrap.Toast(Object.assign(document.createElement('div'), {
                className: 'toast position-fixed bottom-0 end-0 m-3',
                innerHTML: `
                    <div class="toast-header bg-success text-white">
                        <strong class="me-auto">Success</strong>
                        <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
                    </div>
                    <div class="toast-body">
                        Video regenerated successfully
                    </div>
                `
            }));
            document.body.appendChild(toast.element);
            toast.show();
            
            // Redirect to new video
            window.location.href = data.redirect_url;
        } else {
            throw new Error(data.error || 'Failed to regenerate video');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error regenerating video: ' + error.message);
    })
    .finally(() => {
        // Cleanup
        btn.disabled = false;
        btn.innerHTML = originalContent;
        fileItem.classList.remove('processing');
        if (loadingDiv && loadingDiv.parentNode) {
            loadingDiv.parentNode.removeChild(loadingDiv);
        }
    });
}