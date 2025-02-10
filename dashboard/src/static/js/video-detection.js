let videoElement = null;
let detectionCanvas = null;
let detectionContext = null;
let videoResults = null;
let isInitialized = false;
let maxInitAttempts = 20;
let currentAttempt = 0;

function initVideoDetection(results) {
    videoResults = results;
    // Reset initialization state
    isInitialized = false;
    currentAttempt = 0;
    
    // Try initialization immediately or wait for load
    if (document.readyState === 'complete') {
        tryInitialize();
    } else {
        window.addEventListener('load', tryInitialize);
    }
}

function tryInitialize() {
    videoElement = document.getElementById('videoPlayer');
    detectionCanvas = document.getElementById('detectionCanvas');
    
    if (!videoElement && currentAttempt < maxInitAttempts) {
        currentAttempt++;
        console.log(`Attempt ${currentAttempt}: Waiting for video element...`);
        setTimeout(tryInitialize, 100);
        return;
    }

    if (!videoElement) {
        console.error('Failed to initialize video after maximum attempts');
        return;
    }

    console.log('Found video element, setting up detection...');
    setupVideoPlayer();
}

function setupVideoPlayer() {
    // Create canvas if it doesn't exist
    if (!detectionCanvas) {
        detectionCanvas = document.createElement('canvas');
        detectionCanvas.id = 'detectionCanvas';
        detectionCanvas.className = 'position-absolute top-0 start-0';
        detectionCanvas.style.pointerEvents = 'none';
        videoElement.parentElement.appendChild(detectionCanvas);
    }

    detectionContext = detectionCanvas.getContext('2d');

    // Wait for video to be ready
    if (videoElement.readyState === 0) {
        videoElement.addEventListener('loadedmetadata', finalizeSetup);
    } else {
        finalizeSetup();
    }

    // Handle window resize
    window.addEventListener('resize', resizeDetectionCanvas);
}

function finalizeSetup() {
    resizeDetectionCanvas();
    isInitialized = true;
    console.log('Video detection setup completed');
}

// Update time-based navigation
window.navigateToTime = function(timeInSeconds, detections) {
    console.log('Navigate to time called:', timeInSeconds);
    
    // Ensure video element exists
    if (!videoElement) {
        videoElement = document.getElementById('videoPlayer');
    }
    
    // Ensure canvas exists
    if (!detectionCanvas) {
        detectionCanvas = document.getElementById('detectionCanvas');
    }

    if (!videoElement || !detectionCanvas) {
        console.error('Video or canvas not found, retrying initialization...');
        tryInitialize();
        setTimeout(() => navigateToTime(timeInSeconds, detections), 100);
        return;
    }

    // Parse detections if needed
    if (typeof detections === 'string') {
        try {
            detections = JSON.parse(detections);
        } catch (e) {
            console.error('Failed to parse detections:', e);
            return;
        }
    }

    // Navigate to time
    videoElement.pause();
    videoElement.currentTime = timeInSeconds;

    const handleFrameLoaded = () => {
        console.log('Frame loaded, drawing detections');
        resizeDetectionCanvas();
        drawDetections(detections);
        updateCurrentFrameInfo(Math.floor(timeInSeconds * 30), detections);
    };

    videoElement.removeEventListener('seeked', handleFrameLoaded);
    videoElement.addEventListener('seeked', handleFrameLoaded, { once: true });
};

// Modified navigateToFrame function
window.navigateToFrame = async function(frameNumber, detections) {
    // Wait for initialization to complete
    if (!isInitialized && initializationPromise) {
        console.log('Waiting for initialization to complete...');
        await initializationPromise;
    }

    if (!videoElement || !detectionCanvas) {
        console.error('Video or canvas not available after initialization');
        return;
    }

    console.log('Navigating to frame:', frameNumber);

    // Ensure video element exists and is initialized
    if (!isInitialized) {
        console.log('Waiting for initialization...');
        setTimeout(() => window.navigateToFrame(frameNumber, detections), 100);
        return;
    }

    if (!videoElement || !detectionCanvas) {
        console.error('Video or canvas not available');
        return;
    }

    // Parse detections if needed
    if (typeof detections === 'string') {
        try {
            detections = JSON.parse(detections);
        } catch (e) {
            console.error('Failed to parse detections:', e);
            return;
        }
    }

    // Calculate and seek to frame
    const timestamp = frameNumber / 30;
    videoElement.pause();
    videoElement.currentTime = timestamp;

    // Handle frame loaded
    const handleFrameLoaded = () => {
        console.log('Frame loaded, drawing detections');
        resizeDetectionCanvas(); // Ensure canvas is properly sized
        drawDetections(detections);
        updateCurrentFrameInfo(frameNumber, detections);
    };

    videoElement.removeEventListener('seeked', handleFrameLoaded);
    videoElement.addEventListener('seeked', handleFrameLoaded, { once: true });
};

function resizeDetectionCanvas() {
    if (videoElement && detectionCanvas) {
        const rect = videoElement.getBoundingClientRect();
        detectionCanvas.width = rect.width;
        detectionCanvas.height = rect.height;
    }
}

// Update the frame info display to show time
function updateCurrentFrameInfo(frameNumber, detections) {
    const timeInSeconds = Math.floor(frameNumber / 30);
    const timeStr = `${Math.floor(timeInSeconds / 60)}:${(timeInSeconds % 60).toString().padStart(2, '0')}`;
    document.getElementById('currentFrame').textContent = `${timeStr} (Frame ${frameNumber})`;
    
    let detectionsHtml = '<ul class="list-unstyled mb-0">';
    detections.forEach(detection => {
        detectionsHtml += `<li><strong>${detection.label}</strong>: ${detection.confidence}%</li>`;
    });
    detectionsHtml += '</ul>';
    document.getElementById('frameDetections').innerHTML = detectionsHtml;
}

function drawDetections(detections) {
    if (!detectionContext || !videoElement || !detectionCanvas) {
        console.error('Missing required elements for drawing');
        return;
    }

    console.log('Drawing detections:', detections);

    // Get current video dimensions
    const videoRect = videoElement.getBoundingClientRect();
    detectionCanvas.width = videoRect.width;
    detectionCanvas.height = videoRect.height;

    // Clear previous drawings
    detectionContext.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);

    // Calculate scale factors
    const scaleX = detectionCanvas.width / videoElement.videoWidth;
    const scaleY = detectionCanvas.height / videoElement.videoHeight;

    detections.forEach(detection => {
        const [x1, y1, x2, y2] = detection.bbox;
        
        // Scale coordinates
        const scaledX = x1 * scaleX;
        const scaledY = y1 * scaleY;
        const scaledWidth = (x2 - x1) * scaleX;
        const scaledHeight = (y2 - y1) * scaleY;

        // Draw box
        detectionContext.strokeStyle = '#00ff00';
        detectionContext.lineWidth = 2;
        detectionContext.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);

        // Draw label
        const label = `${detection.label} ${detection.confidence}%`;
        detectionContext.font = '14px Arial';
        
        // Background for text
        const padding = 2;
        const textMetrics = detectionContext.measureText(label);
        detectionContext.fillStyle = 'rgba(0, 0, 0, 0.7)';
        detectionContext.fillRect(
            scaledX, 
            scaledY - 20, 
            textMetrics.width + padding * 2, 
            20
        );

        // Text
        detectionContext.fillStyle = '#fff';
        detectionContext.fillText(label, scaledX + padding, scaledY - 5);
    });
}
