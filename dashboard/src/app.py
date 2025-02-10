from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import json  # Add this import
from ultralytics import YOLO
import yaml
import logging
import torch
from utils.cache_manager import PredictionCache
import cv2
import numpy as np
from pathlib import Path
import subprocess
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get the project root directory (parent of src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')

# Select device
device = 'cpu'  # Force CPU usage
logger.info(f"Using device: {device}")

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = os.path.join(SRC_DIR, 'static', 'uploads')
# Increase maximum content length to 500MB
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
# Add chunked upload config
app.config['MAX_CONTENT_LENGTH'] = None  # Disable Flask's content length check
app.config['UPLOAD_CHUNK_SIZE'] = 1024 * 1024  # 1MB chunks

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Default config values
config = {
    'model': {
        'confidence_threshold': 0.5,
        'nms_threshold': 0.4
    }
}

# Try to load config if exists
config_path = os.path.join(SRC_DIR, 'config', 'model_config.yaml')
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

# Load the YOLO model using the correct path (models folder at project root)
model = YOLO(os.path.join(BASE_DIR, 'models', 'best.pt'))
model.to(device)  # Move model to CPU

# Initialize prediction cache
cache = PredictionCache(os.path.join(SRC_DIR, 'static', 'cache'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'GET':
        return render_template('index.html')
    
    try:
        if 'file' not in request.files:
            logger.error("No file part in request")
            flash('No file uploaded')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            flash('No file selected')
            return redirect(request.url)
            
        if file:
            # Save and process file
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            logger.info(f"File saved to {file_path}")
            
            # Run prediction
            logger.info("Running YOLO prediction")
            results = model.predict(
                source=file_path,
                conf=config['model']['confidence_threshold'],
                save=True,
                device=device  # Specify device for prediction
            )
            
            # Process results
            result = results[0]
            detections = []
            for box in result.boxes:
                detection = {
                    'label': f"{result.names[int(box.cls[0])]}",
                    'confidence': f"{float(box.conf[0])*100:.2f}",
                    'bbox': box.xyxy[0].tolist()
                }
                detections.append(detection)
            
            logger.info(f"Detections: {detections}")
            
            return render_template('image.html', 
                                 image_filename=filename,
                                 predictions=detections)
    
    except Exception as e:
        logger.exception("Error processing image")
        flash(f'Error processing image: {str(e)}')
        return redirect(request.url)

def process_uploaded_video(filename, file_path):
    """Helper function to process newly uploaded videos"""
    try:
        logger.info(f"Processing new video upload: {filename}")
        # Run prediction
        results = model.predict(
            source=file_path,
            conf=config['model']['confidence_threshold'],
            device=device
        )
        
        # Generate processed video
        processed_path = process_video_with_detections(file_path, results)
        logger.info(f"Generated processed video: {processed_path}")
        return True
    except Exception as e:
        logger.error(f"Error processing uploaded video: {str(e)}")
        return False

@app.route('/upload_file', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
            
        if file:
            # Clean filename and check extension
            filename = file.filename
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.mp4', '.avi', '.mov')):
                return jsonify({'success': False, 'error': 'Unsupported file type'})
            
            # Save file in chunks
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            chunk_size = app.config['UPLOAD_CHUNK_SIZE']
            
            with open(file_path, 'wb') as f:
                while True:
                    chunk = file.stream.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
            
            logger.info(f"File saved to {file_path}")
            
            # If it's a video, process it immediately
            if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                success = process_uploaded_video(filename, file_path)
                if not success:
                    return jsonify({
                        'success': False,
                        'error': 'Failed to process video'
                    })
            
            return jsonify({
                'success': True,
                'filename': filename,
                'message': 'File uploaded and processed successfully',
                'redirect_url': url_for('predict_file', filename=filename)
            })
            
    except Exception as e:
        logger.exception("Error uploading file")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict/<filename>')
def predict_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Handle video files
        if filename.lower().endswith(('.mp4', '.avi', '.mov')):
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{filename}")
            
            # Create processed version if it doesn't exist
            if not os.path.exists(processed_path):
                logger.info(f"Processing video: {filename}")
                results = model.predict(
                    source=file_path,
                    conf=config['model']['confidence_threshold'],
                    device=device
                )
                process_video_with_detections(file_path, results)
            
            # Return JSON response with processed video path
            return jsonify({
                'success': True,
                'preview_html': f'''
                    <div class="video-container">
                        <video controls autoplay class="img-fluid w-100">
                            <source src="{url_for('static', filename=f'uploads/processed_{filename}')}#t=0.001" type="video/mp4">
                            <source src="{url_for('static', filename=f'uploads/processed_{filename}')}#t=0.001" type="video/avi">
                            Your browser does not support the video tag.
                        </video>
                    </div>
                ''',
                'results_html': '<div class="text-muted">Video processed with detections</div>',
                'cached': True
            })

        # Handle other file types (images)
        # Try to get cached predictions
        cached_data = cache.get_prediction(filename)
        if cached_data:
            if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                return jsonify({
                    'success': True,
                    'preview_html': f'''
                        <div class="video-container">
                            <video controls class="img-fluid w-100">
                                <source src="{url_for('static', filename=f'uploads/{filename}')}" type="video/mp4">
                                Your browser does not support the video tag.
                            </video>
                        </div>
                    ''',
                    'results_html': generate_video_results_html(cached_data['predictions']),
                    'video_results': cached_data['predictions'],
                    'cached': True
                })
            else:
                return jsonify({
                    'success': True,
                    'preview_html': f'''
                        <div class="position-relative">
                            <img src="{url_for('static', filename=f'uploads/{filename}')}" class="img-fluid" alt="Preview">
                            <canvas id="detectionCanvas" class="position-absolute top-0 start-0"></canvas>
                        </div>
                    ''',
                    'results_html': generate_results_html(cached_data['predictions']),
                    'predictions': cached_data['predictions'],
                    'cached': True
                })

        # If no cache, run prediction
        results = model.predict(
            source=file_path,
            conf=config['model']['confidence_threshold'],
            device=device
        )
        
        if filename.lower().endswith(('.mp4', '.avi', '.mov')):
            video_results = process_video_results(results)
            # Cache the results
            cache.save_prediction(filename, video_results)
            return render_template('results/video.html',
                                video_filename=filename,
                                video_results=video_results)
        else:
            predictions = process_image_results(results[0])
            # Cache the results
            cache.save_prediction(filename, predictions)
            return jsonify({
                'success': True,
                'preview_html': f'''
                    <div class="position-relative">
                        <img src="{url_for('static', filename=f'uploads/{filename}')}" class="img-fluid" alt="Preview">
                        <canvas id="detectionCanvas" class="position-absolute top-0 start-0"></canvas>
                    </div>
                ''',
                'results_html': generate_results_html(predictions),
                'predictions': predictions,
                'cached': False
            })

    except Exception as e:
        logger.exception("Error processing file")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/load_model', methods=['POST'])
def load_model():
    try:
        data = request.get_json()
        model_folder = data.get('model_folder')
        if model_folder == 'default':
            model_path = os.path.join(BASE_DIR, 'models', 'best.pt')
        else:
            model_path = os.path.join(BASE_DIR, 'models', model_folder, 'best.pt')
        
        global model
        model = YOLO(model_path)
        model.to(device)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def process_image_results(result):
    predictions = []
    for box in result.boxes:
        prediction = {
            'label': result.names[int(box.cls[0])],
            'confidence': f"{float(box.conf[0])*100:.2f}",
            'bbox': box.xyxy[0].tolist()
        }
        predictions.append(prediction)
    return predictions

def process_video_results(results):
    """Process video detection results grouped by seconds."""
    video_results = []
    
    for frame_idx, result in enumerate(results):
        # Calculate second mark (assuming 30fps)
        second = frame_idx // 30
        
        if hasattr(result.boxes, 'data') and len(result.boxes.data) > 0:
            frame_detections = []
            for box in result.boxes:
                detection = {
                    'label': result.names[int(box.cls[0])],
                    'confidence': f"{float(box.conf[0])*100:.2f}",
                    'bbox': box.xyxy[0].tolist()
                }
                frame_detections.append(detection)
            
            video_results.append({
                'time': second,
                'time_str': f"{second // 60}:{second % 60:02d}",
                'frame': frame_idx + 1,
                'detections': frame_detections
            })
    
    return video_results

def check_ffmpeg():
    """Check if FFmpeg is available on the system"""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        logger.warning("FFmpeg not found on system, using OpenCV fallback")
        return False

def process_video_with_detections(file_path, results):
    """Generate a new video with detection boxes and labels."""
    try:
        # Get original video properties
        cap = cv2.VideoCapture(file_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output paths
        output_path = str(Path(file_path).parent / f"processed_{Path(file_path).name}")
        
        # Try different codecs in order of preference
        codecs = [
            ('XVID', '.avi'),  # Most compatible
            ('mp4v', '.mp4'),  # Fallback
            ('MJPG', '.avi'),  # Another fallback
        ]
        
        out = None
        for codec, ext in codecs:
            try:
                temp_path = str(Path(file_path).parent / f"processed_{Path(file_path).stem}{ext}")
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
                
                if test_out.isOpened():
                    out = test_out
                    output_path = temp_path
                    logger.info(f"Using codec: {codec}")
                    break
                else:
                    test_out.release()
            except Exception as e:
                logger.warning(f"Codec {codec} failed: {e}")
                continue
        
        if out is None:
            raise Exception("No suitable codec found")
        
        # Process frames
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw detections
            if frame_idx < len(results):
                result = results[frame_idx]
                if hasattr(result.boxes, 'data') and len(result.boxes.data) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = f"{result.names[cls]} {conf:.2f}"
                        
                        # Draw box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(frame, (x1, y1-20), (x1+label_w, y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1, y1-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            out.write(frame)
            frame_idx += 1
        
        # Release resources
        cap.release()
        out.release()
        
        logger.info(f"Video saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.exception("Error processing video")
        raise e

def generate_video_results_html(video_results):
    """Generate HTML for video results summary with time-based navigation."""
    html = '<div class="video-results overflow-auto" style="max-height: 60vh;">'
    
    # Handle both old and new format
    def get_total_detections(results):
        return sum(len(r.get('detections', [])) for r in results)
    
    total_detections = get_total_detections(video_results)
    html += f'<div class="mb-3">Total Detections: {total_detections}</div>'
    
    for result in video_results:
        if result.get('detections'):  # Only show frames with detections
            frame_data = json.dumps(result['detections']).replace('"', '&quot;')
            time_str = result.get('time_str', f"{result.get('time', 0) // 60}:{result.get('time', 0) % 60:02d}")
            html += f'''
                <div class="frame-result mb-2 border-bottom pb-2" 
                     onclick="navigateToTime({result.get('time', 0)}, {frame_data})"
                     style="cursor: pointer;">
                    <strong>Time: {time_str}</strong> (Frame {result.get('frame', 0)})
                    <ul class="list-unstyled mb-0 ps-3">
            '''
            for detection in result['detections']:
                html += f'''
                    <li><strong>{detection['label']}</strong>: {detection['confidence']}%</li>
                '''
            html += '</ul></div>'
    html += '</div>'
    return html

@app.route('/clear_files', methods=['POST'])
def clear_files():
    try:
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                # Also clear the cache for this file
                cache.clear_cache(file)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def get_available_models():
    models_dir = os.path.join(BASE_DIR, 'models')
    return [d for d in os.listdir(models_dir) 
            if os.path.isdir(os.path.join(models_dir, d))]

@app.context_processor
def utility_processor():
    def get_uploaded_files():
        files = []
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            # Don't include processed_ files in the list
            if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], file)) and not file.startswith('processed_'):
                files.append(file)
        return files
    
    return {
        'uploaded_files': get_uploaded_files(),
        'available_models': get_available_models(),
        'year': datetime.now().year  # Add current year
    }

# Helper functions for generating HTML
def generate_preview_html(filename, results):
    """Generate HTML for preview section based on file type."""
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        return f'''
            <div class="position-relative">
                <img src="/static/uploads/{filename}" class="img-fluid" alt="Preview">
                <canvas id="detectionCanvas" class="position-absolute top-0 start-0"></canvas>
            </div>
            <script>
                drawDetections({results.boxes.data.tolist()});
            </script>
        '''
    elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
        return f'''
            <div class="video-container">
                <video controls class="img-fluid">
                    <source src="/static/uploads/{filename}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
        '''
    return '<div class="text-center">Unsupported file type</div>'

def generate_results_html(results):
    """Generate HTML for results summary."""
    html = '<div class="detection-list">'
    if hasattr(results, 'boxes') and len(results.boxes) > 0:
        for box in results.boxes:
            label = results.names[int(box.cls[0])]
            conf = float(box.conf[0]) * 100
            html += f'''
                <div class="detection-item">
                    <strong>{label}</strong>: {conf:.2f}%
                </div>
            '''
    else:
        html += '<div>No detections found</div>'
    html += '</div>'
    return html

@app.route('/get_frame_detections/<filename>/<int:frame>')
def get_frame_detections(filename, frame):
    try:
        cached_data = cache.get_prediction(filename)
        if cached_data:
            video_results = cached_data['predictions']
            frame_data = next((r for r in video_results if r['frame'] == frame), None)
            if frame_data:
                return jsonify({
                    'success': True,
                    'detections': frame_data['detections']
                })
    except Exception as e:
        logger.exception("Error getting frame detections")
    return jsonify({'success': False, 'error': 'Frame data not found'})

@app.route('/regenerate_video/<filename>', methods=['POST'])
def regenerate_video(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{filename}")
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'Original video file not found'})
        
        # Clear existing processed video and cache
        try:
            if os.path.exists(processed_path):
                os.remove(processed_path)
                logger.info(f"Removed existing processed video: {processed_path}")
            cache.clear_cache(filename)
        except Exception as e:
            logger.warning(f"Error cleaning up old files: {str(e)}")
        
        # Run new prediction and generate new processed video
        results = model.predict(
            source=file_path,
            conf=config['model']['confidence_threshold'],
            device=device
        )
        
        process_video_with_detections(file_path, results)
        logger.info(f"Generated new processed video: {processed_path}")
        
        return jsonify({
            'success': True,
            'message': 'Video regenerated successfully',
            'redirect_url': url_for('predict_file', filename=filename)
        })

    except Exception as e:
        logger.exception("Error regenerating video")
        # Cleanup any partially processed files
        try:
            if os.path.exists(processed_path):
                os.remove(processed_path)
        except:
            pass
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)