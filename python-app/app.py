from flask import Flask, request, jsonify, render_template, json
from flask_cors import CORS
from flask_socketio import SocketIO
from model import AMRModel
from init_data import DataInitializer
import threading
import random
import re
from pathlib import Path
import numpy as np

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def create_app():
    app = Flask(__name__)
    app.json_encoder = NumpyJSONEncoder  # Use custom JSON encoder
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*", json=json)
    model = AMRModel(socketio=socketio)  # Pass socketio to model
    data_initializer = DataInitializer(socketio)  # Pass socketio to DataInitializer

    # Start data download in background
    def init_background_data():
        data_initializer.initialize_data()
    
    threading.Thread(target=init_background_data).start()

    def parse_fasta(fasta_text):
        sequences = []
        current_header = ''
        current_sequence = ''
        
        for line in fasta_text.split('\n'):
            if line.startswith('>'):
                if current_header and current_sequence:
                    sequences.append({'header': current_header, 'sequence': current_sequence})
                current_header = line[1:].strip()  # Fixed: changed trip() to strip()
                current_sequence = ''
            else:
                current_sequence += line.strip()
        
        if current_header and current_sequence:
            sequences.append({'header': current_header, 'sequence': current_sequence})
        
        return sequences

    def calculate_sequence_features(sequence):
        gc_content = len(re.findall('[GC]', sequence)) / len(sequence)
        length = len(sequence)
        repeats = count_repeats(sequence)
        return {'gc_content': gc_content, 'length': length, 'repeats': repeats}

    def count_repeats(sequence):
        count = 0
        kmer = 3
        kmers = set()
        
        for i in range(len(sequence) - kmer + 1):
            current_kmer = sequence[i:i+kmer]
            if current_kmer in kmers:
                count += 1
            kmers.add(current_kmer)
        
        return count

    def predict_single(sequence, header):
        features = calculate_sequence_features(sequence)
        
        probability = min(0.99, max(0.01,
            features['gc_content'] * 0.4 +
            min(1, features['length'] / 1000) * 0.3 +
            min(1, features['repeats'] / 10) * 0.3 +
            random.uniform(-0.1, 0.1)
        ))
        
        return {
            'header': header,
            'sequence': sequence[:50] + "...",
            'probability': probability,
            'prediction': 'Kháng kháng sinh' if probability > 0.5 else 'Nhạy cảm',
            'confidence': round((abs(0.5 - probability) * 2 * 100), 1),
            'features': features
        }

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/upload-training', methods=['POST'])
    def upload_training():
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        label_file = request.files.get('labels')
        reference_file = request.form.get('reference')
        
        if not all([label_file, reference_file]):
            return jsonify({'error': 'Labels and reference data required'}), 400
            
        fasta_text = file.read().decode('utf-8')
        label_text = label_file.read().decode('utf-8')
        
        sequences = parse_fasta(fasta_text)
        labels = [line.strip() for line in label_text.split('\n') if line.strip()]
        
        if len(sequences) != len(labels):
            return jsonify({'error': 'Sequence and label counts must match'}), 400
            
        return jsonify({
            'status': 'success',
            'message': f'Loaded {len(sequences)} sequences with labels',
            'sequences': sequences
        })

    @app.route('/api/train', methods=['POST'])
    def train_model():
        mode = request.form.get('mode', 'custom')
        
        try:
            if mode == 'reference':
                reference_file = request.form.get('reference')
                if not reference_file:
                    return jsonify({'error': 'No reference file selected'}), 400
                    
                # Train directly with reference data
                history = model.train_with_reference(reference_file)
                
                epochs_completed = len(history['loss'])
                total_epochs = model.last_training_epochs

                return jsonify({
                    'status': 'success',
                    'message': 'Training completed successfully!',
                    'training_history': [
                        {
                            'epoch': i,
                            'loss': float(history['loss'][i]),
                            'accuracy': float(history['accuracy'][i]),
                            'val_loss': float(history.get('val_loss', [0])[i] if 'val_loss' in history else 0),
                            'val_accuracy': float(history.get('val_accuracy', [0])[i] if 'val_accuracy' in history else 0)
                        }
                        for i in range(epochs_completed)
                    ],
                    'total_epochs': total_epochs,
                    'epochs_completed': epochs_completed
                })
            
            else:  # custom mode
                if 'file' not in request.files or 'labels' not in request.files:
                    return jsonify({'error': 'Missing training files'}), 400
                    
                file = request.files['file']
                label_file = request.files['labels']
                
                fasta_text = file.read().decode('utf-8')
                label_text = label_file.read().decode('utf-8')
                
                sequences = parse_fasta(fasta_text)
                labels = [line.strip() for line in label_text.split('\n') if line.strip()]
                
                if len(sequences) != len(labels):
                    return jsonify({'error': 'Sequence and label counts must match'}), 400
                    
                history = model.train([seq['sequence'] for seq in sequences], labels)
            
            # Add total epochs to response
            epochs_completed = len(history['loss'])
            total_epochs = model.last_training_epochs  # We'll add this to the model class
            
            return jsonify({
                'status': 'success',
                'message': 'Training completed successfully!',
                'training_history': [
                    {
                        'epoch': i,
                        'loss': history['loss'][i],
                        'accuracy': history['accuracy'][i]
                    }
                    for i in range(epochs_completed)
                ],
                'total_epochs': total_epochs,
                'epochs_completed': epochs_completed
            })
            
        except Exception as e:
            return jsonify({
                'error': f'Training failed: {str(e)}'
            }), 500

    @app.route('/api/predict', methods=['POST'])
    def predict():
        try:
            # Check if the request is JSON (text prediction) or form data (file prediction)
            if request.is_json:
                data = request.get_json()
                sequence = data.get('sequence')
                if not sequence:
                    return jsonify({'error': 'No sequence provided'}), 400
                
                # Validate sequence
                sequence = sequence.upper()
                if not all(c in 'ATGC' for c in sequence):
                    return jsonify({'error': 'Invalid sequence. Only A, T, G, C allowed'}), 400
                
                # Process single sequence
                prediction = model.predict_sequence(sequence)
                return jsonify({
                    'prediction': int(prediction['class']),
                    'confidence': float(prediction['confidence'])
                })
            else:
                # Handle file upload prediction (existing code)
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                    
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                    
                # ... rest of existing file prediction code ...
                
        except Exception as e:
            app.logger.error(f"Prediction error: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/download-status')
    def download_status():
        return jsonify(data_initializer.get_status())

    @app.route('/api/list-reference-data')
    def list_reference_data():
        # Check download status first
        status = data_initializer.get_status()
        app.logger.debug(f"Download status: {status}")
        
        if status['status'] != 'completed':
            return jsonify({
                'files': [],
                'data_dir_exists': False,
                'data_dir_path': str(Path('data').absolute()),
                'download_status': status
            })

        data_dir = Path('data')
        reference_files = []
        
        if data_dir.exists():
            # Search for both .fasta and .fasta.gz files, including in subdirectories
            for pattern in ['**/*.fasta', '**/*.fasta.gz']:
                app.logger.debug(f"Searching pattern: {pattern}")
                for file in data_dir.glob(pattern):
                    app.logger.debug(f"Found file: {file}")
                    if any(x in file.name.lower() for x in ['card', 'clinical', 'homolog']):
                        reference_files.append({
                            'name': file.stem.replace('_', ' ').title(),
                            'path': str(file.relative_to(data_dir))
                        })
        
        app.logger.info(f"Found {len(reference_files)} reference files: {reference_files}")
        return jsonify({
            'files': reference_files,
            'data_dir_exists': data_dir.exists(),
            'data_dir_path': str(data_dir.absolute()),
            'download_status': status
        })

    @app.route('/api/model-status')
    def model_status():
        """Check if model is trained and ready"""
        return jsonify({
            'trained': model.model is not None,
            'message': 'Model ready for predictions' if model.model else 'Model needs training'
        })

    return app, socketio

app, socketio = create_app()

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)