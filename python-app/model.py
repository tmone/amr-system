import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path
import logging
import json

class AMRModel:
    def __init__(self, max_sequence_length=1000, data_dir="data", socketio=None):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.max_sequence_length = max_sequence_length
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Update model path to include .keras extension
        self.model_path = self.data_dir / "saved_model.keras"
        self.tokenizer_path = self.data_dir / "tokenizer.json"
        self.label_encoder_path = self.data_dir / "label_encoder.npy"
        
        # Initialize objects
        self.tokenizer = None
        self.label_encoder = None
        self.model = None
        self.last_training_epochs = 0
        self.socketio = socketio
        
        # Load or create tokenizer
        self.initialize_tokenizer()
        self.initialize_label_encoder()
        self.load_model()

    def initialize_tokenizer(self):
        """Initialize or load tokenizer"""
        try:
            if self.tokenizer_path.exists():
                with open(self.tokenizer_path, 'r') as f:
                    self.tokenizer = tokenizer_from_json(f.read())
                self.logger.info("Loaded existing tokenizer")
            else:
                self.tokenizer = Tokenizer(char_level=True)
                self.logger.info("Created new tokenizer")
        except Exception as e:
            self.logger.warning(f"Error loading tokenizer: {e}, creating new one")
            self.tokenizer = Tokenizer(char_level=True)

    def initialize_label_encoder(self):
        """Initialize or load label encoder"""
        self.label_encoder = LabelEncoder()
        if self.label_encoder_path.exists():
            try:
                self.label_encoder.classes_ = np.load(self.label_encoder_path)
                self.logger.info("Loaded existing label encoder")
            except Exception as e:
                self.logger.warning(f"Error loading label encoder: {e}")

    def save_tokenizer(self):
        """Save tokenizer state"""
        if self.tokenizer and hasattr(self.tokenizer, 'to_json'):
            with open(self.tokenizer_path, 'w') as f:
                f.write(self.tokenizer.to_json())
            self.logger.info("Saved tokenizer")

    def load_card_data(self):
        """Load CARD reference data for model training"""
        clinical_data = self.data_dir / "clinically_relevant_amr.fasta"
        if not clinical_data.exists():
            raise FileNotFoundError(f"CARD data not found at {clinical_data}")
        # Process FASTA file and return sequences
        # Implementation details would go here

    def load_reference_data(self, reference_file):
        """Load specific reference data file for training"""
        file_path = self.data_dir / reference_file
        if not file_path.exists():
            raise FileNotFoundError(f"Reference data not found: {file_path}")
            
        sequences = []
        current_sequence = ''
        
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('>'):
                    if current_sequence:
                        sequences.append(current_sequence)
                    current_sequence = ''
                else:
                    current_sequence += line
                    
            if current_sequence:  # Add the last sequence
                sequences.append(current_sequence)
                
        return sequences
        
    def create_model(self, vocab_size):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, 32, input_length=self.max_sequence_length),
            tf.keras.layers.Conv1D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
    def prepare_sequences(self, sequences):
        """Prepare sequences with tokenizer"""
        if not self.tokenizer:
            self.initialize_tokenizer()
        
        # Fit tokenizer if it hasn't been fit before
        if not self.tokenizer.word_index:
            self.tokenizer.fit_on_texts(sequences)
            self.save_tokenizer()
            
        sequences = self.tokenizer.texts_to_sequences(sequences)
        return pad_sequences(sequences, maxlen=self.max_sequence_length)
        
    def train(self, sequences, labels, reference_file=None, epochs=50):
        self.last_training_epochs = epochs
        if reference_file:
            reference_sequences = self.load_reference_data(reference_file)
            # Combine reference data with training sequences
            sequences = sequences + reference_sequences
            labels = labels + [1] * len(reference_sequences)  # Assuming reference data is resistant

        X = self.prepare_sequences(sequences)
        y = self.label_encoder.fit_transform(labels)
        
        if not self.model:
            self.create_model(len(self.tokenizer.word_index) + 1)
            
        history = self.model.fit(
            X, y,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3),
                tf.keras.callbacks.ReduceLROnPlateau()
            ]
        )
        self.save_model()  # Save model after training
        return history.history
        
    def train_with_reference(self, reference_file, epochs=50):
        """Train model using only reference data"""
        try:
            self.last_training_epochs = epochs
            sequences = self.load_reference_data(reference_file)
            labels = [1] * len(sequences)  # All reference sequences are resistant
            
            X = self.prepare_sequences(sequences)
            y = self.label_encoder.fit_transform(labels)
            
            if not self.model:
                self.create_model(len(self.tokenizer.word_index) + 1)
                
            history = self.model.fit(
                X, y,
                epochs=epochs,
                validation_split=0.2,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=3,
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.2,
                        patience=2
                    ),
                    tf.keras.callbacks.LambdaCallback(
                        on_epoch_end=lambda epoch, logs: self.emit_progress(epoch, logs)
                    )
                ],
                verbose=1
            )
            
            # Save model after successful training
            self.save_model()
            
            # Convert numpy values to Python types for JSON serialization
            return {
                'loss': [float(x) for x in history.history['loss']],
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']]
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}")

    def emit_progress(self, epoch, logs):
        """Emit training progress with learning rate"""
        if self.socketio:
            self.socketio.emit('training_progress', {
                'epoch': int(epoch),
                'loss': float(logs.get('loss', 0)),
                'val_loss': float(logs.get('val_loss', 0)),
                'accuracy': float(logs.get('accuracy', 0)),
                'val_accuracy': float(logs.get('val_accuracy', 0)),
                'learning_rate': float(
                    self.model.optimizer.learning_rate.numpy()
                ) if self.model else 0
            })

    def predict(self, sequences):
        """Predict with error handling for missing model"""
        if not self.model:
            raise RuntimeError("No trained model available. Please train the model first.")
            
        X = self.prepare_sequences(sequences)
        predictions = self.model.predict(X)
        return predictions.flatten()

    def predict_sequence(self, sequence):
        """Predict AMR for a single sequence."""
        try:
            # Convert sequence to features - Fix: use max_sequence_length instead of max_length
            features = self.tokenizer.texts_to_sequences([sequence])
            features = pad_sequences(features, maxlen=self.max_sequence_length, padding='post')
            
            if not self.model:
                raise ValueError("No trained model available. Please train the model first.")

            # Make prediction
            prediction = self.model.predict(features)[0]
            predicted_class = int(prediction >= 0.5)
            confidence = float(prediction if predicted_class else 1 - prediction) * 100
            
            return {
                'class': predicted_class,
                'confidence': confidence
            }
        except Exception as e:
            raise ValueError(f"Error predicting sequence: {str(e)}")

    def load_model(self):
        """Load saved model and preprocessing objects if they exist"""
        try:
            if self.model_path.exists():
                self.logger.info(f"Loading model from {self.model_path}")
                self.model = tf.keras.models.load_model(self.model_path)
                self.logger.info("Model loaded successfully")
                
            if self.tokenizer_path.exists():
                with open(self.tokenizer_path, 'r') as f:
                    self.tokenizer = tokenizer_from_json(f.read())
                    
            if self.label_encoder_path.exists():
                self.label_encoder = LabelEncoder()
                self.label_encoder.classes_ = np.load(self.label_encoder_path)
                
            return self.model is not None
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def save_model(self):
        """Save model and preprocessing objects"""
        try:
            if self.model:
                # Ensure the directory exists
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                # Save model with .keras extension
                self.model.save(self.model_path, save_format='keras')
                self.logger.info(f"Model saved to {self.model_path}")
                
            self.save_tokenizer()
            if hasattr(self.label_encoder, 'classes_'):
                np.save(self.label_encoder_path, self.label_encoder.classes_)
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
