import os
import yaml
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from pathlib import Path
from metrics import calculate_metrics  # You'll need to create this
from torch.utils.data import DataLoader, Dataset
from custom_model import CustomDetector, DetectionLoss
from download_data import download_dataset
from PIL import Image
import torchvision.transforms as transforms
import torch.cuda.amp as amp
from tqdm import tqdm
import time
import logging

class BloodCellDataset(Dataset):
    def __init__(self, data_path, split='train', transform=None):
        self.data_path = Path(data_path) / split
        self.image_dir = self.data_path / 'images'
        self.label_dir = self.data_path / 'labels'
        self.transform = transform or transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.image_files = list(self.image_dir.glob('*.jpg'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.label_dir / (img_path.stem + '.txt')
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Load labels
        targets = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    targets.append([class_id, x, y, w, h])
        
        targets = torch.tensor(targets if targets else [[0, 0, 0, 0, 0]])
        return image, targets

class BloodCellDetector:
    def __init__(self, data_path):
        # Setup logging first
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize other attributes
        self.data_path = data_path
        self.models = {}
        self.results = {}
        
        # Check CUDA availability and set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Log device info
        if torch.cuda.is_available():
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            self.logger.warning("CUDA is not available. Running on CPU may be slow!")
            self.logger.info("Please ensure CUDA is properly installed for GPU support")

    def setup_data_config(self):
        data_yaml = {
            'path': self.data_path,
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'names': ['RBC', 'WBC', 'Platelets']
        }
        
        with open('data.yaml', 'w') as f:
            yaml.dump(data_yaml, f)

    def train_yolov8(self, epochs=100):
        model = YOLO('yolov8n.pt')
        results = model.train(
            data='data.yaml',
            epochs=epochs,
            imgsz=640,
            batch=16,
            name='yolov8_blood_cells'
        )
        self.models['yolov8'] = model
        return results

    def train_custom_model(self, epochs=100, batch_size=16):
        model = CustomDetector(num_classes=3).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = DetectionLoss()
        
        # Only use GradScaler if CUDA is available
        scaler = amp.GradScaler() if torch.cuda.is_available() else None
        
        train_loader = self.get_data_loader('train', batch_size)
        val_loader = self.get_data_loader('val', batch_size)
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Total batches per epoch: {len(train_loader)}")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            model.train()
            total_loss = 0
            
            # Use tqdm for progress bar
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_idx, (images, targets) in enumerate(pbar):
                try:
                    images = images.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    
                    # Mixed precision training only with CUDA
                    if torch.cuda.is_available():
                        with amp.autocast():
                            predictions = model(images)
                            loss = criterion(predictions, targets)
                        
                        optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        predictions = model(images)
                        loss = criterion(predictions, targets)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                    # Log GPU memory usage periodically
                    if batch_idx % 50 == 0:
                        gpu_memory = torch.cuda.memory_allocated() / 1e9
                        self.logger.debug(f"GPU Memory used: {gpu_memory:.2f} GB")
                except RuntimeError as e:
                    self.logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    self.logger.debug(f"Images shape: {images.shape}")
                    self.logger.debug(f"Targets shape: {targets.shape}")
                    continue
            
            # Validation
            val_loss = self.validate_model(model, val_loader, criterion)
            epoch_time = time.time() - epoch_start
            
            # Log epoch statistics
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {total_loss/len(train_loader):.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pt')
                self.logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/60:.2f} minutes")
        self.models['custom'] = model
        return model

    def get_data_loader(self, split, batch_size):
        dataset = BloodCellDataset(self.data_path, split)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=4,
            pin_memory=True,  # Enable pin memory for faster GPU transfer
            collate_fn=self.collate_fn,
            prefetch_factor=2  # Enable prefetching
        )

    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        # Convert targets to a padded tensor
        max_targets = max(t.shape[0] for t in targets)
        padded_targets = []
        for t in targets:
            pad_size = max_targets - t.shape[0]
            if pad_size > 0:
                # Pad with zeros if needed
                padding = torch.zeros((pad_size, t.shape[1]), dtype=t.dtype)
                padded_target = torch.cat([t, padding], dim=0)
            else:
                padded_target = t
            padded_targets.append(padded_target)
        targets = torch.stack(padded_targets)
        return images, targets

    def validate_model(self, model, val_loader, criterion):
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                with amp.autocast():
                    predictions = model(images)
                    loss = criterion(predictions, targets)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)

    def evaluate_model(self, model_name, test_images_path):
        model = self.models[model_name]
        results = []
        
        for img_path in Path(test_images_path).glob('*.jpg'):
            prediction = model(str(img_path))
            results.append(prediction)
            
        metrics = calculate_metrics(results)
        self.results[model_name] = metrics
        return metrics

    def compare_models(self):
        # Update comparison to include both custom and YOLO models
        custom_metrics = self.evaluate_model('custom', self.data_path + '/test/images')
        yolo_metrics = self.evaluate_model('yolov8', self.data_path + '/test/images')
        
        return {
            'custom_model': custom_metrics,
            'yolov8': yolo_metrics
        }

    def visualize_results(self, image_path, model_name):
        model = self.models[model_name]
        results = model(image_path)
        
        # Plot results
        results.plot()
        cv2.imshow(f'Detection Results - {model_name}', results.plot())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    # Get dataset path from download script
    dataset_path = download_dataset()
    if (dataset_path is None):
        print("Failed to download dataset. Exiting...")
        return

    # Initialize detector with downloaded dataset path
    detector = BloodCellDetector(data_path=dataset_path)
    detector.setup_data_config()

    # Train both models
    detector.train_custom_model(epochs=100)
    detector.train_yolov8(epochs=100)

    # Compare results
    comparison = detector.compare_models()
    print("Model Comparison Results:")
    print(comparison)

if __name__ == "__main__":
    main()
