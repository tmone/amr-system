import torch
import torch.nn as nn
import torchvision.models as models
import torch.cuda.amp as amp

class CustomDetector(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Check CUDA availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Use efficient backbone
        backbone = models.efficientnet_b0(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        
        # Use more efficient detection head
        self.det_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes * 5)
        )
        
        # Initialize weights for faster convergence
        for m in self.det_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    @torch.cuda.amp.autocast(enabled=torch.cuda.is_available())  # Enable automatic mixed precision
    def forward(self, x):
        features = self.features(x)
        detections = self.det_head(features)
        return detections.view(x.shape[0], -1, 5)

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.reg_loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, predictions, targets):
        """
        Args:
            predictions: [batch_size, num_predictions, 5]
            targets: [batch_size, max_targets, 5]
        """
        batch_size = predictions.shape[0]
        num_predictions = predictions.shape[1]
        
        # Split predictions into classification and regression
        pred_cls = predictions[..., 0]  # [batch_size, num_predictions]
        pred_box = predictions[..., 1:]  # [batch_size, num_predictions, 4]
        
        # Initialize targets with same size as predictions
        target_cls = torch.zeros_like(pred_cls)
        target_box = torch.zeros_like(pred_box)
        
        # For each item in batch, fill in actual targets
        for i in range(batch_size):
            # Get valid targets (non-zero rows)
            valid_mask = targets[i].sum(dim=1) > 0
            valid_targets = targets[i][valid_mask]
            
            if len(valid_targets) > 0:
                # Limit number of targets to num_predictions
                num_valid = min(len(valid_targets), num_predictions)
                target_cls[i, :num_valid] = valid_targets[:num_valid, 0]
                target_box[i, :num_valid] = valid_targets[:num_valid, 1:]

        # Calculate losses
        cls_loss = self.bce_loss(pred_cls, target_cls)
        box_loss = self.reg_loss(pred_box, target_box)
        
        return cls_loss + box_loss
