import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from einops import rearrange

class CNNModel(nn.Module):
    """Pure CNN model for mammogram classification."""
    
    def __init__(self, num_classes=3, pretrained=True):
        super(CNNModel, self).__init__()
        
        # Use ResNet50 as the backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace the first conv layer to accept grayscale images if needed
        # self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class TransformerModel(nn.Module):
    """Pure Transformer model for mammogram classification."""
    
    def __init__(self, num_classes=3, pretrained=True):
        super(TransformerModel, self).__init__()
        
        # Use Vision Transformer (ViT) as the backbone
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        
        # Replace the final classification head
        self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class HybridSequentialModel(nn.Module):
    """
    Hybrid CNN + Transformer model (sequential approach).
    CNN extracts features, then Transformer processes them.
    """
    
    def __init__(self, num_classes=3, pretrained=True):
        super(HybridSequentialModel, self).__init__()
        
        # CNN backbone (ResNet without final FC layer)
        resnet = models.resnet50(pretrained=pretrained)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Transformer encoder
        self.transformer_dim = 512
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Projection from CNN feature dimension to transformer dimension
        self.projection = nn.Conv2d(2048, self.transformer_dim, kernel_size=1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.transformer_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # CNN feature extraction
        x = self.cnn_backbone(x)  # [batch_size, 2048, h, w]
        
        # Project to transformer dimension
        x = self.projection(x)  # [batch_size, transformer_dim, h, w]
        
        # Reshape for transformer: [batch_size, seq_len, transformer_dim]
        batch_size, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # [h*w, batch_size, transformer_dim]
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # [h*w, batch_size, transformer_dim]
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=0)  # [batch_size, transformer_dim]
        
        # Classification
        x = self.classifier(x)
        
        return x

class HybridParallelModel(nn.Module):
    """
    Hybrid CNN + Transformer model (parallel approach).
    CNN and Transformer process the input in parallel, then features are combined.
    """
    
    def __init__(self, num_classes=3, pretrained=True):
        super(HybridParallelModel, self).__init__()
        
        # CNN branch
        resnet = models.resnet50(pretrained=pretrained)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_feature_dim = 2048
        
        # Transformer branch
        vit = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=0)
        self.transformer_backbone = vit
        self.transformer_feature_dim = vit.embed_dim
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(self.cnn_feature_dim + self.transformer_feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # CNN branch
        cnn_features = self.cnn_backbone(x)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)  # Flatten
        
        # Transformer branch
        transformer_features = self.transformer_backbone(x)
        
        # Concatenate features
        combined_features = torch.cat([cnn_features, transformer_features], dim=1)
        
        # Classification
        output = self.fusion(combined_features)
        
        return output