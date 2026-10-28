import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None


class DINOv2MetricEmbedder(nn.Module):
    """
    Mô hình trích xuất đặc trưng Fine-Grained Pill Embedding dựa trên DINOv2 và Linear Projector.
    - Backbone: Vision Transformer (DINOv2 Small/Base patch 14) với độ phân giải native 518x518.
    - Projector: 2-layer MLP chiếu về không gian metric 384 chiều, L2 normalize.
    """
    def __init__(self, embedding_dim: int = 384, pretrained: bool = True, img_size: int = 518, backbone_name: str = "vit_small_patch14_dinov2.lvd142m"):
        super().__init__()
        if timm is None:
            raise ImportError("Vui lòng cài đặt timm (`pip install timm`) để sử dụng DINOv2MetricEmbedder.")
            
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            dynamic_img_size=False
        )
        in_features = self.backbone.num_features
        self.projector = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.projector(features)
        return F.normalize(embeddings, p=2, dim=1)

    def load_weights(self, weights_path: str, device: str = "cpu"):
        state_dict = torch.load(weights_path, map_location=device)
        self.load_state_dict(state_dict)
        self.eval()
