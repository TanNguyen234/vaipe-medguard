from dataclasses import dataclass
from pathlib import Path


@dataclass
class VerificationConfig:
    """
    Cấu hình tiêu chuẩn cho Engine Đối soát Lâm sàng VAIPE-MedGuard:
    - k_neighbors: Số láng giềng k-NN cho mỗi lớp để bao phủ cả 2 mặt viên thuốc (Face A & Face B).
    - rejection_threshold: Ngưỡng Cosine tối thiểu để xác nhận viên thuốc thuộc danh mục kê đơn.
    - foreign_drug_margin: Độ chênh lệch điểm (Top-1 toàn cục vs Top-1 đơn) để chặn thuốc lạ tráo vào.
    - img_size: Kích thước ảnh chuẩn sau letterboxing (518x518 cho DINOv2 ViT-Base).
    """
    k_neighbors: int = 2
    rejection_threshold: float = 0.60
    foreign_drug_margin: float = 0.15
    img_size: int = 518
    device: str = "cuda"


@dataclass
class PathConfig:
    """Đường dẫn dữ liệu và trọng số chuẩn."""
    weights_path: Path = Path("/kaggle/input/vaipe-medguard-weights/best_metric_embedder.pt")
    output_dir: Path = Path("/kaggle/working/output")
