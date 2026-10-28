from typing import Tuple, List, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
from collections import defaultdict


class LetterboxPad:
    """
    Tiền xử lý bảo toàn tỷ lệ khung hình (Aspect Ratio) của viên thuốc:
    - Resize ảnh sao cho cạnh lớn nhất vừa khít kích thước mục tiêu (518x518).
    - Đệm (pad) đều 2 bên bằng màu xám trung tính (fill=128), triệt tiêu méo mó hình ảnh.
    """
    def __init__(self, target_size: Tuple[int, int] = (518, 518), fill: int = 128):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        target_w, target_h = self.target_size
        scale = min(target_w / max(1, w), target_h / max(1, h))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        fill_color = (self.fill, self.fill, self.fill) if img.mode == 'RGB' else self.fill
        new_img = Image.new(img.mode, self.target_size, fill_color)
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        new_img.paste(img_resized, (pad_left, pad_top))
        return new_img


def get_eval_transforms(img_size: int = 518):
    """Pipeline chuẩn hóa ảnh đầu vào cho mô hình DINOv2."""
    return v2.Compose([
        LetterboxPad(target_size=(img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class FastPillDataset(Dataset):
    """Dataset tốc độ cao nạp ảnh viên thuốc vật lý."""
    def __init__(self, image_paths: List[str], labels: List[str], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (518, 518), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], path


def build_multi_exemplar_gallery(
    gallery_pills: List[Dict],
    path_to_embedding: Dict[str, torch.Tensor],
    max_exemplars_per_class: int = 8,
    device: str = "cpu"
):
    """
    Xây dựng Ngân hàng Tham chiếu Đa Vector (Multi-Exemplar Gallery Bank):
    - Thay vì dùng 1 centroid trung bình làm loãng đặc trưng, lưu trữ tối đa N vector mẫu thật
      cho từng lớp thuốc, bảo lưu trọn vẹn cả Face A (có ký hiệu dập) và Face B (mặt trơn).
    """
    class_to_embs = defaultdict(list)
    for p in gallery_pills:
        emb = path_to_embedding.get(p["path"])
        if emb is not None:
            class_to_embs[p["true_class_id"]].append(emb)

    gallery_ref_tensors = []
    gallery_ref_labels = []

    for c, embs in sorted(class_to_embs.items()):
        if not embs:
            continue
        stack_embs = torch.stack(embs)
        if len(embs) <= max_exemplars_per_class:
            selected_embs = stack_embs
        else:
            # Chọn các vector đa dạng nhất thông qua khoảng cách tới centroid
            centroid = stack_embs.mean(dim=0, keepdim=True)
            dists = torch.norm(stack_embs - centroid, dim=1)
            sorted_idx = torch.argsort(dists)
            half = max_exemplars_per_class // 2
            # Lấy các mẫu gần centroid (đại diện chung) và xa centroid (mẫu đặc thù/góc cạnh)
            chosen_idx = torch.cat([sorted_idx[:half], sorted_idx[-half:]])
            selected_embs = stack_embs[chosen_idx]

        for emb in selected_embs:
            gallery_ref_tensors.append(emb)
            gallery_ref_labels.append(c)

    gallery_ref_tensor = torch.stack(gallery_ref_tensors).to(device)
    unique_gallery_classes = sorted(list(set(gallery_ref_labels)))
    class_to_ref_indices = {
        c: [idx for idx, lbl in enumerate(gallery_ref_labels) if lbl == c]
        for c in unique_gallery_classes
    }

    return gallery_ref_tensor, gallery_ref_labels, unique_gallery_classes, class_to_ref_indices
