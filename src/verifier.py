from typing import List, Dict, Optional
import torch
from .config import VerificationConfig


class ClinicalVerificationEngine:
    """
    Engine Đối soát An toàn Y tế Tinh gọn (VAIPE-MedGuard):
    - So khớp ảnh viên thuốc dựa trên ngân hàng tham chiếu đa vector (Top-k k-NN).
    - Giới hạn không gian tìm kiếm trong danh mục thuốc được chỉ định (Prescription-Constrained).
    - Tự động phát hiện và chặn đứng thuốc lạ nguy hiểm tráo vào (Foreign / Counterfeit Drug Rejection).
    """
    def __init__(
        self,
        gallery_ref_tensor: torch.Tensor,
        gallery_ref_labels: List[str],
        unique_gallery_classes: List[str],
        class_to_ref_indices: Dict[str, List[int]],
        config: VerificationConfig = VerificationConfig()
    ):
        self.gallery_ref_tensor = gallery_ref_tensor
        self.gallery_ref_labels = gallery_ref_labels
        self.unique_gallery_classes = unique_gallery_classes
        self.class_to_ref_indices = class_to_ref_indices
        self.config = config

    def verify_pills(
        self,
        pill_embeddings: torch.Tensor,
        true_labels: Optional[List[str]] = None,
        candidate_classes: Optional[List[str]] = None
    ) -> Dict:
        """
        Đối soát một nhóm viên thuốc (trong cùng 1 cữ uống hoặc 1 đơn thuốc).
        
        Args:
            pill_embeddings: Tensor chứa vector đặc trưng (N_pills, D).
            true_labels: Danh sách nhãn thực tế (dùng khi benchmark/eval).
            candidate_classes: Danh sách mã thuốc trong đơn bác sĩ kê.
        """
        device = self.gallery_ref_tensor.device
        pill_embeddings = pill_embeddings.to(device)

        # Tính ma trận tương đồng Cosine: (N_pills, N_gallery_exemplars)
        sims_to_refs = torch.mm(pill_embeddings, self.gallery_ref_tensor.t())

        records = []
        n_pills = pill_embeddings.size(0)

        for i in range(n_pills):
            true_cls = true_labels[i] if true_labels and i < len(true_labels) else None
            row_sims = sims_to_refs[i]

            # 1. Tính điểm tương đồng từng loại thuốc bằng Top-k k-NN mean
            class_scores = {}
            for c in self.unique_gallery_classes:
                ref_idx = self.class_to_ref_indices.get(c, [])
                if ref_idx:
                    c_sims = row_sims[ref_idx]
                    k_val = min(self.config.k_neighbors, len(c_sims))
                    class_scores[c] = torch.topk(c_sims, k=k_val).values.mean().item()
                else:
                    class_scores[c] = -1.0

            # 2. Nhận diện Mở Toàn cục (Unconstrained)
            uncon_pred = max(class_scores.keys(), key=lambda c: class_scores[c])
            uncon_score = class_scores[uncon_pred]

            # 3. Đối soát theo Danh mục Đơn thuốc (Prescription-Constrained)
            if candidate_classes and len(candidate_classes) > 0:
                valid_cands = [c for c in candidate_classes if c in class_scores]
                cand_pred = max(valid_cands, key=lambda c: class_scores[c]) if valid_cands else uncon_pred
                cand_score = class_scores[cand_pred]

                # Phòng vệ thuốc lạ ngoài đơn (Foreign Drug Detection)
                is_foreign = (
                    (uncon_score - cand_score > self.config.foreign_drug_margin) and 
                    (uncon_score >= self.config.rejection_threshold)
                )

                if is_foreign:
                    final_pred = uncon_pred
                    final_score = uncon_score
                    status = "WRONG_UNAUTHORIZED_DRUG"
                    is_safe = False
                elif cand_score >= self.config.rejection_threshold:
                    final_pred = cand_pred
                    final_score = cand_score
                    status = "MATCH_CONFIRMED"
                    is_safe = True
                else:
                    final_pred = "UNKNOWN"
                    final_score = cand_score
                    status = "UNKNOWN_MEDICATION"
                    is_safe = False
            else:
                final_pred = uncon_pred
                final_score = uncon_score
                is_safe = (uncon_score >= self.config.rejection_threshold)
                status = "MATCH_CONFIRMED" if is_safe else "UNKNOWN_MEDICATION"

            con_correct = (final_pred == true_cls) if true_cls else None
            uncon_correct = (uncon_pred == true_cls) if true_cls else None

            records.append({
                "true_class": true_cls,
                "uncon_pred": uncon_pred,
                "uncon_score": uncon_score,
                "uncon_correct": uncon_correct,
                "con_pred": final_pred,
                "con_score": final_score,
                "con_correct": con_correct,
                "status": status,
                "is_safe": is_safe
            })

        session_safe = all(r["is_safe"] for r in records)
        return {
            "pills_count": len(records),
            "uncon_correct_count": sum(1 for r in records if r["uncon_correct"]) if true_labels else None,
            "con_correct_count": sum(1 for r in records if r["con_correct"]) if true_labels else None,
            "session_safe": session_safe,
            "details": records
        }
