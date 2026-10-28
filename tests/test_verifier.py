import unittest
import torch
import torch.nn.functional as F

from src.config import VerificationConfig
from src.verifier import ClinicalVerificationEngine


class TestClinicalVerificationEngine(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = VerificationConfig(
            k_neighbors=2,
            rejection_threshold=0.60,
            foreign_drug_margin=0.15
        )
        # Create synthetic gallery with 3 classes, 2 exemplars each, dim=32
        # Class A: label_0 (aligned with dim 0)
        # Class B: label_1 (aligned with dim 1)
        # Class C: label_2 (aligned with dim 2)
        dim = 32
        v_0_a = torch.zeros(dim); v_0_a[0] = 1.0
        v_0_b = torch.zeros(dim); v_0_b[0] = 0.9; v_0_b[3] = 0.1; v_0_b = F.normalize(v_0_b, dim=0)

        v_1_a = torch.zeros(dim); v_1_a[1] = 1.0
        v_1_b = torch.zeros(dim); v_1_b[1] = 0.9; v_1_b[3] = 0.1; v_1_b = F.normalize(v_1_b, dim=0)

        v_2_a = torch.zeros(dim); v_2_a[2] = 1.0
        v_2_b = torch.zeros(dim); v_2_b[2] = 0.9; v_2_b[3] = 0.1; v_2_b = F.normalize(v_2_b, dim=0)

        self.gallery_tensor = torch.stack([v_0_a, v_0_b, v_1_a, v_1_b, v_2_a, v_2_b])
        self.gallery_labels = ["label_0", "label_0", "label_1", "label_1", "label_2", "label_2"]
        self.unique_classes = ["label_0", "label_1", "label_2"]
        self.class_to_indices = {
            "label_0": [0, 1],
            "label_1": [2, 3],
            "label_2": [4, 5]
        }
        self.engine = ClinicalVerificationEngine(
            gallery_ref_tensor=self.gallery_tensor,
            gallery_ref_labels=self.gallery_labels,
            unique_gallery_classes=self.unique_classes,
            class_to_ref_indices=self.class_to_indices,
            config=self.config
        )

    def test_single_dose_safe_verification(self):
        """Kiểm tra đối soát cữ uống đơn (1-3 viên) thuộc danh mục kê đơn."""
        # Query 1: label_0, Query 2: label_1
        q0 = self.gallery_tensor[0:1] # exact label_0
        q1 = self.gallery_tensor[2:3] # exact label_1
        query_embs = torch.cat([q0, q1], dim=0)
        true_labels = ["label_0", "label_1"]
        prescribed_candidates = ["label_0", "label_1", "label_2"]

        result = self.engine.verify_pills(
            pill_embeddings=query_embs,
            true_labels=true_labels,
            candidate_classes=prescribed_candidates
        )

        self.assertEqual(result["pills_count"], 2)
        self.assertEqual(result["con_correct_count"], 2)
        self.assertEqual(result["uncon_correct_count"], 2)
        self.assertTrue(result["session_safe"])
        self.assertEqual(result["details"][0]["status"], "MATCH_CONFIRMED")
        self.assertEqual(result["details"][1]["status"], "MATCH_CONFIRMED")

    def test_foreign_drug_rejection(self):
        """Kiểm tra khả năng chặn đứng thuốc lạ (label_2) khi đơn chỉ kê label_0 và label_1."""
        # Bệnh nhân uống nhầm viên label_2 (thuốc lạ nguy hiểm)
        q_foreign = self.gallery_tensor[4:5] # exact label_2
        prescribed_candidates = ["label_0", "label_1"] # Không có label_2

        result = self.engine.verify_pills(
            pill_embeddings=q_foreign,
            true_labels=["label_2"],
            candidate_classes=prescribed_candidates
        )

        self.assertEqual(result["pills_count"], 1)
        # Bị phát hiện là thuốc lạ không được phép
        self.assertFalse(result["session_safe"])
        self.assertEqual(result["details"][0]["status"], "WRONG_UNAUTHORIZED_DRUG")
        self.assertEqual(result["details"][0]["con_pred"], "label_2")

    def test_unknown_medication_rejection(self):
        """Kiểm tra từ chối khi thuốc có độ tương đồng quá thấp (< 0.60)."""
        dim = 32
        v_noise = torch.zeros(1, dim)
        v_noise[0, 15] = 1.0 # orthogonal noise vector
        prescribed_candidates = ["label_0", "label_1"]

        result = self.engine.verify_pills(
            pill_embeddings=v_noise,
            true_labels=["unknown_drug"],
            candidate_classes=prescribed_candidates
        )

        self.assertFalse(result["session_safe"])
        self.assertEqual(result["details"][0]["status"], "UNKNOWN_MEDICATION")
        self.assertEqual(result["details"][0]["con_pred"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
