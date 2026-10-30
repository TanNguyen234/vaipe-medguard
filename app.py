"""
VAIPE-MedGuard: Clinical Medication Safety Verification Web Demo
Interactive interface powered by Gradio for real-time prescription-constrained pill verification
and open-set foreign drug interception using trained DINOv2 Deep Metric Learning.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn.functional as F
from PIL import Image
import gradio as gr

# Ensure project root is on sys.path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import VerificationConfig
from src.dataset import get_eval_transforms
from src.model import DINOv2MetricEmbedder
from src.verifier import ClinicalVerificationEngine

# Paths
WEIGHTS_PATH = repo_root / "weights" / "best_metric_embedder.pt"
GALLERY_BANK_PATH = repo_root / "weights" / "gallery_bank.pt"
SAMPLES_DIR = repo_root / "docs" / "samples"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_verification_system():
    """Load real trained DINOv2 weights and multi-exemplar reference gallery."""
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {WEIGHTS_PATH}")
    if not GALLERY_BANK_PATH.exists():
        raise FileNotFoundError(f"Gallery bank not found at: {GALLERY_BANK_PATH}")

    # Load Embedder
    model = DINOv2MetricEmbedder(pretrained=False, embedding_dim=384)
    model.load_weights(str(WEIGHTS_PATH), device=str(device))
    model.eval()

    # Load Gallery Bank
    bank_data = torch.load(str(GALLERY_BANK_PATH), map_location=device)
    gallery_tensor = bank_data["gallery_tensor"].to(device)
    gallery_labels = bank_data["gallery_labels"]
    unique_classes = bank_data["unique_classes"]
    class_to_indices = bank_data["class_to_indices"]
    class_names = bank_data["class_names"]

    return model, gallery_tensor, gallery_labels, unique_classes, class_to_indices, class_names


# Initialize production model and gallery
embedder_model, gallery_tensor, gallery_labels, unique_classes, class_to_indices, catalog_names = load_verification_system()
eval_transforms = get_eval_transforms()


def extract_real_embedding(image: Image.Image) -> torch.Tensor:
    """Extract fine-grained 384-dimensional metric embedding using real DINOv2 ViT."""
    tensor = eval_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = embedder_model(tensor)
    return emb


def run_verification(
    input_image: Image.Image,
    prescribed_drugs: List[str],
    rejection_thresh: float,
    foreign_margin: float
):
    if input_image is None:
        return "<div style='padding: 12px; color: #856404; background-color: #fff3cd; border-radius: 6px;'>Please upload or select a pill image to verify.</div>", {}, {}

    if not prescribed_drugs:
        return "<div style='padding: 12px; color: #721c24; background-color: #f8d7da; border-radius: 6px;'>Please select at least one prescribed drug from the patient prescription.</div>", {}, {}

    # 1. Extract real embedding using trained DINOv2 backbone
    emb = extract_real_embedding(input_image)

    # 2. Map human-readable drug names back to internal labels
    name_to_label = {v: k for k, v in catalog_names.items()}
    cand_labels = [name_to_label[d] for d in prescribed_drugs if d in name_to_label]

    # 3. Configure verification engine
    cfg = VerificationConfig(
        k_neighbors=2,
        rejection_threshold=rejection_thresh,
        foreign_drug_margin=foreign_margin
    )

    engine = ClinicalVerificationEngine(
        gallery_ref_tensor=gallery_tensor,
        gallery_ref_labels=gallery_labels,
        unique_gallery_classes=unique_classes,
        class_to_ref_indices=class_to_indices,
        config=cfg
    )

    # 4. Perform prescription-constrained matching & open-set rejection
    result = engine.verify_pills(
        pill_embeddings=emb,
        candidate_classes=cand_labels
    )

    detail = result["details"][0]
    status = detail["status"]
    final_pred = detail["con_pred"]
    score = detail["con_score"]
    uncon_pred = detail["uncon_pred"]
    uncon_score = detail["uncon_score"]

    pred_name = catalog_names.get(final_pred, final_pred)
    uncon_name = catalog_names.get(uncon_pred, uncon_pred)

    # 5. Format Clinical Status Card HTML (Professional, without AI icons)
    if status == "MATCH_CONFIRMED":
        status_html = f"""
        <div style="background-color: #d4edda; border: 2px solid #28a745; border-radius: 8px; padding: 18px; color: #155724; font-family: sans-serif;">
            <h2 style="margin: 0 0 8px 0; font-size: 20px; font-weight: bold; color: #155724;">[VERIFIED] MATCH CONFIRMED - AUTHORIZED MEDICATION</h2>
            <p style="margin: 0; font-size: 15px;"><strong>Identified Drug:</strong> {pred_name}</p>
            <p style="margin: 4px 0 0 0; font-size: 14px;"><strong>Cosine Similarity Score:</strong> {score:.4f} (Threshold: &ge; {rejection_thresh})</p>
            <p style="margin: 8px 0 0 0; font-size: 13px; color: #1e7e34;">Patient safety confirmed: Pill strictly matches active patient prescription.</p>
        </div>
        """
    elif status == "WRONG_UNAUTHORIZED_DRUG":
        status_html = f"""
        <div style="background-color: #f8d7da; border: 2px solid #dc3545; border-radius: 8px; padding: 18px; color: #721c24; font-family: sans-serif;">
            <h2 style="margin: 0 0 8px 0; font-size: 20px; font-weight: bold; color: #721c24;">[CRITICAL ALERT] FOREIGN / UNAUTHORIZED DRUG INTERCEPTED</h2>
            <p style="margin: 0; font-size: 15px;"><strong>Detected External Pill:</strong> {uncon_name}</p>
            <p style="margin: 4px 0 0 0; font-size: 14px;"><strong>Global Similarity:</strong> {uncon_score:.4f} (Margin to prescription best: +{uncon_score - score:.4f})</p>
            <p style="margin: 8px 0 0 0; font-size: 13px; font-weight: bold; color: #bd2130;">Action required: Do not ingest. This pill was not prescribed in the patient's active prescription.</p>
        </div>
        """
    else: # UNKNOWN_MEDICATION
        status_html = f"""
        <div style="background-color: #fff3cd; border: 2px solid #ffc107; border-radius: 8px; padding: 18px; color: #856404; font-family: sans-serif;">
            <h2 style="margin: 0 0 8px 0; font-size: 20px; font-weight: bold; color: #856404;">[WARNING] UNKNOWN MEDICATION / LOW CONFIDENCE</h2>
            <p style="margin: 0; font-size: 15px;"><strong>Best Candidate Match:</strong> {pred_name}</p>
            <p style="margin: 4px 0 0 0; font-size: 14px;"><strong>Similarity Score:</strong> {score:.4f} (Below safe threshold &lt; {rejection_thresh})</p>
            <p style="margin: 8px 0 0 0; font-size: 13px; color: #856404;">Visual information insufficient (e.g. plain featureless surface or severe occlusion). Pharmacist review required.</p>
        </div>
        """

    # 6. Similarity breakdown across all prescribed candidates
    similarities = {}
    row_sims = torch.mm(emb, gallery_tensor.t())[0]
    for c in cand_labels:
        idxs = class_to_indices.get(c, [])
        if idxs:
            s_val = torch.topk(row_sims[idxs], k=min(2, len(idxs))).values.mean().item()
            c_name = catalog_names.get(c, c)
            similarities[c_name] = round(float(s_val), 4)

    # Sort descending
    similarities = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))

    metrics_text = {
        "Status Code": status,
        "Prescription Safe": result["session_safe"],
        "Constrained Best Match": pred_name,
        "Constrained Similarity": f"{score:.4f}",
        "Global Unconstrained Match": uncon_name,
        "Global Similarity": f"{uncon_score:.4f}",
        "Delta Margin": f"{uncon_score - score:.4f}",
        "Device": str(device)
    }

    return status_html, similarities, metrics_text


# Build Gradio UI
examples = []
if SAMPLES_DIR.exists():
    for f in ["sample_paracetamol_500mg.jpg", "sample_klamentin_1g.jpg", "sample_alphachoay.jpg", "sample_foreign_drug.jpg"]:
        p = SAMPLES_DIR / f
        if p.exists():
            examples.append([str(p)])

# Candidate choices (exclude foreign label by default from prescription list)
selectable_choices = [v for k, v in catalog_names.items() if k != "label_107"]

with gr.Blocks(title="VAIPE-MedGuard: Medication Safety Verification", theme=gr.themes.Base()) as demo:
    gr.Markdown(
        """
        # VAIPE-MedGuard: Clinical-Grade Medication Safety Verification Engine
        ### Deep Metric Learning (DINOv2) + Multi-Exemplar k-NN Gallery + Open-Set Defense
        Real-time physical pill verification against patient-prescribed drug candidates with automated foreign drug interception.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Input Pill Crop (518x518)")
            prescribed_selector = gr.CheckboxGroup(
                choices=selectable_choices,
                value=selectable_choices, # By default all authorized prescription drugs are active
                label="Prescribed Medications in Active Patient Prescription"
            )
            with gr.Accordion("Clinical Safety Parameters", open=False):
                rejection_slider = gr.Slider(minimum=0.40, maximum=0.85, value=0.60, step=0.05, label="Rejection Threshold (tau)")
                margin_slider = gr.Slider(minimum=0.05, maximum=0.30, value=0.15, step=0.01, label="Foreign Drug Margin (Delta)")

            verify_btn = gr.Button("Verify Medication Safety", variant="primary")

            if examples:
                gr.Examples(examples=examples, inputs=input_img, label="Quick Load Clinical Sample Pill Crops")

        with gr.Column(scale=1):
            status_box = gr.HTML(label="Clinical Decision")
            chart_output = gr.Label(label="Similarity Scores across Prescribed Candidates")
            metrics_box = gr.JSON(label="Detailed Verification Telemetry")

    verify_btn.click(
        fn=run_verification,
        inputs=[input_img, prescribed_selector, rejection_slider, margin_slider],
        outputs=[status_box, chart_output, metrics_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
