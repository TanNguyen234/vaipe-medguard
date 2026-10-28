"""
VAIPE-MedGuard: Clinical Medication Safety Verification Web Demo
Interactive interface powered by Gradio for testing prescription-constrained pill verification
and open-set foreign drug interception.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Ensure project root is on sys.path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import VerificationConfig
from src.dataset import get_eval_transforms
from src.verifier import ClinicalVerificationEngine

import gradio as gr

# Setup default catalog classes
DEMO_CATALOG = {
    "label_0": "Paracetamol 500mg (White Round Tablet)",
    "label_1": "Klamentin 1000mg (White Oblong Tablet)",
    "label_10": "Alpha Choay (Red / White Chymotrypsin)",
    "label_47": "Cetirizine 10mg (Small White Film Tablet)",
    "label_64": "Amoxicillin 500mg (Red / Yellow Capsule)",
    "label_89": "Vitamin C 500mg (Orange Circular Tablet)",
    "label_91": "Panadol Extra (White / Red Tablet)",
    "label_107": "Foreign / Out-of-Prescription Unregistered Drug"
}

# Pre-generate synthetic reference vectors (dim=384) for fast offline demo
torch.manual_seed(42)
DEMO_DIM = 384
gallery_vectors = []
gallery_labels = []

# Create consistent distinctive vectors per class
class_bases = {}
for i, c in enumerate(DEMO_CATALOG.keys()):
    v = torch.zeros(DEMO_DIM)
    v[i * 45 : (i + 1) * 45] = 1.0
    v = F.normalize(v, dim=0)
    class_bases[c] = v
    # 2 exemplars per class (representing Face A and Face B)
    v_a = v.clone()
    v_b = v.clone()
    v_b[300:] = 0.05
    v_b = F.normalize(v_b, dim=0)
    gallery_vectors.extend([v_a, v_b])
    gallery_labels.extend([c, c])

gallery_tensor = torch.stack(gallery_vectors)
unique_classes = sorted(list(DEMO_CATALOG.keys()))
class_to_indices = {
    c: [idx for idx, lbl in enumerate(gallery_labels) if lbl == c]
    for c in unique_classes
}


def mock_extract_embedding(image: Image.Image, filename: Optional[str] = None) -> torch.Tensor:
    """Extract or map embedding for demo visualization."""
    # Check if filename indicates a known sample
    if filename:
        fn = filename.lower()
        if "paracetamol" in fn:
            base = class_bases["label_0"].clone()
            base += torch.randn(DEMO_DIM) * 0.03
            return F.normalize(base, dim=0)
        elif "klamentin" in fn:
            base = class_bases["label_1"].clone()
            base += torch.randn(DEMO_DIM) * 0.03
            return F.normalize(base, dim=0)
        elif "alpha" in fn or "chymo" in fn:
            base = class_bases["label_10"].clone()
            base += torch.randn(DEMO_DIM) * 0.03
            return F.normalize(base, dim=0)
        elif "foreign" in fn:
            base = class_bases["label_107"].clone()
            base += torch.randn(DEMO_DIM) * 0.03
            return F.normalize(base, dim=0)

    # General image color-based deterministic seed
    np_img = np.array(image.resize((64, 64)))
    mean_color = np_img.mean(axis=(0, 1))
    
    # Yellowish / Foreign
    if mean_color[0] > 200 and mean_color[1] > 180 and mean_color[2] < 100:
        base = class_bases["label_107"].clone()
    # Reddish / Alpha Choay
    elif mean_color[0] > 180 and mean_color[1] < 120:
        base = class_bases["label_10"].clone()
    # White / Paracetamol
    else:
        base = class_bases["label_0"].clone()

    base += torch.randn(DEMO_DIM) * 0.05
    return F.normalize(base, dim=0)


def run_verification(
    input_image: Image.Image,
    prescribed_drugs: List[str],
    rejection_thresh: float,
    foreign_margin: float
):
    if input_image is None:
        return "<h3>Please upload or select a pill image to verify.</h3>", {}, {}

    if not prescribed_drugs:
        return "<h3>Please select at least one prescribed drug from the patient prescription.</h3>", {}, {}

    # Extract embedding
    emb = mock_extract_embedding(input_image).unsqueeze(0)

    # Map readable drug names back to labels
    cand_labels = [k for k, v in DEMO_CATALOG.items() if v in prescribed_drugs]

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

    pred_name = DEMO_CATALOG.get(final_pred, final_pred)
    uncon_name = DEMO_CATALOG.get(uncon_pred, uncon_pred)

    # Format Status Card HTML (Professional, without AI icons)
    if status == "MATCH_CONFIRMED":
        status_html = f"""
        <div style="background-color: #d4edda; border: 2px solid #28a745; border-radius: 8px; padding: 18px; color: #155724; font-family: sans-serif;">
            <h2 style="margin: 0 0 8px 0; font-size: 20px; font-weight: bold; color: #155724;">[VERIFIED] MATCH CONFIRMED - AUTHORIZED MEDICATION</h2>
            <p style="margin: 0; font-size: 15px;"><strong>Identified Drug:</strong> {pred_name}</p>
            <p style="margin: 4px 0 0 0; font-size: 14px;"><strong>Cosine Similarity Score:</strong> {score:.4f} (Threshold: &ge; {rejection_thresh})</p>
            <p style="margin: 8px 0 0 0; font-size: 13px; color: #1e7e34;">Patient safety confirmed: Pill strictly matches patient prescription.</p>
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
            <p style="margin: 8px 0 0 0; font-size: 13px; color: #856404;">Visual information insufficient (e.g. smooth plain surface). Pharmacist review required.</p>
        </div>
        """

    # Similarity breakdown across all prescribed candidates
    similarities = {}
    row_sims = torch.mm(emb, gallery_tensor.t())[0]
    for c in cand_labels:
        idxs = class_to_indices.get(c, [])
        if idxs:
            s_val = torch.topk(row_sims[idxs], k=min(2, len(idxs))).values.mean().item()
            c_name = DEMO_CATALOG.get(c, c)
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
        "Delta Margin": f"{uncon_score - score:.4f}"
    }

    return status_html, similarities, metrics_text


# Build Gradio UI
sample_dir = Path("projects/vaipe_medguard/docs/samples")
examples = []
if sample_dir.exists():
    for f in ["sample_paracetamol_500mg.jpg", "sample_klamentin_1g.jpg", "sample_alphachoay.jpg", "sample_foreign_drug.jpg"]:
        p = sample_dir / f
        if p.exists():
            examples.append([str(p)])

with gr.Blocks(title="VAIPE-MedGuard: Medication Safety Verification", theme=gr.themes.Base()) as demo:
    gr.Markdown(
        """
        # VAIPE-MedGuard: Clinical-Grade Medication Safety Verification Engine
        ### Deep Metric Learning (DINOv2) + Multi-Exemplar k-NN + Open-Set Defense
        Verify physical pill crops against patient-prescribed drug candidates with real-time foreign drug interception.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Input Pill Crop (518x518)")
            prescribed_selector = gr.CheckboxGroup(
                choices=list(DEMO_CATALOG.values())[:-1], # exclude foreign
                value=[
                    DEMO_CATALOG["label_0"], # Paracetamol
                    DEMO_CATALOG["label_1"], # Klamentin
                    DEMO_CATALOG["label_10"] # Alpha Choay
                ],
                label="Prescribed Medications in Active Patient Prescription"
            )
            with gr.Accordion("Verification Parameters", open=False):
                rejection_slider = gr.Slider(minimum=0.40, maximum=0.85, value=0.60, step=0.05, label="Rejection Threshold (tau)")
                margin_slider = gr.Slider(minimum=0.05, maximum=0.30, value=0.15, step=0.01, label="Foreign Drug Margin (Delta)")

            verify_btn = gr.Button("Verify Medication Safety", variant="primary")

            if examples:
                gr.Examples(examples=examples, inputs=input_img, label="Quick Load Sample Pill Crops")

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
