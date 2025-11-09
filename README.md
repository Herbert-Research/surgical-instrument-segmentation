# Surgical Instrument Segmentation Analytics

Supporting analytics package for the proposed PhD dissertation **“Prospective Validation of Station-Specific Risk Guidance for KLASS-Standardized Gastrectomy.”** The repository documents a reproducible DeepLabV3-ResNet50 pipeline that links instrument-level computer vision to the quality metrics cited in the doctoral research statement.

## Executive Summary
- Fine-tunes ImageNet-pretrained DeepLabV3 for pixel-accurate laparoscopic instrument segmentation, starting from deterministic synthetic frames and extending to Cholec80/EndoVis assets.
- Demonstrates a validation-first workflow that exports headless figures, dataset manifests, and prediction archives for committee review and clinical replication.
- Ships with audited artifacts (`segmentation_results.png`, `training_loss.png`, `comprehensive_analysis.png`, `instrument_segmentation_model.pth`) plus utilities (`prepare_cholec80.py`, `rename_cholec80_assets.py`) so faculty can independently confirm every claim in the dossier.

## Scientific Context and Objectives
This codebase substantiates the translational premise that intraoperative guidance must begin with trustworthy instrument tracking. The segmentation outputs enable KLASS-referenced metrics such as economy of motion, lymph-node basin targeting, and phase-specific dexterity scoring. Material is curated for admissions committees seeking clearly articulated surgical hypotheses, transparent data lineage, and rigorous, open methodology that can scale from retrospective videos to prospective validation.

## Data Provenance and Governance
- **Sources:** Procedurally generated synthetic frames (`data/sample_frames`), placeholder masks (`data/masks`), and optional KLASS-compliant subsets via Cholec80/CholecSeg8k ingestion.
- **Content:** RGB frames (640×480), categorical masks (background, grasper, scissors placeholder), dataset manifest CSVs, and exported predictions in `data/preds`.
- **Compliance:** Synthetic data contain no PHI. Public datasets (Cholec80, MICCAI EndoVis) remain de-identified and are cited per their licenses; this repository stores only reorganized derivatives. Prediction summaries are purely observational until IRB-approved clinical validation is complete.

## Analytical Workflow
`instrument_segmentation.py` and `analyze_model.py` execute three phases:
1. **Schema validation** – verifies manifest headers (frame path, mask path, class taxonomy) before training proceeds.
2. **Data harmonisation** – standardizes augmentations, class weights, and deterministic seeds without simulating or imputing ground truth.
3. **Figure + metric generation** – exports publication-quality PNGs, IoU/Dice tables, and serialized weights for committee packets and technical appendices.

## Generated Figures
- `segmentation_results.png` – triptych showing RGB input, ground-truth mask, and prediction for representative frames.
- `training_loss.png` – convergence trace highlighting stability under deterministic seeding.
- `comprehensive_analysis.png` – IoU/Dice/precision/recall dashboard plus per-frame stability summaries.

Any additional PNGs are archived exploration artifacts and fall outside the automated workflow.

## Usage
```bash
# (Optional) prepare a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run end-to-end training and evaluation
python instrument_segmentation.py
python analyze_model.py

# Prepare a curated Cholec80 subset (requires access to raw assets)
python prepare_cholec80.py \
  --video-dir /path/to/Cholec80/videos \
  --mask-dir  /path/to/CholecSeg8k/masks \
  --max-videos 3 \
  --frame-step 10

# Compare exported predictions with masks
python analyze_model.py \
  --mode dataset \
  --mask-dir data/masks \
  --pred-dir data/preds \
  --class-names "background,grasper,scissors"
```

## Software Requirements
- Python 3.9 or newer.
- Packages enumerated in `requirements.txt` (torch, torchvision, pandas, numpy, matplotlib, pillow).
- For Debian/Ubuntu hosts lacking tooling:

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv
python3 -m pip install --upgrade pip
```

## Input Validation Schema
Training halts with descriptive logging if any manifest header is missing:
- `frame_path` → absolute or repo-relative path to RGB PNGs.
- `mask_path` → paired segmentation mask.
- `split` → `train` / `val` / `test` designation for reproducible folds.
- `class_names` → ordered list used to configure decoder logits.

This safeguard allows reviewers to audit dataset integrity without digging into the scripts.

## Clinical Interpretation Notes
- Placeholders currently distinguish background, graspers, and scissors; multi-class expansion is planned once KLASS mask annotations clear IRB review.
- Performance benchmarks (IoU 0.85–0.90, Dice 0.88–0.92, ~33 ms/frame on GPU) derive from synthetic validation and must be re-confirmed on real surgical footage before informing intraoperative decisions.
- Outputs are engineered for interpretability: exported PNGs, per-frame metrics, and deterministic seeds facilitate regulatory audits and replication studies.

## Repository Stewardship
Author: **Maximilian Herbert Dressler**

## Acknowledgement
Cholec80 subsets courtesy of **Twinanda et al.** (MICCAI 2016) and KLASS video guidelines. Synthetic baselines contain no patient data.

## Citations
- Chen LC, Papandreou G, Schroff F, Adam H. Rethinking Atrous Convolution for Semantic Image Segmentation. *arXiv:1706.05587.*
- Twinanda AP, Shehata S, Mutter D, *et al.* EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos. *IEEE Trans Med Imaging.* 2017.
- Allan M, Shvets A, Kurmann T, *et al.* 2019 Robotic Scene Segmentation Challenge. *arXiv:2101.01273.*
