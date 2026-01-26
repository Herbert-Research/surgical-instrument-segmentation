# Artifacts Guide

## Authoritative Outputs (Cite These)

- `outputs/run_<id>/metrics.json` — Verified benchmark results
- `outputs/run_<id>/predictions/` — Full validation set predictions

## Demo/Illustrative Outputs (Do NOT Cite as Evidence)

- `scripts/demo_best_cases.py` output — Cherry-picked easy frames for visualization
- `notebooks/*.ipynb` — Exploratory analysis, not reproducible

## How to Generate Authoritative Results

```bash
python -m surgical_segmentation.training.trainer \
    --config configs/default.yaml \
    --frame-dir /path/to/cholec80/frames \
    --mask-dir /path/to/cholec80/masks \
    --output-dir outputs/run_$(date +%Y%m%d_%H%M%S)
```
