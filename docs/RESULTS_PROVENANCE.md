# Results Provenance

This document provides audit trails for all reported metrics.

## Metric Claims in README.md

| Metric | Value | Run ID | Git SHA | Dataset | Config | Verified By | Date |
|--------|-------|--------|---------|---------|--------|-------------|------|
| IoU    | TBD   | TBD    | TBD     | TBD     | TBD    | TBD         | TBD  |
| Dice   | TBD   | TBD    | TBD     | TBD     | TBD    | TBD         | TBD  |

## How to Verify

1. Check out the git SHA listed
2. Run: `python -m surgical_segmentation.training.trainer --config configs/default.yaml`
3. Compare output metrics to the values above

## Evidence Packets

Each run generates an evidence packet in `outputs/run_<id>/` containing:
- `config.yaml` — exact configuration used
- `environment.txt` — Python version, package versions
- `dataset_manifest.json` — file hashes of training data
- `splits.json` — exact train/val split
- `metrics.json` — computed metrics
- `model_checkpoint.pth` — trained weights
- `training.log` — full training log
