# Dataset Setup Guide

## CholecSeg8k Dataset

### Prerequisites

1. Create a Kaggle account: https://www.kaggle.com
2. Accept the dataset license: https://www.kaggle.com/datasets/newslab/cholecseg8k
3. Install Kaggle CLI:
   ```bash
   pip install kaggle
   ```
4. Configure Kaggle credentials:
   - Generate an API token at https://www.kaggle.com/settings → API → Create New Token
   - Save `kaggle.json` to `~/.kaggle/` (Linux/macOS) or `%USERPROFILE%\.kaggle\` (Windows)
   - Set permissions (Linux/macOS): `chmod 600 ~/.kaggle/kaggle.json`

### Download Commands

```bash
# Create datasets directory
mkdir -p datasets/CholecSeg8k

# Download dataset (≈2.3 GB)
kaggle datasets download -d newslab/cholecseg8k -p datasets/

# Extract
unzip datasets/cholecseg8k.zip -d datasets/CholecSeg8k

# Verify structure
ls datasets/CholecSeg8k/
# Expected: video01/, video09/, video12/, ... video55/
```

### Data Organization

After downloading, standardize the frames and masks using the provided script:

```bash
python scripts/prepare_full_dataset.py \
    --source-dir datasets/CholecSeg8k \
    --output-frame-dir data/full_dataset/frames \
    --output-mask-dir data/full_dataset/masks
```

### Expected Directory Structure

```
datasets/
└── CholecSeg8k/
    ├── video01/
    │   ├── video01_00000/
    │   │   ├── frame_00000_endo.png
    │   │   ├── frame_00000_endo_color_mask.png
    │   │   └── frame_00000_endo_watershed_mask.png
    │   └── ...
    ├── video09/
    └── ... (17 videos total)

data/
└── full_dataset/
    ├── frames/
    │   ├── video01_frame_000000.png
    │   └── ...
    └── masks/
        ├── video01_mask_000000.png
        └── ...
```

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total frames | 8,080 |
| Videos | 17 |
| Resolution | 854×480 |
| Instrument classes | Grasper (31), L-hook (32) |
| Annotation type | Pixel-level watershed masks |

### Citation

```bibtex
@dataset{CholecSeg8k,
  author = {Hong, W.-Y. and Kao, C.-L. and Kuo, Y.-H. and Wang, J.-R. and Chang, W.-L. and Shih, C.-S.},
  title = {CholecSeg8k: A Semantic Segmentation Dataset for Laparoscopic Cholecystectomy},
  year = {2020},
  url = {https://www.kaggle.com/datasets/newslab/cholecseg8k}
}
```

### Troubleshooting

**Error: "403 - Forbidden"**
- Ensure you have accepted the dataset license on Kaggle.

**Error: "Could not find kaggle.json"**
- Verify the credentials file is in the correct location with proper permissions.

**Error: "No space left on device"**
- The dataset requires roughly 5 GB (2.3 GB compressed + extracted). Ensure sufficient disk space.
