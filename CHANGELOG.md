# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CI status badges in README (CI, codecov, Python version, License)
- CHANGELOG.md for version history tracking
- Reproducibility verification script (`scripts/verify_reproducibility.py`)

### Changed
- (Planned) Increased test coverage to 80%
- (Planned) Statistical significance testing for model comparisons

### Fixed
- (Planned) Remove suppressed flake8 warnings (F401, F841, F541)

---

## [0.1.0] - 2024-12-01

### Added

#### Core Features
- DeepLabV3-ResNet50 segmentation model with ImageNet pretrained backbone
- U-Net baseline implementation for comparative analysis
- Binary surgical instrument segmentation (background vs. instrument)
- Transfer learning approach achieving competitive benchmark performance

#### Dataset Support
- CholecSeg8k dataset preparation scripts
- Automatic video-to-frame extraction pipeline
- Binary mask generation from multi-class annotations
- Support for custom dataset formats

#### Training Pipeline
- Configurable training with YAML configuration files
- Mixed precision training support (AMP)
- Learning rate scheduling (StepLR, ReduceLROnPlateau)
- Early stopping with configurable patience
- Checkpoint saving with best model tracking
- Reproducible training with seed control

#### Evaluation & Metrics
- Intersection over Union (IoU / Jaccard Index)
- Dice Coefficient (F1 Score)
- Pixel-wise Precision and Recall
- Per-class and mean metrics computation
- Confusion matrix generation

#### Infrastructure
- Docker and docker-compose support for containerized execution
- GitHub Actions CI/CD pipeline with multi-Python version testing
- Pre-commit hooks (Black, isort, mypy, flake8)
- Codecov integration for coverage tracking
- Comprehensive test suite (167 tests)

#### Documentation
- Detailed README with installation and usage instructions
- Model Card following responsible AI guidelines
- Ethics statement (IRB, HIPAA, regulatory considerations)
- Methods documentation with scientific methodology
- Dataset setup guide
- Contributing guidelines with code standards

### Technical Details

#### Model Performance (15 epochs, batch size 4)

| Model | IoU | Dice | Precision | Recall |
|-------|-----|------|-----------|--------|
| DeepLabV3-ResNet50 | 87.1% | 93.1% | 89.1% | 97.4% |
| U-Net | 78.3% | 87.6% | 82.1% | 94.2% |

#### Training Configuration
- Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
- Loss: Cross-Entropy with class weighting
- Input resolution: 256×256 RGB
- Augmentation: Random horizontal/vertical flip, rotation, color jitter

#### Dependencies
- Python 3.9-3.11
- PyTorch 2.0+
- torchvision 0.15+
- OpenCV, Pillow, NumPy, scikit-learn

### Known Limitations
- Trained only on cholecystectomy procedures (CholecSeg8k)
- Binary segmentation only (no multi-instrument differentiation)
- Not validated for real-time clinical deployment
- Domain shift observed when generalizing across videos (~10× performance drop)

### Security
- No patient identifiable information in repository
- All training data from de-identified public dataset
- Model weights do not encode patient-specific features

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2024-12-01 | Initial release with DeepLabV3 and U-Net models |

---

## Upgrade Notes

### Migrating to future versions

When upgrading to future versions, please note:

1. **Configuration changes**: Check `config/default.yaml` for new parameters
2. **API changes**: Review the CHANGELOG for breaking changes marked with ⚠️
3. **Model compatibility**: Pre-trained weights may not be compatible across major versions

---

## Contributors

- Maximilian Herbert Dressler - Initial development and documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
