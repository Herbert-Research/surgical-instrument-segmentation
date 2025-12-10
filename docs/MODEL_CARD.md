# Model Card: Surgical Instrument Segmentation

## Model Details

| Property | Value |
|----------|-------|
| **Model Name** | InstrumentSegmentationModel |
| **Architecture** | DeepLabV3+ with ResNet-50 backbone |
| **Version** | 0.1.0 |
| **Release Date** | December 2024 |
| **License** | MIT |
| **Author** | Maximilian Herbert Dressler |
| **Contact** | maximilian.dressler@stud.uni-heidelberg.de |
| **Institution** | Heidelberg University |

### Model Architecture

The model uses a DeepLabV3+ architecture with the following specifications:

- **Backbone**: ResNet-50 (pretrained on ImageNet)
- **Output Stride**: 16
- **ASPP Dilation Rates**: [6, 12, 18]
- **Decoder**: Low-level feature fusion with 4× upsampling
- **Output Classes**: 2 (background, instrument)
- **Input Size**: 256×256×3 (RGB)
- **Output Size**: 256×256 (class predictions)

### Training Configuration

```yaml
optimizer: Adam
learning_rate: 0.0001
weight_decay: 0.0001
batch_size: 4
epochs: 15
loss_function: CrossEntropyLoss
class_weights:
  background: 1.0
  instrument: 3.0  # Compensates for class imbalance
```

---

## Intended Use

### Primary Use Cases

- **Research**: Academic research in surgical computer vision
- **Education**: Teaching deep learning for medical imaging
- **Development**: Baseline model for surgical instrument detection systems
- **Prototyping**: Foundation for procedure-specific segmentation models

### Primary Users

- Medical AI researchers
- Surgical data scientists
- Computer vision engineers working in healthcare
- Graduate students in medical informatics

### Out-of-Scope Uses

> ⚠️ **This model is NOT intended for:**

- **Clinical diagnosis** or treatment decisions
- **Real-time intraoperative guidance** without additional validation
- **Medical device applications** without regulatory approval (FDA/CE)
- **Patient safety-critical systems** without extensive validation
- **Deployment in clinical workflows** without institutional review

---

## Training Data

### Dataset: CholecSeg8k

| Property | Value |
|----------|-------|
| **Source** | CholecSeg8k Dataset (publicly available) |
| **Total Frames** | 8,080 annotated frames |
| **Procedure Type** | Cholecystectomy (gallbladder removal) |
| **Videos** | 17 laparoscopic surgery videos |
| **Annotation Type** | Pixel-level semantic segmentation |
| **Original Classes** | 13 semantic classes |
| **Used Classes** | 2 (binary: background/instrument) |

### Class Mapping

The original CholecSeg8k dataset uses class IDs 31 and 32 for surgical instruments:

| Original Class ID | Original Label | Mapped Class |
|-------------------|----------------|--------------|
| 0-30, 33+ | Various anatomical structures | 0 (Background) |
| 31 | Grasper | 1 (Instrument) |
| 32 | L-hook Electrocautery | 1 (Instrument) |

> **Note**: Published CholecSeg8k documentation indicates class IDs 5 and 9 for instruments. Our empirical analysis discovered the actual mask values are 31 and 32. This discrepancy is documented in our methodology.

### Data Split

| Split | Frames | Purpose |
|-------|--------|---------|
| Training | 80% | Model optimization |
| Validation | 20% | Hyperparameter tuning |

### Validation Methodology

**Strategy**: Leave-videos-out 5-fold cross-validation

Each fold holds out 2-3 surgical videos (~950 frames) for validation while training on the remaining videos (~7,130 frames). This ensures no temporal leakage between train and validation sets.

**Rationale**: Video-level splits prevent the model from memorizing patient-specific or procedure-specific features, providing a more realistic estimate of generalization performance. Frame-level random splits would allow temporal neighbors to appear in both train and validation, artificially inflating metrics.

**Statistical Rigor**: All comparative results include:
- Mean ± standard deviation across folds
- 95% confidence intervals via bootstrap
- Paired t-tests for significance testing
- Cohen's d effect sizes for practical significance

### Data Preprocessing

1. **Resize**: 256×256 pixels
2. **Normalization**: ImageNet mean/std
3. **Augmentation** (training only):
   - Random horizontal flip
   - Color jitter (brightness, contrast, saturation)
   - Gaussian noise injection

---

## Evaluation Results

### Overall Performance (Single Split)

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 99.47% |
| **Mean IoU (Instrument)** | 87.1% |
| **Mean Dice (Instrument)** | 93.1% |
| **Precision (Instrument)** | 89.1% |
| **Recall (Instrument)** | 97.4% |

### Cross-Validation Performance (5-Fold)

| Architecture | IoU (mean ± std) | 95% CI | Dice (mean ± std) | 95% CI |
|--------------|------------------|--------|-------------------|--------|
| **U-Net** | 0.8999 ± 0.0234 | [0.8794, 0.9204] | 0.9473 ± 0.0156 | [0.9336, 0.9610] |
| **DeepLabV3-ResNet50** | 0.8587 ± 0.0312 | [0.8314, 0.8860] | 0.9240 ± 0.0198 | [0.9066, 0.9414] |

### Per-Class Metrics

| Class | IoU | Dice | Precision | Recall | Support |
|-------|-----|------|-----------|--------|---------|
| Background | 99.1% | 99.5% | 99.8% | 99.7% | ~98% pixels |
| Instrument | 87.1% | 93.1% | 89.1% | 97.4% | ~2% pixels |

### Statistical Comparison: U-Net vs DeepLabV3

| Metric | t-statistic | p-value | Significant | Cohen's d | Interpretation |
|--------|-------------|---------|-------------|-----------|----------------|
| **IoU** | 3.21 | 0.032 | Yes (p < 0.05) | 1.44 | Large effect |
| **Dice** | 2.89 | 0.044 | Yes (p < 0.05) | 1.29 | Large effect |

**Key Finding**: U-Net significantly outperforms DeepLabV3-ResNet50 on this dataset with a large effect size. Despite having fewer parameters (17.3M vs 42.0M), U-Net achieves higher IoU (+4.1%) and Dice (+2.3%) scores.

### Comparative Analysis Summary

| Model | IoU | Dice | Parameters | Training Time |
|-------|-----|------|------------|---------------|
| **U-Net (Best)** | 90.0% | 94.7% | 17.3M | 8.2 hrs |
| DeepLabV3-ResNet50 | 85.9% | 92.4% | 42.0M | 39.1 hrs |

### Cross-Video Generalization

| Evaluation Scenario | IoU | Notes |
|---------------------|-----|-------|
| Same-video validation | 87.1% | Train/val from same video pool |
| Cross-video (Video 01) | 85.2% | Held-out during training |
| Cross-video (Video 80) | 8.7% | Different surgical conditions |

> ⚠️ **Critical Finding**: Performance drops significantly (~10×) on video 80, which contains different lighting conditions and instrument appearances. This demonstrates the domain shift challenge in surgical AI.

---

## Limitations

### Technical Limitations

1. **Single Procedure Type**: Trained exclusively on cholecystectomy procedures
2. **Binary Classification**: Does not distinguish between instrument types
3. **Resolution Constraint**: Optimized for 256×256, may lose detail at higher resolutions
4. **Temporal Independence**: No temporal modeling across video frames

### Generalization Limitations

1. **Domain Shift**: Significant performance degradation on:
   - Different surgical procedures (gastrectomy, colorectal)
   - Different camera systems or lighting conditions
   - Different surgical teams or institutions

2. **Instrument Coverage**: Limited to:
   - Grasper forceps
   - L-hook electrocautery
   - Does not include: staplers, clip appliers, suction/irrigation

3. **Edge Cases**: Known challenges with:
   - Smoke/vapor occlusion
   - Specular reflections
   - Instrument overlap
   - Extreme motion blur

### Data Limitations

- Single dataset source (Cholec80/CholecSeg8k)
- French hospital system only
- 2014-2016 video capture equipment
- No external validation set

---

## Ethical Considerations

### Privacy and Data Protection

- Uses de-identified public research data only
- No patient identifiable information in model or training
- Dataset complies with original IRB approval from source institution

### Potential Harms

1. **Misuse Risk**: Model outputs could be misinterpreted as clinical recommendations
2. **Overconfidence**: High accuracy on limited domain may create false confidence
3. **Automation Bias**: Risk of over-reliance on model predictions in clinical research

### Mitigation Strategies

- Clear documentation that this is a research tool only
- Explicit warnings against clinical deployment
- Transparency about limitations and failure modes
- Open-source release for academic scrutiny

### Regulatory Status

| Jurisdiction | Status |
|--------------|--------|
| FDA (USA) | Not submitted; not a medical device |
| CE Mark (EU) | Not applicable; research use only |
| HIPAA | Compliant (no PHI in training data) |

---

## Usage

### Loading the Model

```python
from surgical_segmentation.models import InstrumentSegmentationModel
import torch

# Initialize model
model = InstrumentSegmentationModel(num_classes=2)

# Load trained weights
model.load_state_dict(torch.load("outputs/models/instrument_segmentation_model.pth"))
model.eval()
```

### Inference

```python
from PIL import Image
from torchvision import transforms

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load and preprocess image
image = Image.open("surgical_frame.png").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).squeeze().numpy()

# prediction is now a 256x256 array where:
# 0 = background
# 1 = instrument
```

### Command Line

```bash
# Train model
train-segmentation --frame-dir data/frames --mask-dir data/masks

# Evaluate model
evaluate-model --model-path outputs/models/instrument_segmentation_model.pth
```

---

## Citation

If you use this model in your research, please cite:

```bibtex
@software{dressler2024surgical,
  author = {Dressler, Maximilian Herbert},
  title = {Surgical Instrument Segmentation using Deep Learning},
  year = {2024},
  url = {https://github.com/Herbert-Research/surgical-instrument-segmentation},
  version = {0.1.0}
}
```

### Related Publications

- CholecSeg8k Dataset: Hong et al., "CholecSeg8k: A Semantic Segmentation Dataset for Laparoscopic Cholecystectomy Based on Cholec80"
- DeepLabV3+: Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"

---

## Model Card Authors

- **Maximilian Herbert Dressler** - Primary author and developer

## Model Card Contact

For questions or feedback about this model card:
- Email: maximilian.dressler@stud.uni-heidelberg.de
- GitHub Issues: https://github.com/Herbert-Research/surgical-instrument-segmentation/issues

---

*Last Updated: December 2024*
