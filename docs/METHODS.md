# Methodology Documentation

## Loss Function

We use weighted cross-entropy loss to handle class imbalance:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} w_c \cdot y_{i,c} \cdot \log(\hat{y}_{i,c})$$

Where:
- $N$ = number of pixels
- $C$ = number of classes (2 for binary segmentation)
- $w_c$ = class weight (background: 1.0, instrument: 3.0)
- $y_{i,c}$ = ground truth one-hot encoding
- $\hat{y}_{i,c}$ = predicted probability

## Evaluation Metrics

### Intersection over Union (IoU / Jaccard Index)

$$\text{IoU} = \frac{|P \cap G|}{|P \cup G|} = \frac{TP}{TP + FP + FN}$$

### Dice Coefficient (F1-Score)

$$\text{Dice} = \frac{2|P \cap G|}{|P| + |G|} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

### Relationship

$$\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$$

## Architectures

### DeepLabV3-ResNet50

- **Backbone**: ResNet-50 pretrained on ImageNet
- **ASPP**: Atrous Spatial Pyramid Pooling with rates [6, 12, 18]
- **Parameters**: ~42M

### U-Net

- **Encoder**: 4 downsampling blocks (64->128->256->512->1024)
- **Decoder**: 4 upsampling blocks with skip connections
- **Parameters**: ~17M (bilinear) / ~31M (transposed conv)

## Data Augmentation

| Augmentation | Probability | Parameters |
|--------------|-------------|------------|
| Horizontal Flip | 0.5 | - |
| Rotation | 0.2 | +/-12 deg |
| Brightness | 0.2 | [0.85, 1.15] |
| Contrast | 0.2 | [0.8, 1.2] |
| Color Jitter | 1.0 | B=0.2, C=0.2, S=0.1, H=0.02 |
| Gaussian Noise | 1.0 | sigma=0.02 |
