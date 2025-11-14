## Model Comparison Results

| Architecture | IoU (Instrument) | Dice (Instrument) | Accuracy | Parameters | Training Time |
|-------------|-----------------|-------------------|----------|------------|---------------|
| UNET        | 0.8999 | 0.9473 | 0.9960 | 17.3M | 490.8 min |
| DEEPLABV3   | 0.8587 | 0.9240 | 0.9941 | 42.0M | 2345.5 min |

**Analysis**: U-Net achieves 0.0412 higher IoU on instrument segmentation. DeepLabV3 has 2.4× more parameters than U-Net, providing stronger feature representations at the cost of increased computational requirements.
