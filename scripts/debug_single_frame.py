"""Test model predictions on a single frame with different thresholds."""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50

MODEL_PATH = Path("outputs/models/instrument_segmentation_model.pth")
FIGURES_DIR = Path("outputs/figures")

# Load model
print("Loading model...")
model = deeplabv3_resnet50(num_classes=2, weights=None, aux_loss=False)
checkpoint = torch.load(MODEL_PATH, map_location="cuda", weights_only=False)

# Remove 'model.' prefix and filter aux_classifier
new_state_dict = {}
for key, value in checkpoint.items():
    if key.startswith("model."):
        new_key = key[6:]
    else:
        new_key = key
    if "aux_classifier" in new_key:
        continue
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict, strict=False)
model = model.to("cuda")
model.eval()
print("Model loaded!")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load a sample frame
frame_path = "datasets/Cholec80/video01_generated_frames/video01_frame_000200.png"
frame = cv2.imread(frame_path)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Prepare for model
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

pil_image = Image.fromarray(frame_rgb)
input_tensor = transform(pil_image)
input_batch = input_tensor.unsqueeze(0).to("cuda")

# Run inference
print("\nRunning inference...")
with torch.no_grad():
    output = model(input_batch)["out"]
    probabilities = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()

    # Get instrument class probability (class 1)
    instrument_prob = probabilities[1]
    background_prob = probabilities[0]

    print(
        f"Instrument probability range: {instrument_prob.min():.4f} to {instrument_prob.max():.4f}"
    )
    print(
        f"Background probability range: {background_prob.min():.4f} to {background_prob.max():.4f}"
    )
    print(f"Mean instrument probability: {instrument_prob.mean():.4f}")

    # Test different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("\nPixels detected at different thresholds:")
    for thresh in thresholds:
        mask = (instrument_prob >= thresh).astype(np.uint8)
        num_pixels = np.sum(mask)
        percentage = (num_pixels / mask.size) * 100
        print(f"  Threshold {thresh:.1f}: {num_pixels:6d} pixels ({percentage:5.2f}%)")

    # Save visualization with threshold 0.5
    mask_05 = ((instrument_prob >= 0.5) * 255).astype(np.uint8)
    mask_05_path = FIGURES_DIR / "test_mask_thresh05.png"
    cv2.imwrite(str(mask_05_path), mask_05)

    # Save visualization with threshold 0.3
    mask_03 = ((instrument_prob >= 0.3) * 255).astype(np.uint8)
    mask_03_path = FIGURES_DIR / "test_mask_thresh03.png"
    cv2.imwrite(str(mask_03_path), mask_03)

    print(f"\nSaved {mask_05_path} and {mask_03_path} for visualization")
