"""
Diagnose what's in the saved prediction files.
Check if predictions are being saved correctly.
"""

from pathlib import Path
import numpy as np
from PIL import Image

pred_dir = Path(r"C:\Users\m4rti\Documents\GitHub\surgical-instrument-segmentation\datasets\Cholec80\preds")
mask_dir = Path(r"C:\Users\m4rti\Documents\GitHub\surgical-instrument-segmentation\datasets\Cholec80\masks")

print("\n" + "="*70)
print("DIAGNOSING SAVED PREDICTIONS")
print("="*70 + "\n")

# Check predictions exist
pred_files = sorted(list(pred_dir.glob("*.png")))
print(f"Found {len(pred_files)} prediction files")

if not pred_files:
    print("\n❌ NO PREDICTION FILES FOUND!")
    print(f"   Expected location: {pred_dir}")
    exit()

# Check first 5 predictions
print(f"\nChecking first 5 predictions:\n")

for i, pred_file in enumerate(pred_files[:5]):
    # Load prediction
    pred = np.array(Image.open(pred_file).convert('L'))
    
    # Find corresponding mask
    mask_name = pred_file.name
    mask_file = mask_dir / mask_name
    
    if mask_file.exists():
        mask = np.array(Image.open(mask_file).convert('L'))
        
        # Analyze
        pred_values = np.unique(pred)
        mask_values = np.unique(mask)
        
        print(f"{i+1}. {pred_file.name}")
        print(f"   Prediction values: {pred_values.tolist()}")
        print(f"   Ground truth values: {mask_values.tolist()}")
        print(f"   Pred shape: {pred.shape}")
        print(f"   Mask shape: {mask.shape}")
        
        # Check if prediction has instruments
        instrument_pixels_pred = np.sum(pred > 0)
        instrument_pixels_mask = np.sum(mask > 0)
        
        print(f"   Instrument pixels in pred: {instrument_pixels_pred} ({100*instrument_pixels_pred/pred.size:.1f}%)")
        print(f"   Instrument pixels in mask: {instrument_pixels_mask} ({100*instrument_pixels_mask/mask.size:.1f}%)")
        
        if instrument_pixels_pred == 0 and instrument_pixels_mask > 0:
            print(f"   ❌ PROBLEM: Prediction has NO instruments but mask has {instrument_pixels_mask}!")
        elif instrument_pixels_pred > 0:
            print(f"   ✓ Prediction has instruments")
        print()

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70 + "\n")

# Check all predictions
all_pred_values = set()
has_instruments = 0
no_instruments = 0

for pred_file in pred_files:
    pred = np.array(Image.open(pred_file).convert('L'))
    all_pred_values.update(np.unique(pred))
    
    if np.sum(pred > 0) > 0:
        has_instruments += 1
    else:
        no_instruments += 1

print(f"All unique values across {len(pred_files)} predictions: {sorted(all_pred_values)}")
print(f"Predictions with instruments: {has_instruments}")
print(f"Predictions with NO instruments: {no_instruments}")

if no_instruments == len(pred_files):
    print("\n❌ CRITICAL: ALL predictions are empty (no instruments)!")
    print("   This means the prediction export code has a bug.")
    print("\n   The model trained correctly (65.6% IoU during training)")
    print("   But predictions are not being saved properly.")
elif sorted(all_pred_values) == [0]:
    print("\n❌ CRITICAL: All predictions only have value 0 (background)!")
    print("   The model output is not being saved correctly.")
elif len(all_pred_values) == 3:
    print("\n⚠️  WARNING: Predictions have 3 classes instead of 2!")
    print(f"   Values: {sorted(all_pred_values)}")
    print("   This suggests NUM_CLASSES mismatch during prediction save.")
else:
    print("\n✓ Predictions look reasonable")

print("\n" + "="*70)
print("RECOMMENDED FIX")
print("="*70 + "\n")

if no_instruments == len(pred_files) or sorted(all_pred_values) == [0]:
    print("""
The issue is in the evaluate_model function in instrument_segmentation.py.

The predictions during training show 65.6% IoU (working correctly),
but when saved to disk, they become all zeros.

CAUSE: The argmax operation or file saving is incorrect.

FIX: In evaluate_model(), change the prediction saving code:

OLD (probably has bug):
    pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    Image.fromarray(pred_mask).save(prediction_dir / pred_name)

NEW (correct):
    pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    # Verify it has instruments before saving
    print(f"Saving {pred_name}: unique values = {np.unique(pred_mask)}")
    Image.fromarray(pred_mask).save(prediction_dir / pred_name)

Then re-run ONLY the evaluation (not full training):
    python instrument_segmentation.py --skip-training --evaluate-only
    
Or delete the preds folder and re-run full training.
""")
elif len(all_pred_values) == 3:
    print("""
Predictions have 3 classes but you set NUM_CLASSES=2.

This means the saved model still has 3 output classes.

FIX:
1. Delete the old model file: instrument_segmentation_model.pth
2. Delete predictions: datasets/Cholec80/preds/*.png
3. Re-train with NUM_CLASSES=2

The model architecture must match NUM_CLASSES.
""")
