#!/usr/bin/env python
"""
Verify training reproducibility by running training twice with the same seed.

This script trains a small model twice and asserts that results are identical,
ensuring the seed_everything() function works correctly and that training
is deterministic when using the same random seed.

Usage:
    python scripts/verify_reproducibility.py
    python scripts/verify_reproducibility.py --seed 123 --epochs 3

Exit Codes:
    0: Reproducibility verified successfully
    1: Reproducibility check failed

Author: Maximilian Herbert Dressler
Purpose: Ensure scientific reproducibility in medical AI research
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Import torch and related modules
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
except ImportError as e:
    print(f"Error: PyTorch not installed. {e}")
    sys.exit(1)

# Import project modules
try:
    from surgical_segmentation.datasets import SurgicalDataset
    from surgical_segmentation.models.deeplabv3 import InstrumentSegmentationModel
    from surgical_segmentation.training.trainer import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        AdditiveGaussianNoise,
        create_synthetic_surgical_frames,
        seed_everything,
    )
except ImportError as e:
    print(f"Error: Could not import project modules. {e}")
    print("Make sure the package is installed: pip install -e .")
    sys.exit(1)


def get_train_transform():
    """Get the standard training transform."""
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            AdditiveGaussianNoise(std=0.02),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def compute_model_checksum(model: nn.Module) -> float:
    """
    Compute a deterministic checksum of model parameters.

    Args:
        model: PyTorch model to checksum

    Returns:
        Sum of all parameter values (deterministic for same weights)
    """
    return sum(p.sum().item() for p in model.parameters())


def train_deterministic(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[nn.Module, list[float]]:
    """
    Deterministic training loop for reproducibility testing.

    This is a simplified training loop that ensures deterministic behavior
    by using the specified device and avoiding non-deterministic operations.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on

    Returns:
        Tuple of (trained model, list of epoch losses)
    """
    model = model.to(device)

    # Use class weights to handle imbalance
    class_weights = torch.tensor([1.0, 3.0], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for frames, masks in train_loader:
            frames = frames.to(device)
            masks = masks.to(device)

            outputs = model(frames)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.6f}")

    return model, losses


def run_single_training(
    frame_dir: Path,
    mask_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    """
    Run a single training session with the specified seed.

    Args:
        frame_dir: Directory containing training frames
        mask_dir: Directory containing training masks
        seed: Random seed for reproducibility
        epochs: Number of training epochs
        batch_size: Training batch size
        device: Device to train on ('cpu' or 'cuda')

    Returns:
        Dictionary containing training results:
        - final_loss: Loss value from final epoch
        - losses: List of all epoch losses
        - model_checksum: Sum of model parameters
    """
    # Set seed before creating model and dataloader
    seed_everything(seed)

    # Set device
    device_obj = torch.device(device)

    # Create model
    model = InstrumentSegmentationModel(num_classes=2)

    # Create dataset and dataloader
    transform = get_train_transform()
    dataset = SurgicalDataset(
        frame_dir=str(frame_dir),
        mask_dir=str(mask_dir),
        transform=transform,
    )

    # Use a generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single worker for determinism
        generator=generator,
    )

    # Train model using deterministic training loop
    trained_model, losses = train_deterministic(
        model=model,
        train_loader=train_loader,
        num_epochs=epochs,
        learning_rate=1e-4,
        device=device_obj,
    )

    # Compute results
    final_loss = losses[-1] if losses else 0.0
    model_checksum = compute_model_checksum(trained_model)

    return {
        "final_loss": final_loss,
        "losses": losses,
        "model_checksum": model_checksum,
    }


def verify_reproducibility(
    seed: int = 42,
    epochs: int = 2,
    batch_size: int = 2,
    num_frames: int = 10,
    device: str = "cpu",
    tolerance: float = 1e-5,
) -> bool:
    """
    Verify training reproducibility by running training twice.

    Args:
        seed: Random seed for both training runs
        epochs: Number of training epochs per run
        batch_size: Training batch size
        num_frames: Number of synthetic frames to generate
        device: Device to train on ('cpu' recommended for determinism)
        tolerance: Relative tolerance for floating point comparison

    Returns:
        True if results are identical within tolerance, False otherwise
    """
    print("=" * 60)
    print("REPRODUCIBILITY VERIFICATION TEST")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Seed: {seed}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Synthetic frames: {num_frames}")
    print(f"  - Device: {device}")
    print(f"  - Tolerance: {tolerance}")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        frame_dir = tmpdir / "frames"
        mask_dir = tmpdir / "masks"

        # Generate synthetic data (deterministic with seed)
        print("\n[Step 1/4] Generating synthetic training data...")
        seed_everything(seed)
        create_synthetic_surgical_frames(frame_dir, mask_dir, force=True)
        print(f"  Generated {len(list(frame_dir.glob('*.png')))} frames")

        results = []

        for run_idx in range(2):
            print(f"\n[Step {run_idx + 2}/4] Training Run {run_idx + 1}/2...")
            print("-" * 40)

            result = run_single_training(
                frame_dir=frame_dir,
                mask_dir=mask_dir,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
            )

            results.append(result)

            print(f"  Final loss: {result['final_loss']:.8f}")
            print(f"  Model checksum: {result['model_checksum']:.8f}")

        # Compare results
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)

        # Compare losses
        loss_match = np.isclose(
            results[0]["final_loss"],
            results[1]["final_loss"],
            rtol=tolerance,
        )

        # Compare model checksums
        checksum_match = np.isclose(
            results[0]["model_checksum"],
            results[1]["model_checksum"],
            rtol=tolerance,
        )

        # Compare all epoch losses
        all_losses_match = True
        if len(results[0]["losses"]) == len(results[1]["losses"]):
            for i, (l1, l2) in enumerate(zip(results[0]["losses"], results[1]["losses"])):
                if not np.isclose(l1, l2, rtol=tolerance):
                    all_losses_match = False
                    print(f"  ⚠ Epoch {i+1} loss mismatch: {l1:.8f} vs {l2:.8f}")
        else:
            all_losses_match = False

        # Print comparison table
        print("\n┌─────────────────────┬────────────────────┬────────────────────┬────────┐")
        print("│ Metric              │ Run 1              │ Run 2              │ Status │")
        print("├─────────────────────┼────────────────────┼────────────────────┼────────┤")
        print(
            f"│ Final Loss          │ {results[0]['final_loss']:>18.8f} │ {results[1]['final_loss']:>18.8f} │ {'✓ PASS' if loss_match else '✗ FAIL'} │"
        )
        print(
            f"│ Model Checksum      │ {results[0]['model_checksum']:>18.8f} │ {results[1]['model_checksum']:>18.8f} │ {'✓ PASS' if checksum_match else '✗ FAIL'} │"
        )
        print(
            f"│ All Epoch Losses    │ {'(matched)':>18} │ {'(matched)':>18} │ {'✓ PASS' if all_losses_match else '✗ FAIL'} │"
        )
        print("└─────────────────────┴────────────────────┴────────────────────┴────────┘")

        # Final verdict
        all_passed = loss_match and checksum_match and all_losses_match

        print("\n" + "=" * 60)
        if all_passed:
            print("✓ REPRODUCIBILITY VERIFIED SUCCESSFULLY")
            print("=" * 60)
            print("\nThe training pipeline produces identical results when")
            print("using the same random seed. This confirms:")
            print("  1. seed_everything() correctly initializes all RNGs")
            print("  2. DataLoader shuffling is deterministic")
            print("  3. Model initialization is deterministic")
            print("  4. Training loop produces consistent gradients")
        else:
            print("✗ REPRODUCIBILITY CHECK FAILED")
            print("=" * 60)
            print("\nThe training pipeline produced different results between runs.")
            print("Potential causes:")
            print("  1. Non-deterministic operations in the model")
            print("  2. CUDA operations without deterministic mode")
            print("  3. Data loading with multiple workers")
            print("  4. Floating point accumulation order differences")

            if device != "cpu":
                print("\nTry running with --device cpu for more deterministic behavior.")

        return all_passed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify training reproducibility with identical seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic reproducibility check (recommended settings)
    python scripts/verify_reproducibility.py

    # Custom seed and epochs
    python scripts/verify_reproducibility.py --seed 123 --epochs 3

    # Test on GPU (may be less deterministic)
    python scripts/verify_reproducibility.py --device cuda
        """,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility test (default: 42)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs per run (default: 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size (default: 2)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="Number of synthetic frames to generate (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to train on (default: cpu, recommended for determinism)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Relative tolerance for floating point comparison (default: 1e-5)",
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for reproducibility verification.

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    args = parse_args()

    # Warn if using CUDA
    if args.device == "cuda":
        print("⚠ Warning: CUDA may introduce non-determinism even with seeding.")
        print("  For strictest reproducibility testing, use --device cpu")
        print()

    success = verify_reproducibility(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        device=args.device,
        tolerance=args.tolerance,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
