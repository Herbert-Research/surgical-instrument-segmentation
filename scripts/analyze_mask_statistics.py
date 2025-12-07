"""
Simple statistics summary for generated masks.

Provides quick metrics about the generated masks:
- How many frames have instruments
- Distribution of instrument coverage
- Overall quality indicators
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def analyze_mask_statistics(generated_dir: Path) -> None:
    """Analyze statistics of generated masks."""

    mask_files = sorted([f for f in generated_dir.glob("*_mask.png")])

    if not mask_files:
        print("❌ No mask files found!")
        return

    print(f"\n{'='*70}")
    print(f"MASK STATISTICS SUMMARY")
    print(f"{'='*70}")
    print(f"Directory: {generated_dir}")
    print(f"Total masks: {len(mask_files)}")

    # Analyze each mask
    instrument_coverages = []
    frames_with_instruments = 0

    for mask_path in mask_files:
        mask = np.array(Image.open(mask_path).convert("L"))
        total_pixels = mask.size
        instrument_pixels = (mask > 127).sum()
        coverage = (instrument_pixels / total_pixels) * 100

        instrument_coverages.append(coverage)
        if instrument_pixels > 0:
            frames_with_instruments += 1

    # Statistics
    print(f"\nInstrument Detection:")
    print(
        f"  Frames with instruments: {frames_with_instruments}/{len(mask_files)} ({frames_with_instruments/len(mask_files)*100:.1f}%)"
    )
    print(f"  Frames without instruments: {len(mask_files) - frames_with_instruments}")

    if instrument_coverages:
        print(f"\nInstrument Coverage (% of pixels):")
        print(f"  Mean:   {np.mean(instrument_coverages):.2f}%")
        print(f"  Median: {np.median(instrument_coverages):.2f}%")
        print(f"  Min:    {np.min(instrument_coverages):.2f}%")
        print(f"  Max:    {np.max(instrument_coverages):.2f}%")
        print(f"  Std:    {np.std(instrument_coverages):.2f}%")

    # Create histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Coverage distribution
    ax1.hist(instrument_coverages, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
    ax1.axvline(
        np.mean(instrument_coverages),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(instrument_coverages):.2f}%",
    )
    ax1.set_xlabel("Instrument Coverage (%)", fontsize=12)
    ax1.set_ylabel("Number of Frames", fontsize=12)
    ax1.set_title("Distribution of Instrument Coverage", fontweight="bold", fontsize=14)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Frame-by-frame plot
    ax2.plot(
        range(len(instrument_coverages)),
        instrument_coverages,
        "o-",
        linewidth=2,
        markersize=6,
        color="steelblue",
    )
    ax2.axhline(
        np.mean(instrument_coverages),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(instrument_coverages):.2f}%",
    )
    ax2.set_xlabel("Frame Number", fontsize=12)
    ax2.set_ylabel("Instrument Coverage (%)", fontsize=12)
    ax2.set_title("Instrument Coverage Over Time", fontweight="bold", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = generated_dir.parent / f"{generated_dir.name}_statistics.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved statistics plot to: {output_path}")
    print(f"{'='*70}\n")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze mask statistics")
    parser.add_argument(
        "--generated-dir", type=Path, required=True, help="Directory with generated masks"
    )
    args = parser.parse_args()

    analyze_mask_statistics(args.generated_dir)


if __name__ == "__main__":
    main()
