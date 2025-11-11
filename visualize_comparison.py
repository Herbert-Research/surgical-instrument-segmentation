"""
Generate comprehensive comparison visualizations
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_results(output_dir: Path):
    """Load all training histories."""

    results = {}
    for arch in ["unet", "deeplabv3"]:
        history_file = output_dir / f"{arch}_history.json"
        if history_file.exists():
            with open(history_file, "r", encoding="utf-8") as f:
                results[arch] = json.load(f)
    return results


def plot_training_curves(results: dict, output_path: Path) -> None:
    """Plot training loss curves."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for arch, data in results.items():
        losses = data["train_losses"]
        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, losses, "o-", label=arch.upper(), linewidth=2, markersize=6)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("Training Loss Comparison", fontweight="bold", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for arch, data in results.items():
        val_metrics = data["val_metrics"]
        epochs = [m["epoch"] for m in val_metrics]
        iou_scores = [m["iou_instrument"] for m in val_metrics]
        ax.plot(epochs, iou_scores, "o-", label=f"{arch.upper()} IoU", linewidth=2, markersize=6)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("IoU (Instrument Class)", fontsize=12)
    ax.set_title("Validation IoU Comparison", fontweight="bold", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved training curves: {output_path}")
    plt.close()


def plot_final_comparison(results: dict, output_path: Path) -> None:
    """Plot final metrics comparison."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    architectures = list(results.keys())
    arch_labels = [a.upper() for a in architectures]

    iou_scores = [results[a]["final_metrics"]["iou_instrument"] for a in architectures]
    dice_scores = [results[a]["final_metrics"]["dice_instrument"] for a in architectures]
    accuracies = [results[a]["final_metrics"]["accuracy"] for a in architectures]
    params = [results[a]["parameters"] / 1e6 for a in architectures]

    colors = ["#3498db", "#e74c3c"][: len(architectures)]

    ax = axes[0, 0]
    bars = ax.bar(arch_labels, iou_scores, color=colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("IoU Score", fontsize=12)
    ax.set_title("Intersection over Union (Instrument Class)", fontweight="bold", fontsize=13)
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)
    for bar, score in zip(bars, iou_scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax = axes[0, 1]
    bars = ax.bar(arch_labels, dice_scores, color=colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Dice Coefficient", fontsize=12)
    ax.set_title("Dice Coefficient (Instrument Class)", fontweight="bold", fontsize=13)
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)
    for bar, score in zip(bars, dice_scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax = axes[1, 0]
    bars = ax.bar(arch_labels, accuracies, color=colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Overall Pixel Accuracy", fontweight="bold", fontsize=13)
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)
    for bar, score in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax = axes[1, 1]
    bars = ax.bar(arch_labels, params, color=colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("Parameters (Millions)", fontsize=12)
    ax.set_title("Model Complexity", fontweight="bold", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{param:.1f}M",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved final comparison: {output_path}")
    plt.close()


def generate_comparison_table(results: dict, output_path: Path):
    """Generate markdown table for README."""

    table = "## Model Comparison Results\n\n"
    table += (
        "| Architecture | IoU (Instrument) | Dice (Instrument) | Accuracy | Parameters | Training Time |\n"
    )
    table += "|-------------|-----------------|-------------------|----------|------------|---------------|\n"

    for arch, data in results.items():
        metrics = data["final_metrics"]
        params_m = data["parameters"] / 1e6
        time_min = data["training_time_seconds"] / 60

        table += f"| {arch.upper():<11} | "
        table += f"{metrics['iou_instrument']:.4f} | "
        table += f"{metrics['dice_instrument']:.4f} | "
        table += f"{metrics['accuracy']:.4f} | "
        table += f"{params_m:.1f}M | "
        table += f"{time_min:.1f} min |\n"

    iou_diff = abs(
        results["deeplabv3"]["final_metrics"]["iou_instrument"]
        - results["unet"]["final_metrics"]["iou_instrument"]
    )
    better = (
        "DeepLabV3"
        if results["deeplabv3"]["final_metrics"]["iou_instrument"]
        > results["unet"]["final_metrics"]["iou_instrument"]
        else "U-Net"
    )

    table += (
        f"\n**Analysis**: {better} achieves {iou_diff:.4f} higher IoU on instrument segmentation. "
    )

    param_ratio = results["deeplabv3"]["parameters"] / results["unet"]["parameters"]
    table += f"DeepLabV3 has {param_ratio:.1f}× more parameters than U-Net, "
    table += (
        "providing stronger feature representations at the cost of increased computational requirements.\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(table)

    print(f"✓ Saved comparison table: {output_path}")
    return table


def main() -> None:
    output_dir = Path("outputs/comparative")

    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        print("Run train_comparative.py first")
        return

    sns.set_theme(style="whitegrid")

    print("=" * 70)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("=" * 70)

    results = load_results(output_dir)

    if len(results) < 2:
        print(f"❌ Need at least 2 models, found {len(results)}")
        return

    print(f"Loaded results for: {', '.join(results.keys())}")

    plot_training_curves(results, output_dir / "training_comparison.png")
    plot_final_comparison(results, output_dir / "final_comparison.png")
    table = generate_comparison_table(results, output_dir / "comparison_table.md")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(table)


if __name__ == "__main__":
    main()
