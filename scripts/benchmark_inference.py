#!/usr/bin/env python
"""
Benchmark Inference Speed for Surgical Instrument Segmentation.

This script measures inference latency and throughput for real-time feasibility
analysis of the surgical instrument segmentation model. Critical for evaluating
whether the model can meet real-time requirements for intraoperative guidance.

Usage:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --model-path outputs/models/my_model.pth
    python scripts/benchmark_inference.py --input-sizes 256 512 --batch-sizes 1 2 4

Outputs:
    - Console report with latency statistics
    - Optional JSON file with detailed benchmark results

Author: Maximilian Herbert Dressler
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class BenchmarkResult:
    """Container for inference benchmark results."""

    device: str
    input_size: int
    batch_size: int
    num_iterations: int
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_fps: float
    throughput_images_per_sec: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class SystemInfo:
    """System information for reproducibility."""

    python_version: str
    pytorch_version: str
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    gpu_memory_gb: float | None
    timestamp: str


@dataclass
class ProvenanceInfo:
    """Provenance metadata for benchmark results."""

    git_sha: str | None
    model_checksum: str | None
    dataset_manifest_hash: str | None
    config_hash: str | None

    def to_dict(self) -> dict[str, str | None]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def get_system_info() -> SystemInfo:
    """Collect system information for benchmark reproducibility."""
    import torch

    cuda_version = None
    gpu_name = None
    gpu_memory_gb = None

    if torch.cuda.is_available():
        torch_version = getattr(torch, "version", None)
        cuda_version = getattr(torch_version, "cuda", None) if torch_version else None
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    return SystemInfo(
        python_version=sys.version.split()[0],
        pytorch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        gpu_memory_gb=round(gpu_memory_gb, 2) if gpu_memory_gb else None,
        timestamp=datetime.now().isoformat(),
    )


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash for a file.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Hex-encoded SHA256 hash string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_sha() -> str | None:
    """Get current git SHA if available.

    Returns:
        Git SHA string or None if not available.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def collect_provenance(
    model_path: Path | None,
    dataset_manifest_path: Path | None,
    config_path: Path | None,
) -> ProvenanceInfo:
    """Collect provenance metadata for benchmark output.

    Args:
        model_path: Path to model weights if provided.
        dataset_manifest_path: Path to dataset manifest if provided.
        config_path: Path to config file if provided.

    Returns:
        ProvenanceInfo with available hashes and git SHA.
    """
    model_checksum = compute_sha256(model_path) if model_path and model_path.exists() else None
    dataset_manifest_hash = (
        compute_sha256(dataset_manifest_path)
        if dataset_manifest_path and dataset_manifest_path.exists()
        else None
    )
    config_hash = compute_sha256(config_path) if config_path and config_path.exists() else None

    return ProvenanceInfo(
        git_sha=get_git_sha(),
        model_checksum=model_checksum,
        dataset_manifest_hash=dataset_manifest_hash,
        config_hash=config_hash,
    )


def benchmark_inference(
    model,
    input_size: int = 256,
    batch_size: int = 1,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = "cuda",
) -> BenchmarkResult:
    """
    Measure inference latency statistics for a segmentation model.

    Performs controlled benchmarking with warmup iterations to ensure
    stable measurements. Uses CUDA synchronization for accurate GPU timing.

    Args:
        model: PyTorch segmentation model to benchmark.
        input_size: Square input image dimension (e.g., 256 for 256x256).
        batch_size: Number of images per inference batch.
        num_iterations: Number of timed inference iterations.
        warmup_iterations: Number of warmup iterations before timing.
        device: Compute device ("cuda" or "cpu").

    Returns:
        BenchmarkResult containing comprehensive latency statistics.

    Example:
        >>> model = InstrumentSegmentationModel(num_classes=2)
        >>> result = benchmark_inference(model, input_size=256, device="cuda")
        >>> print(f"Mean latency: {result.mean_latency_ms:.2f} ms")
        >>> print(f"Throughput: {result.throughput_fps:.1f} FPS")
    """
    import torch

    model = model.to(device).eval()
    dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)

    # Warmup phase - allows JIT compilation and memory allocation
    print(f"  Warming up ({warmup_iterations} iterations)...", end=" ", flush=True)
    for _ in range(warmup_iterations):
        with torch.no_grad():
            _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()
    print("done")

    # Timed phase
    print(f"  Benchmarking ({num_iterations} iterations)...", end=" ", flush=True)
    latencies = []

    for _ in range(num_iterations):
        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            _ = model(dummy_input)

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms

    print("done")

    latencies = np.array(latencies)
    mean_latency = float(np.mean(latencies))

    return BenchmarkResult(
        device=device,
        input_size=input_size,
        batch_size=batch_size,
        num_iterations=num_iterations,
        mean_latency_ms=float(np.mean(latencies)),
        std_latency_ms=float(np.std(latencies)),
        min_latency_ms=float(np.min(latencies)),
        max_latency_ms=float(np.max(latencies)),
        median_latency_ms=float(np.median(latencies)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        p99_latency_ms=float(np.percentile(latencies, 99)),
        throughput_fps=float(1000.0 / mean_latency),
        throughput_images_per_sec=float((1000.0 / mean_latency) * batch_size),
    )


def print_benchmark_report(
    results: list[BenchmarkResult],
    system_info: SystemInfo,
) -> None:
    """Print formatted benchmark report to console."""

    print("\n" + "=" * 80)
    print("SURGICAL INSTRUMENT SEGMENTATION - INFERENCE BENCHMARK REPORT")
    print("=" * 80)

    print("\n📊 SYSTEM INFORMATION")
    print("-" * 40)
    print(f"  Python Version:    {system_info.python_version}")
    print(f"  PyTorch Version:   {system_info.pytorch_version}")
    print(f"  CUDA Available:    {system_info.cuda_available}")
    if system_info.cuda_available:
        print(f"  CUDA Version:      {system_info.cuda_version}")
        print(f"  GPU:               {system_info.gpu_name}")
        print(f"  GPU Memory:        {system_info.gpu_memory_gb:.1f} GB")
    print(f"  Timestamp:         {system_info.timestamp}")

    print("\n📈 BENCHMARK RESULTS")
    print("-" * 80)

    # Group results by device
    devices = sorted({r.device for r in results})

    for device in devices:
        device_results = [r for r in results if r.device == device]
        print(f"\n  Device: {device.upper()}")
        print("  " + "-" * 76)

        # Table header
        print(
            f"  {'Input':<8} {'Batch':<6} {'Mean (ms)':<12} {'Std (ms)':<10} "
            f"{'P95 (ms)':<10} {'FPS':<10} {'Img/s':<10}"
        )
        print("  " + "-" * 76)

        for r in device_results:
            print(
                f"  {r.input_size:<8} {r.batch_size:<6} {r.mean_latency_ms:<12.2f} "
                f"{r.std_latency_ms:<10.2f} {r.p95_latency_ms:<10.2f} "
                f"{r.throughput_fps:<10.1f} {r.throughput_images_per_sec:<10.1f}"
            )

    print("\n" + "=" * 80)

    # Real-time feasibility assessment
    print("\n🎯 REAL-TIME FEASIBILITY ASSESSMENT")
    print("-" * 40)

    # Standard surgical video framerates
    target_fps_30 = 30.0  # Standard video
    target_fps_60 = 60.0  # High-speed capture

    for r in results:
        if r.batch_size == 1:  # Only assess single-image latency
            meets_30fps = r.throughput_fps >= target_fps_30
            meets_60fps = r.throughput_fps >= target_fps_60

            status_30 = "✅ YES" if meets_30fps else "❌ NO"
            status_60 = "✅ YES" if meets_60fps else "❌ NO"

            print(f"\n  {r.device.upper()} @ {r.input_size}x{r.input_size}:")
            print(
                f"    30 FPS (standard video):     {status_30} "
                f"({r.throughput_fps:.1f} FPS achieved)"
            )
            print(
                f"    60 FPS (high-speed capture): {status_60} "
                f"({r.throughput_fps:.1f} FPS achieved)"
            )

            if meets_30fps:
                headroom = ((r.throughput_fps - target_fps_30) / target_fps_30) * 100
                print(f"    Headroom for 30 FPS:         {headroom:.1f}%")

    print("\n" + "=" * 80)
    print("Note: Real-time performance depends on actual deployment environment.")
    print("These benchmarks provide baseline estimates for feasibility assessment.")
    print("=" * 80 + "\n")


def save_benchmark_results(
    results: list[BenchmarkResult],
    system_info: SystemInfo,
    provenance: ProvenanceInfo,
    output_path: Path,
) -> None:
    """Save benchmark results to JSON file."""

    output_data = {
        "system_info": asdict(system_info),
        "provenance": provenance.to_dict(),
        "results": [r.to_dict() for r in results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Benchmark results saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark inference speed for surgical instrument segmentation model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run with defaults
  %(prog)s --model-path outputs/models/best.pth
  %(prog)s --input-sizes 256 512 --batch-sizes 1 4
  %(prog)s --num-iterations 500 --output results.json
        """,
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to trained model weights (.pth file). If not provided, "
        "uses randomly initialized model.",
    )

    parser.add_argument(
        "--input-sizes",
        type=int,
        nargs="+",
        default=[256],
        help="Input image sizes to benchmark (default: 256)",
    )

    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Batch sizes to benchmark (default: 1)",
    )

    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of timed iterations per configuration (default: 100)",
    )

    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="Number of warmup iterations before timing (default: 10)",
    )

    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=None,
        help="Devices to benchmark (default: cuda if available, else cpu)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save JSON results (optional)",
    )

    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=None,
        help="Path to dataset manifest JSON for provenance hashing (optional)",
    )

    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Path to config file for provenance hashing (optional)",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of output classes (default: 2)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for inference benchmarking."""
    import torch

    args = parse_args()

    # Import model
    from surgical_segmentation.models import InstrumentSegmentationModel

    print("\n" + "=" * 80)
    print("SURGICAL INSTRUMENT SEGMENTATION - INFERENCE BENCHMARK")
    print("=" * 80)

    # Collect system information
    system_info = get_system_info()
    provenance = collect_provenance(
        model_path=args.model_path,
        dataset_manifest_path=args.dataset_manifest,
        config_path=args.config_path,
    )

    # Determine devices to benchmark
    if args.devices:
        devices = args.devices
    elif torch.cuda.is_available():
        devices = ["cuda", "cpu"]
    else:
        devices = ["cpu"]

    print("\nBenchmark Configuration:")
    print(f"  Input sizes:  {args.input_sizes}")
    print(f"  Batch sizes:  {args.batch_sizes}")
    print(f"  Devices:      {devices}")
    print(f"  Iterations:   {args.num_iterations}")
    print(f"  Warmup:       {args.warmup_iterations}")

    # Initialize model
    print("\nInitializing model...")
    model = InstrumentSegmentationModel(num_classes=args.num_classes)

    if args.model_path and args.model_path.exists():
        print(f"Loading weights from: {args.model_path}")
        state_dict = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        print("Using randomly initialized weights (no --model-path provided)")

    # Run benchmarks
    results: list[BenchmarkResult] = []

    for device in devices:
        if device == "cuda" and not torch.cuda.is_available():
            print(f"\nSkipping {device} (CUDA not available)")
            continue

        for input_size in args.input_sizes:
            for batch_size in args.batch_sizes:
                print(
                    f"\nBenchmarking: {device} | "
                    f"{input_size}x{input_size} | batch_size={batch_size}"
                )

                try:
                    result = benchmark_inference(
                        model=model,
                        input_size=input_size,
                        batch_size=batch_size,
                        num_iterations=args.num_iterations,
                        warmup_iterations=args.warmup_iterations,
                        device=device,
                    )
                    results.append(result)
                    print(
                        f"    Mean: {result.mean_latency_ms:.2f} ms | "
                        f"FPS: {result.throughput_fps:.1f}"
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("    ⚠️ Out of memory - skipping this configuration")
                        if device == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        raise

    # Print report
    print_benchmark_report(results, system_info)

    # Save results if requested
    if args.output:
        save_benchmark_results(results, system_info, provenance, args.output)


if __name__ == "__main__":
    main()
