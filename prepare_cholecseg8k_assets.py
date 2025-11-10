"""Rename and organize CholecSeg8k dataset frames and watershed masks.

This script processes endoscopic frames and their corresponding watershed
segmentation masks from the CholecSeg8k dataset, organizing them into a
standardized directory structure for model training.

Uses watershed masks to ensure accurate pixel-level class labels for
surgical instrument segmentation.

Example usage:
    python prepare_cholecseg8k_assets.py \
        --frame-dir /path/to/frame_pngs \
        --mask-dir  /path/to/mask_pngs \
        --video-stem video01
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


FRAME_PATTERN = re.compile(r"^frame_(\d+)_endo$", re.IGNORECASE)
MASK_PATTERN = re.compile(r"^frame_(\d+)_endo_watershed_mask$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy/rename CholecSeg8k frames/masks")
    parser.add_argument("--frame-dir", type=Path, required=True, help="Directory with *_endo.png files")
    parser.add_argument("--mask-dir", type=Path, required=True, help="Directory with *_endo_watershed_mask.png files")
    parser.add_argument("--output-frame-dir", type=Path, default=Path("data/sample_frames"))
    parser.add_argument("--output-mask-dir", type=Path, default=Path("data/masks"))
    parser.add_argument("--video-stem", type=str, default="video01", help="Stem prefix (e.g., video01)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return parser.parse_args()


def collect_sources(directory: Path, pattern: re.Pattern[str]) -> dict[int, Path]:
    """Recursively collect image files matching the specified pattern.
    
    Args:
        directory: Root directory to search
        pattern: Compiled regex pattern to match file stems
        
    Returns:
        Dictionary mapping frame indices to file paths
    """
    mapping: dict[int, Path] = {}
    for path in directory.rglob("*.png"):
        match = pattern.match(path.stem)
        if not match:
            continue
        idx = int(match.group(1))
        mapping[idx] = path
    return mapping


def main() -> None:
    args = parse_args()
    args.output_frame_dir.mkdir(parents=True, exist_ok=True)
    args.output_mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for frames in: {args.frame_dir}")
    frame_sources = collect_sources(args.frame_dir, FRAME_PATTERN)
    print(f"Found {len(frame_sources)} valid frames.")

    print(f"Scanning for watershed masks in: {args.mask_dir}")
    mask_sources = collect_sources(args.mask_dir, MASK_PATTERN)
    print(f"Found {len(mask_sources)} valid watershed masks.")

    if not frame_sources:
        raise FileNotFoundError(f"No frame_*_endo.png files found in {args.frame_dir}")
    if not mask_sources:
        raise FileNotFoundError(
            f"No frame_*_endo_watershed_mask.png files found in {args.mask_dir}. "
            "Please check that you have the correct CholecSeg8k subfolder."
        )

    paired_indices = sorted(set(frame_sources.keys()) & set(mask_sources.keys()))
    if not paired_indices:
        raise RuntimeError("No overlapping indices between frames and watershed masks")

    print(f"Processing {len(paired_indices)} paired images...")

    for idx in paired_indices:
        frame_src = frame_sources[idx]
        mask_src = mask_sources[idx]
        
        frame_dst = args.output_frame_dir / f"{args.video_stem}_frame_{idx:06d}.png"
        mask_dst = args.output_mask_dir / f"{args.video_stem}_mask_{idx:06d}.png"

        if not args.overwrite and frame_dst.exists():
             print(f"Skipping existing: {frame_dst.name}")
             continue

        shutil.copy2(frame_src, frame_dst)
        shutil.copy2(mask_src, mask_dst)

    print(f"✓ Successfully copied {len(paired_indices)} pairs with stem '{args.video_stem}'.")

    print(f"Copied {len(paired_indices)} paired frames/masks with stem '{args.video_stem}'.")


if __name__ == "__main__":
    main()
