"""Prepare the entire Full Dataset by processing all videos.

This script processes all videos in the Full Dataset directory, organizing
frames and watershed masks into standardized directories for model training.

Example usage:
    python scripts/prepare_full_dataset.py \
        --source-dir "datasets/Full Dataset" \
        --output-frame-dir "datasets/Full Dataset/frames" \
        --output-mask-dir "datasets/Full Dataset/masks"
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict

FRAME_PATTERN = re.compile(r"^frame_(\d+)_endo$", re.IGNORECASE)
MASK_PATTERN = re.compile(r"^frame_(\d+)_endo_watershed_mask$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Full Dataset for training")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("datasets/Full Dataset"),
        help="Root directory containing video folders",
    )
    parser.add_argument(
        "--output-frame-dir",
        type=Path,
        default=Path("datasets/Full Dataset/frames"),
        help="Output directory for processed frames",
    )
    parser.add_argument(
        "--output-mask-dir",
        type=Path,
        default=Path("datasets/Full Dataset/masks"),
        help="Output directory for processed masks",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually copying files",
    )
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


def process_video_directory(
    video_dir: Path,
    video_stem: str,
    output_frame_dir: Path,
    output_mask_dir: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> int:
    """Process a single video directory.

    Args:
        video_dir: Directory containing video subdirectories with frames/masks
        video_stem: Name of the video (e.g., "video01")
        output_frame_dir: Destination for processed frames
        output_mask_dir: Destination for processed masks
        overwrite: Whether to overwrite existing files
        dry_run: If True, only show what would be done

    Returns:
        Number of frame-mask pairs processed
    """
    print(f"\nProcessing {video_stem}...")
    print(f"  Scanning for frames in: {video_dir}")

    frame_sources = collect_sources(video_dir, FRAME_PATTERN)
    print(f"  Found {len(frame_sources)} valid frames")

    mask_sources = collect_sources(video_dir, MASK_PATTERN)
    print(f"  Found {len(mask_sources)} valid watershed masks")

    if not frame_sources:
        print(f"  ⚠ No frames found in {video_dir}")
        return 0

    if not mask_sources:
        print(f"  ⚠ No masks found in {video_dir}")
        return 0

    paired_indices = sorted(set(frame_sources.keys()) & set(mask_sources.keys()))
    if not paired_indices:
        print(f"  ⚠ No overlapping indices between frames and masks")
        return 0

    print(f"  Processing {len(paired_indices)} paired images...")

    copied_count = 0
    skipped_count = 0

    for idx in paired_indices:
        frame_src = frame_sources[idx]
        mask_src = mask_sources[idx]

        frame_dst = output_frame_dir / f"{video_stem}_frame_{idx:06d}.png"
        mask_dst = output_mask_dir / f"{video_stem}_mask_{idx:06d}.png"

        if not overwrite and frame_dst.exists():
            skipped_count += 1
            continue

        if dry_run:
            print(f"    Would copy: {frame_src.name} -> {frame_dst.name}")
            copied_count += 1
        else:
            shutil.copy2(frame_src, frame_dst)
            shutil.copy2(mask_src, mask_dst)
            copied_count += 1

    if skipped_count > 0:
        print(f"  Skipped {skipped_count} existing pairs")

    action = "Would copy" if dry_run else "Copied"
    print(f"  ✓ {action} {copied_count} pairs for {video_stem}")

    return copied_count


def main() -> None:
    args = parse_args()

    if not args.source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {args.source_dir}")

    # Create output directories
    if not args.dry_run:
        args.output_frame_dir.mkdir(parents=True, exist_ok=True)
        args.output_mask_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directories:")
        print(f"  Frames: {args.output_frame_dir}")
        print(f"  Masks:  {args.output_mask_dir}")
    else:
        print("DRY RUN MODE - No files will be copied")
        print(f"Would create output directories:")
        print(f"  Frames: {args.output_frame_dir}")
        print(f"  Masks:  {args.output_mask_dir}")

    # Find all video directories
    video_dirs = sorted(
        [d for d in args.source_dir.iterdir() if d.is_dir() and d.name.startswith("video")]
    )

    if not video_dirs:
        raise FileNotFoundError(f"No video directories found in {args.source_dir}")

    print(f"\nFound {len(video_dirs)} video directories:")
    for vdir in video_dirs:
        print(f"  - {vdir.name}")

    print(f"\n{'='*70}")
    print("Starting processing...")
    print(f"{'='*70}")

    total_pairs = 0
    processed_videos = 0

    for video_dir in video_dirs:
        video_stem = video_dir.name

        pairs_processed = process_video_directory(
            video_dir=video_dir,
            video_stem=video_stem,
            output_frame_dir=args.output_frame_dir,
            output_mask_dir=args.output_mask_dir,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

        if pairs_processed > 0:
            total_pairs += pairs_processed
            processed_videos += 1

    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Videos processed: {processed_videos}/{len(video_dirs)}")
    print(f"Total frame-mask pairs: {total_pairs}")

    if not args.dry_run:
        print(f"\nProcessed files saved to:")
        print(f"  Frames: {args.output_frame_dir}")
        print(f"  Masks:  {args.output_mask_dir}")
        print(f"\nYou can now train the model using:")
        print(f"  python -m surgical_segmentation.training.trainer \\")
        print(f'      --frame-dir "{args.output_frame_dir}" \\')
        print(f'      --mask-dir "{args.output_mask_dir}"')
    else:
        print(f"\nThis was a dry run. Run without --dry-run to actually copy files.")


if __name__ == "__main__":
    main()
