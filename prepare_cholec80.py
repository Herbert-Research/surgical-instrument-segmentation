"""
Utility script to reorganize a subset of the Cholec80 dataset into the
`data/sample_frames` + `data/masks` layout used by instrument_segmentation.py.

This does NOT download the dataset for you (Cholec80 access requires credentialed
approval). Instead, point the script to your locally downloaded videos and
mask annotations and it will:
 1. Sample frames from the first N videos (configurable).
 2. Resize frames to match the demo resolution (longest side = 640 px).
 3. Attempt to locate the corresponding mask PNGs and copy/resize them.
 4. Emit a CSV manifest so downstream experiments know the provenance.

Example:
    python prepare_cholec80.py \
        --video-dir /path/to/Cholec80/videos \
        --mask-dir /path/to/CholecSeg8k/masks \
        --output-frame-dir data/sample_frames \
        --output-mask-dir data/masks \
        --max-videos 3 \
        --frame-step 10
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a Cholec80 subset for the segmentation demo."
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        required=True,
        help="Directory containing the Cholec80 MP4 videos.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        help=(
            "Directory containing per-frame instrument masks (e.g., CholecSeg8k). "
            "If omitted, blank masks are generated so training remains possible."
        ),
    )
    parser.add_argument(
        "--output-frame-dir",
        type=Path,
        default=Path("data/sample_frames"),
        help="Where to store resized RGB frames.",
    )
    parser.add_argument(
        "--output-mask-dir",
        type=Path,
        default=Path("data/masks"),
        help="Where to store resized mask PNGs.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/cholec80_manifest.csv"),
        help="CSV manifest capturing the frame-to-source mapping.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=3,
        help="Number of videos to process (starting from the lexicographically first).",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=10,
        help="Sample every Nth frame to keep the subset lightweight.",
    )
    parser.add_argument(
        "--long-side",
        type=int,
        default=640,
        help="Resize so that the longest side equals this value (aspect preserved).",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=500,
        help="Safety cap that limits how many frames are extracted per video.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resize_with_aspect(image: np.ndarray, long_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = long_side / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def resolve_mask_path(mask_root: Optional[Path], video_stem: str, frame_idx: int) -> Optional[Path]:
    if mask_root is None:
        return None
    candidates = [
        mask_root / video_stem / f"{frame_idx:06d}.png",
        mask_root / video_stem / f"{video_stem}_{frame_idx:06d}.png",
        mask_root / f"{video_stem}_{frame_idx:06d}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def extract_video(
    video_path: Path,
    mask_root: Optional[Path],
    output_frame_dir: Path,
    output_mask_dir: Path,
    long_side: int,
    frame_step: int,
    max_frames: int,
    manifest_writer: csv.DictWriter,
) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open {video_path}, skipping.")
        return 0, 0

    frame_idx = 0
    saved = 0
    mask_hits = 0
    video_stem = video_path.stem

    while saved < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = resize_with_aspect(frame_rgb, long_side)

        frame_name = f"{video_stem}_frame_{frame_idx:06d}.png"
        cv2.imwrite(str(output_frame_dir / frame_name), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        mask_path = resolve_mask_path(mask_root, video_stem, frame_idx)
        if mask_path and mask_path.exists():
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask_img is None:
                print(f"[WARN] Failed to read mask {mask_path}, using blank mask.")
                mask_img = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
            else:
                mask_hits += 1
        else:
            mask_img = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)

        mask_resized = resize_with_aspect(mask_img, long_side)
        if mask_resized.ndim == 3:
            mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(output_mask_dir / frame_name.replace("frame", "mask")), mask_resized)

        manifest_writer.writerow(
            {
                "video": video_path.name,
                "frame_name": frame_name,
                "source_frame_index": frame_idx,
                "mask_found": bool(mask_path and mask_path.exists()),
            }
        )

        saved += 1
        frame_idx += 1

    cap.release()
    return saved, mask_hits


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_frame_dir)
    ensure_dir(args.output_mask_dir)
    ensure_dir(args.manifest_path.parent)

    if args.mask_dir is None:
        print(
            "[WARN] --mask-dir not provided. Blank masks will be generated;"
            " metrics will remain background-only until true annotations are supplied."
        )

    videos = sorted(
        [p for p in args.video_dir.iterdir() if p.suffix.lower() in {".mp4", ".avi", ".mov"}]
    )[: args.max_videos]
    if not videos:
        raise FileNotFoundError(f"No videos found in {args.video_dir}")

    with args.manifest_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["video", "frame_name", "source_frame_index", "mask_found"],
        )
        writer.writeheader()
        total_frames = 0
        total_mask_hits = 0
        for video_path in videos:
            print(f"[INFO] Processing {video_path.name}")
            saved, mask_hits = extract_video(
                video_path=video_path,
                mask_root=args.mask_dir,
                output_frame_dir=args.output_frame_dir,
                output_mask_dir=args.output_mask_dir,
                long_side=args.long_side,
                frame_step=args.frame_step,
                max_frames=args.max_frames_per_video,
                manifest_writer=writer,
            )
            print(f"       → Saved {saved} frames from {video_path.name}")
            total_frames += saved
            total_mask_hits += mask_hits

    print(f"\n✓ Finished preparing subset. Total frames saved: {total_frames}")
    print(f"   Frames directory: {args.output_frame_dir}")
    print(f"   Masks directory:  {args.output_mask_dir}")
    print(f"   Manifest:         {args.manifest_path}")
    if total_frames:
        coverage = 100.0 * total_mask_hits / total_frames
        print(f"   Masks with annotations: {total_mask_hits} ({coverage:.1f}% coverage)")
    else:
        print("   Masks with annotations: 0 (no frames exported)")


if __name__ == "__main__":
    main()
