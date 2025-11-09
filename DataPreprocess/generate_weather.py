#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import os
import sys
import cv2
import numpy as np
import albumentations as A
import random

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_outputs(img_bgr: np.ndarray, out_png: Path, save_npy: bool):
    ensure_dir(out_png.parent)
    cv2.imwrite(str(out_png), img_bgr)
    if save_npy:
        npy_path = out_png.with_suffix(".npy")
        np.save(str(npy_path), img_bgr)

def _odd_int(x, lo=3, hi=11):
    """Clamp to [lo,hi], round to int, and make it odd (Albumentations blur_value must be odd)."""
    x = int(np.clip(round(x), lo, hi))
    return x if x % 2 == 1 else x + 1 if x < hi else x - 1

def build_transforms(intensity: float):
    """
    intensity in (0, 1]; scales typical params.
    All parameters passed as scalars (no tuples) to satisfy Albumentations validation.
    """

    # ---- Fog ----
    fog_alpha = float(np.clip(0.05 + 0.10 * intensity, 0.0, 1.0))
    fog = A.RandomFog(
        fog_coef_lower=float(max(0.02, 0.08 * intensity)),
        fog_coef_upper=float(max(0.04, 0.15 * intensity)),
        alpha_coef=fog_alpha,
        p=1.0
    )

    # ---- Snow ----
    # keep lower < upper and both in (0,1]
    sp_low  = float(np.clip(0.25 * intensity, 0.05, 0.6))
    sp_high = float(np.clip(sp_low + 0.15, 0.2, 0.9))
    snow = A.RandomSnow(
        snow_point_lower=sp_low,
        snow_point_upper=sp_high,
        brightness_coeff=float(np.clip(1.0 + 0.6 * intensity, 1.0, 3.0)),
        p=1.0
    )

    # ---- Rain ----
    # Scalars only (no tuples)
    drop_length = int(np.clip(20 + 40 * intensity, 10, 80))   # pixels
    drop_width  = int(np.clip(1 + 2 * intensity, 1, 5))       # pixels, albumentations requires 1..5
    blur_value  = _odd_int(3 + 4 * intensity, lo=3, hi=11)    # odd integer
    rain = A.RandomRain(
        slant_lower=-10, slant_upper=10,
        drop_length=drop_length,
        drop_width=drop_width,
        drop_color=(180, 180, 180),
        blur_value=blur_value,
        brightness_coefficient=float(np.clip(1.0 - 0.2 * intensity, 0.0, 1.0)),
        rain_type=None,
        p=1.0
    )

    return {"fog": fog, "snow": snow, "rain": rain}

def main():
    parser = argparse.ArgumentParser(description="Generate fog/rain/snow images using Albumentations while preserving folder structure.")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root containing 'normal/' folder (e.g., /data/myset).")
    parser.add_argument("--normal_subdir", type=str, default="normal",
                        help="Name of the subfolder with clean images. Default: normal")
    parser.add_argument("--out_root", type=str, default=None,
                        help="Output root. Default: dataset_root (creates fog/rain/snow alongside 'normal').")
    parser.add_argument("--effects", type=str, nargs="+", default=["fog", "rain", "snow"],
                        help="Which effects to generate. Any of: fog rain snow. Default: all three.")
    parser.add_argument("--png_ext", type=str, default=".png",
                        help="Output image extension (must be .png, .jpg, etc.). Default: .png")
    parser.add_argument("--save_npy", action="store_true",
                        help="Also save a .npy array next to each PNG.")
    parser.add_argument("--variants", type=int, default=1,
                        help="How many variants per effect per image. If >1, adds _v{idx} suffix. Default: 1")
    parser.add_argument("--intensity", type=float, default=0.8,
                        help="Global intensity in (0,1]. Higher = stronger effects. Default: 0.8")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Basic checks
    if not args.png_ext.startswith("."):
        print("png_ext must start with a dot, e.g. .png", file=sys.stderr)
        sys.exit(1)
    effects = [e.lower() for e in args.effects]
    allowed = {"fog", "rain", "snow"}
    for e in effects:
        if e not in allowed:
            print(f"Unknown effect '{e}'. Use any of: fog rain snow", file=sys.stderr)
            sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_root = Path(args.dataset_root).resolve()
    normal_root = dataset_root / args.normal_subdir
    if not normal_root.exists():
        print(f"normal folder not found: {normal_root}", file=sys.stderr)
        sys.exit(1)

    out_root = Path(args.out_root).resolve() if args.out_root else dataset_root

    # Build transforms (one set we reuse; Albumentations will randomize per call)
    transforms = build_transforms(args.intensity)

    images = list_images(normal_root)
    if not images:
        print(f"No images found under {normal_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(images)} images under {normal_root}")
    print(f"Generating effects: {effects}")
    print(f"Output root: {out_root}")

    total_written = 0
    for src in images:
        # Read in BGR (OpenCV) and pass directly to Albumentations (works fine with uint8)
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not read image: {src}")
            continue

        rel = src.relative_to(normal_root)  # path after 'normal/'
        stem = src.stem

        for eff in effects:
            aug = transforms[eff]
            for vidx in range(args.variants):
                # Run the transform
                out_img = aug(image=img)["image"]

                # Decide filename
                if args.variants == 1:
                    out_name = stem + args.png_ext
                else:
                    out_name = f"{stem}_v{vidx+1}{args.png_ext}"

                # Mirror subfolders: out_root/<effect>/<same_rel_dir>/<filename>
                out_path = out_root / eff / rel.with_suffix("")  # remove original suffix
                out_path = out_path.with_name(out_name)  # set new name + ext
                write_outputs(out_img, out_path, args.save_npy)
                total_written += 1

    print(f"Done. Wrote {total_written} file(s){' (+ .npy arrays)' if args.save_npy else ''}.")

if __name__ == "__main__":
    main()
