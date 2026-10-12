#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 26 — Rice mask / rice fraction animations (MP4)
----------------------------------------------------
Purpose
  - Scan Step 21 outputs recursively
  - Group yearly GeoTIFFs into animation series
  - Render one MP4 per series
  - Write the MP4 into the same series folder as the source TIFF set
    (or one level above if the TIFFs are stored under year subfolders)

Assumptions
  - Input is under:
      /opt/foss4g_2026/processed/21_cdl_rice
  - Files are annual TIFFs for rice mask / rice fraction
  - Output MP4 is for presentation/slides
  - FPS is fixed to 2 unless changed by CLI

Examples
  python 26_rice_mask_fraction_animation.py
  python 26_rice_mask_fraction_animation.py --overwrite
  python 26_rice_mask_fraction_animation.py --root /opt/foss4g_2026/processed/21_cdl_rice
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap
from rasterio.enums import Resampling

# ------------------------------------------------------------
# Defaults
# ------------------------------------------------------------

PROJECT_DIR = Path("/opt/foss4g_2026")
DEFAULT_ROOT = PROJECT_DIR / "processed" / "21_cdl_rice"
DEFAULT_FPS = 2
DEFAULT_MAX_DIM = 1400   # display downsample cap, not changing raw data
DEFAULT_DPI = 150

YEAR_RE = re.compile(r"(20\d{2})")
VALID_YEARS = set(range(2000, 2101))


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def is_year_token(text: str) -> bool:
    if re.fullmatch(r"\d{4}", text):
        y = int(text)
        return y in VALID_YEARS
    return False


def extract_year(text: str) -> Optional[int]:
    m = YEAR_RE.search(text)
    if not m:
        return None
    y = int(m.group(1))
    if y in VALID_YEARS:
        return y
    return None


def discover_tifs(root: Path) -> List[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]
    )


def infer_series_dir(tif_path: Path, root: Path) -> Path:
    """
    If the TIFF sits inside a year subfolder, use the parent of that year folder
    as the series dir, so all years become one animation.
    Otherwise use the TIFF's own parent.
    """
    parent = tif_path.parent
    if is_year_token(parent.name) and parent.parent.exists():
        return parent.parent
    return parent


def build_series(root: Path) -> Dict[Path, List[Path]]:
    series = defaultdict(list)
    for tif in discover_tifs(root):
        series_dir = infer_series_dir(tif, root)
        series[series_dir].append(tif)
    return dict(series)


def sort_key_tif(path: Path):
    y = extract_year(path.name)
    return (y if y is not None else 9999, path.name)


def classify_series(series_dir: Path, files: List[Path]) -> Dict[str, str]:
    """
    Guess human-friendly label and rendering style from path/file names.
    """
    joined = " ".join([series_dir.as_posix().lower()] + [f.name.lower() for f in files])

    if "mask" in joined and "fraction" not in joined:
        kind = "mask"
        label = "Rice Mask"
    elif "fraction" in joined:
        kind = "fraction"
        label = "Rice Fraction"
    else:
        # fallback
        if any("mask" in f.name.lower() for f in files):
            kind = "mask"
            label = "Rice Mask"
        else:
            kind = "fraction"
            label = "Rice Fraction"

    # infer resolution
    if "30m" in joined:
        res = "30m"
    elif "250m" in joined:
        res = "250m"
    elif "500m" in joined:
        res = "500m"
    elif "1000m" in joined or "1km" in joined:
        res = "1000m"
    elif "4km" in joined:
        res = "4km"
    else:
        res = "unknown-res"

    return {"kind": kind, "label": label, "resolution": res}


def make_green_cmap():
    # white -> pale green -> dark green
    return LinearSegmentedColormap.from_list(
        "rice_green",
        ["#ffffff", "#d9f0d3", "#78c679", "#238443"],
        N=256,
    )


RICE_CMAP = make_green_cmap()


def read_display_raster(
    tif_path: Path,
    max_dim: int,
    kind: str,
) -> Tuple[np.ma.MaskedArray, Tuple[float, float, float, float]]:
    """
    Read a display-friendly downsampled version for rendering only.
    RAW/processed TIFF is not modified.
    """
    with rasterio.open(tif_path) as src:
        height = src.height
        width = src.width

        if max(height, width) <= max_dim:
            out_h, out_w = height, width
        else:
            scale = max(height / max_dim, width / max_dim)
            out_h = max(1, int(round(height / scale)))
            out_w = max(1, int(round(width / scale)))

        if kind == "mask":
            resampling = Resampling.nearest
        else:
            resampling = Resampling.bilinear

        arr = src.read(
            1,
            out_shape=(1, out_h, out_w),
            resampling=resampling,
            masked=False,
        ).astype("float32")

        nodata = src.nodata
        mask = np.zeros(arr.shape, dtype=bool)

        if nodata is not None:
            mask |= np.isclose(arr, nodata)

        mask |= ~np.isfinite(arr)
        marr = np.ma.masked_array(arr, mask=mask)

        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        return marr, extent


def decide_value_range(files: List[Path], kind: str, max_dim: int) -> Tuple[float, float]:
    """
    Compute a stable display range for the whole series.
    For masks/fractions we prefer 0..1 when possible.
    """
    observed_max = None
    observed_min = None

    sample_paths = files[: min(len(files), 5)] + files[max(0, len(files) - 5):]
    seen = []
    for p in sample_paths:
        if p not in seen:
            seen.append(p)

    for p in seen:
        arr, _ = read_display_raster(p, max_dim=max_dim, kind=kind)
        if arr.count() == 0:
            continue
        vmin = float(arr.min())
        vmax = float(arr.max())
        observed_min = vmin if observed_min is None else min(observed_min, vmin)
        observed_max = vmax if observed_max is None else max(observed_max, vmax)

    if observed_min is None or observed_max is None:
        return (0.0, 1.0)

    # Most likely binary mask or 0-1 fraction
    if observed_min >= 0 and observed_max <= 1.001:
        return (0.0, 1.0)

    # If stored as percentage-like 0..100, keep that
    if observed_min >= 0 and observed_max <= 100.0:
        return (0.0, float(np.ceil(observed_max)))

    return (float(observed_min), float(observed_max))


def figure_to_rgb_array(fig) -> np.ndarray:
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    rgb = buf[..., :3]
    return rgb


def render_frame(
    tif_path: Path,
    series_dir: Path,
    series_meta: Dict[str, str],
    vmin: float,
    vmax: float,
    max_dim: int,
    dpi: int,
) -> np.ndarray:
    arr, extent = read_display_raster(tif_path, max_dim=max_dim, kind=series_meta["kind"])
    year = extract_year(tif_path.name)
    year_text = str(year) if year is not None else "Unknown year"

    title_main = f'{series_meta["label"]} — {series_meta["resolution"]}'
    title_sub = f"{series_dir.name} | {year_text}"

    fig, ax = plt.subplots(figsize=(9.5, 7.2), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(
        arr,
        extent=extent,
        origin="upper",
        cmap=RICE_CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_title(title_main, fontsize=16, pad=12, weight="bold")
    ax.text(
        0.5,
        1.01,
        title_sub,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    if series_meta["kind"] == "mask" and vmin == 0 and vmax == 1:
        cbar.set_label("Mask (0/1)")
    elif vmin == 0 and vmax == 1:
        cbar.set_label("Rice fraction")
    else:
        cbar.set_label("Value")

    fig.tight_layout()
    frame = figure_to_rgb_array(fig)
    plt.close(fig)
    return frame


def make_output_name(series_dir: Path, files: List[Path], series_meta: Dict[str, str]) -> str:
    years = sorted({extract_year(f.name) for f in files if extract_year(f.name) is not None})
    if years:
        y0, y1 = years[0], years[-1]
        return f"{series_dir.name}_{series_meta['label'].lower().replace(' ', '_')}_{series_meta['resolution']}_{y0}_{y1}.mp4"
    return f"{series_dir.name}_{series_meta['label'].lower().replace(' ', '_')}_{series_meta['resolution']}.mp4"


def create_animation_for_series(
    series_dir: Path,
    files: List[Path],
    fps: int,
    max_dim: int,
    dpi: int,
    overwrite: bool = False,
) -> Optional[Path]:
    files = sorted(files, key=sort_key_tif)

    if len(files) == 0:
        return None

    series_meta = classify_series(series_dir, files)
    output_name = make_output_name(series_dir, files, series_meta)
    output_mp4 = series_dir / output_name

    if output_mp4.exists() and not overwrite:
        print(f"[SKIP EXISTS] {output_mp4}")
        return output_mp4

    print("=" * 80)
    print(f"[SERIES] {series_dir}")
    print(f"  files      : {len(files)}")
    print(f"  type       : {series_meta['label']}")
    print(f"  resolution : {series_meta['resolution']}")
    print(f"  output     : {output_mp4}")

    vmin, vmax = decide_value_range(files, kind=series_meta["kind"], max_dim=max_dim)
    print(f"  display range: vmin={vmin} vmax={vmax}")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(
        output_mp4.as_posix(),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    ) as writer:
        for idx, tif_path in enumerate(files, start=1):
            print(f"  [{idx}/{len(files)}] frame: {tif_path.name}")
            frame = render_frame(
                tif_path=tif_path,
                series_dir=series_dir,
                series_meta=series_meta,
                vmin=vmin,
                vmax=vmax,
                max_dim=max_dim,
                dpi=dpi,
            )
            frame = ensure_even_frame(frame)
            writer.append_data(frame)

    print(f"[WROTE] {output_mp4}")
    return output_mp4

def ensure_even_frame(frame: np.ndarray) -> np.ndarray:
    """
    libx264/yuv420p requires even width and height.
    Pad by at most one pixel without changing the map content.
    """
    h, w = frame.shape[:2]

    pad_h = h % 2
    pad_w = w % 2

    if pad_h == 0 and pad_w == 0:
        return np.ascontiguousarray(frame)

    frame = np.pad(
        frame,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="constant",
        constant_values=255,
    )

    return np.ascontiguousarray(frame)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Create MP4 animations from Step 21 rice TIFF outputs.")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help=f"Root folder to scan recursively (default: {DEFAULT_ROOT})")
    p.add_argument("--fps", type=int, default=DEFAULT_FPS,
                   help="Frames per second for MP4 (default: 2)")
    p.add_argument("--max-dim", type=int, default=DEFAULT_MAX_DIM,
                   help="Max display dimension for rendering each frame (default: 1400)")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                   help="Matplotlib DPI (default: 150)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing MP4 files")
    p.add_argument("--limit-series", type=int, default=None,
                   help="Process only first N series (for smoke test)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"[ERROR] root does not exist: {root}", file=sys.stderr)
        return 1

    print("=" * 80)
    print("STEP 26 — Rice mask / fraction MP4 animation")
    print(f"Root       : {root}")
    print(f"FPS        : {args.fps}")
    print(f"Max dim    : {args.max_dim}")
    print(f"DPI        : {args.dpi}")
    print(f"Overwrite  : {args.overwrite}")
    print("=" * 80)

    series_map = build_series(root)
    # keep only groups with at least 2 files, so we make real animations
    series_items = [(d, sorted(fs, key=sort_key_tif)) for d, fs in series_map.items() if len(fs) >= 2]
    series_items.sort(key=lambda x: x[0].as_posix())

    if args.limit_series is not None:
        series_items = series_items[: args.limit_series]

    if not series_items:
        print("[WARN] no multi-frame series found under root")
        return 0

    print(f"Series found: {len(series_items)}")

    wrote = 0
    skipped = 0
    failed = 0

    for series_dir, files in series_items:
        try:
            result = create_animation_for_series(
                series_dir=series_dir,
                files=files,
                fps=args.fps,
                max_dim=args.max_dim,
                dpi=args.dpi,
                overwrite=args.overwrite,
            )
            if result is None:
                skipped += 1
            else:
                if result.exists():
                    wrote += 1
                else:
                    failed += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] {series_dir} -> {e}", file=sys.stderr)

    print("=" * 80)
    print("STEP 26 SUMMARY")
    print(f"Series total : {len(series_items)}")
    print(f"MP4 written  : {wrote}")
    print(f"Skipped      : {skipped}")
    print(f"Failed       : {failed}")
    print("=" * 80)

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
