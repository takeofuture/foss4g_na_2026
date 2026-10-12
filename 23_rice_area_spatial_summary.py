#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
23_rice_area_spatial_summary.py

California Rice Monitoring — Step 23

Purpose
-------
Apply the Step-22 physical-value / QA recipes to the downloaded MODIS/NTSG
RAW data, but only over cells intersecting the *actual annual* California
rice area from Step 21.

This step intentionally does NOT duplicate tens of thousands of full-size
Float32 rasters.  Instead it performs the expensive native-grid pixel work on
EC2 and writes compact observation-level spatial summaries that Step 24 can
turn into monthly / seasonal feature tables.

Pipeline
--------
    Step 20 RAW archive (read only)
           +
    Step 21 annual CDL rice mask / native MODIS-grid rice fraction
           +
    Step 22 scale / offset / nodata / QA recipe catalog
           |
           v
    Step 23 native-grid rice-area weighted spatial summaries
           |
           v
    Step 24 temporal aggregation + PRISM + final Parquet feature tables

Scientific policy
-----------------
* RAW under ./data is READ ONLY.
* No proxy CDL is used.  California CDL is available here for 2007-2025, so
  Step 23 analyzes 2007-2025 only even though Step 20 downloaded 2005-2025.
* Each remote-sensing product stays on its own native grid.
* Rice spatial weights are annual rice fraction * native pixel area.
* QA and scale rules come from Step 22.
* UMT/NTSG GPP and MOD17 GPP remain separate products.
* MOD09 derived wetness indices are calculated per pixel AFTER QA:
      NDWI  = (green - nir)   / (green + nir)
      MNDWI = (green - swir1) / (green + swir1)
      NDMI  = (nir - swir1)   / (nir + swir1)

Output
------
./processed/23_rice_spatial_summary/
    _manifests/
        tile_observation_summary.csv
        statewide_observation_summary.csv
        run_summary.json
        processing_errors.jsonl
    _work/
        observations.jsonl
        completed_files.txt
        run_signature.json
    _cache/
        native_rice_fraction/...   # only when a source grid is not one of
                                   # the prebuilt Step-21 MODIS grids

Resume behavior
---------------
A RAW file is checkpointed only after all requested variables for that file
have been processed and appended.  Re-running the script skips completed RAW
files.  Duplicate observation rows caused by a crash between append and
checkpoint are deterministically deduplicated during final consolidation.

Typical run
-----------
mkdir -p processed/23_rice_spatial_summary/_manifests
nohup python -u 23_rice_area_spatial_summary.py \
  > processed/23_rice_spatial_summary/_manifests/23.log 2>&1 &

The script is intentionally single-process.  GDAL/HDF4 handles and a large
EC2/EBS workload are more reliable this way; resume checkpoints protect the
long run.  If Rasterio was installed from a wheel without HDF4 support, Step 23
automatically uses the system gdalinfo/gdal_translate HDF4 driver for only the
rice-area source window, then reads the temporary GeoTIFF with Rasterio.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS as PyCRS
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds, transform as window_transform


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PIPELINE_VERSION = "23.2"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = PROJECT_DIR / "processed"

STEP21_ROOT = PROCESSED_DIR / "21_cdl_rice"
STEP21_MASK_DIR = STEP21_ROOT / "native_30m"
STEP21_FRACTION_DIR = STEP21_ROOT / "modis_fraction"
STEP21_FRACTION_MANIFEST = STEP21_ROOT / "_manifests" / "rice_fraction_manifest.csv"

STEP22_ROOT = PROCESSED_DIR / "22_physical_rules"
STEP22_RECIPE_CSV = STEP22_ROOT / "_manifests" / "band_recipe_manifest.csv"
STEP22_AUDIT_CSV = STEP22_ROOT / "_manifests" / "file_recipe_audit.csv"

OUT_ROOT = PROCESSED_DIR / "23_rice_spatial_summary"
MANIFEST_DIR = OUT_ROOT / "_manifests"
WORK_DIR = OUT_ROOT / "_work"
CACHE_DIR = OUT_ROOT / "_cache" / "native_rice_fraction"

OBS_JSONL = WORK_DIR / "observations.jsonl"
COMPLETED_TXT = WORK_DIR / "completed_files.txt"
RUN_SIGNATURE_JSON = WORK_DIR / "run_signature.json"
ERROR_JSONL = MANIFEST_DIR / "processing_errors.jsonl"

TILE_SUMMARY_CSV = MANIFEST_DIR / "tile_observation_summary.csv"
STATE_SUMMARY_CSV = MANIFEST_DIR / "statewide_observation_summary.csv"
RUN_SUMMARY_JSON = MANIFEST_DIR / "run_summary.json"

ANALYSIS_START_YEAR = 2007
ANALYSIS_END_YEAR = 2025
MONTHS = set(range(4, 12))
RICE_FRACTION_MIN = 0.0
STEP21_FRACTION_NODATA = -9999.0
STEP21_MASK_NODATA = 255

# Step-21 native fraction grid to use for each standard MODIS product.
PRODUCT_RESOLUTION = {
    "mod13_vi": "250m",
    "mod09_sr": "500m",
    "mod15_lai_fpar": "500m",
    "mod11_lst": "1000m",
    "mod16_et": "500m",
    "mod17_gpp": "500m",
    # UMT/NTSG is a separate GeoTIFF grid; fraction is cached on that grid.
    "umt_gpp": None,
}

PRODUCT_NOMINAL_M = {
    "mod13_vi": 250,
    "mod09_sr": 500,
    "mod15_lai_fpar": 500,
    "mod11_lst": 1000,
    "mod16_et": 500,
    "mod17_gpp": 500,
    "umt_gpp": 250,
}

PRODUCT_CADENCE = {
    "mod13_vi": "16-day composite",
    "mod09_sr": "8-day composite",
    "mod15_lai_fpar": "8-day composite",
    "mod11_lst": "8-day composite",
    "mod16_et": "8-day composite",
    "mod17_gpp": "8-day composite",
    "umt_gpp": "8-day composite",
}

# Step 24 uses this to choose temporal mean vs sum.
SUM_TEMPORAL_VARIABLES = {"et", "pet", "gpp", "psn_net", "gpp_ntsg"}

# Keep a handful of rice-fraction crops in RAM; these are reused across many
# dates from the same year/tile/resolution.
FRACTION_LRU_MAX = 8


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class Recipe:
    product: str
    source_product: str
    variable: str
    acquisition_date: str
    year: int
    month: int
    local_path: str
    dataset_locator: str
    band_index: int
    source_description: str
    source_nodata: Optional[float]
    applied_scale: float
    applied_offset: float
    output_unit: str
    valid_range: Optional[Tuple[float, float]]
    qa_key: Optional[str]
    qa_dataset_locator: str
    qa_band_index: int
    qa_rule: str


@dataclass
class FractionBundle:
    fraction: np.ndarray
    science_window: Window
    transform: Affine
    crs: rasterio.crs.CRS
    total_rice_area_m2: float
    rice_positive_cells: int
    source_fraction_path: str
    fraction_mode: str  # step21_native_modis or dynamic_native_grid


@dataclass(frozen=True)
class GridMeta:
    """Minimal raster-grid metadata used without requiring Rasterio to open HDF4."""
    name: str
    width: int
    height: int
    transform: Affine
    crs: rasterio.crs.CRS


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    for p in (OUT_ROOT, MANIFEST_DIR, WORK_DIR, CACHE_DIR):
        p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".part")
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def atomic_write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".part")
    tmp.unlink(missing_ok=True)
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def append_completed(path: Path, local_path: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(local_path + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    }


def safe_float(v: object) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"none", "nan", "null"}:
        return None
    try:
        x = float(s)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


def safe_int(v: object, default: int = 1) -> int:
    try:
        return int(float(str(v).strip()))
    except Exception:
        return default


def safe_json_range(v: object) -> Optional[Tuple[float, float]]:
    s = str(v or "").strip()
    if not s or s.lower() in {"none", "null", "nan"}:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, list) and len(obj) == 2:
            a, b = float(obj[0]), float(obj[1])
            return (a, b)
    except Exception:
        pass
    return None


def tile_from_path(path: str) -> str:
    m = re.search(r"(?i)(h\d{2}v\d{2})", Path(path).name)
    return m.group(1).lower() if m else "native"


def grids_match(a, b, tol: float = 1e-7) -> bool:
    if a.crs != b.crs or a.width != b.width or a.height != b.height:
        return False
    ta = tuple(a.transform)
    tb = tuple(b.transform)
    return all(abs(float(x) - float(y)) <= tol * max(1.0, abs(float(x)), abs(float(y))) for x, y in zip(ta, tb))


def clamp_window(window: Window, width: int, height: int) -> Optional[Window]:
    col0 = max(0, int(math.floor(window.col_off)))
    row0 = max(0, int(math.floor(window.row_off)))
    col1 = min(width, int(math.ceil(window.col_off + window.width)))
    row1 = min(height, int(math.ceil(window.row_off + window.height)))
    if col1 <= col0 or row1 <= row0:
        return None
    return Window(col0, row0, col1 - col0, row1 - row0)


def trim_to_positive(a: np.ndarray, base_window: Window) -> Tuple[np.ndarray, Optional[Window]]:
    positive = np.isfinite(a) & (a > RICE_FRACTION_MIN) & (a != STEP21_FRACTION_NODATA)
    rr, cc = np.where(positive)
    if rr.size == 0:
        return a[:0, :0], None
    r0, r1 = int(rr.min()), int(rr.max()) + 1
    c0, c1 = int(cc.min()), int(cc.max()) + 1
    trimmed = a[r0:r1, c0:c1]
    w = Window(
        int(base_window.col_off) + c0,
        int(base_window.row_off) + r0,
        c1 - c0,
        r1 - r0,
    )
    return trimmed, w


def dataset_candidates(locator: str, local_path: str) -> List[str]:
    candidates: List[str] = []
    loc = str(locator or "").strip()
    lp = str(local_path or "").strip()
    if loc:
        candidates.append(loc)
    if lp and lp not in candidates:
        candidates.append(lp)
    return candidates


def is_hdf4_locator(locator: str) -> bool:
    return str(locator or "").lstrip().upper().startswith(("HDF4_", "HDF4:"))


def _run_checked(cmd: List[str], *, purpose: str) -> subprocess.CompletedProcess:
    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"{purpose} requires the system GDAL command {cmd[0]!r}, but it was not found. "
            "Install gdal-bin and ensure it is on PATH."
        ) from e
    if cp.returncode != 0:
        msg = (cp.stderr or cp.stdout or "").strip()
        raise RuntimeError(f"{purpose} failed (exit {cp.returncode}): {msg}")
    return cp


_GRID_META_CACHE: Dict[str, GridMeta] = {}


def _grid_from_rasterio(candidate: str) -> GridMeta:
    with rasterio.open(candidate) as ds:
        if ds.crs is None:
            raise RuntimeError(f"Raster has no CRS: {candidate}")
        return GridMeta(
            name=candidate,
            width=int(ds.width),
            height=int(ds.height),
            transform=ds.transform,
            crs=ds.crs,
        )


def _grid_from_system_gdal(locator: str) -> GridMeta:
    cached = _GRID_META_CACHE.get(locator)
    if cached is not None:
        return cached

    cp = _run_checked(["gdalinfo", "-json", locator], purpose="gdalinfo HDF4 metadata read")
    try:
        info = json.loads(cp.stdout)
        size = info["size"]
        gt = info["geoTransform"]
        wkt = (info.get("coordinateSystem") or {}).get("wkt")
        if not wkt:
            raise KeyError("coordinateSystem.wkt")
        meta = GridMeta(
            name=locator,
            width=int(size[0]),
            height=int(size[1]),
            transform=Affine.from_gdal(*[float(x) for x in gt]),
            crs=rasterio.crs.CRS.from_wkt(wkt),
        )
    except Exception as e:
        raise RuntimeError(f"Could not parse gdalinfo JSON for {locator}: {e}") from e

    _GRID_META_CACHE[locator] = meta
    return meta


def get_grid_meta(locator: str, local_path: str) -> GridMeta:
    """Return grid metadata; use system GDAL when Rasterio lacks HDF4 support."""
    errors: List[str] = []
    for candidate in dataset_candidates(locator, local_path):
        try:
            meta = _grid_from_rasterio(candidate)
            _GRID_META_CACHE[candidate] = meta
            return meta
        except Exception as e:
            errors.append(f"Rasterio {candidate!r}: {e}")

        if is_hdf4_locator(candidate):
            try:
                return _grid_from_system_gdal(candidate)
            except Exception as e:
                errors.append(f"system GDAL {candidate!r}: {e}")

    raise RuntimeError("Unable to read raster grid metadata. Tried: " + " | ".join(errors))


def _read_hdf4_window_with_system_gdal(
    locator: str,
    window: Window,
    *,
    masked: bool = True,
) -> np.ma.MaskedArray:
    """
    Read only the requested HDF4 subdataset window using the system GDAL build.

    Rasterio wheels may omit HDF4 even when /usr/bin/gdalinfo has it.  We avoid
    full-raster conversion: gdal_translate extracts only the rice-area srcwin to
    a temporary GeoTIFF, Rasterio reads that TIFF, and it is immediately deleted.
    """
    xoff = int(window.col_off)
    yoff = int(window.row_off)
    xsize = int(window.width)
    ysize = int(window.height)
    if xsize <= 0 or ysize <= 0:
        raise RuntimeError(f"Invalid read window for {locator}: {window}")

    fd, tmp_name = tempfile.mkstemp(prefix="foss4g23_hdf4_", suffix=".tif")
    os.close(fd)
    tmp = Path(tmp_name)
    tmp.unlink(missing_ok=True)
    try:
        cmd = [
            "gdal_translate",
            "-q",
            "-of", "GTiff",
            "-srcwin", str(xoff), str(yoff), str(xsize), str(ysize),
            "-co", "TILED=YES",
            "-co", "COMPRESS=LZW",
            locator,
            str(tmp),
        ]
        _run_checked(cmd, purpose="gdal_translate HDF4 rice-window extraction")
        with rasterio.open(tmp) as ds:
            if ds.width != xsize or ds.height != ysize:
                raise RuntimeError(
                    f"System-GDAL HDF4 crop shape mismatch for {locator}: "
                    f"expected {xsize}x{ysize}, got {ds.width}x{ds.height}"
                )
            return ds.read(1, masked=masked)
    finally:
        tmp.unlink(missing_ok=True)


def read_dataset_window(
    locator: str,
    local_path: str,
    band_index: int,
    window: Window,
) -> np.ma.MaskedArray:
    """Read a native-grid window via Rasterio, falling back to system GDAL for HDF4."""
    errors: List[str] = []
    loc = str(locator or "").strip()

    # Prefer Rasterio when its GDAL build supports the format.
    for candidate in dataset_candidates(locator, local_path):
        try:
            with rasterio.open(candidate) as ds:
                return ds.read(band_index, window=window, masked=True)
        except Exception as e:
            errors.append(f"Rasterio {candidate!r}: {e}")

        # The Step-22 HDF4 subdataset locator already selects one SDS, so the
        # extracted temporary GeoTIFF is single-band regardless of band_index.
        if is_hdf4_locator(candidate):
            try:
                return _read_hdf4_window_with_system_gdal(candidate, window, masked=True)
            except Exception as e:
                errors.append(f"system GDAL {candidate!r}: {e}")

    raise RuntimeError("Unable to read raster window. Tried: " + " | ".join(errors))


def recipe_signature() -> dict:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "step22_recipe_sha256": sha256_file(STEP22_RECIPE_CSV),
        "analysis_start_year": ANALYSIS_START_YEAR,
        "analysis_end_year": ANALYSIS_END_YEAR,
        "months": sorted(MONTHS),
        "rice_fraction_min": RICE_FRACTION_MIN,
        "mod09_derived_indices": ["ndwi", "mndwi", "ndmi"],
        "hdf4_io_backend": "system_gdal_srcwin_fallback",
    }


def validate_or_create_signature(reset_work: bool) -> None:
    if reset_work:
        # Do not delete OUT_ROOT itself: the shell may already have opened
        # _manifests/23.log for stdout redirection before Python starts.
        # Remove only Step-23 state/data artifacts and preserve arbitrary logs.
        for d in (WORK_DIR, CACHE_DIR):
            if d.exists():
                shutil.rmtree(d)
        for f in (TILE_SUMMARY_CSV, STATE_SUMMARY_CSV, RUN_SUMMARY_JSON, ERROR_JSONL):
            f.unlink(missing_ok=True)
        ensure_dirs()

    current = recipe_signature()
    if RUN_SIGNATURE_JSON.exists():
        old = json.loads(RUN_SIGNATURE_JSON.read_text(encoding="utf-8"))
        if old != current:
            raise RuntimeError(
                "Step-23 work checkpoint was created with a different recipe/code signature. "
                "Run again with --reset-work to intentionally rebuild Step 23."
            )
    else:
        atomic_write_json(RUN_SIGNATURE_JSON, current)


# -----------------------------------------------------------------------------
# Step-22 recipe loading
# -----------------------------------------------------------------------------

def verify_step22_complete() -> None:
    if not STEP22_RECIPE_CSV.exists():
        raise FileNotFoundError(f"Missing Step-22 recipe CSV: {STEP22_RECIPE_CSV}")
    if not STEP22_AUDIT_CSV.exists():
        raise FileNotFoundError(f"Missing Step-22 audit CSV: {STEP22_AUDIT_CSV}")

    problems = 0
    with STEP22_AUDIT_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("status") or "").strip() != "OK":
                problems += 1
                if problems >= 5:
                    break
    if problems:
        raise RuntimeError(
            "Step 22 audit is not clean. Fix/rerun Step 22 before Step 23."
        )


def load_recipes(
    selected_products: Optional[set[str]] = None,
    start_year: int = ANALYSIS_START_YEAR,
    end_year: int = ANALYSIS_END_YEAR,
) -> Tuple[Dict[str, List[Recipe]], dict]:
    by_file: Dict[str, List[Recipe]] = defaultdict(list)
    counts = {
        "recipe_rows_total": 0,
        "recipe_rows_used": 0,
        "recipe_rows_skipped_no_cdl_year": 0,
        "recipe_rows_skipped_product": 0,
    }

    with STEP22_RECIPE_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts["recipe_rows_total"] += 1
            product = str(row.get("product") or "").strip()
            if selected_products and product not in selected_products:
                counts["recipe_rows_skipped_product"] += 1
                continue

            year = safe_int(row.get("year"), 0)
            month = safe_int(row.get("month"), 0)
            if year < start_year or year > end_year or month not in MONTHS:
                counts["recipe_rows_skipped_no_cdl_year"] += 1
                continue

            lp = str(row.get("local_path") or "").strip()
            if not lp:
                continue

            rec = Recipe(
                product=product,
                source_product=str(row.get("source_product") or ""),
                variable=str(row.get("canonical_variable") or ""),
                acquisition_date=str(row.get("acquisition_date") or ""),
                year=year,
                month=month,
                local_path=lp,
                dataset_locator=str(row.get("dataset_locator") or ""),
                band_index=safe_int(row.get("band_index"), 1),
                source_description=str(row.get("source_description") or ""),
                source_nodata=safe_float(row.get("source_nodata")),
                applied_scale=float(row.get("applied_scale") or 1.0),
                applied_offset=float(row.get("applied_offset") or 0.0),
                output_unit=str(row.get("output_unit") or ""),
                valid_range=safe_json_range(row.get("valid_range_json")),
                qa_key=(str(row.get("qa_key") or "").strip() or None),
                qa_dataset_locator=str(row.get("qa_dataset_locator") or ""),
                qa_band_index=safe_int(row.get("qa_band_index"), 1),
                qa_rule=str(row.get("qa_rule") or ""),
            )
            by_file[lp].append(rec)
            counts["recipe_rows_used"] += 1

    # Deterministic variable order within each raw file.
    for lp in by_file:
        by_file[lp].sort(key=lambda r: r.variable)

    return dict(by_file), counts


# -----------------------------------------------------------------------------
# Annual rice mask bounds and dynamic native-grid fraction cache
# -----------------------------------------------------------------------------

_MASK_POSITIVE_BOUNDS_CACHE: Dict[int, Tuple[float, float, float, float]] = {}
_FRACTION_WINDOW_CACHE: Dict[str, Optional[Window]] = {}
_FRACTION_LRU: "OrderedDict[str, FractionBundle]" = OrderedDict()


def annual_mask_path(year: int) -> Path:
    return STEP21_MASK_DIR / str(year) / f"rice_mask_{year}.tif"


def positive_data_window(ds, *, threshold: float = 0.0, nodata: Optional[float] = None) -> Optional[Window]:
    min_row = ds.height
    min_col = ds.width
    max_row = -1
    max_col = -1

    # block_windows works for tiled and striped rasters.
    for _, w in ds.block_windows(1):
        a = ds.read(1, window=w)
        good = np.isfinite(a) & (a > threshold)
        if nodata is not None:
            good &= a != nodata
        rr, cc = np.where(good)
        if rr.size == 0:
            continue
        r0 = int(w.row_off) + int(rr.min())
        r1 = int(w.row_off) + int(rr.max())
        c0 = int(w.col_off) + int(cc.min())
        c1 = int(w.col_off) + int(cc.max())
        min_row = min(min_row, r0)
        max_row = max(max_row, r1)
        min_col = min(min_col, c0)
        max_col = max(max_col, c1)

    if max_row < 0:
        return None
    return Window(min_col, min_row, max_col - min_col + 1, max_row - min_row + 1)


def rice_positive_bounds(year: int) -> Tuple[float, float, float, float]:
    if year in _MASK_POSITIVE_BOUNDS_CACHE:
        return _MASK_POSITIVE_BOUNDS_CACHE[year]

    p = annual_mask_path(year)
    if not p.exists():
        raise FileNotFoundError(
            f"Missing actual annual CDL rice mask for {year}: {p}. "
            "Step 23 does not use a proxy year."
        )

    with rasterio.open(p) as ds:
        w = positive_data_window(ds, threshold=0.0, nodata=STEP21_MASK_NODATA)
        if w is None:
            raise RuntimeError(f"No rice pixels found in annual mask: {p}")
        b = rasterio.windows.bounds(w, ds.transform)
        _MASK_POSITIVE_BOUNDS_CACHE[year] = b
        return b


def grid_hash(ds) -> str:
    payload = {
        "crs": ds.crs.to_wkt() if ds.crs else "",
        "transform": [float(x) for x in tuple(ds.transform)],
        "width": ds.width,
        "height": ds.height,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def dynamic_fraction_cache_path(product: str, year: int, ds) -> Path:
    return CACHE_DIR / product / str(year) / f"rice_fraction_{year}_{grid_hash(ds)}.tif"


def build_dynamic_fraction_cache(product: str, year: int, science_ds) -> Path:
    target = dynamic_fraction_cache_path(product, year, science_ds)
    if target.exists() and target.stat().st_size > 0:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(target) + ".part")
    tmp.unlink(missing_ok=True)

    mask_path = annual_mask_path(year)
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing Step-21 rice mask: {mask_path}")
    if science_ds.crs is None:
        raise RuntimeError(f"Science raster has no CRS: {science_ds.name}")

    with rasterio.open(mask_path) as src:
        if src.crs is None:
            raise RuntimeError(f"Rice mask has no CRS: {mask_path}")

        rb = rice_positive_bounds(year)
        rb_dst = transform_bounds(src.crs, science_ds.crs, *rb, densify_pts=21)
        float_w = from_bounds(*rb_dst, transform=science_ds.transform)
        # Expand by one target cell to avoid edge clipping from transformed bounds.
        float_w = Window(
            float_w.col_off - 1,
            float_w.row_off - 1,
            float_w.width + 2,
            float_w.height + 2,
        )
        w = clamp_window(float_w, science_ds.width, science_ds.height)
        if w is None:
            raise RuntimeError(
                f"Annual rice area does not overlap {product} science grid for {year}: {science_ds.name}"
            )

        dst_transform = window_transform(w, science_ds.transform)
        dst = np.full((int(w.height), int(w.width)), STEP21_FRACTION_NODATA, dtype="float32")

        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=STEP21_MASK_NODATA,
            dst_transform=dst_transform,
            dst_crs=science_ds.crs,
            dst_nodata=STEP21_FRACTION_NODATA,
            resampling=Resampling.average,
            init_dest_nodata=True,
            num_threads=2,
        )

        # Clamp tiny numerical noise then trim to actual positive rice fraction.
        valid = np.isfinite(dst) & (dst != STEP21_FRACTION_NODATA)
        dst[valid] = np.clip(dst[valid], 0.0, 1.0)
        dst_trim, w_trim = trim_to_positive(dst, w)
        if w_trim is None:
            raise RuntimeError(
                f"No positive annual rice fraction on {product} science grid for {year}: {science_ds.name}"
            )

        out_transform = window_transform(w_trim, science_ds.transform)
        profile = {
            "driver": "GTiff",
            "width": int(w_trim.width),
            "height": int(w_trim.height),
            "count": 1,
            "dtype": "float32",
            "crs": science_ds.crs,
            "transform": out_transform,
            "nodata": STEP21_FRACTION_NODATA,
            "compress": "DEFLATE",
            "predictor": 3,
            "BIGTIFF": "IF_SAFER",
        }
        with rasterio.open(tmp, "w", **profile) as out:
            out.write(dst_trim.astype("float32"), 1)
            out.update_tags(
                derived_product="rice_fraction",
                source_mask=str(mask_path),
                year=str(year),
                target_product=product,
                target_grid_hash=grid_hash(science_ds),
                target_row_off=str(int(w_trim.row_off)),
                target_col_off=str(int(w_trim.col_off)),
                target_full_width=str(science_ds.width),
                target_full_height=str(science_ds.height),
                raw_source_modified="false",
            )

    tmp.replace(target)
    print(f"[CACHE FRACTION] {target}")
    return target


# -----------------------------------------------------------------------------
# Pixel area and rice-fraction loading
# -----------------------------------------------------------------------------

def pixel_area_for_window(ds, window: Window) -> np.ndarray | float:
    """Return m² per pixel as scalar (projected) or per-row column vector."""
    if ds.crs is None:
        raise RuntimeError(f"Raster has no CRS: {ds.name}")

    pycrs = PyCRS.from_user_input(ds.crs)
    if pycrs.is_projected:
        # MODIS sinusoidal and most projected grids use metres.  Convert linear
        # axis units to metres when metadata provides another projected unit.
        unit_factor = 1.0
        try:
            axes = pycrs.axis_info
            if axes and axes[0].unit_conversion_factor:
                unit_factor = float(axes[0].unit_conversion_factor)
        except Exception:
            unit_factor = 1.0
        det = abs(ds.transform.a * ds.transform.e - ds.transform.b * ds.transform.d)
        return float(det * unit_factor * unit_factor)

    if not pycrs.is_geographic:
        raise RuntimeError(f"Unsupported non-projected/non-geographic CRS: {ds.crs}")

    # Geographic grid: calculate geodesic cell area for each row.  Grids used
    # here are north-up; reject rotated geographic grids rather than silently
    # approximating them.
    if abs(ds.transform.b) > 1e-12 or abs(ds.transform.d) > 1e-12:
        raise RuntimeError("Rotated geographic grids are not supported for geodesic area weighting")

    geod = pycrs.get_geod()
    col0 = int(window.col_off)
    row0 = int(window.row_off)
    width = int(window.width)
    height = int(window.height)

    x_left = ds.transform.c + col0 * ds.transform.a
    x_right = ds.transform.c + (col0 + 1) * ds.transform.a
    areas = np.empty((height, 1), dtype="float64")

    for i in range(height):
        r = row0 + i
        y_top = ds.transform.f + r * ds.transform.e
        y_bottom = ds.transform.f + (r + 1) * ds.transform.e
        area, _ = geod.polygon_area_perimeter(
            [x_left, x_right, x_right, x_left],
            [y_top, y_top, y_bottom, y_bottom],
        )
        areas[i, 0] = abs(float(area))
    return areas


def bundle_from_fraction_dataset(frac_ds, science_ds, science_window: Window, mode: str, source_path: str) -> FractionBundle:
    frac = frac_ds.read(1).astype("float32")
    valid = np.isfinite(frac) & (frac != STEP21_FRACTION_NODATA)
    frac[~valid] = 0.0
    frac = np.clip(frac, 0.0, 1.0)

    if frac.shape != (int(science_window.height), int(science_window.width)):
        raise RuntimeError(
            f"Fraction/science window shape mismatch: fraction={frac.shape}, science_window="
            f"({int(science_window.height)}, {int(science_window.width)})"
        )

    area = pixel_area_for_window(science_ds, science_window)
    weights = frac.astype("float64") * area
    total = float(np.sum(weights, dtype=np.float64))
    positive_cells = int(np.count_nonzero(frac > RICE_FRACTION_MIN))

    return FractionBundle(
        fraction=frac,
        science_window=science_window,
        transform=window_transform(science_window, science_ds.transform),
        crs=science_ds.crs,
        total_rice_area_m2=total,
        rice_positive_cells=positive_cells,
        source_fraction_path=source_path,
        fraction_mode=mode,
    )


def lru_get(key: str) -> Optional[FractionBundle]:
    b = _FRACTION_LRU.get(key)
    if b is not None:
        _FRACTION_LRU.move_to_end(key)
    return b


def lru_put(key: str, value: FractionBundle) -> None:
    _FRACTION_LRU[key] = value
    _FRACTION_LRU.move_to_end(key)
    while len(_FRACTION_LRU) > FRACTION_LRU_MAX:
        _FRACTION_LRU.popitem(last=False)


def modis_fraction_path(product: str, year: int, tile: str) -> Optional[Path]:
    res = PRODUCT_RESOLUTION.get(product)
    if not res:
        return None
    return STEP21_FRACTION_DIR / res / tile / str(year) / f"rice_fraction_{year}_{tile}_{res}.tif"


def load_fraction_bundle(product: str, year: int, tile: str, science_ds) -> Optional[FractionBundle]:
    # Standard MODIS: prefer Step-21 precomputed native-grid fraction.
    fp = modis_fraction_path(product, year, tile)
    if fp is not None and fp.exists():
        key = f"step21|{fp}|{grid_hash(science_ds)}"
        cached = lru_get(key)
        if cached is not None:
            return cached

        with rasterio.open(fp) as frac_ds:
            if grids_match(frac_ds, science_ds):
                w = _FRACTION_WINDOW_CACHE.get(str(fp))
                if str(fp) not in _FRACTION_WINDOW_CACHE:
                    w = positive_data_window(
                        frac_ds,
                        threshold=RICE_FRACTION_MIN,
                        nodata=STEP21_FRACTION_NODATA,
                    )
                    _FRACTION_WINDOW_CACHE[str(fp)] = w
                if w is None:
                    return None

                # Re-open on the positive crop so bundle_from_fraction_dataset
                # can read the fraction crop only, not the full tile.
                a = frac_ds.read(1, window=w).astype("float32")
                valid = np.isfinite(a) & (a != STEP21_FRACTION_NODATA)
                a[~valid] = 0.0
                a = np.clip(a, 0.0, 1.0)
                area = pixel_area_for_window(science_ds, w)
                total = float(np.sum(a.astype("float64") * area, dtype=np.float64))
                bundle = FractionBundle(
                    fraction=a,
                    science_window=w,
                    transform=window_transform(w, science_ds.transform),
                    crs=science_ds.crs,
                    total_rice_area_m2=total,
                    rice_positive_cells=int(np.count_nonzero(a > RICE_FRACTION_MIN)),
                    source_fraction_path=str(fp),
                    fraction_mode="step21_native_modis",
                )
                lru_put(key, bundle)
                return bundle

        # Standard product but native grids are unexpectedly not identical.
        # Do not resample the science product.  Rebuild annual rice fraction on
        # THIS exact science grid from the native 30 m annual CDL mask.
        print(
            f"[GRID FALLBACK] {product} {year} {tile}: Step-21 fraction grid differs; "
            "building fraction on exact science grid"
        )

    # NTSG, or a standard product fallback.
    cache_path = build_dynamic_fraction_cache(product, year, science_ds)
    key = f"dynamic|{cache_path}|{grid_hash(science_ds)}"
    cached = lru_get(key)
    if cached is not None:
        return cached

    with rasterio.open(cache_path) as frac_ds:
        tags = frac_ds.tags()
        row_off = int(tags["target_row_off"])
        col_off = int(tags["target_col_off"])
        w = Window(col_off, row_off, frac_ds.width, frac_ds.height)
        bundle = bundle_from_fraction_dataset(
            frac_ds,
            science_ds,
            w,
            mode="dynamic_native_grid",
            source_path=str(cache_path),
        )
        lru_put(key, bundle)
        return bundle


# -----------------------------------------------------------------------------
# Science / QA pixel interpretation
# -----------------------------------------------------------------------------

def read_raw_window(recipe: Recipe, window: Window, reference_ds=None) -> Tuple[np.ndarray, np.ndarray]:
    ds_meta = get_grid_meta(recipe.dataset_locator, recipe.local_path)
    if reference_ds is not None and not grids_match(ds_meta, reference_ds):
        raise RuntimeError(
            f"Science variable grids differ inside one raw file: {Path(recipe.local_path).name} "
            f"variable={recipe.variable}"
        )
    if int(window.col_off + window.width) > ds_meta.width or int(window.row_off + window.height) > ds_meta.height:
        raise RuntimeError(f"Science window exceeds raster extent: {recipe.dataset_locator or recipe.local_path}")

    ma = read_dataset_window(
        recipe.dataset_locator,
        recipe.local_path,
        recipe.band_index,
        window,
    )
    raw = np.asarray(ma.data)
    valid = ~np.ma.getmaskarray(ma)
    if recipe.source_nodata is not None:
        valid &= raw != recipe.source_nodata

    val = raw.astype("float32") * np.float32(recipe.applied_scale) + np.float32(recipe.applied_offset)
    valid &= np.isfinite(val)
    if recipe.valid_range is not None:
        lo, hi = recipe.valid_range
        valid &= val >= lo
        valid &= val <= hi
    return val, valid


def qa_mask_for_recipe(recipe: Recipe, window: Window, reference_ds) -> np.ndarray:
    if not recipe.qa_key:
        return np.ones((int(window.height), int(window.width)), dtype=bool)

    # MOD15 in the established project pipeline retains FparLai_QC for
    # diagnostics, while science values are filtered by physical range.
    if recipe.qa_key == "fpar_lai_qc":
        return np.ones((int(window.height), int(window.width)), dtype=bool)

    if not recipe.qa_dataset_locator:
        raise RuntimeError(f"QA locator missing for {recipe.product}/{recipe.variable}")

    qa_meta = get_grid_meta(recipe.qa_dataset_locator, recipe.local_path)
    if not grids_match(qa_meta, reference_ds):
        raise RuntimeError(
            f"Science/QA grid mismatch for {recipe.local_path}: "
            f"science={reference_ds.name}, qa={qa_meta.name}"
        )

    ma = read_dataset_window(
        recipe.qa_dataset_locator,
        recipe.local_path,
        recipe.qa_band_index,
        window,
    )
    q = np.asarray(ma.data)
    valid = ~np.ma.getmaskarray(ma)
    q64 = q.astype("int64", copy=False)

    if recipe.qa_key == "summary_qa":
        good = q64 <= 1
    elif recipe.qa_key == "state_qa":
        cloud_state = q64 & 3
        cloud_shadow = (q64 >> 2) & 1
        good = (cloud_state == 0) & (cloud_shadow == 0)
    elif recipe.qa_key in {"qc_day", "qc_night"}:
        good = (q64 & 3) <= 1
    elif recipe.qa_key in {"et_qc", "psn_qc"}:
        good = (q64 & 1) == 0
    else:
        raise RuntimeError(f"Unsupported QA key in Step-23 implementation: {recipe.qa_key}")

    return valid & good


def weighted_stats(
    values: np.ndarray,
    value_valid: np.ndarray,
    bundle: FractionBundle,
    pixel_area: np.ndarray | float,
) -> dict:
    frac = bundle.fraction
    rice = np.isfinite(frac) & (frac > RICE_FRACTION_MIN)
    valid = rice & value_valid & np.isfinite(values)

    if np.isscalar(pixel_area):
        all_weights = frac.astype("float64") * float(pixel_area)
    else:
        all_weights = frac.astype("float64") * pixel_area

    total_w = float(np.sum(all_weights[rice], dtype=np.float64)) if np.any(rice) else 0.0
    if not np.any(valid):
        return {
            "rice_cells": int(np.count_nonzero(rice)),
            "valid_cells": 0,
            "rice_area_m2_total": total_w,
            "valid_weight_m2": 0.0,
            "coverage_fraction": 0.0,
            "sum_wx": 0.0,
            "sum_wx2": 0.0,
            "weighted_mean": None,
            "weighted_std": None,
            "value_min": None,
            "value_max": None,
        }

    w = all_weights[valid]
    x = values[valid].astype("float64")
    den = float(np.sum(w, dtype=np.float64))
    num = float(np.sum(w * x, dtype=np.float64))
    sum2 = float(np.sum(w * x * x, dtype=np.float64))
    mean = num / den if den > 0 else None
    variance = max(0.0, sum2 / den - mean * mean) if den > 0 and mean is not None else None

    return {
        "rice_cells": int(np.count_nonzero(rice)),
        "valid_cells": int(np.count_nonzero(valid)),
        "rice_area_m2_total": total_w,
        "valid_weight_m2": den,
        "coverage_fraction": (den / total_w) if total_w > 0 else None,
        "sum_wx": num,
        "sum_wx2": sum2,
        "weighted_mean": mean,
        "weighted_std": math.sqrt(variance) if variance is not None else None,
        "value_min": float(np.min(x)),
        "value_max": float(np.max(x)),
    }


def normalized_difference(
    a: np.ndarray,
    a_valid: np.ndarray,
    b: np.ndarray,
    b_valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    denom = a.astype("float64") + b.astype("float64")
    valid = a_valid & b_valid & np.isfinite(denom) & (np.abs(denom) > 1e-12)
    out = np.full(a.shape, np.nan, dtype="float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        out[valid] = ((a[valid].astype("float64") - b[valid].astype("float64")) / denom[valid]).astype("float32")
    valid &= np.isfinite(out)
    return out, valid


# -----------------------------------------------------------------------------
# Observation records
# -----------------------------------------------------------------------------

def observation_record(
    recipe: Recipe,
    tile: str,
    variable: str,
    output_unit: str,
    stats: dict,
    bundle: FractionBundle,
    source_kind: str = "step22_science_variable",
) -> dict:
    record_key_text = "|".join([
        recipe.product,
        variable,
        recipe.acquisition_date,
        recipe.local_path,
        tile,
    ])
    return {
        "record_key": hashlib.sha1(record_key_text.encode("utf-8")).hexdigest(),
        "recorded_utc": utc_now(),
        "product": recipe.product,
        "source_product": recipe.source_product,
        "variable": variable,
        "output_unit": output_unit,
        "source_kind": source_kind,
        "acquisition_date": recipe.acquisition_date,
        "year": recipe.year,
        "month": recipe.month,
        "tile": tile,
        "nominal_m": PRODUCT_NOMINAL_M.get(recipe.product),
        "cadence": PRODUCT_CADENCE.get(recipe.product, ""),
        "temporal_aggregation": "sum" if variable in SUM_TEMPORAL_VARIABLES else "mean",
        "local_path": recipe.local_path,
        "rice_fraction_source": bundle.source_fraction_path,
        "rice_fraction_mode": bundle.fraction_mode,
        "rice_cells": stats["rice_cells"],
        "valid_cells": stats["valid_cells"],
        "rice_area_m2_total": stats["rice_area_m2_total"],
        "valid_weight_m2": stats["valid_weight_m2"],
        "coverage_fraction": stats["coverage_fraction"],
        "sum_wx": stats["sum_wx"],
        "sum_wx2": stats["sum_wx2"],
        "weighted_mean": stats["weighted_mean"],
        "weighted_std": stats["weighted_std"],
        "value_min": stats["value_min"],
        "value_max": stats["value_max"],
    }


def process_raw_file(recipes: List[Recipe]) -> List[dict]:
    if not recipes:
        return []
    first = recipes[0]
    tile = tile_from_path(first.local_path)

    if not Path(first.local_path).is_file() or Path(first.local_path).stat().st_size <= 0:
        raise FileNotFoundError(f"RAW missing/empty: {first.local_path}")

    # First science variable defines the native grid for this raw product file.
    # Grid metadata are read through Rasterio when possible, otherwise through
    # the system GDAL build (needed when the Rasterio wheel omits HDF4).
    ref_ds = get_grid_meta(first.dataset_locator, first.local_path)
    if ref_ds.crs is None:
        raise RuntimeError(f"Science raster has no CRS: {ref_ds.name}")
    bundle = load_fraction_bundle(first.product, first.year, tile, ref_ds)
    if bundle is None or bundle.rice_positive_cells == 0:
        print(f"[NO RICE] {first.product} {first.year} {tile} {Path(first.local_path).name}")
        return []
    pixel_area = pixel_area_for_window(ref_ds, bundle.science_window)

    output_rows: List[dict] = []
    mod09_keep: Dict[str, Tuple[np.ndarray, np.ndarray, Recipe]] = {}
    qa_cache: Dict[Tuple[str, str, int], np.ndarray] = {}

    for rec in recipes:
        values, source_valid = read_raw_window(rec, bundle.science_window, ref_ds)
        qa_cache_key = (rec.qa_key or "", rec.qa_dataset_locator or "", rec.qa_band_index)
        if qa_cache_key not in qa_cache:
            qa_cache[qa_cache_key] = qa_mask_for_recipe(rec, bundle.science_window, ref_ds)
        qa_good = qa_cache[qa_cache_key]
        final_valid = source_valid & qa_good
        stats = weighted_stats(values, final_valid, bundle, pixel_area)
        output_rows.append(
            observation_record(
                rec,
                tile,
                rec.variable,
                rec.output_unit,
                stats,
                bundle,
            )
        )

        if rec.product == "mod09_sr" and rec.variable in {"green", "nir", "swir1"}:
            mod09_keep[rec.variable] = (values, final_valid, rec)

    # Derived water/wetness indices are spatially nonlinear, so derive them
    # per pixel here, not later from statewide band means.
    if first.product == "mod09_sr" and {"green", "nir", "swir1"}.issubset(mod09_keep):
        green, green_v, green_r = mod09_keep["green"]
        nir, nir_v, _ = mod09_keep["nir"]
        swir1, swir1_v, _ = mod09_keep["swir1"]

        derived = {
            "ndwi": normalized_difference(green, green_v, nir, nir_v),
            "mndwi": normalized_difference(green, green_v, swir1, swir1_v),
            "ndmi": normalized_difference(nir, nir_v, swir1, swir1_v),
        }
        for name, (arr, valid) in derived.items():
            stats = weighted_stats(arr, valid, bundle, pixel_area)
            output_rows.append(
                observation_record(
                    green_r,
                    tile,
                    name,
                    "1",
                    stats,
                    bundle,
                    source_kind="derived_from_qa_filtered_mod09_reflectance",
                )
            )

    return output_rows


# -----------------------------------------------------------------------------
# Consolidation and statewide aggregation
# -----------------------------------------------------------------------------

TILE_FIELDS = [
    "record_key", "recorded_utc",
    "product", "source_product", "variable", "output_unit", "source_kind",
    "acquisition_date", "year", "month", "tile", "nominal_m", "cadence",
    "temporal_aggregation", "local_path",
    "rice_fraction_source", "rice_fraction_mode",
    "rice_cells", "valid_cells",
    "rice_area_m2_total", "valid_weight_m2", "coverage_fraction",
    "sum_wx", "sum_wx2", "weighted_mean", "weighted_std",
    "value_min", "value_max",
]

STATE_FIELDS = [
    "product", "source_product", "variable", "output_unit", "source_kind",
    "acquisition_date", "year", "month", "nominal_m", "cadence",
    "temporal_aggregation", "tiles", "source_files",
    "rice_cells", "valid_cells",
    "rice_area_m2_total", "valid_weight_m2", "coverage_fraction",
    "sum_wx", "sum_wx2", "weighted_mean", "weighted_std",
    "value_min", "value_max",
]


def load_observations_dedup(path: Path) -> Tuple[List[dict], int]:
    dedup: Dict[str, dict] = {}
    malformed = 0
    if not path.exists():
        return [], 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = str(row.get("record_key") or "")
                if key:
                    dedup[key] = row
            except Exception:
                malformed += 1
    rows = sorted(
        dedup.values(),
        key=lambda r: (
            str(r.get("product")),
            int(r.get("year") or 0),
            int(r.get("month") or 0),
            str(r.get("acquisition_date")),
            str(r.get("variable")),
            str(r.get("tile")),
            str(r.get("local_path")),
        ),
    )
    return rows, malformed


def aggregate_statewide(tile_rows: List[dict]) -> List[dict]:
    groups: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for r in tile_rows:
        key = (
            str(r["product"]),
            str(r["variable"]),
            str(r["acquisition_date"]),
        )
        groups[key].append(r)

    out: List[dict] = []
    for key, rows in sorted(groups.items()):
        first = rows[0]
        rice_area = sum(float(r.get("rice_area_m2_total") or 0.0) for r in rows)
        den = sum(float(r.get("valid_weight_m2") or 0.0) for r in rows)
        num = sum(float(r.get("sum_wx") or 0.0) for r in rows)
        sum2 = sum(float(r.get("sum_wx2") or 0.0) for r in rows)
        mean = num / den if den > 0 else None
        var = max(0.0, sum2 / den - mean * mean) if den > 0 and mean is not None else None
        mins = [float(r["value_min"]) for r in rows if r.get("value_min") is not None]
        maxs = [float(r["value_max"]) for r in rows if r.get("value_max") is not None]

        out.append({
            "product": first["product"],
            "source_product": first["source_product"],
            "variable": first["variable"],
            "output_unit": first["output_unit"],
            "source_kind": first["source_kind"],
            "acquisition_date": first["acquisition_date"],
            "year": int(first["year"]),
            "month": int(first["month"]),
            "nominal_m": first["nominal_m"],
            "cadence": first["cadence"],
            "temporal_aggregation": first["temporal_aggregation"],
            "tiles": len({str(r.get("tile")) for r in rows}),
            "source_files": len({str(r.get("local_path")) for r in rows}),
            "rice_cells": sum(int(r.get("rice_cells") or 0) for r in rows),
            "valid_cells": sum(int(r.get("valid_cells") or 0) for r in rows),
            "rice_area_m2_total": rice_area,
            "valid_weight_m2": den,
            "coverage_fraction": (den / rice_area) if rice_area > 0 else None,
            "sum_wx": num,
            "sum_wx2": sum2,
            "weighted_mean": mean,
            "weighted_std": math.sqrt(var) if var is not None else None,
            "value_min": min(mins) if mins else None,
            "value_max": max(maxs) if maxs else None,
        })

    return out


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 23: native-grid rice-area weighted spatial summaries for MODIS/NTSG."
    )
    p.add_argument(
        "--products",
        default="",
        help="Optional comma-separated product keys, e.g. mod13_vi,mod09_sr. Default: all Step-22 products.",
    )
    p.add_argument("--start-year", type=int, default=ANALYSIS_START_YEAR)
    p.add_argument("--end-year", type=int, default=ANALYSIS_END_YEAR)
    p.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="Optional smoke-test limit on number of unfinished RAW files. 0 = all.",
    )
    p.add_argument(
        "--reset-work",
        action="store_true",
        help="Delete Step-23 outputs/checkpoints and rebuild intentionally.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    verify_step22_complete()
    validate_or_create_signature(args.reset_work)

    if args.start_year < ANALYSIS_START_YEAR:
        raise ValueError(
            f"Step 23 cannot start before {ANALYSIS_START_YEAR}: no actual annual California CDL mask is available in Step 21."
        )
    if args.end_year > ANALYSIS_END_YEAR:
        raise ValueError(f"Step 23 end year cannot exceed {ANALYSIS_END_YEAR} in this project run.")

    selected_products = {
        x.strip() for x in args.products.split(",") if x.strip()
    } or None

    by_file, recipe_counts = load_recipes(
        selected_products=selected_products,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    completed = load_completed(COMPLETED_TXT)

    all_files = sorted(by_file)
    pending = [lp for lp in all_files if lp not in completed]
    if args.limit_files > 0:
        pending = pending[: args.limit_files]

    print("=" * 78)
    print("STEP 23 — NATIVE-GRID RICE-AREA SPATIAL SUMMARY")
    print("Project              :", PROJECT_DIR)
    print("Analysis years       :", args.start_year, "-", args.end_year)
    print("Season               : Apr-Nov")
    print("Products             :", ", ".join(sorted(selected_products)) if selected_products else "ALL")
    print("Step-22 recipe rows  :", recipe_counts["recipe_rows_total"])
    print("Recipe rows used     :", recipe_counts["recipe_rows_used"])
    print("RAW files targeted   :", len(all_files))
    print("Already completed    :", len(completed.intersection(all_files)))
    print("Pending this run     :", len(pending))
    print("RAW modified         : NO")
    print("Full raster duplicate: NO")
    print("Output               :", OUT_ROOT)
    print("=" * 78)

    errors_this_run = 0
    files_completed_this_run = 0
    observation_rows_this_run = 0

    for idx, lp in enumerate(pending, start=1):
        recipes = by_file[lp]
        first = recipes[0]
        try:
            rows = process_raw_file(recipes)
            for row in rows:
                append_jsonl(OBS_JSONL, row)
                observation_rows_this_run += 1
            append_completed(COMPLETED_TXT, lp)
            completed.add(lp)
            files_completed_this_run += 1

            if idx <= 10 or idx % 100 == 0 or idx == len(pending):
                print(
                    f"[{idx}/{len(pending)}] OK {first.product} {first.acquisition_date} "
                    f"{tile_from_path(lp)} {Path(lp).name} rows={len(rows)}"
                )
        except KeyboardInterrupt:
            print("\nInterrupted by user. Completed-file checkpoints are preserved.")
            return 130
        except Exception as e:
            errors_this_run += 1
            err = {
                "recorded_utc": utc_now(),
                "local_path": lp,
                "product": first.product,
                "acquisition_date": first.acquisition_date,
                "year": first.year,
                "month": first.month,
                "error_type": type(e).__name__,
                "error": str(e),
            }
            append_jsonl(ERROR_JSONL, err)
            print(
                f"[{idx}/{len(pending)}] ERROR {first.product} {Path(lp).name}: "
                f"{type(e).__name__}: {e}"
            )

    # Consolidate whatever valid work exists.  This is useful even after a
    # partial run; final success still requires every target RAW file complete.
    tile_rows, malformed_jsonl = load_observations_dedup(OBS_JSONL)
    atomic_write_csv(TILE_SUMMARY_CSV, tile_rows, TILE_FIELDS)
    state_rows = aggregate_statewide(tile_rows)
    atomic_write_csv(STATE_SUMMARY_CSV, state_rows, STATE_FIELDS)

    completed_now = load_completed(COMPLETED_TXT)
    unfinished = [lp for lp in all_files if lp not in completed_now]

    summary = {
        "generated_utc": utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "analysis_years": [args.start_year, args.end_year],
        "months": sorted(MONTHS),
        "selected_products": sorted(selected_products) if selected_products else "ALL",
        "raw_files_targeted": len(all_files),
        "raw_files_completed": len(completed_now.intersection(all_files)),
        "raw_files_unfinished": len(unfinished),
        "files_completed_this_run": files_completed_this_run,
        "errors_this_run": errors_this_run,
        "tile_observation_rows": len(tile_rows),
        "statewide_observation_rows": len(state_rows),
        "malformed_work_jsonl_lines_ignored": malformed_jsonl,
        "raw_modified": False,
        "full_raster_duplication": False,
        "step24_ready": len(unfinished) == 0 and errors_this_run == 0,
    }
    atomic_write_json(RUN_SUMMARY_JSON, summary)

    print("\n" + "=" * 78)
    print("STEP 23 SUMMARY")
    print("RAW files targeted       :", len(all_files))
    print("RAW files completed      :", len(completed_now.intersection(all_files)))
    print("RAW files unfinished     :", len(unfinished))
    print("Completed this run       :", files_completed_this_run)
    print("Errors this run          :", errors_this_run)
    print("Tile observation rows    :", len(tile_rows))
    print("Statewide observation rows:", len(state_rows))
    print("Tile CSV                 :", TILE_SUMMARY_CSV)
    print("Statewide CSV            :", STATE_SUMMARY_CSV)
    print("Run summary              :", RUN_SUMMARY_JSON)
    print("RAW modified             : NO")
    print("Full raster duplication  : NO")
    print("=" * 78)

    if unfinished:
        print("\nSTEP 23 INCOMPLETE — rerun the same command; completed RAW files will be skipped.")
        print("First unfinished files:")
        for p in unfinished[:20]:
            print(" ", p)
        return 2

    if errors_this_run:
        print("\nSTEP 23 INCOMPLETE — processing errors occurred; inspect:", ERROR_JSONL)
        return 2

    print("\n✓ STEP 23 COMPLETE — native-grid rice-area spatial summaries are ready for Step 24")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
