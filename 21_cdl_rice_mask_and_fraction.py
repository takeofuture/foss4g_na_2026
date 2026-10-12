#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21_cdl_rice_mask_and_fraction.py

California Rice monitoring preprocessing step 21.

Purpose
-------
1) Preserve the RAW CDL GeoTIFFs unchanged.
2) Build an annual binary rice mask from USDA CDL class 3 (Rice), 2007-2025.
3) Build annual rice-fraction rasters on the *actual native MODIS tile grids*
   already recorded in data/_manifests/spatial_manifest.csv:
      - 250 m grid: MOD13 (mod13_vi)
      - 500 m grid: MOD09 (mod09_sr)
      - 1 km grid:  MOD11 (mod11_lst)
4) Write QC/summary manifests for later spatial alignment and feature extraction.

The rice-fraction products are spatial weights, not new classifications.
Each value is in [0, 1] and represents the average of the binary 30 m rice mask
contributing to the target MODIS grid cell during reprojection.

RAW data are never modified.

Expected project layout
-----------------------
/opt/foss4g_2026/
  21_cdl_rice_mask_and_fraction.py
  data/
    usda_cdl_california/<year>/cdl_california_<year>.tif
    _manifests/spatial_manifest.csv
  processed/
    21_cdl_rice/
      native_30m/<year>/rice_mask_<year>.tif
      modis_fraction/250m/<tile>/<year>/rice_fraction_<year>_<tile>_250m.tif
      modis_fraction/500m/<tile>/<year>/rice_fraction_<year>_<tile>_500m.tif
      modis_fraction/1000m/<tile>/<year>/rice_fraction_<year>_<tile>_1000m.tif
      _manifests/
        rice_mask_manifest.csv
        rice_fraction_manifest.csv
        rice_area_by_year.csv

Resume behavior
---------------
- Existing, valid outputs are reused and skipped.
- Temporary *.part.tif files are removed before retry.
- New rasters are written to a temporary file, validated, then atomically renamed.
- Use --force to rebuild outputs.

Example
-------
nohup python -u 21_cdl_rice_mask_and_fraction.py \
  >> processed/21_cdl_rice/_manifests/21_cdl_rice.log 2>&1 &
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR

DATA_DIR = PROJECT_DIR / "data"
CDL_DIR = DATA_DIR / "usda_cdl_california"
SPATIAL_MANIFEST = DATA_DIR / "_manifests" / "spatial_manifest.csv"

PROCESSED_ROOT = PROJECT_DIR / "processed" / "21_cdl_rice"
NATIVE_MASK_DIR = PROCESSED_ROOT / "native_30m"
FRACTION_DIR = PROCESSED_ROOT / "modis_fraction"
MANIFEST_DIR = PROCESSED_ROOT / "_manifests"

MASK_MANIFEST_CSV = MANIFEST_DIR / "rice_mask_manifest.csv"
FRACTION_MANIFEST_CSV = MANIFEST_DIR / "rice_fraction_manifest.csv"
AREA_SUMMARY_CSV = MANIFEST_DIR / "rice_area_by_year.csv"

START_YEAR = 2007
END_YEAR = 2025
RICE_CLASS = 3
MASK_NODATA = 255
FRACTION_NODATA = -9999.0

# Use the actual native MODIS grids recorded during step 20.
# One source product is enough to define each stable tile grid/resolution.
MODIS_TEMPLATE_PRODUCTS = {
    "250m": "mod13_vi",
    "500m": "mod09_sr",
    "1000m": "mod11_lst",
}

# Output creation options. DEFLATE is broadly supported by GDAL/rasterio.
MASK_PROFILE_UPDATES = {
    "driver": "GTiff",
    "dtype": "uint8",
    "count": 1,
    "nodata": MASK_NODATA,
    "compress": "DEFLATE",
    "predictor": 2,
    "tiled": True,
    "blockxsize": 512,
    "blockysize": 512,
    "BIGTIFF": "IF_SAFER",
}

FRACTION_PROFILE_BASE = {
    "driver": "GTiff",
    "dtype": "float32",
    "count": 1,
    "nodata": FRACTION_NODATA,
    "compress": "DEFLATE",
    "predictor": 3,
    "tiled": True,
    "blockxsize": 512,
    "blockysize": 512,
    "BIGTIFF": "IF_SAFER",
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    for p in (PROCESSED_ROOT, NATIVE_MASK_DIR, FRACTION_DIR, MANIFEST_DIR):
        p.mkdir(parents=True, exist_ok=True)


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def raster_ok(
    path: Path,
    *,
    expected_width: Optional[int] = None,
    expected_height: Optional[int] = None,
    expected_crs: Optional[str] = None,
) -> bool:
    path = Path(path)
    if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
        return False

    try:
        with rasterio.open(path) as src:
            if src.count < 1 or src.width <= 0 or src.height <= 0:
                return False
            if expected_width is not None and src.width != expected_width:
                return False
            if expected_height is not None and src.height != expected_height:
                return False
            if expected_crs is not None and src.crs is not None:
                # Compare WKT-normalized CRS through rasterio where possible.
                if rasterio.crs.CRS.from_wkt(expected_crs) != src.crs:
                    return False
    except Exception:
        return False

    return True


def atomic_temp_path(final_path: Path) -> Path:
    final_path = Path(final_path)
    # Keep .tif extension so GDAL driver detection is deterministic.
    return final_path.with_name(final_path.stem + ".part.tif")


def remove_stale_temp(final_path: Path) -> Path:
    tmp = atomic_temp_path(final_path)
    if tmp.exists():
        print("  removing stale partial:", tmp)
        tmp.unlink()
    return tmp


def append_csv_row(path: Path, fieldnames: List[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_csv_replace(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def safe_json_loads(value: str):
    try:
        return json.loads(value)
    except Exception:
        return None


def windows_for_dataset(src: rasterio.io.DatasetReader) -> Iterator[Tuple[Tuple[int, int], Window]]:
    # Prefer the source's own internal block windows.
    yield from src.block_windows(1)


def bounds_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    al, ab, ar, at = a
    bl, bb, br, bt = b
    return (al < br) and (ar > bl) and (ab < bt) and (at > bb)


# -----------------------------------------------------------------------------
# MODIS native grid templates from spatial_manifest.csv
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GridTemplate:
    resolution_label: str
    product: str
    tile: str
    crs_wkt: str
    transform: Affine
    width: int
    height: int
    reference_local_path: str
    reference_dataset_locator: str

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return array_bounds(self.height, self.width, self.transform)


def load_modis_grid_templates() -> Dict[str, List[GridTemplate]]:
    if not SPATIAL_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing spatial manifest: {SPATIAL_MANIFEST}. "
            "Run the step-20 downloader/georeference workflow first."
        )

    wanted = {product: res for res, product in MODIS_TEMPLATE_PRODUCTS.items()}
    found: Dict[str, Dict[str, GridTemplate]] = {res: {} for res in MODIS_TEMPLATE_PRODUCTS}

    # Stream the CSV rather than pandas-reading the multi-GB manifest.
    with SPATIAL_MANIFEST.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            product = (row.get("product") or "").strip()
            if product not in wanted:
                continue
            if not truthy(row.get("primary_grid")):
                continue

            tile = (row.get("tile") or "").strip()
            if not tile:
                continue

            crs_wkt = row.get("native_crs_wkt") or ""
            gt = safe_json_loads(row.get("geo_transform") or "")

            try:
                width = int(float(row.get("width") or 0))
                height = int(float(row.get("height") or 0))
            except Exception:
                continue

            if not crs_wkt or not gt or len(gt) != 6 or width <= 0 or height <= 0:
                continue

            resolution_label = wanted[product]
            if tile in found[resolution_label]:
                continue

            found[resolution_label][tile] = GridTemplate(
                resolution_label=resolution_label,
                product=product,
                tile=tile,
                crs_wkt=crs_wkt,
                transform=Affine.from_gdal(*[float(x) for x in gt]),
                width=width,
                height=height,
                reference_local_path=row.get("local_path") or "",
                reference_dataset_locator=row.get("dataset_locator") or "",
            )

    result = {
        res: sorted(tile_map.values(), key=lambda x: x.tile)
        for res, tile_map in found.items()
    }

    for res, templates in result.items():
        product = MODIS_TEMPLATE_PRODUCTS[res]
        print(f"MODIS template grids {res} ({product}): {len(templates)} tile(s)")
        for t in templates:
            print(f"  {t.tile}: {t.width} x {t.height}")
        if not templates:
            raise RuntimeError(
                f"No primary-grid rows found for {product} in {SPATIAL_MANIFEST}"
            )

    return result


# -----------------------------------------------------------------------------
# Step 21A: native CDL 30 m binary rice mask
# -----------------------------------------------------------------------------


MASK_MANIFEST_FIELDS = [
    "recorded_utc",
    "year",
    "source_cdl",
    "output_mask",
    "rice_class",
    "width",
    "height",
    "crs",
    "transform",
    "source_nodata",
    "mask_nodata",
    "valid_pixels",
    "rice_pixels",
    "rice_fraction_of_valid_pixels",
    "pixel_area_native_units2",
    "rice_area_hectares_if_meters",
    "rice_area_acres_if_meters",
]


def build_native_rice_mask(year: int, *, force: bool = False) -> Dict[str, object]:
    source = CDL_DIR / str(year) / f"cdl_california_{year}.tif"
    target = NATIVE_MASK_DIR / str(year) / f"rice_mask_{year}.tif"
    target.parent.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(f"Missing CDL source for {year}: {source}")

    with rasterio.open(source) as src:
        if src.count < 1:
            raise RuntimeError(f"No raster band in CDL source: {source}")

        if not force and raster_ok(
            target,
            expected_width=src.width,
            expected_height=src.height,
            expected_crs=src.crs.to_wkt() if src.crs else None,
        ):
            print(f"[EXISTS MASK] {target}")
        else:
            tmp = remove_stale_temp(target)
            if force and target.exists():
                target.unlink()

            profile = src.profile.copy()
            profile.update(MASK_PROFILE_UPDATES)
            profile.update(width=src.width, height=src.height, transform=src.transform, crs=src.crs)

            print(f"[BUILD MASK] {year} <- {source.name}")
            with rasterio.open(tmp, "w", **profile) as dst:
                for _, window in windows_for_dataset(src):
                    data = src.read(1, window=window)
                    valid = src.read_masks(1, window=window) > 0

                    out = np.full(data.shape, MASK_NODATA, dtype=np.uint8)
                    out[valid] = 0
                    out[valid & (data == RICE_CLASS)] = 1
                    dst.write(out, 1, window=window)

                dst.update_tags(
                    source="USDA_NASS_CDL",
                    source_file=str(source),
                    year=str(year),
                    derived_product="binary_rice_mask",
                    rice_class=str(RICE_CLASS),
                    value_0="valid_non_rice",
                    value_1="rice",
                    value_255="nodata",
                    raw_source_modified="false",
                )

            if not raster_ok(
                tmp,
                expected_width=src.width,
                expected_height=src.height,
                expected_crs=src.crs.to_wkt() if src.crs else None,
            ):
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"Validation failed for temporary rice mask: {tmp}")

            tmp.replace(target)
            print(f"[WROTE MASK] {target}")

    # Compute summary from the compact derived mask, not the original CDL.
    valid_pixels = 0
    rice_pixels = 0
    with rasterio.open(target) as mask_src:
        for _, window in windows_for_dataset(mask_src):
            a = mask_src.read(1, window=window)
            valid_pixels += int(np.count_nonzero(a != MASK_NODATA))
            rice_pixels += int(np.count_nonzero(a == 1))

        transform = mask_src.transform
        pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
        crs_string = mask_src.crs.to_string() if mask_src.crs else ""
        crs_wkt = mask_src.crs.to_wkt() if mask_src.crs else ""
        source_nodata = None
        with rasterio.open(source) as src:
            source_nodata = src.nodata

        # CDL California is normally in projected meter units. Only publish
        # hectare/acre values when rasterio reports meter/metre units.
        linear_units = ""
        try:
            linear_units = (mask_src.crs.linear_units or "").lower() if mask_src.crs else ""
        except Exception:
            linear_units = ""

        rice_area_ha = None
        rice_area_acres = None
        if linear_units in {"metre", "meter", "metres", "meters", "m"}:
            rice_area_m2 = rice_pixels * pixel_area
            rice_area_ha = rice_area_m2 / 10_000.0
            rice_area_acres = rice_area_m2 / 4046.8564224

        row = {
            "recorded_utc": utc_now(),
            "year": year,
            "source_cdl": str(source),
            "output_mask": str(target),
            "rice_class": RICE_CLASS,
            "width": mask_src.width,
            "height": mask_src.height,
            "crs": crs_string,
            "transform": json.dumps(tuple(transform), ensure_ascii=False),
            "source_nodata": source_nodata,
            "mask_nodata": MASK_NODATA,
            "valid_pixels": valid_pixels,
            "rice_pixels": rice_pixels,
            "rice_fraction_of_valid_pixels": (
                rice_pixels / valid_pixels if valid_pixels else None
            ),
            "pixel_area_native_units2": pixel_area,
            "rice_area_hectares_if_meters": rice_area_ha,
            "rice_area_acres_if_meters": rice_area_acres,
        }

    return row


# -----------------------------------------------------------------------------
# Step 21B: rice fraction on actual MODIS native tile grids
# -----------------------------------------------------------------------------


FRACTION_MANIFEST_FIELDS = [
    "recorded_utc",
    "year",
    "resolution_label",
    "template_product",
    "tile",
    "source_mask",
    "output_fraction",
    "width",
    "height",
    "crs",
    "transform",
    "fraction_nodata",
    "valid_cells",
    "rice_positive_cells",
    "fraction_min",
    "fraction_max",
    "fraction_mean",
    "reference_local_path",
    "reference_dataset_locator",
]


def build_fraction_for_grid(
    year: int,
    mask_path: Path,
    grid: GridTemplate,
    *,
    force: bool = False,
) -> Optional[Dict[str, object]]:
    out_dir = FRACTION_DIR / grid.resolution_label / grid.tile / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"rice_fraction_{year}_{grid.tile}_{grid.resolution_label}.tif"

    with rasterio.open(mask_path) as src:
        if src.crs is None:
            raise RuntimeError(f"Rice mask has no CRS: {mask_path}")

        dst_crs = rasterio.crs.CRS.from_wkt(grid.crs_wkt)

        # Skip template tiles that do not intersect the annual CDL mask extent.
        src_bounds_in_dst = transform_bounds(
            src.crs,
            dst_crs,
            *src.bounds,
            densify_pts=21,
        )
        if not bounds_overlap(src_bounds_in_dst, grid.bounds):
            print(f"[NO OVERLAP] {year} {grid.resolution_label} {grid.tile}")
            return None

        if not force and raster_ok(
            target,
            expected_width=grid.width,
            expected_height=grid.height,
            expected_crs=grid.crs_wkt,
        ):
            print(f"[EXISTS FRACTION] {target}")
        else:
            tmp = remove_stale_temp(target)
            if force and target.exists():
                target.unlink()

            profile = FRACTION_PROFILE_BASE.copy()
            profile.update(
                width=grid.width,
                height=grid.height,
                crs=dst_crs,
                transform=grid.transform,
            )

            print(
                f"[BUILD FRACTION] year={year} res={grid.resolution_label} "
                f"tile={grid.tile}"
            )

            # Write the destination first. reproject() writes through GDAL and
            # does not require loading the full CDL mask into memory.
            with rasterio.open(tmp, "w", **profile) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=MASK_NODATA,
                    dst_transform=grid.transform,
                    dst_crs=dst_crs,
                    dst_nodata=FRACTION_NODATA,
                    resampling=Resampling.average,
                    init_dest_nodata=True,
                    num_threads=2,
                )

                dst.update_tags(
                    source_mask=str(mask_path),
                    year=str(year),
                    derived_product="rice_fraction",
                    fraction_definition=(
                        "average of binary CDL rice mask during reprojection "
                        "onto actual native MODIS tile grid"
                    ),
                    template_product=grid.product,
                    template_tile=grid.tile,
                    template_reference_local_path=grid.reference_local_path,
                    raw_source_modified="false",
                )

            if not raster_ok(
                tmp,
                expected_width=grid.width,
                expected_height=grid.height,
                expected_crs=grid.crs_wkt,
            ):
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"Validation failed for temporary fraction raster: {tmp}")

            tmp.replace(target)
            print(f"[WROTE FRACTION] {target}")

    # QC statistics from output, block by block.
    valid_cells = 0
    positive_cells = 0
    value_min = math.inf
    value_max = -math.inf
    value_sum = 0.0

    with rasterio.open(target) as frac:
        for _, window in windows_for_dataset(frac):
            a = frac.read(1, window=window)
            valid = np.isfinite(a) & (a != FRACTION_NODATA)
            if not np.any(valid):
                continue

            v = a[valid]
            # Numerical guard; average resampling should already yield [0,1].
            if np.nanmin(v) < -1e-6 or np.nanmax(v) > 1.000001:
                raise RuntimeError(
                    f"Rice fraction outside [0,1] in {target}: "
                    f"min={float(np.nanmin(v))}, max={float(np.nanmax(v))}"
                )

            valid_cells += int(v.size)
            positive_cells += int(np.count_nonzero(v > 0.0))
            value_min = min(value_min, float(np.min(v)))
            value_max = max(value_max, float(np.max(v)))
            value_sum += float(np.sum(v, dtype=np.float64))

        mean_value = value_sum / valid_cells if valid_cells else None

        return {
            "recorded_utc": utc_now(),
            "year": year,
            "resolution_label": grid.resolution_label,
            "template_product": grid.product,
            "tile": grid.tile,
            "source_mask": str(mask_path),
            "output_fraction": str(target),
            "width": frac.width,
            "height": frac.height,
            "crs": frac.crs.to_string() if frac.crs else "",
            "transform": json.dumps(tuple(frac.transform), ensure_ascii=False),
            "fraction_nodata": FRACTION_NODATA,
            "valid_cells": valid_cells,
            "rice_positive_cells": positive_cells,
            "fraction_min": None if valid_cells == 0 else value_min,
            "fraction_max": None if valid_cells == 0 else value_max,
            "fraction_mean": mean_value,
            "reference_local_path": grid.reference_local_path,
            "reference_dataset_locator": grid.reference_dataset_locator,
        }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_years(text: Optional[str]) -> List[int]:
    if not text:
        return list(range(START_YEAR, END_YEAR + 1))

    years: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(part))

    years = sorted(set(years))
    invalid = [y for y in years if y < START_YEAR or y > END_YEAR]
    if invalid:
        raise ValueError(f"Years outside {START_YEAR}-{END_YEAR}: {invalid}")
    return years


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build California annual CDL rice masks and MODIS-grid rice fractions."
    )
    p.add_argument(
        "--years",
        default=None,
        help="Years such as 2007-2025 or 2007,2008,2015. Default: all 2007-2025.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild existing processed outputs.",
    )
    p.add_argument(
        "--mask-only",
        action="store_true",
        help="Build/verify native 30 m rice masks only; skip MODIS-grid fractions.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    years = parse_years(args.years)
    ensure_dirs()

    print("=" * 72)
    print("Step 21 — CDL annual rice masks + MODIS native-grid rice fractions")
    print("Project   :", PROJECT_DIR)
    print("RAW CDL   :", CDL_DIR)
    print("Processed :", PROCESSED_ROOT)
    print("Years     :", years)
    print("Force     :", args.force)
    print("Mask only :", args.mask_only)
    print("=" * 72)

    missing_sources = [
        CDL_DIR / str(y) / f"cdl_california_{y}.tif"
        for y in years
        if not (CDL_DIR / str(y) / f"cdl_california_{y}.tif").exists()
    ]
    if missing_sources:
        print("ERROR: missing CDL source files:")
        for p in missing_sources:
            print(" ", p)
        return 2

    templates = None if args.mask_only else load_modis_grid_templates()

    mask_rows: List[Dict[str, object]] = []
    fraction_rows: List[Dict[str, object]] = []

    for year in years:
        print("\n" + "-" * 72)
        print("YEAR", year)
        print("-" * 72)

        mask_row = build_native_rice_mask(year, force=args.force)
        mask_rows.append(mask_row)

        if args.mask_only:
            continue

        mask_path = Path(str(mask_row["output_mask"]))
        assert templates is not None

        for resolution_label in ("250m", "500m", "1000m"):
            for grid in templates[resolution_label]:
                row = build_fraction_for_grid(
                    year,
                    mask_path,
                    grid,
                    force=args.force,
                )
                if row is not None:
                    fraction_rows.append(row)

    # Replace summary manifests atomically so reruns do not duplicate rows.
    write_csv_replace(MASK_MANIFEST_CSV, MASK_MANIFEST_FIELDS, mask_rows)
    write_csv_replace(AREA_SUMMARY_CSV, MASK_MANIFEST_FIELDS, mask_rows)

    if not args.mask_only:
        write_csv_replace(
            FRACTION_MANIFEST_CSV,
            FRACTION_MANIFEST_FIELDS,
            fraction_rows,
        )

    print("\n" + "=" * 72)
    print("STEP 21 COMPLETE")
    print("Annual masks      :", len(mask_rows))
    print("Fraction rasters  :", len(fraction_rows))
    print("Mask manifest     :", MASK_MANIFEST_CSV)
    if not args.mask_only:
        print("Fraction manifest :", FRACTION_MANIFEST_CSV)
    print("Area summary      :", AREA_SUMMARY_CSV)
    print("RAW CDL modified  : NO")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
