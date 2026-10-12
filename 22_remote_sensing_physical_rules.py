#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22_remote_sensing_physical_rules.py

California Rice Monitoring — Step 22
Build a reproducible RAW -> physical-value / QA interpretation catalog
WITHOUT duplicating the full MODIS/NTSG rasters.

Why this step is metadata/recipe-first
--------------------------------------
The RAW archive is already very large. Materializing every MODIS HDF band as
full-size Float32 GeoTIFFs would duplicate hundreds of GB before we even crop
to rice areas. Instead Step 22 creates an explicit, auditable recipe for each
raw file/band:

    raw dataset locator
      + scale / offset / nodata
      + QA companion and bit rule
      + output variable / unit
      = analysis-ready interpretation used by Step 23

Step 23 will apply these recipes WHILE cropping to California rice areas, so
we do the expensive pixel conversion only where it is actually needed.

RAW files under ./data are READ ONLY.  Outputs go under:

    ./processed/22_physical_rules/

This script is resume-safe and can be re-run.

Product rules intentionally match the earlier California-rice preparation
logic used in this project:

- MOD13Q1: NDVI/EVI * 0.0001; good if pixel reliability / SummaryQA is 0 or 1
- MOD09A1: reflectance bands * 0.0001; clear cloud state and no cloud shadow
- MOD15A2H: LAI * 0.1, FPAR * 0.01; range checks; retain FparLai_QC
- MOD11A2: LST * 0.02 K -> deg C; QC mandatory bits <= 1
- MOD16A2GF: ET/PET * 0.1; ET_QC bit 0 == 0
- MOD17A2H: GPP/PsnNet * 0.0001; Psn_QC bit 0 == 0
- UMT/NTSG MODIS 250 m GPP: GPP * 0.0001 kg C m-2 per 8-day

No reprojection or resampling is performed here.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import math
import re
import sys


# -----------------------------------------------------------------------------
# Paths / configuration
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
DATA_DIR = PROJECT_DIR / "data"
MANIFEST_DIR = DATA_DIR / "_manifests"
BAND_GRID_CSV = MANIFEST_DIR / "band_grid_manifest.csv"

OUT_ROOT = PROJECT_DIR / "processed" / "22_physical_rules"
OUT_MANIFEST_DIR = OUT_ROOT / "_manifests"
OUT_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

RULES_JSON = OUT_MANIFEST_DIR / "product_rules.json"
BAND_RECIPE_CSV = OUT_MANIFEST_DIR / "band_recipe_manifest.csv"
FILE_AUDIT_CSV = OUT_MANIFEST_DIR / "file_recipe_audit.csv"
SUMMARY_CSV = OUT_MANIFEST_DIR / "recipe_summary.csv"
UNMAPPED_CSV = OUT_MANIFEST_DIR / "unmapped_relevant_rows.csv"

START_YEAR = 2005
END_YEAR = 2025
MONTHS = set(range(4, 12))


# -----------------------------------------------------------------------------
# Scientific interpretation rules
# -----------------------------------------------------------------------------

PRODUCT_RULES = {
    "mod13_vi": {
        "source_product": "MOD13Q1.061",
        "variables": {
            "ndvi": {
                "match": [r"\bndvi\b"],
                "scale": 0.0001,
                "offset": 0.0,
                "unit": "1",
                "qa_key": "summary_qa",
                "qa_rule": "0 <= QA <= 1",
                "valid_range": [-1.0, 1.0],
            },
            "evi": {
                "match": [r"\bevi\b"],
                "scale": 0.0001,
                "offset": 0.0,
                "unit": "1",
                "qa_key": "summary_qa",
                "qa_rule": "0 <= QA <= 1",
                "valid_range": [-1.0, 1.0],
            },
        },
        "qa": {
            "summary_qa": {
                "match": [r"pixel reliability", r"summaryqa", r"summary_qa"],
                "interpretation": {
                    "0": "good data",
                    "1": "marginal data",
                    "2": "snow/ice",
                    "3": "cloudy",
                },
            }
        },
        "expected_variables_per_file": ["ndvi", "evi"],
    },

    "mod09_sr": {
        "source_product": "MOD09A1.061",
        "variables": {
            "red":       {"match": [r"sur_refl_b01"], "scale": 0.0001, "offset": 0.0, "unit": "reflectance", "qa_key": "state_qa", "qa_rule": "cloud_state(bits0-1)==0 AND cloud_shadow(bit2)==0"},
            "nir":       {"match": [r"sur_refl_b02"], "scale": 0.0001, "offset": 0.0, "unit": "reflectance", "qa_key": "state_qa", "qa_rule": "cloud_state(bits0-1)==0 AND cloud_shadow(bit2)==0"},
            "blue":      {"match": [r"sur_refl_b03"], "scale": 0.0001, "offset": 0.0, "unit": "reflectance", "qa_key": "state_qa", "qa_rule": "cloud_state(bits0-1)==0 AND cloud_shadow(bit2)==0"},
            "green":     {"match": [r"sur_refl_b04"], "scale": 0.0001, "offset": 0.0, "unit": "reflectance", "qa_key": "state_qa", "qa_rule": "cloud_state(bits0-1)==0 AND cloud_shadow(bit2)==0"},
            "swir_1230": {"match": [r"sur_refl_b05"], "scale": 0.0001, "offset": 0.0, "unit": "reflectance", "qa_key": "state_qa", "qa_rule": "cloud_state(bits0-1)==0 AND cloud_shadow(bit2)==0"},
            "swir1":     {"match": [r"sur_refl_b06"], "scale": 0.0001, "offset": 0.0, "unit": "reflectance", "qa_key": "state_qa", "qa_rule": "cloud_state(bits0-1)==0 AND cloud_shadow(bit2)==0"},
            "swir2":     {"match": [r"sur_refl_b07"], "scale": 0.0001, "offset": 0.0, "unit": "reflectance", "qa_key": "state_qa", "qa_rule": "cloud_state(bits0-1)==0 AND cloud_shadow(bit2)==0"},
        },
        "qa": {
            "state_qa": {
                "match": [r"\bstateqa\b", r"state_qa", r"sur_refl_state_500m", r"\bstate_500m\b", r"reflectance data state"],
                "interpretation": {
                    "cloud_state": "bits 0-1; keep 0 (clear)",
                    "cloud_shadow": "bit 2; keep 0 (no shadow)",
                    "aerosol_quantity": "bits 6-7; retain for diagnostics",
                },
            }
        },
        "expected_variables_per_file": ["red", "nir", "blue", "green", "swir_1230", "swir1", "swir2"],
    },

    "mod15_lai_fpar": {
        "source_product": "MOD15A2H.061",
        "variables": {
            "lai":  {"match": [r"\blai_500m\b", r"\blai\b"],  "scale": 0.1,  "offset": 0.0, "unit": "m2/m2", "qa_key": "fpar_lai_qc", "qa_rule": "range check 0..10; retain FparLai_QC", "valid_range": [0.0, 10.0]},
            "fpar": {"match": [r"\bfpar_500m\b", r"\bfpar\b"], "scale": 0.01, "offset": 0.0, "unit": "fraction", "qa_key": "fpar_lai_qc", "qa_rule": "range check 0..1; retain FparLai_QC", "valid_range": [0.0, 1.0]},
        },
        "qa": {
            "fpar_lai_qc": {
                "match": [r"fparlai_qc", r"fpar_lai_qc"],
                "interpretation": "retained for diagnostics; science values use physical-range validity in the established project pipeline",
            }
        },
        "expected_variables_per_file": ["lai", "fpar"],
    },

    "mod11_lst": {
        "source_product": "MOD11A2.061",
        "variables": {
            "lst_day_c":   {"match": [r"lst_day_1km"],   "scale": 0.02, "offset": -273.15, "unit": "degC", "qa_key": "qc_day",   "qa_rule": "mandatory QA bits0-1 <= 1"},
            "lst_night_c": {"match": [r"lst_night_1km"], "scale": 0.02, "offset": -273.15, "unit": "degC", "qa_key": "qc_night", "qa_rule": "mandatory QA bits0-1 <= 1"},
        },
        "qa": {
            "qc_day":   {"match": [r"\bqc_day\b"],   "interpretation": "keep when (QA & 3) <= 1"},
            "qc_night": {"match": [r"\bqc_night\b"], "interpretation": "keep when (QA & 3) <= 1"},
        },
        "expected_variables_per_file": ["lst_day_c", "lst_night_c"],
    },

    "mod16_et": {
        "source_product": "MOD16A2GF.061",
        "variables": {
            "et":  {"match": [r"(^|[^a-z])et_500m([^a-z]|$)", r"(^|[^a-z])et([^a-z]|$)"],  "exclude": [r"pet", r"qc"], "scale": 0.1, "offset": 0.0, "unit": "kg/m2/8-day (equiv. mm/8-day water)", "qa_key": "et_qc", "qa_rule": "ET_QC bit0 == 0"},
            "pet": {"match": [r"pet_500m", r"\bpet\b"], "scale": 0.1, "offset": 0.0, "unit": "kg/m2/8-day (equiv. mm/8-day water)", "qa_key": "et_qc", "qa_rule": "ET_QC bit0 == 0"},
        },
        "qa": {
            "et_qc": {"match": [r"et_qc_500m", r"\bet_qc\b"], "interpretation": "keep when (QA & 1) == 0"}
        },
        "expected_variables_per_file": ["et", "pet"],
    },

    "mod17_gpp": {
        "source_product": "MOD17A2H.061",
        "variables": {
            "gpp":     {"match": [r"gpp_500m", r"\bgpp\b"],       "scale": 0.0001, "offset": 0.0, "unit": "kg C/m2/8-day", "qa_key": "psn_qc", "qa_rule": "Psn_QC bit0 == 0"},
            "psn_net": {"match": [r"psnnet_500m", r"psn_net", r"psnnet"], "scale": 0.0001, "offset": 0.0, "unit": "kg C/m2/8-day", "qa_key": "psn_qc", "qa_rule": "Psn_QC bit0 == 0"},
        },
        "qa": {
            "psn_qc": {"match": [r"psn_qc_500m", r"\bpsn_qc\b"], "interpretation": "keep when (QA & 1) == 0"}
        },
        "expected_variables_per_file": ["gpp", "psn_net"],
    },

    "umt_gpp": {
        "source_product": "UMT/NTSG MODIS 250 m GPP v2",
        "variables": {
            "gpp_ntsg": {
                "match": [r".*"],
                "scale": 0.0001,
                "offset": 0.0,
                "unit": "kg C/m2/8-day",
                "qa_key": None,
                "qa_rule": "no separate QA band is present in the downloaded gpp GeoTIFF branch; preserve source nodata",
            }
        },
        "qa": {},
        "expected_variables_per_file": ["gpp_ntsg"],
    },
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def clean_text(*parts: object) -> str:
    return " ".join(str(p or "") for p in parts).lower()


def match_rule(text: str, rule: dict) -> bool:
    for pat in rule.get("exclude", []):
        if re.search(pat, text, flags=re.I):
            return False
    return any(re.search(pat, text, flags=re.I) for pat in rule.get("match", []))


def parse_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"none", "nan", "null"}:
        return None
    try:
        x = float(s)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


def infer_product(row: dict) -> str | None:
    p = str(row.get("product") or "").strip()
    local = str(row.get("local_path") or "")

    if p in PRODUCT_RULES:
        return p

    # NTSG rows in the Step-20 band manifest use product=umt_gpp.
    if "ntsg_umt_gpp" in local:
        return "umt_gpp"

    # Fallback path-based inference.
    for key in PRODUCT_RULES:
        if f"/{key}/" in local:
            return key

    return None


def infer_date(local_path: str) -> tuple[str | None, int | None, int | None]:
    name = Path(local_path).name

    # MODIS AYYYYDDD.
    m = re.search(r"\.A(\d{4})(\d{3})\.", name)
    if m:
        from datetime import date, timedelta
        year = int(m.group(1))
        doy = int(m.group(2))
        dt = date(year, 1, 1) + timedelta(days=doy - 1)
        return dt.isoformat(), dt.year, dt.month

    # NTSG YYYYDDD.
    m = re.search(r"(?<!\d)(20\d{2})(\d{3})(?!\d)", name)
    if m:
        from datetime import date, timedelta
        year = int(m.group(1))
        doy = int(m.group(2))
        dt = date(year, 1, 1) + timedelta(days=doy - 1)
        return dt.isoformat(), dt.year, dt.month

    # Path fallback .../<year>/<month>/...
    parts = Path(local_path).parts
    for i in range(len(parts) - 1):
        if re.fullmatch(r"20\d{2}", parts[i]) and i + 1 < len(parts):
            try:
                year = int(parts[i])
                month = int(parts[i + 1]) if re.fullmatch(r"\d{2}", parts[i + 1]) else None
                return None, year, month
            except Exception:
                pass

    return None, None, None


def atomic_write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".part")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def atomic_write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".part")
    if tmp.exists():
        tmp.unlink()
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def rule_formula(scale: float, offset: float) -> str:
    if offset:
        return f"physical = raw * {scale:g} + ({offset:g})"
    return f"physical = raw * {scale:g}"


# -----------------------------------------------------------------------------
# Main catalog build
# -----------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("STEP 22 — RAW -> PHYSICAL VALUE / QA RECIPE CATALOG")
    print("Project      :", PROJECT_DIR)
    print("RAW          :", DATA_DIR)
    print("Band manifest:", BAND_GRID_CSV)
    print("Output       :", OUT_ROOT)
    print("RAW modified : NO")
    print("Full raster duplication: NO")
    print("=" * 72)

    if not BAND_GRID_CSV.exists():
        print("ERROR: missing", BAND_GRID_CSV)
        return 2

    # Save human/machine-readable rule registry first.
    registry = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project": str(PROJECT_DIR),
        "policy": {
            "raw_read_only": True,
            "reproject_here": False,
            "resample_here": False,
            "materialize_full_physical_rasters_here": False,
            "materialization_stage": "Step 23, while clipping to rice areas",
        },
        "product_rules": PRODUCT_RULES,
    }
    atomic_write_json(RULES_JSON, registry)

    # First pass: retain relevant rows only, grouped by raw local_path.
    by_file: dict[str, dict] = {}
    relevant_rows = 0
    total_rows = 0

    with BAND_GRID_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            product = infer_product(row)
            if product not in PRODUCT_RULES:
                continue

            local_path = str(row.get("local_path") or "").strip()
            if not local_path:
                continue

            # Keep only the project period and Apr-Nov when date/path allows it.
            date_iso, year, month = infer_date(local_path)
            if year is not None and not (START_YEAR <= year <= END_YEAR):
                continue
            if month is not None and month not in MONTHS:
                continue

            relevant_rows += 1
            bucket = by_file.setdefault(
                local_path,
                {
                    "product": product,
                    "date": date_iso,
                    "year": year,
                    "month": month,
                    "rows": [],
                },
            )
            bucket["rows"].append(row)

    print("Band manifest rows total   :", total_rows)
    print("Relevant band rows         :", relevant_rows)
    print("Relevant raw files         :", len(by_file))

    recipe_rows: list[dict] = []
    audit_rows: list[dict] = []
    unmapped_rows: list[dict] = []

    for idx, (local_path, bucket) in enumerate(sorted(by_file.items()), start=1):
        product = bucket["product"]
        rules = PRODUCT_RULES[product]
        rows = bucket["rows"]

        raw_exists = Path(local_path).is_file() and Path(local_path).stat().st_size > 0

        # Discover QA locators first.
        qa_found: dict[str, dict] = {}
        for qa_key, qa_rule in rules.get("qa", {}).items():
            candidates = []
            for row in rows:
                text = clean_text(row.get("dataset_locator"), row.get("description"))
                if match_rule(text, qa_rule):
                    candidates.append(row)
            if candidates:
                qa_found[qa_key] = candidates[0]

        vars_found: dict[str, dict] = {}

        # NTSG is typically a one-band GeoTIFF. Treat band 1 as GPP.
        if product == "umt_gpp":
            if rows:
                # Prefer band 1 if present.
                chosen = sorted(
                    rows,
                    key=lambda r: int(float(r.get("band_index") or 1)),
                )[0]
                vars_found["gpp_ntsg"] = chosen
        else:
            for var_name, var_rule in rules["variables"].items():
                candidates = []
                for row in rows:
                    text = clean_text(row.get("dataset_locator"), row.get("description"))
                    if match_rule(text, var_rule):
                        candidates.append(row)
                if candidates:
                    # Science HDF subdatasets are single-band; first match is canonical.
                    vars_found[var_name] = candidates[0]

        expected_vars = list(rules.get("expected_variables_per_file", []))
        missing_vars = [v for v in expected_vars if v not in vars_found]

        # QA absence is only an audit warning for products that define QA.
        expected_qa_keys = sorted({
            rules["variables"][v].get("qa_key")
            for v in expected_vars
            if v in rules.get("variables", {}) and rules["variables"][v].get("qa_key")
        })
        missing_qa = [q for q in expected_qa_keys if q not in qa_found]

        audit_status = "OK"
        if not raw_exists:
            audit_status = "RAW_MISSING"
        elif missing_vars:
            audit_status = "VARIABLE_MAPPING_INCOMPLETE"
        elif missing_qa:
            audit_status = "QA_MAPPING_INCOMPLETE"

        audit_rows.append({
            "local_path": local_path,
            "product": product,
            "acquisition_date": bucket["date"],
            "year": bucket["year"],
            "month": bucket["month"],
            "raw_exists_nonempty": raw_exists,
            "variables_expected": ";".join(expected_vars),
            "variables_found": ";".join(sorted(vars_found)),
            "variables_missing": ";".join(missing_vars),
            "qa_expected": ";".join(expected_qa_keys),
            "qa_found": ";".join(sorted(qa_found)),
            "qa_missing": ";".join(missing_qa),
            "status": audit_status,
        })

        # Recipe row per science variable.
        for var_name, source_row in vars_found.items():
            var_rule = rules["variables"][var_name]
            qa_key = var_rule.get("qa_key")
            qa_row = qa_found.get(qa_key) if qa_key else None

            raw_scale = parse_float(source_row.get("scale"))
            raw_offset = parse_float(source_row.get("offset"))
            raw_nodata = parse_float(source_row.get("no_data_value"))

            applied_scale = float(var_rule["scale"])
            applied_offset = float(var_rule.get("offset", 0.0))

            recipe_rows.append({
                "product": product,
                "source_product": rules["source_product"],
                "canonical_variable": var_name,
                "acquisition_date": bucket["date"],
                "year": bucket["year"],
                "month": bucket["month"],
                "local_path": local_path,
                "dataset_locator": source_row.get("dataset_locator"),
                "band_index": source_row.get("band_index"),
                "source_description": source_row.get("description"),
                "source_data_type": source_row.get("data_type"),
                "source_nodata": raw_nodata,
                "source_scale_metadata": raw_scale,
                "source_offset_metadata": raw_offset,
                "applied_scale": applied_scale,
                "applied_offset": applied_offset,
                "formula": rule_formula(applied_scale, applied_offset),
                "output_unit": var_rule.get("unit"),
                "valid_range_json": json.dumps(var_rule.get("valid_range"), ensure_ascii=False),
                "qa_key": qa_key,
                "qa_dataset_locator": qa_row.get("dataset_locator") if qa_row else None,
                "qa_band_index": qa_row.get("band_index") if qa_row else None,
                "qa_rule": var_rule.get("qa_rule"),
                "raw_read_only": True,
                "materialize_in_step": 23,
            })

        # Preserve relevant rows that did not map to either a requested science var or QA.
        mapped_ids = {
            (str(r.get("dataset_locator")), str(r.get("band_index")))
            for r in vars_found.values()
        } | {
            (str(r.get("dataset_locator")), str(r.get("band_index")))
            for r in qa_found.values()
        }

        for row in rows:
            rid = (str(row.get("dataset_locator")), str(row.get("band_index")))
            if rid not in mapped_ids:
                unmapped_rows.append({
                    "product": product,
                    "local_path": local_path,
                    "dataset_locator": row.get("dataset_locator"),
                    "band_index": row.get("band_index"),
                    "description": row.get("description"),
                    "note": "Not required by the current California-rice feature set; RAW remains available.",
                })

        if idx % 1000 == 0 or idx == len(by_file):
            print(f"Mapped files: {idx}/{len(by_file)}")

    # Deterministic ordering.
    recipe_rows.sort(key=lambda r: (
        str(r.get("product")),
        int(r.get("year") or 0),
        int(r.get("month") or 0),
        str(r.get("local_path")),
        str(r.get("canonical_variable")),
    ))
    audit_rows.sort(key=lambda r: str(r.get("local_path")))
    unmapped_rows.sort(key=lambda r: (str(r.get("product")), str(r.get("local_path")), str(r.get("dataset_locator"))))

    recipe_fields = [
        "product", "source_product", "canonical_variable",
        "acquisition_date", "year", "month",
        "local_path", "dataset_locator", "band_index",
        "source_description", "source_data_type",
        "source_nodata", "source_scale_metadata", "source_offset_metadata",
        "applied_scale", "applied_offset", "formula", "output_unit",
        "valid_range_json",
        "qa_key", "qa_dataset_locator", "qa_band_index", "qa_rule",
        "raw_read_only", "materialize_in_step",
    ]
    audit_fields = [
        "local_path", "product", "acquisition_date", "year", "month",
        "raw_exists_nonempty",
        "variables_expected", "variables_found", "variables_missing",
        "qa_expected", "qa_found", "qa_missing", "status",
    ]
    unmapped_fields = [
        "product", "local_path", "dataset_locator", "band_index", "description", "note"
    ]

    atomic_write_csv(BAND_RECIPE_CSV, recipe_rows, recipe_fields)
    atomic_write_csv(FILE_AUDIT_CSV, audit_rows, audit_fields)
    atomic_write_csv(UNMAPPED_CSV, unmapped_rows, unmapped_fields)

    # Summary by product / canonical variable.
    counts: dict[tuple[str, str], int] = {}
    for r in recipe_rows:
        k = (str(r["product"]), str(r["canonical_variable"]))
        counts[k] = counts.get(k, 0) + 1

    summary_rows = [
        {"product": k[0], "canonical_variable": k[1], "recipe_rows": n}
        for k, n in sorted(counts.items())
    ]
    atomic_write_csv(
        SUMMARY_CSV,
        summary_rows,
        ["product", "canonical_variable", "recipe_rows"],
    )

    bad = [r for r in audit_rows if r["status"] != "OK"]

    print("\n" + "=" * 72)
    print("STEP 22 SUMMARY")
    print("Raw files audited      :", len(audit_rows))
    print("Science recipe rows    :", len(recipe_rows))
    print("Unmapped ancillary rows:", len(unmapped_rows))
    print("Audit problems         :", len(bad))
    print("Rules JSON             :", RULES_JSON)
    print("Recipe CSV             :", BAND_RECIPE_CSV)
    print("Audit CSV              :", FILE_AUDIT_CSV)
    print("Summary CSV            :", SUMMARY_CSV)
    print("RAW modified           : NO")
    print("Full raster duplication: NO")
    print("=" * 72)

    if bad:
        print("\nFirst audit problems:")
        for r in bad[:30]:
            print(
                " ", r["status"], r["product"], Path(r["local_path"]).name,
                "missing vars=", r["variables_missing"],
                "missing qa=", r["qa_missing"],
            )
        print("\nSTEP 22 INCOMPLETE — inspect file_recipe_audit.csv")
        return 2

    print("\n✓ STEP 22 COMPLETE — physical-value/QA recipes are ready for Step 23")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
