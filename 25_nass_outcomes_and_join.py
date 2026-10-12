#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
25_nass_outcomes_and_join.py

California Rice Monitoring — Step 25

Purpose
-------
Download/cache USDA NASS Quick Stats annual California rice outcomes, clean
only the statewide annual TOTAL/SURVEY series, and join those annual outcomes
to the compact Step-24 feature tables.

Step 25 deliberately does NOT do regression, feature selection, imputation, or
ML.  The output is the final handoff to Colab.

NASS target variables
---------------------
* planted acres
* harvested acres
* yield (LB / ACRE)
* production (CWT)

The default API query is intentionally broad (RICE, CA, STATE, >=2007) and the
scientific selection is performed locally and audited.  Duplicate candidate
rows are never silently averaged.

API key
-------
The script looks for a NASS Quick Stats API key in this order:

1. environment variable NASS_API_KEY
2. NASS_API_KEY in ./config.py

No key is needed after the raw response has been cached, unless
--refresh-nass is used.

Example:
    export NASS_API_KEY='...'
    python -u 25_nass_outcomes_and_join.py

Inputs
------
processed/24_analysis_ready/
    monthly_features.parquet
    seasonal_features.parquet
    _manifests/run_summary.json

Outputs
-------
processed/25_nass_outcomes/
    nass_raw_quickstats.csv
    nass_yearly.parquet / .csv
    monthly_features_with_nass.parquet / .csv
    seasonal_features_with_nass.parquet / .csv
    model_yearly.parquet / .csv
    _manifests/
        nass_selected_rows.csv
        nass_quality.csv
        nass_cdl_area_comparison.csv
        feature_dictionary_targets.csv
        run_summary.json

``model_yearly.parquet`` is the primary small Colab ML table: one row per year
(2007-2025), seasonal satellite/climate predictors from Step 24, annual CDL
rice area, and annual NASS outcomes.  The NASS columns are targets/outcomes,
not predictor recommendations.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


PIPELINE_VERSION = "25.1"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
PROCESSED_DIR = PROJECT_DIR / "processed"

STEP24_ROOT = PROCESSED_DIR / "24_analysis_ready"
STEP24_RUN_SUMMARY = STEP24_ROOT / "_manifests" / "run_summary.json"
STEP24_MONTHLY = STEP24_ROOT / "monthly_features.parquet"
STEP24_SEASONAL = STEP24_ROOT / "seasonal_features.parquet"
STEP24_FEATURE_DICT = STEP24_ROOT / "_manifests" / "feature_dictionary.csv"

OUT_ROOT = PROCESSED_DIR / "25_nass_outcomes"
MANIFEST_DIR = OUT_ROOT / "_manifests"
NASS_RAW_CSV = OUT_ROOT / "nass_raw_quickstats.csv"
NASS_YEARLY_CSV = OUT_ROOT / "nass_yearly.csv"
NASS_YEARLY_PARQUET = OUT_ROOT / "nass_yearly.parquet"
MONTHLY_JOINED_CSV = OUT_ROOT / "monthly_features_with_nass.csv"
MONTHLY_JOINED_PARQUET = OUT_ROOT / "monthly_features_with_nass.parquet"
SEASONAL_JOINED_CSV = OUT_ROOT / "seasonal_features_with_nass.csv"
SEASONAL_JOINED_PARQUET = OUT_ROOT / "seasonal_features_with_nass.parquet"
MODEL_YEARLY_CSV = OUT_ROOT / "model_yearly.csv"
MODEL_YEARLY_PARQUET = OUT_ROOT / "model_yearly.parquet"

SELECTED_ROWS_CSV = MANIFEST_DIR / "nass_selected_rows.csv"
QUALITY_CSV = MANIFEST_DIR / "nass_quality.csv"
DUPLICATE_CANDIDATES_CSV = MANIFEST_DIR / "nass_duplicate_candidates.csv"
AREA_COMPARISON_CSV = MANIFEST_DIR / "nass_cdl_area_comparison.csv"
TARGET_DICTIONARY_CSV = MANIFEST_DIR / "feature_dictionary_targets.csv"
RUN_SUMMARY_JSON = MANIFEST_DIR / "run_summary.json"

ANALYSIS_START_YEAR = 2007
ANALYSIS_END_YEAR = 2025
EXPECTED_YEARS = list(range(ANALYSIS_START_YEAR, ANALYSIS_END_YEAR + 1))

NASS_API_URL = "https://quickstats.nass.usda.gov/api/api_GET/"

METRIC_SPECS = {
    "planted_acres": {
        "pattern": r"^RICE - ACRES PLANTED$",
        "unit": "ACRES",
        "output": "nass_planted_acres",
    },
    "harvested_acres": {
        "pattern": r"^RICE - ACRES HARVESTED$",
        "unit": "ACRES",
        "output": "nass_harvested_acres",
    },
    "yield": {
        "pattern": r"^RICE - YIELD",
        "unit": "LB / ACRE",
        "output": "nass_yield_lb_acre",
    },
    "production": {
        "pattern": r"^RICE - PRODUCTION",
        "unit": "CWT",
        "output": "nass_production_cwt",
    },
}



def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


def atomic_json(path: Path, obj: object) -> None:
    tmp = Path(str(path) + ".part")
    tmp.unlink(missing_ok=True)
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = Path(str(path) + ".part")
    tmp.unlink(missing_ok=True)
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = Path(str(path) + ".part")
    tmp.unlink(missing_ok=True)
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def write_pair(df: pd.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    atomic_csv(df, csv_path)
    atomic_parquet(df, parquet_path)


def require_parquet_engine() -> None:
    try:
        import pyarrow  # noqa: F401
        return
    except Exception:
        pass
    try:
        import fastparquet  # noqa: F401
        return
    except Exception as e:
        raise RuntimeError("Step 25 requires pyarrow or fastparquet; e.g. pip install pyarrow") from e


def verify_step24() -> None:
    required = [STEP24_RUN_SUMMARY, STEP24_MONTHLY, STEP24_SEASONAL]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Step-24 output(s):\n" + "\n".join(str(p) for p in missing))
    s = json.loads(STEP24_RUN_SUMMARY.read_text(encoding="utf-8"))
    if not s.get("analysis_ready"):
        raise RuntimeError("Step 24 is not marked analysis_ready; finish Step 24 before Step 25.")


def load_config_api_key() -> Optional[str]:
    config_path = PROJECT_DIR / "config.py"
    if not config_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("foss4g_project_config", config_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        value = getattr(module, "NASS_API_KEY", None)
        return str(value).strip() if value else None
    except Exception:
        return None


def get_api_key() -> Optional[str]:
    v = os.environ.get("NASS_API_KEY")
    if v and v.strip():
        return v.strip()
    return load_config_api_key()


def download_nass(api_key: str) -> pd.DataFrame:
    try:
        import requests
    except Exception as e:
        raise RuntimeError("requests is required for NASS download: pip install requests") from e

    params = {
        "key": api_key,
        "commodity_desc": "RICE",
        "state_alpha": "CA",
        "agg_level_desc": "STATE",
        "year__GE": str(ANALYSIS_START_YEAR),
        "year__LE": str(ANALYSIS_END_YEAR),
        "format": "JSON",
    }
    print("[GET] USDA NASS Quick Stats — California statewide rice", ANALYSIS_START_YEAR, "-", ANALYSIS_END_YEAR)
    r = requests.get(NASS_API_URL, params=params, timeout=180)
    r.raise_for_status()
    payload = r.json()
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected NASS API response: {str(payload)[:1000]}")
    df = pd.DataFrame(data)
    if df.empty:
        raise RuntimeError("NASS API returned zero rows for California RICE.")
    return df


def value_to_numeric(series: pd.Series) -> pd.Series:
    # Quick Stats uses commas for thousands.  Suppressed values such as (D),
    # (Z), or other non-numeric symbols deliberately become NaN rather than
    # being guessed/imputed.
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def optional_eq(df: pd.DataFrame, column: str, value: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(True, index=df.index)
    return df[column].astype(str).eq(value)


def clean_nass(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"year", "short_desc", "Value", "unit_desc"}
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"NASS response missing required columns: {sorted(missing)}")

    d = raw.copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")
    d = d[d["year"].between(ANALYSIS_START_YEAR, ANALYSIS_END_YEAR)].copy()
    d["value_num"] = value_to_numeric(d["Value"])

    # Preserve only short_desc entries for total RICE, not class-specific
    # strings such as "RICE, MEDIUM GRAIN - ...".
    main = d[d["short_desc"].astype(str).str.match(r"^RICE - ", na=False)].copy()

    selected: List[pd.DataFrame] = []
    for metric, spec in METRIC_SPECS.items():
        x = main[main["short_desc"].astype(str).str.contains(spec["pattern"], regex=True, na=False)].copy()
        x["metric"] = metric
        selected.append(x)
    cand = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()

    if cand.empty:
        raise RuntimeError("No NASS rice target candidate rows matched expected short_desc patterns.")

    keep = pd.Series(True, index=cand.index)
    keep &= optional_eq(cand, "agg_level_desc", "STATE")
    keep &= optional_eq(cand, "freq_desc", "ANNUAL")
    keep &= optional_eq(cand, "reference_period_desc", "YEAR")
    keep &= optional_eq(cand, "source_desc", "SURVEY")
    keep &= optional_eq(cand, "domain_desc", "TOTAL")
    cand = cand[keep].copy()

    unit_ok = pd.Series(False, index=cand.index)
    for metric, spec in METRIC_SPECS.items():
        unit_ok |= cand["metric"].eq(metric) & cand["unit_desc"].astype(str).eq(spec["unit"])
    selected_clean = cand[unit_ok].copy()

    duplicate_sizes = selected_clean.groupby(["year", "metric"], dropna=False).size()
    bad = duplicate_sizes[duplicate_sizes > 1]
    duplicate_candidates = pd.DataFrame()
    if len(bad):
        bad_keys = set((int(y), str(m)) for y, m in bad.index)
        duplicate_candidates = selected_clean[
            selected_clean.apply(lambda r: (int(r["year"]), str(r["metric"])) in bad_keys, axis=1)
        ].copy()

    # One row per year/metric is a hard invariant; no silent mean/first choice.
    if len(bad):
        cols = [c for c in [
            "year", "metric", "short_desc", "Value", "unit_desc", "statisticcat_desc",
            "class_desc", "prodn_practice_desc", "util_practice_desc", "domain_desc",
            "source_desc", "reference_period_desc",
        ] if c in duplicate_candidates.columns]
        atomic_csv(duplicate_candidates[cols], DUPLICATE_CANDIDATES_CSV)
        raise RuntimeError(
            f"NASS duplicate year/metric rows remain ({len(bad)} groups). "
            f"Inspect {DUPLICATE_CANDIDATES_CSV}; no rows were silently averaged."
        )

    selected_clean["year"] = selected_clean["year"].astype(int)

    # Build a compact yearly target table.
    pivot = selected_clean.pivot(index="year", columns="metric", values="value_num").reset_index()
    pivot.columns.name = None
    rename = {metric: spec["output"] for metric, spec in METRIC_SPECS.items()}
    yearly = pivot.rename(columns=rename)
    years = pd.DataFrame({"year": EXPECTED_YEARS})
    yearly = years.merge(yearly, on="year", how="left")

    quality_rows: List[dict] = []
    for metric, spec in METRIC_SPECS.items():
        col = spec["output"]
        missing_years = yearly.loc[pd.to_numeric(yearly[col], errors="coerce").isna(), "year"].astype(int).tolist()
        quality_rows.append({
            "metric": metric,
            "column": col,
            "expected_years": len(EXPECTED_YEARS),
            "nonmissing_years": len(EXPECTED_YEARS) - len(missing_years),
            "missing_years": ",".join(map(str, missing_years)),
            "unit": spec["unit"],
        })
    quality = pd.DataFrame(quality_rows)

    return selected_clean, yearly, quality


def join_outcomes(features: pd.DataFrame, yearly: pd.DataFrame) -> pd.DataFrame:
    x = features.copy()
    x["year"] = pd.to_numeric(x["year"], errors="raise").astype(int)
    y = yearly.copy()
    y["year"] = pd.to_numeric(y["year"], errors="raise").astype(int)
    return x.merge(y, on="year", how="left", validate="m:1")


def build_area_comparison(seasonal_joined: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "year", "cdl_rice_area_acres", "nass_planted_acres", "nass_harvested_acres",
        "nass_yield_lb_acre", "nass_production_cwt",
    ]
    existing = [c for c in cols if c in seasonal_joined.columns]
    q = seasonal_joined[existing].copy()
    if {"cdl_rice_area_acres", "nass_harvested_acres"}.issubset(q.columns):
        denom = pd.to_numeric(q["cdl_rice_area_acres"], errors="coerce")
        harvest = pd.to_numeric(q["nass_harvested_acres"], errors="coerce")
        q["nass_harvested_to_cdl_area_ratio"] = harvest / denom.where(denom != 0)
    if {"nass_harvested_acres", "nass_yield_lb_acre", "nass_production_cwt"}.issubset(q.columns):
        harvest = pd.to_numeric(q["nass_harvested_acres"], errors="coerce")
        yld = pd.to_numeric(q["nass_yield_lb_acre"], errors="coerce")
        prod = pd.to_numeric(q["nass_production_cwt"], errors="coerce")
        q["implied_production_cwt_from_harvest_yield"] = harvest * yld / 100.0
        implied = q["implied_production_cwt_from_harvest_yield"]
        q["reported_to_implied_production_ratio"] = prod / implied.where(implied != 0)
    return q


def build_target_dictionary() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "feature": "nass_planted_acres",
            "role": "annual outcome/context",
            "source": "USDA NASS Quick Stats",
            "unit": "ACRES",
            "notes": "Statewide annual rice acres planted; repeated across months only for convenient joins.",
        },
        {
            "feature": "nass_harvested_acres",
            "role": "annual outcome/context",
            "source": "USDA NASS Quick Stats",
            "unit": "ACRES",
            "notes": "Statewide annual rice acres harvested.",
        },
        {
            "feature": "nass_yield_lb_acre",
            "role": "candidate ML target",
            "source": "USDA NASS Quick Stats",
            "unit": "LB / ACRE",
            "notes": "Statewide annual yield; useful target when focusing on biological/productivity response rather than planted area decisions.",
        },
        {
            "feature": "nass_production_cwt",
            "role": "candidate ML target",
            "source": "USDA NASS Quick Stats",
            "unit": "CWT",
            "notes": "Statewide annual production; combines yield with harvested-area decisions.",
        },
    ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 25: USDA NASS outcomes + join to Step-24 analysis-ready tables.")
    p.add_argument("--refresh-nass", action="store_true", help="Redownload NASS even when cached raw CSV exists.")
    p.add_argument("--raw-csv", default="", help="Optional existing NASS Quick Stats raw CSV; copied/used instead of API download.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    require_parquet_engine()
    verify_step24()

    if args.raw_csv:
        src = Path(args.raw_csv).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        raw = pd.read_csv(src, dtype=str)
        atomic_csv(raw, NASS_RAW_CSV)
        raw_source = str(src)
        print("[USE] NASS raw CSV:", src)
    elif NASS_RAW_CSV.exists() and not args.refresh_nass:
        raw = pd.read_csv(NASS_RAW_CSV, dtype=str)
        raw_source = str(NASS_RAW_CSV)
        print("[CACHE] NASS raw:", NASS_RAW_CSV)
    else:
        key = get_api_key()
        if not key:
            raise RuntimeError(
                "NASS_API_KEY is not configured. Set environment variable NASS_API_KEY or add "
                "NASS_API_KEY to /opt/foss4g_2026/config.py. The key is not written to outputs."
            )
        raw = download_nass(key)
        atomic_csv(raw, NASS_RAW_CSV)
        raw_source = "USDA NASS Quick Stats API"
        print("[WROTE] NASS raw cache:", NASS_RAW_CSV, "rows=", len(raw))

    selected, yearly, quality = clean_nass(raw)
    atomic_csv(selected, SELECTED_ROWS_CSV)
    atomic_csv(quality, QUALITY_CSV)

    missing_metrics = quality[quality["nonmissing_years"] < len(EXPECTED_YEARS)]
    if len(missing_metrics):
        print("\nNASS quality problem — expected all 2007-2025 annual outcomes:")
        print(missing_metrics.to_string(index=False))
        atomic_json(RUN_SUMMARY_JSON, {
            "generated_utc": utc_now(),
            "pipeline_version": PIPELINE_VERSION,
            "complete": False,
            "raw_source": raw_source,
            "reason": "missing annual NASS metric values",
            "quality": quality.to_dict(orient="records"),
        })
        return 2

    write_pair(yearly, NASS_YEARLY_CSV, NASS_YEARLY_PARQUET)

    monthly = pd.read_parquet(STEP24_MONTHLY)
    seasonal = pd.read_parquet(STEP24_SEASONAL)
    monthly_joined = join_outcomes(monthly, yearly)
    seasonal_joined = join_outcomes(seasonal, yearly)
    model_yearly = seasonal_joined.copy()

    write_pair(monthly_joined, MONTHLY_JOINED_CSV, MONTHLY_JOINED_PARQUET)
    write_pair(seasonal_joined, SEASONAL_JOINED_CSV, SEASONAL_JOINED_PARQUET)
    write_pair(model_yearly, MODEL_YEARLY_CSV, MODEL_YEARLY_PARQUET)

    area_qc = build_area_comparison(seasonal_joined)
    atomic_csv(area_qc, AREA_COMPARISON_CSV)
    target_dict = build_target_dictionary()
    atomic_csv(target_dict, TARGET_DICTIONARY_CSV)

    target_cols = [spec["output"] for spec in METRIC_SPECS.values()]
    missing_after_join = {c: int(pd.to_numeric(model_yearly[c], errors="coerce").isna().sum()) for c in target_cols}
    if any(missing_after_join.values()):
        raise RuntimeError(f"NASS targets became missing after Step-24 join: {missing_after_join}")

    summary = {
        "generated_utc": utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "complete": True,
        "analysis_years": [ANALYSIS_START_YEAR, ANALYSIS_END_YEAR],
        "nass_raw_rows": len(raw),
        "nass_selected_rows": len(selected),
        "nass_yearly_rows": len(yearly),
        "monthly_rows": len(monthly_joined),
        "seasonal_rows": len(seasonal_joined),
        "model_yearly_rows": len(model_yearly),
        "target_columns": target_cols,
        "target_missing_after_join": missing_after_join,
        "raw_source": raw_source,
        "outputs": {
            "nass_yearly": str(NASS_YEARLY_PARQUET),
            "monthly_with_nass": str(MONTHLY_JOINED_PARQUET),
            "seasonal_with_nass": str(SEASONAL_JOINED_PARQUET),
            "model_yearly": str(MODEL_YEARLY_PARQUET),
        },
    }
    atomic_json(RUN_SUMMARY_JSON, summary)

    print("\n" + "=" * 78)
    print("STEP 25 SUMMARY")
    print("NASS raw rows          :", len(raw))
    print("NASS selected rows     :", len(selected))
    print("NASS yearly rows       :", len(yearly), "(expected 19)")
    print("Monthly joined rows    :", len(monthly_joined), "(expected 152)")
    print("Seasonal/model rows    :", len(model_yearly), "(expected 19)")
    print("Model yearly Parquet   :", MODEL_YEARLY_PARQUET)
    print("Monthly joined Parquet :", MONTHLY_JOINED_PARQUET)
    print("Area QC                :", AREA_COMPARISON_CSV)
    print("=" * 78)
    print("\n✓ STEP 25 COMPLETE — final EC2 preprocessing is ready for Colab analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
