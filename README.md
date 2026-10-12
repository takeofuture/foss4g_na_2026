# foss4g_na_2026

## 2026/11/03 — `20261103_LT_California_Rice_PUBLIC.ipynb` (Google Colab)

FOSS4G North America 2026 lightning talk notebook for California rice monitoring.

### Requirements

- Google Colab
- Google Drive
- Google Earth Engine account / Cloud Project
- USDA NASS Quick Stats API key

### Before running

1. Set your Google Drive project folder:

```python
PROJECT_DIR = Path(
    "/content/drive/MyDrive/YOUR_PROJECT_FOLDER"
)
```

2. Set your Google Earth Engine Cloud Project ID:

```python
EE_PROJECT = "YOUR_EARTH_ENGINE_PROJECT_ID"
```

3. Add a Colab Secret named:

```text
NASS_API_KEY
```

4. Authenticate Google Earth Engine when prompted.

### Main data sources

- USDA NASS Quick Stats
- California DWR Crop Mapping
- U.S. Drought Monitor
- Google Earth Engine
  - Sentinel-2
  - Sentinel-1

The notebook caches downloaded and processed files in Google Drive so later runs can reuse them.

### Notes

- The notebook is designed to run in Google Colab.
- Google Earth Engine is used for Sentinel-1 and Sentinel-2 processing.
- The USDA NASS API key is not hard-coded into the public notebook.
- See the notebook itself for the full workflow, processing details, and analysis.
