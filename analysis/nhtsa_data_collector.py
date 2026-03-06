"""
NHTSA Airbag Recall Data Collector

Fetches airbag-related recall data from the NHTSA Recalls API.
Falls back to cached sample data when the API is unavailable.

Primary API endpoint:
    https://api.nhtsa.gov/recalls/recallsBySearch?query=air%20bags&type=equipment

Data source: National Highway Traffic Safety Administration (NHTSA)
"""

import json
import logging
import os
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

NHTSA_API_BASE = "https://api.nhtsa.gov/recalls/recallsBySearch"
SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"
SAMPLE_DATA_FILE = SAMPLE_DATA_DIR / "nhtsa_airbag_recalls.json"

# Column mapping from NHTSA API response fields to our dataset columns
COLUMN_MAP = {
    "NHTSACampaignNumber": "recall_id",
    "ReportReceivedDate": "recall_date",
    "Component": "component",
    "Summary": "defect_description",
    "Consequence": "safety_risk",
    "Remedy": "remedy",
    "ModelYear": "model_year",
    "Make": "manufacturer",
    "Model": "vehicle_model",
    "Manufacturer": "recall_manufacturer",
    "ParkIt": "park_it",
    "ParkOutSide": "park_outside",
    "NHTSAActionNumber": "action_number",
    "Notes": "notes",
}


def fetch_from_api(timeout: int = 30) -> list[dict]:
    """Fetch airbag recall data from the NHTSA API.

    Args:
        timeout: Request timeout in seconds.

    Returns:
        List of recall record dicts.

    Raises:
        requests.RequestException: If the API request fails.
    """
    params = {"query": "air bags", "type": "equipment"}
    logger.info("Fetching airbag recalls from NHTSA API: %s", NHTSA_API_BASE)
    response = requests.get(NHTSA_API_BASE, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    results = data.get("results", [])
    logger.info("Retrieved %d recall records from NHTSA API", len(results))
    return results


def load_sample_data() -> list[dict]:
    """Load cached sample data from disk.

    Returns:
        List of recall record dicts.
    """
    logger.info("Loading sample data from %s", SAMPLE_DATA_FILE)
    with open(SAMPLE_DATA_FILE, "r") as f:
        data = json.load(f)
    logger.info("Loaded %d sample recall records", len(data))
    return data


def save_sample_data(records: list[dict]) -> None:
    """Cache fetched data to disk for offline use.

    Args:
        records: List of recall record dicts to save.
    """
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_DATA_FILE, "w") as f:
        json.dump(records, f, indent=2)
    logger.info("Saved %d records to %s", len(records), SAMPLE_DATA_FILE)


def normalize_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convert raw NHTSA records into a clean, normalized DataFrame.

    Args:
        records: List of raw recall record dicts.

    Returns:
        Cleaned pandas DataFrame with standardized column names.
    """
    df = pd.DataFrame(records)

    # Rename columns using our mapping
    rename_cols = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_cols)

    # Parse dates
    if "recall_date" in df.columns:
        df["recall_date"] = pd.to_datetime(df["recall_date"], errors="coerce")
        df["recall_year"] = df["recall_date"].dt.year

    # Normalize text fields
    for col in ["manufacturer", "vehicle_model", "component", "defect_description"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Ensure model_year is numeric
    if "model_year" in df.columns:
        df["model_year"] = pd.to_numeric(df["model_year"], errors="coerce")

    return df


def collect_data(use_cache: bool = True) -> pd.DataFrame:
    """Main data collection entry point.

    Attempts to fetch data from the NHTSA API first. If the API is
    unavailable, falls back to cached sample data.

    Args:
        use_cache: If True, try cached data when API fails.

    Returns:
        DataFrame of airbag recall records.
    """
    records = None

    # Try fetching from API first
    try:
        records = fetch_from_api()
        save_sample_data(records)
    except (requests.RequestException, Exception) as e:
        logger.warning("NHTSA API unavailable: %s", e)
        if use_cache and SAMPLE_DATA_FILE.exists():
            records = load_sample_data()
        else:
            logger.error("No cached data available. Cannot proceed.")
            raise

    df = normalize_dataframe(records)
    logger.info("Final dataset: %d records, %d columns", len(df), len(df.columns))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = collect_data()
    print(f"Collected {len(df)} airbag recall records")
    print(df.head())
