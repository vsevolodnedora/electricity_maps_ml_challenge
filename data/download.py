import logging
import os

import pandas as pd

from src.config import DATA_BUCKET_PATH, ZONES

DATA_PATH = "data"


def download_features_for_zone(zone_key: str) -> pd.DataFrame:
    """Download features for a zone from a bucket."""
    return pd.read_parquet(f"{DATA_BUCKET_PATH}/features/{zone_key}.parquet")


def download_targets_for_zone(zone_key: str) -> pd.DataFrame:
    """Download targets for a zone from a bucket."""
    return pd.read_parquet(f"{DATA_BUCKET_PATH}/targets/{zone_key}.parquet")


if __name__ == "__main__":
    # Set logging level to info
    logging.getLogger().setLevel(logging.INFO)
    os.makedirs(f"{DATA_PATH}/features", exist_ok=True)
    os.makedirs(f"{DATA_PATH}/targets", exist_ok=True)
    for zone_key in ZONES:
        df = download_features_for_zone(zone_key)
        df.to_parquet(f"data/features/{zone_key}.parquet")
        logging.info(f"✅ Successfully retrieved features for {zone_key}!")
        df = download_targets_for_zone(zone_key)
        df.to_parquet(f"data/targets/{zone_key}.parquet")
        logging.info(f"✅ Successfully retrieved targets for {zone_key}!")
