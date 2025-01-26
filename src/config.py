from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent

ZONES = ["US-CAL-CISO", "US-TEX-ERCO"]
FORECAST_KEY_SUBKEYS = {"production": {"solar"}}

DATA_BUCKET_PATH = "https://storage.googleapis.com/mlengineer-challenge"
