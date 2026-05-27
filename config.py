"""
Central configuration for paths, GCS settings, and column naming conventions.
Set GCS_BUCKET as an environment variable when running on GCP.
"""
import os

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_DIR, "data"))
CSV_DIR = os.path.join(DATA_DIR, "csv")
PARQUET_DIR = os.path.join(DATA_DIR, "parquet")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# GCS (set GCS_BUCKET when running on GCP, leave empty for local)
GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
GCS_DATA_PREFIX = os.environ.get("GCS_DATA_PREFIX", "taxi-benchmark")

def parquet_path(filename: str = "yellow_taxi.parquet") -> str:
    """Return GCS or local parquet path depending on environment."""
    if GCS_BUCKET:
        return f"gs://{GCS_BUCKET}/{GCS_DATA_PREFIX}/{filename}"
    return os.path.join(PARQUET_DIR, filename)


# ---------------------------------------------------------------------------
# NYC Taxi download URLs
# ---------------------------------------------------------------------------
# TLC CloudFront CDN (Parquet, 2022+):
#   https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet
# TLC older CSVs (2009-2021):
#   https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet
#   (TLC migrated older files to Parquet; check availability before using CSV fallback)
TLC_CDN_BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"

# Start small: 1 month.  Expand TAXI_MONTHS to add more data.
TAXI_YEAR = 2009
TAXI_MONTHS = [1]   # [1] = Jan 2009 only; extend to [1,2,3,...] for more data

# ---------------------------------------------------------------------------
# Column name normalisation
# The blog uses lowercase names (fare_amt, tip_amt, vendor_name).
# The 2009-2013 TLC CSVs use mixed-case names.
# ---------------------------------------------------------------------------
COLUMN_RENAME = {
    # 2009-2013 CSV columns → blog-style names
    "Fare_Amt":               "fare_amt",
    "Tip_Amt":                "tip_amt",
    "vendor_name":            "vendor_name",
    "Trip_Distance":          "trip_distance",
    "Payment_Type":           "payment_type",
    "Passenger_Count":        "passenger_count",
    "Trip_Pickup_DateTime":   "pickup_datetime",
    "Trip_Dropoff_DateTime":  "dropoff_datetime",
    "Start_Lon":              "pickup_longitude",
    "Start_Lat":              "pickup_latitude",
    "End_Lon":                "dropoff_longitude",
    "End_Lat":                "dropoff_latitude",
    "Tolls_Amt":              "tolls_amt",
    "Total_Amt":              "total_amt",
    "Rate_Code":              "rate_code",
    "surcharge":              "surcharge",
    "mta_tax":                "mta_tax",
    "store_and_forward":      "store_and_fwd_flag",
    # 2014+ Parquet columns → blog-style names
    "fare_amount":            "fare_amt",
    "tip_amount":             "tip_amt",
    "VendorID":               "vendor_name",
    "vendor_id":              "vendor_name",
    "tolls_amount":           "tolls_amt",
    "total_amount":           "total_amt",
    "tpep_pickup_datetime":   "pickup_datetime",
    "tpep_dropoff_datetime":  "dropoff_datetime",
    "pickup_longitude":       "pickup_longitude",
    "pickup_latitude":        "pickup_latitude",
    "dropoff_longitude":      "dropoff_longitude",
    "dropoff_latitude":       "dropoff_latitude",
    "RatecodeID":             "rate_code",
    "store_and_fwd_flag":     "store_and_fwd_flag",
}

# ---------------------------------------------------------------------------
# Blog benchmark parameters
# ---------------------------------------------------------------------------
TIP_FILTER_MIN = 1      # filter: tip_amt >= 1
TIP_FILTER_MAX = 5      # filter: tip_amt < 5

# Vendor lookup used for join operations
VENDOR_LOOKUP = {
    "vendor_name": ["CMT", "VTS", "DDS", "1", "2"],
    "vendor_full":  [
        "Creative Mobile Technologies",
        "VeriFone Transportation Systems",
        "Digital Dispatch Systems",
        "Creative Mobile Technologies",
        "VeriFone Transportation Systems",
    ],
}

# ---------------------------------------------------------------------------
# ML settings
# ---------------------------------------------------------------------------
TARGET_COL = "fare_amt"
FEATURE_COLS = [
    "trip_distance", "passenger_count", "tip_amt",
    "tolls_amt", "surcharge", "mta_tax",
]
RANDOM_STATE = 42
N_CV_FOLDS = 5
