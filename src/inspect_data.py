
import pandas as pd
from pathlib import Path

# Paths to our raw datasets
DATASETS = {
    "MODIS": Path("data/raw/modis/modis_haryana_delhi.csv"),
    "VIIRS S-NPP": Path("data/raw/viirs/viirs_snpp_haryana_delhi.csv")
}

# Read only 10,000 rows initially
SAMPLE_SIZE = 10_000


def inspect_dataset(sensor, file_path):

    print("\n" + "=" * 70)
    print(f"INSPECTING {sensor}")
    print("=" * 70)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    # Load a small sample
    df = pd.read_csv(file_path, nrows=SAMPLE_SIZE)

    print("\n1. SAMPLE ROW COUNT")
    print("Rows loaded:", len(df))

    print("\n2. COLUMN NAMES")
    print(df.columns.tolist())

    print("\n3. DATA TYPES")
    print(df.dtypes)

    print("\n4. DATE RANGE")

    if "acq_date" in df.columns:
        dates = pd.to_datetime(df["acq_date"], errors="coerce")

        print("Start date:", dates.min())
        print("End date:", dates.max())
    else:
        print("acq_date column not found")

    print("\n5. FIRST FIVE RECORDS")
    print(df.head())

    print("\n6. MISSING VALUES")
    print(df.isnull().sum())

    print("\n7. CONFIDENCE DISTRIBUTION")

    if "confidence" in df.columns:
        print(df["confidence"].value_counts(dropna=False).head(15))

    print("\n8. ACQUISITION TIME FORMAT")

    if "acq_time" in df.columns:
        print(df["acq_time"].head(10).tolist())


for sensor, path in DATASETS.items():
    inspect_dataset(sensor, path)
