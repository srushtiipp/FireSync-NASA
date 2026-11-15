"""
preprocessing.py - FireSync
Clean MODIS + VIIRS active-fire detections and aggregate them to a weekly,
gridded table.

Run from the project root:   python src/preprocessing.py
Outputs:
    data/processed/detections_clean.parquet   (one row per detection)
    data/processed/weekly_grid.parquet        (sensor x platform x week x cell)
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ config
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

SOURCES = {
    "MODIS": "modis/*.csv",
    "VIIRS": "viirs/*.csv",
}

# Region of interest (approx. Punjab + Haryana) - adjust to match your downloads
LAT_MIN, LAT_MAX = 27.5, 32.6
LON_MIN, LON_MAX = 73.8, 77.7

GRID_DEG = 0.25                  # grid cell size in degrees
MODIS_MIN_CONF = 30              # drops MODIS "low" confidence (0-29)
VIIRS_KEEP_CONF = {"n", "h"}     # drops VIIRS "low"

USECOLS = ["latitude", "longitude", "scan", "acq_date", "acq_time",
           "satellite", "confidence", "frp", "daynight", "type"]


# ------------------------------------------------------------------ steps
def load_sensor(pattern: str) -> pd.DataFrame:
    files = sorted(RAW_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match {RAW_DIR / pattern}")
    return pd.concat((pd.read_csv(f, usecols=USECOLS) for f in files),
                     ignore_index=True)


def clean(df: pd.DataFrame, sensor: str):
    """Filter and derive columns. Returns (clean_df, attrition_log)."""
    log = {"raw": len(df)}

    df = df[df["latitude"].between(LAT_MIN, LAT_MAX)
            & df["longitude"].between(LON_MIN, LON_MAX)]
    log["in_region"] = len(df)

    df = df[df["type"] == 0]                      # presumed vegetation fire only
    log["type_0"] = len(df)

    if sensor == "MODIS":
        df = df[df["confidence"].astype(int) >= MODIS_MIN_CONF]
    else:
        df = df[df["confidence"].astype(str).str.lower().isin(VIIRS_KEEP_CONF)]
    log["confidence_ok"] = len(df)

    df = df.copy()
    df["sensor"] = sensor
    df["platform"] = df["satellite"]
    df["confidence"] = df["confidence"].astype(str)   # avoid mixed int/str in parquet

    # acq_time is UTC HHMM without zero padding (753 -> 07:53)
    hhmm = df["acq_time"].astype(int).astype(str).str.zfill(4)
    df["acq_datetime"] = pd.to_datetime(
        df["acq_date"] + " " + hhmm.str[:2] + ":" + hhmm.str[2:]
    )

    # Weekly bin (Monday start) and grid cell
    day = pd.to_datetime(df["acq_date"])
    df["week_start"] = day - pd.to_timedelta(day.dt.day_of_week, unit="D")
    df["cell_lat"] = np.floor(df["latitude"] / GRID_DEG) * GRID_DEG
    df["cell_lon"] = np.floor(df["longitude"] / GRID_DEG) * GRID_DEG

    keep = ["sensor", "platform", "acq_datetime", "week_start", "latitude",
            "longitude", "cell_lat", "cell_lon", "frp", "scan", "daynight",
            "confidence"]
    return df[keep], log


def aggregate(det: pd.DataFrame) -> pd.DataFrame:
    return (
        det.groupby(["sensor", "platform", "week_start", "cell_lat", "cell_lon"],
                    as_index=False)
           .agg(count=("frp", "size"),
                frp_sum=("frp", "sum"),
                frp_mean=("frp", "mean"),
                frp_max=("frp", "max"),
                scan_mean=("scan", "mean"))
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cleaned, attrition = [], {}
    for sensor, pattern in SOURCES.items():
        print(f"Loading {sensor} ...")
        df, log = clean(load_sensor(pattern), sensor)
        cleaned.append(df)
        attrition[sensor] = log

    det = pd.concat(cleaned, ignore_index=True)
    det.to_parquet(OUT_DIR / "detections_clean.parquet", index=False)

    weekly = aggregate(det)
    weekly.to_parquet(OUT_DIR / "weekly_grid.parquet", index=False)

    # ---------------------------------------------------------- reports
    print("\nFILTER ATTRITION (rows remaining after each step)")
    print(pd.DataFrame(attrition).T.to_string())

    print("\nDATE RANGE PER SENSOR")
    print(det.groupby("sensor")["acq_datetime"].agg(["min", "max"]).to_string())

    print("\nDETECTIONS PER PLATFORM")
    print(det.groupby(["sensor", "platform"]).size().to_string())

    # First MODIS vs VIIRS overlap check (region-level weekly totals)
    totals = (weekly.groupby(["week_start", "sensor"])["count"]
                    .sum().unstack(fill_value=0))
    first = det.groupby("sensor")["week_start"].min().max()
    last = det.groupby("sensor")["week_start"].max().min()
    overlap = totals.loc[first:last]

    print(f"\nOVERLAP CHECK: {len(overlap)} weeks with both sensors "
          f"({first.date()} to {last.date()})")
    if len(overlap) > 2 and overlap["MODIS"].sum() > 0:
        print("Correlation of weekly detection counts:")
        print(overlap.corr().round(3).to_string())
        ratio = overlap["VIIRS"].sum() / overlap["MODIS"].sum()
        print(f"VIIRS / MODIS total detections: {ratio:.2f}x")

    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()