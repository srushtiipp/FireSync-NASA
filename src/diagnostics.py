"""
diagnostics.py - FireSync
First look at how well MODIS and VIIRS agree, and where they don't.

Run from the project root:   python src/diagnostics.py
Reads   data/processed/*.parquet
Writes  reports/*.html  (open in a browser)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

PROC = Path("data/processed")
REPORTS = Path("reports")
TRAIN_END_YEAR = 2023            # train on <= 2023, test on later years


def weekly_series(weekly, sensor, platform=None):
    w = weekly[weekly["sensor"] == sensor]
    if platform:
        w = w[w["platform"] == platform]
    return w.groupby("week_start")["count"].sum()


def iso_year(index):
    # week_start is a Monday; +3 days = Thursday, which gives the ISO year
    return (index + pd.Timedelta(days=3)).year.to_numpy()


def evaluate(name, modis, viirs, idx):
    ts = pd.DataFrame({"MODIS": modis.reindex(idx, fill_value=0),
                       "VIIRS": viirs.reindex(idx, fill_value=0)}).astype(float)
    years = iso_year(ts.index)

    print(f"\n=== {name} ===")
    woy = ts.index.isocalendar().week.astype(int).to_numpy()
    resid = ts - ts.groupby(woy).transform("mean")   # remove seasonal pattern
    print(f"Weekly correlation:              {ts['MODIS'].corr(ts['VIIRS']):.3f}")
    print(f"After removing seasonal pattern: {resid['MODIS'].corr(resid['VIIRS']):.3f}")
    print(f"VIIRS/MODIS ratio overall:       {ts['VIIRS'].sum() / ts['MODIS'].sum():.2f}x")

    by_year = ts.groupby(years).sum()
    print("\nVIIRS/MODIS ratio by year:")
    print((by_year["VIIRS"] / by_year["MODIS"]).round(2).to_string())

    bins = pd.qcut(ts["VIIRS"], 4, duplicates="drop")
    g = ts.groupby(bins, observed=True).sum()
    print("\nVIIRS/MODIS ratio by activity level (quiet -> busy weeks):")
    print((g["VIIRS"] / g["MODIS"]).round(2).to_string())

    tr, te = years <= TRAIN_END_YEAR, years > TRAIN_END_YEAR
    x, y = ts["MODIS"].to_numpy(), ts["VIIRS"].to_numpy()
    ratio = y[tr].sum() / x[tr].sum()
    slope, intercept = np.polyfit(x[tr], y[tr], 1)

    def wape(pred):   # weighted absolute percentage error on the test years
        return np.abs(pred[te] - y[te]).sum() / y[te].sum()

    print(f"\nHoldout (train <= {TRAIN_END_YEAR}, test after), WAPE, lower is better:")
    print(f"  constant ratio ({ratio:.2f}x): {wape(ratio * x):.1%}")
    print(f"  linear fit:               {wape(slope * x + intercept):.1%}")
    return ts, ratio


def main():
    REPORTS.mkdir(exist_ok=True)
    weekly = pd.read_parquet(PROC / "weekly_grid.parquet")
    det = pd.read_parquet(PROC / "detections_clean.parquet")

    rng = weekly.groupby("sensor")["week_start"].agg(["min", "max"])
    idx = pd.date_range(rng["min"].max(), rng["max"].min(), freq="7D")
    viirs = weekly_series(weekly, "VIIRS")

    ts, ratio = evaluate("MODIS (Terra + Aqua) vs VIIRS",
                         weekly_series(weekly, "MODIS"), viirs, idx)
    evaluate("MODIS Aqua only vs VIIRS",
             weekly_series(weekly, "MODIS", "Aqua"), viirs, idx)

    # Overpass-time drift: median local solar hour of daytime detections
    day = det[det["daynight"] == "D"].copy()
    hour = day["acq_datetime"].dt.hour + day["acq_datetime"].dt.minute / 60
    day["solar_hour"] = (hour + day["longitude"] / 15) % 24
    year = day["acq_datetime"].dt.year.rename("year")
    drift = day.groupby([year, "platform"])["solar_hour"].median().unstack().round(2)
    print("\nMedian local solar hour of daytime detections, by year:")
    print(drift.to_string())

    # Plots
    scaled = pd.DataFrame({"VIIRS": ts["VIIRS"],
                           f"MODIS x {ratio:.2f}": ts["MODIS"] * ratio})
    px.line(scaled, title="Weekly detections: VIIRS vs scaled MODIS"
            ).write_html(str(REPORTS / "weekly_series.html"))

    pos = ts[(ts > 0).all(axis=1)]
    px.scatter(pos, x="MODIS", y="VIIRS", log_x=True, log_y=True,
               color=iso_year(pos.index).astype(str).tolist(),
               title="Weekly MODIS vs VIIRS detections (log-log)"
               ).write_html(str(REPORTS / "scatter.html"))

    px.line(drift, markers=True,
            title="Median local solar hour of daytime detections"
            ).write_html(str(REPORTS / "overpass_drift.html"))

    print(f"\nPlots saved to {REPORTS}/")


if __name__ == "__main__":
    main()