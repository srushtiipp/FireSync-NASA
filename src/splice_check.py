"""
splice_check.py - does the calibrated MODIS estimate reproduce observed VIIRS season
totals and timing in the overlap years (2012-2021)?

Run from the project root:
    python src/splice_check.py
    python src/splice_check.py --target frp

Caveat: the final calibration was fitted on 2012-2021, so this is an in-sample check.
It catches a systematic bias in totals or timing; it does not prove the estimate
is accurate for years before 2012.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seasons import season_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/harmonized_weekly.csv")
    ap.add_argument("--target", choices=["count", "frp"], default="count")
    ap.add_argument("--start", type=int, default=2012)
    ap.add_argument("--end", type=int, default=2021)
    args = ap.parse_args()

    d = pd.read_csv(args.input, parse_dates=["week_start"], index_col="week_start")
    c_obs, c_est = f"{args.target}_viirs_observed", f"{args.target}_viirs_equiv"
    missing = [c for c in (c_obs, c_est) if c not in d.columns]
    if missing:
        print(f"Missing columns {missing}. Columns present: {list(d.columns)}")
        return

    obs = season_table(d[c_obs].dropna())
    est = season_table(d[c_est].dropna())
    m = obs.merge(est, on=["season", "year"], suffixes=("_obs", "_est"))
    m = m[m.year.between(args.start, args.end)].copy()
    m["total_ratio"] = m.total_est / m.total_obs
    m["centroid_diff_days"] = m.centroid_off_est - m.centroid_off_obs
    m["peak_diff_days"] = m.peak_off_est - m.peak_off_obs

    print("estimate minus observed, per season "
          "(positive days = the MODIS estimate is later than observed VIIRS)")
    for name, g in m.groupby("season"):
        print(f"\n=== {name.upper()} {args.start}-{args.end} ===")
        cols = ["year", "total_obs", "total_est", "total_ratio",
                "centroid_diff_days", "peak_diff_days"]
        print(g[cols].round({"total_obs": 0, "total_est": 0, "total_ratio": 2,
                             "centroid_diff_days": 1, "peak_diff_days": 1})
              .to_string(index=False))
        print(f"median total ratio (est/obs): {g.total_ratio.median():.2f}")
        print(f"centroid diff: mean {g.centroid_diff_days.mean():+.1f} days, "
              f"mean absolute {g.centroid_diff_days.abs().mean():.1f} days")
        print(f"peak diff:     mean {g.peak_diff_days.mean():+.1f} days, "
              f"mean absolute {g.peak_diff_days.abs().mean():.1f} days")


if __name__ == "__main__":
    main()