"""
anomalies.py - seasonal anomalies and critical periods for a weekly activity series.

Library use (e.g. from the Streamlit app, for any AOI series):
    from anomalies import detect_anomalies, climatology, peak_window, describe
    res = detect_anomalies(series)        # series: pd.Series indexed by week_start

CLI (uses data/processed/harmonized_weekly.csv):
    python src/anomalies.py
    python src/anomalies.py --target frp --z-thresh 3 --min-count 100

Method
    1. Seasonal baseline: for each week, the median and robust scale (1.4826*MAD)
       of log1p(activity) in the same time of year (+/- half_window weeks) in
       OTHER years. Leaving the year out stops a big year from hiding itself.
    2. Primary flag: robust z-score = (log1p(value) - median) / scale, |z| >= z_thresh.
       Scale has a floor so near-constant quiet seasons don't flag on noise.
    3. Secondary flag: Isolation Forest on [z, 4-week mean z, week-over-week change],
       which catches sustained or abrupt shifts the z-score alone misses. It always
       flags ~contamination of weeks, so a flag only counts if |z| or the 4-week
       mean |z| is at least if_min_z (default 1.5).
    4. Count floor: weeks where both the value and the baseline are below
       min_count are never flagged (removes noise like "0 vs 4").
    5. 'trigger' says which flagged the week: z, iforest or both.
    6. Critical periods: top fraction of weeks within each year, plus a peak-season
       window from the climatology.
    7. Terra-only weeks (before Aqua launched) are left unscored and excluded from
       the baseline, because their magnitudes are low confidence.

Caveat: there is no trend removal. If activity trends up or down over the record,
late years will be flagged as consistently high or low. The CLI prints flags per
year so you can see this.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

PROCESSED = Path("data/processed")
REPORTS = Path("reports")


# ----------------------------------------------------------------- helpers
def _woy(idx: pd.DatetimeIndex) -> np.ndarray:
    """ISO week of year, folded to 1..52."""
    return np.minimum(idx.isocalendar().week.astype(int).to_numpy(), 52)


def _iso_year(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.isocalendar().year.astype(int).to_numpy()


def woy_label(w: int) -> str:
    """Approximate calendar date of the Monday of ISO week w (for readable text)."""
    return pd.Timestamp.fromisocalendar(2021, int(w), 1).strftime("%d %b")


# ---------------------------------------------------------------- baseline
def seasonal_baseline(s: pd.Series, half_window: int = 2, min_pool: int = 8,
                      min_scale: float = 0.3):
    """Leave-one-year-out seasonal median and robust scale on the log1p scale."""
    y = np.log1p(s.astype(float)).to_numpy()
    woy, yr = _woy(s.index), _iso_year(s.index)
    med = np.full(len(s), np.nan)
    scale = np.full(len(s), np.nan)
    for i in range(len(s)):
        d = np.abs(woy - woy[i])
        d = np.minimum(d, 52 - d)  # circular distance across the year boundary
        mask = (d <= half_window) & (yr != yr[i]) & np.isfinite(y)
        if mask.sum() < min_pool:
            continue
        pool = y[mask]
        m = np.median(pool)
        mad = 1.4826 * np.median(np.abs(pool - m))
        med[i], scale[i] = m, max(mad, min_scale)
    return pd.Series(med, index=s.index), pd.Series(scale, index=s.index)


# --------------------------------------------------------------- detection
def critical_weeks(s: pd.Series, top_frac: float = 0.10) -> pd.Series:
    """True for the top `top_frac` of weeks within each year."""
    yr = _iso_year(s.index)
    rank = s.groupby(yr).transform(lambda g: g.rank(pct=True, ascending=False))
    return rank <= top_frac


def detect_anomalies(s: pd.Series, z_thresh: float = 2.5, contamination: float = 0.05,
                     half_window: int = 2, top_frac: float = 0.10,
                     if_min_z: float = 1.5, min_count: float = 20) -> pd.DataFrame:
    s = s.astype(float).sort_index()
    med, scale = seasonal_baseline(s, half_window=half_window)
    y = np.log1p(s)
    z = (y - med) / scale

    out = pd.DataFrame({"value": s, "baseline": np.expm1(med), "scale": scale, "z": z})
    out["direction"] = np.where(z > 0, "high", "low")

    # ignore tiny-count weeks (NaN weeks compare False, so they never flag)
    big = np.maximum(s, np.expm1(med)) >= min_count
    out["z_flag"] = (z.abs() >= z_thresh) & big

    feat = pd.DataFrame({
        "z": z.clip(-6, 6),
        "z4": z.rolling(4, min_periods=1).mean().clip(-6, 6),
        "dy": y.diff(),
    })
    ok = feat.notna().all(axis=1)
    out["if_flag"] = False
    if ok.sum() >= 50:
        iso = IsolationForest(n_estimators=200, contamination=contamination,
                              random_state=0)
        raw = pd.Series(False, index=s.index)
        raw[ok] = iso.fit_predict(feat[ok]) == -1
        # Isolation Forest always flags `contamination` of weeks, even on pure noise,
        # so only keep flags backed by at least a moderate deviation.
        moderate = (feat.z.abs() >= if_min_z) | (feat.z4.abs() >= if_min_z)
        out["if_flag"] = raw & moderate & big

    out["trigger"] = np.select(
        [out.z_flag & out.if_flag, out.z_flag, out.if_flag],
        ["both", "z", "iforest"], default="")
    out["flagged"] = out.trigger != ""
    out["critical"] = critical_weeks(s, top_frac)
    return out


# ------------------------------------------------------------ climatology
def climatology(s: pd.Series) -> pd.DataFrame:
    g = pd.Series(s.to_numpy(float), index=_woy(s.index)).groupby(level=0)
    return pd.DataFrame({"median": g.median(), "p25": g.quantile(0.25),
                         "p75": g.quantile(0.75)})


def peak_window(clim: pd.DataFrame, frac: float = 0.5):
    """Contiguous weeks around the climatological peak with median >= frac * peak."""
    peak = int(clim["median"].idxmax())
    thr = frac * clim["median"].max()
    lo = hi = peak
    while (lo - 1) in clim.index and clim.loc[lo - 1, "median"] >= thr:
        lo -= 1
    while (hi + 1) in clim.index and clim.loc[hi + 1, "median"] >= thr:
        hi += 1
    return lo, hi, peak


def describe(ts, row) -> str:
    """Template text for one flagged week (no LLM: every number is checkable)."""
    side = "above" if row.z > 0 else "below"
    return (f"Week of {ts:%d %b %Y}: {row.value:,.0f} vs typical {row.baseline:,.0f} "
            f"for this time of year ({abs(row.z):.1f} sigma {side} baseline; "
            f"flagged by {row.trigger}).")


# --------------------------------------------------------------------- CLI
def load_series(path: Path, target: str):
    df = pd.read_csv(path, parse_dates=["week_start"], index_col="week_start")
    obs, eq = df[f"{target}_viirs_observed"], df[f"{target}_viirs_equiv"]
    s = obs.fillna(eq)
    ct = df["calib_source"] if "calib_source" in df else pd.Series("", index=df.index)
    source = pd.Series(np.where(obs.notna(), "observed",
                       np.where(ct == "terra_only", "terra_only", "harmonized")),
                       index=df.index)
    # the first and last weeks are partial (data start/end mid-week) and read as false "lows"
    return s.iloc[1:-1], source.iloc[1:-1]


def plot(res: pd.DataFrame, target: str):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed; skipping plot.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res.index, y=res.value, name="activity",
                             line=dict(color="black", width=1.5)))
    fig.add_trace(go.Scatter(x=res.index, y=res.baseline, name="seasonal baseline",
                             line=dict(color="steelblue", dash="dot")))
    for trig, color in [("z", "orange"), ("iforest", "purple"), ("both", "crimson")]:
        f = res[res.trigger == trig]
        fig.add_trace(go.Scatter(x=f.index, y=f.value, mode="markers",
                                 name=f"flag: {trig}", marker=dict(color=color, size=8)))
    fig.update_yaxes(type="log")
    fig.update_layout(title=f"Weekly {target} activity with anomaly flags (log scale)",
                      hovermode="x unified", height=550)
    out = REPORTS / f"anomalies_{target}.html"
    fig.write_html(out)
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(PROCESSED / "harmonized_weekly.csv"))
    ap.add_argument("--target", choices=["count", "frp"], default="count")
    ap.add_argument("--z-thresh", type=float, default=2.5)
    ap.add_argument("--contamination", type=float, default=0.05)
    ap.add_argument("--min-count", type=float, default=20,
                    help="ignore weeks where value and baseline are below this "
                         "(use ~100 for --target frp)")
    args = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)

    s, source = load_series(Path(args.input), args.target)
    print(f"Series: {len(s)} weeks, {s.index.min().date()} to {s.index.max().date()}; "
          f"sources: {source.value_counts().to_dict()}")

    # Terra-only weeks are low confidence: leave them unscored and out of the baseline
    s_scored = s.where(source != "terra_only")

    res = detect_anomalies(s_scored, args.z_thresh, args.contamination,
                           min_count=args.min_count)
    res["source"] = source
    res.to_csv(PROCESSED / f"anomalies_{args.target}.csv")

    print(f"\nFlagged weeks: {int(res.flagged.sum())} of {res.z.notna().sum()} scored "
          f"(by trigger: {res[res.flagged].trigger.value_counts().to_dict()})")

    yr = _iso_year(res.index)
    by_year = (res.assign(year=yr).groupby("year")
                  .apply(lambda g: pd.Series({
                      "flagged": int(g.flagged.sum()),
                      "high": int((g.flagged & (g.direction == "high")).sum()),
                      "low": int((g.flagged & (g.direction == "low")).sum())}),
                         include_groups=False))
    print("\nFlags per year (a drift to all-high or all-low suggests a trend, not anomalies):")
    print(by_year.to_string())

    mnames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    mt = s.groupby([s.index.year, s.index.month]).sum().unstack().reindex(columns=range(1, 13))
    mt.columns = mnames
    print("\nMonthly totals by year (thousands; week-start month) - shows seasonal regime changes:")
    print((mt / 1000).round(1).to_string())

    pw = (res.assign(month=res.index.month).groupby("month")
             .agg(median_scale=("scale", "median"), flagged=("flagged", "sum")))
    pw.index = [mnames[m - 1] for m in pw.index]
    print("\nDetection power by month (a large median_scale means weak sensitivity that month):")
    print(pw.round(2).T.to_string())

    top = res[res.flagged].reindex(
        res[res.flagged].z.abs().sort_values(ascending=False).index).head(15)
    print("\nTop flagged weeks:")
    for ts, row in top.iterrows():
        print("  " + describe(ts, row))

    clim = climatology(s_scored.dropna())
    lo, hi, peak = peak_window(clim)
    print(f"\nPeak season (climatology median >= 50% of peak): weeks {lo}-{hi} "
          f"(~{woy_label(lo)} to {woy_label(hi)}), peak week {peak} (~{woy_label(peak)})")

    peaks = (res.assign(year=yr).groupby("year")
            .apply(lambda g: ", ".join(
                f"{t:%d %b}" for t in g.value.dropna().nlargest(3).sort_index().index)
                or "n/a (Terra-only)",
                   include_groups=False))
    print("\nTop-3 weeks per year (Terra-only weeks excluded):")
    print(peaks.to_string())

    plot(res, args.target)
    print(f"\nSaved {PROCESSED / f'anomalies_{args.target}.csv'}")


if __name__ == "__main__":
    main()