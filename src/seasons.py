"""
seasons.py - season-level burning metrics and unusual-season flags.

Library use (e.g. from the Streamlit app, for any weekly series):
    from seasons import season_table, add_baselines, describe_season
    tbl = add_baselines(season_table(series, source))

CLI (uses data/processed/harmonized_weekly.csv):
    python src/seasons.py
    python src/seasons.py --target frp --trailing 5 --z-thresh 2 --timing-days 7

Seasons (by week-start date; edit SEASONS to change):
    rabi    1 Apr - 31 May   (wheat residue)
    kharif  1 Oct - 30 Nov   (paddy residue)

Per season and year:
    total, peak week, onset (10% of season total reached), end (90% reached),
    centroid (activity-weighted mean date), coverage and data confidence.

Baselines (two views of "typical"):
    all-years  : the same season in every OTHER valid year (leave-one-out).
    trailing   : the same season in the previous N valid years.

Flags:
    level_flag  : robust z of log1p(total) vs the all-years baseline, |z| >= z_thresh.
    timing_flag : centroid at least timing_days earlier/later than the TRAILING median.
                  (Against the all-years median, a steady drift in timing would flag
                  the early years just for being early, so the trailing baseline is used.)
    run         : number of consecutive valid seasons with the same level flag, so a
                  lasting change (e.g. 3 low seasons in a row) is visible.

Scoring rules:
    - A season is "valid" only if it is complete (>= min_cov of its weeks present)
      and has no Terra-only weeks. Invalid seasons are kept in the table but not
      scored, and are excluded from every baseline and rank.
    - With ~25 years, rank and ratio are more honest than a z-score; the z-score is
      only a screening flag.
    - A detection count is not burned area. A low season means fewer detections.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from anomalies import load_series          # python src/seasons.py
except ImportError:                              # imported as a package
    from src.anomalies import load_series

PROCESSED = Path("data/processed")
REPORTS = Path("reports")

SEASONS = {"rabi": ((4, 1), (5, 31)), "kharif": ((10, 1), (11, 30))}
MID = pd.Timedelta(days=3)   # week_start (Monday) -> mid-week date


# ------------------------------------------------------------------ metrics
def _window(year: int, season: str, seasons=SEASONS):
    (m1, d1), (m2, d2) = seasons[season]
    return pd.Timestamp(year, m1, d1), pd.Timestamp(year, m2, d2)


def season_table(s: pd.Series, source: pd.Series | None = None,
                 seasons=SEASONS, min_cov: float = 0.9) -> pd.DataFrame:
    """One row per (season, year) with timing and total metrics."""
    s = s.astype(float).sort_index()
    if source is None:
        source = pd.Series("observed", index=s.index)
    rows = []
    for year in range(s.index.min().year, s.index.max().year + 1):
        for name in seasons:
            start, end = _window(year, name, seasons)
            expected = pd.date_range(start, end, freq="W-MON")
            if len(expected) == 0:
                continue
            vals = s.reindex(expected)
            src = source.reindex(expected)
            v = vals.dropna()
            if v.empty:
                continue
            cov = len(v) / len(expected)
            total = float(v.sum())
            mid = v.index + MID
            row = dict(season=name, year=year, total=total, weeks=len(v),
                       coverage=round(cov, 2),
                       n_terra=int((src == "terra_only").sum()),
                       n_harmonized=int((src == "harmonized").sum()))
            if total > 0:
                cum = (v.cumsum() / total).to_numpy()
                offs = np.asarray((mid - start).days, dtype=float)
                cen_off = float(np.average(offs, weights=v.to_numpy()))
                row.update(
                    peak_date=v.idxmax() + MID, peak_value=float(v.max()),
                    onset=mid[int(np.argmax(cum >= 0.10))],
                    end=mid[int(np.argmax(cum >= 0.90))],
                    centroid=start + pd.Timedelta(days=cen_off),
                    centroid_off=cen_off,
                    peak_off=float((v.idxmax() + MID - start).days))
            rows.append(row)
    tbl = pd.DataFrame(rows)
    tbl["complete"] = tbl["coverage"] >= min_cov
    tbl["confidence"] = np.where(tbl.n_terra > 0, "low (Terra-only)",
                         np.where(tbl.n_harmonized > 0, "estimated", "observed"))
    tbl["valid"] = tbl["complete"] & (tbl["n_terra"] == 0)
    return tbl


# ---------------------------------------------------------------- baselines
def add_baselines(tbl: pd.DataFrame, trailing: int = 5, z_thresh: float = 2.0,
                  timing_days: float = 7.0, min_scale: float = 0.3,
                  min_others: int = 5, min_trail: int = 3) -> pd.DataFrame:
    out = tbl.copy()
    for c in ["baseline_all", "ratio_all", "z_all", "rank_high", "n_valid",
              "baseline_trail", "ratio_trail", "n_trail",
              "centroid_shift_all", "centroid_shift_trail"]:
        out[c] = np.nan

    for _, g in out.groupby("season"):
        gv = g[g["valid"]]
        if gv.empty:
            continue
        ranks = gv["total"].rank(ascending=False, method="min")
        out.loc[ranks.index, "rank_high"] = ranks
        out.loc[gv.index, "n_valid"] = len(gv)
        for idx, row in gv.iterrows():
            others = gv.drop(idx)
            if len(others) >= min_others:
                lt = np.log1p(others["total"].to_numpy(float))
                med = np.median(lt)
                scale = max(1.4826 * np.median(np.abs(lt - med)), min_scale)
                base = float(np.expm1(med))
                out.loc[idx, "baseline_all"] = base
                out.loc[idx, "ratio_all"] = row["total"] / max(base, 1e-9)
                out.loc[idx, "z_all"] = (np.log1p(row["total"]) - med) / scale
                out.loc[idx, "centroid_shift_all"] = (
                    row["centroid_off"] - others["centroid_off"].median())
            prev = gv[gv["year"] < row["year"]].tail(trailing)
            if len(prev) >= min_trail:
                base = float(prev["total"].median())
                out.loc[idx, "baseline_trail"] = base
                out.loc[idx, "ratio_trail"] = row["total"] / max(base, 1e-9)
                out.loc[idx, "n_trail"] = len(prev)
                out.loc[idx, "centroid_shift_trail"] = (
                    row["centroid_off"] - prev["centroid_off"].median())

    out["level_flag"] = np.where(out.z_all >= z_thresh, "high",
                         np.where(out.z_all <= -z_thresh, "low", ""))

    # timing is judged against the trailing median so a steady drift is not flagged
    sh = out["centroid_shift_trail"]
    out["timing_flag"] = np.where(sh >= timing_days, "later",
                          np.where(sh <= -timing_days, "earlier", ""))

    # consecutive seasons with the same level flag (e.g. 3 low seasons in a row)
    out["run"] = 0
    for _, g in out.groupby("season"):
        n, prev_flag = 0, ""
        for idx, row in g.sort_values("year").iterrows():
            flag = row["level_flag"]
            if flag and flag == prev_flag:
                n += 1
            elif flag:
                n = 1
            else:
                n = 0
            out.loc[idx, "run"] = n
            prev_flag = flag
    return out


# -------------------------------------------------------------------- text
def describe_season(row) -> str:
    """Template sentence for one season (every number is checkable)."""
    name = f"{row.season.capitalize()} {int(row.year)}"
    if not row.valid:
        why = ("incomplete season" if not row.complete
               else "Terra-only estimate, low confidence")
        return f"{name}: {row.total:,.0f} detections, not scored ({why})."
    parts = [f"{name}: {row.total:,.0f} detections"]
    if pd.notna(row.ratio_all):
        parts.append(f"{row.ratio_all:.2f}x the typical season ({row.baseline_all:,.0f})")
    if pd.notna(row.rank_high):
        parts.append(f"rank {int(row.rank_high)} of {int(row.n_valid)} (1 = highest)")
    if pd.notna(row.ratio_trail):
        parts.append(f"{row.ratio_trail:.2f}x the previous {int(row.n_trail)} seasons")
    text = "; ".join(parts) + "."
    if row.timing_flag:
        text += (f" Activity was centred {abs(row.centroid_shift_trail):.0f} days "
                 f"{row.timing_flag} than in the previous seasons.")
    if row.run >= 2:
        text += f" That is {int(row.run)} {row.level_flag} seasons in a row."
    return text


# --------------------------------------------------------------------- CLI
def _fmt_dates(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_datetime(df[c]).dt.strftime("%d %b").fillna("")
    return df


def plot(res: pd.DataFrame, target: str):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed; skipping plot.")
        return
    names = list(res["season"].unique())
    fig = make_subplots(rows=len(names), cols=2, horizontal_spacing=0.08,
                        subplot_titles=[t for n in names for t in
                                        (f"{n.capitalize()}: season total",
                                         f"{n.capitalize()}: centroid (days after season start)")])
    for r, name in enumerate(names, start=1):
        g = res[res.season == name]
        colors = np.where(~g.valid, "lightgrey",
                  np.where(g.level_flag == "high", "orange",
                  np.where(g.level_flag == "low", "crimson", "steelblue")))
        fig.add_trace(go.Bar(x=g.year, y=g.total, marker_color=list(colors),
                             name="season total", showlegend=False), row=r, col=1)
        fig.add_trace(go.Scatter(x=g.year, y=g.baseline_trail, mode="lines",
                                 line=dict(color="black", dash="dot"),
                                 name="trailing median", showlegend=(r == 1)),
                      row=r, col=1)
        fig.add_trace(go.Scatter(x=g.year, y=g.centroid_off, mode="lines+markers",
                                 line=dict(color="darkgreen"), showlegend=False),
                      row=r, col=2)
    fig.update_layout(title=f"Season-level {target} activity "
                            "(orange/red = flagged, grey = not scored)",
                      height=330 * len(names) + 80, hovermode="x unified")
    out = REPORTS / f"seasons_{target}.html"
    fig.write_html(out)
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(PROCESSED / "harmonized_weekly.csv"))
    ap.add_argument("--target", choices=["count", "frp"], default="count")
    ap.add_argument("--trailing", type=int, default=5)
    ap.add_argument("--z-thresh", type=float, default=2.0)
    ap.add_argument("--timing-days", type=float, default=7.0)
    ap.add_argument("--min-cov", type=float, default=0.9)
    args = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)

    s, source = load_series(Path(args.input), args.target)
    tbl = season_table(s, source, min_cov=args.min_cov)
    res = add_baselines(tbl, trailing=args.trailing, z_thresh=args.z_thresh,
                        timing_days=args.timing_days)
    res.to_csv(PROCESSED / f"seasons_{args.target}.csv", index=False)

    print(f"Series: {len(s)} weeks, {s.index.min().date()} to {s.index.max().date()}")
    print(f"Seasons: {len(res)} rows, {int(res.valid.sum())} valid "
          f"(complete and no Terra-only weeks)")
    bad = res[~res.valid]
    if len(bad):
        print("Not scored: " + ", ".join(
            f"{r.season} {int(r.year)}" for r in bad.itertuples()))

    cols = ["year", "total", "rank_high", "ratio_all", "ratio_trail", "centroid",
            "peak_date", "onset", "end", "level_flag", "run", "timing_flag", "confidence"]
    for name in res["season"].unique():
        d = _fmt_dates(res[res.season == name][cols],
                       ["centroid", "peak_date", "onset", "end"])
        d[["ratio_all", "ratio_trail"]] = d[["ratio_all", "ratio_trail"]].round(2)
        d["total"] = d["total"].round(0).astype(int)
        print(f"\n=== {name.upper()} (rank 1 = highest total; ratios vs median of other / "
              f"previous {args.trailing} valid years) ===")
        print(d.to_string(index=False))

    flagged = res[(res.level_flag != "") | (res.timing_flag != "")].copy()
    flagged["_a"] = flagged.z_all.abs().fillna(0)
    print("\nUnusual seasons (level or timing flag), strongest first:")
    for _, row in flagged.sort_values("_a", ascending=False).iterrows():
        print("  " + describe_season(row))

    v = res[res.valid].copy()
    v["period"] = (v.year // 10 * 10).astype(str) + "s"
    era = (v.groupby(["season", "period"])
             .agg(seasons=("year", "count"), median_total=("total", "median"),
                  median_centroid_day=("centroid_off", "median"),
                  median_peak_day=("peak_off", "median")).round(0))
    print("\nBy decade (descriptive only; day = days after season start):")
    print(era.to_string())

    plot(res, args.target)
    print(f"\nSaved {PROCESSED / f'seasons_{args.target}.csv'}")


if __name__ == "__main__":
    main()