"""
harmonize.py (v4) - MODIS -> VIIRS-equivalent calibration, with Terra-only handling.

Run from the project root:
    python src/harmonize.py
    python src/harmonize.py --cal-start 2015 --cal-end 2019
    python src/harmonize.py --modes backcast,frozen

Validation modes
    backcast : calibrate on cal_start..cal_end, test on earlier years.
    frozen   : calibrate on everything up to cal_end, test each later year
               (shows what 2022+ overpass drift does to a stable-era calibration).
    rolling  : train on all years before Y, test on Y.

Models (Terra+Aqua)
    constant, loglinear, ridge_drift, gbm_resid, gbm_season

Terra-only calibration (NEW in v4)
    Aqua only flies from mid-2002, so MODIS weeks before that contain Terra alone.
    A model fitted on combined Terra+Aqua counts would badly undercount them.
    v4 fits a second calibration using Terra counts/FRP only (models: constant,
    loglinear, gbm_season), validates it with its own backcast (2012-14 from
    2015-19), and uses it for every week up to the first Aqua week.
    It also predicts Terra-only equivalents for all weeks, so the two
    calibrations can be cross-checked on 2003-2011 where both exist.

Outputs
    reports/harmonization_<mode>.html, reports/harmonization_terra_backcast.html
    reports/harmonization_results_<mode>.csv, reports/harmonization_oof_<mode>.csv
    data/processed/harmonized_weekly.csv  (adds calib_source + *_terra_equiv columns)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

MODELS = ["constant", "loglinear", "ridge_drift", "gbm_resid", "gbm_season"]
TERRA_MODELS = ["constant", "loglinear", "gbm_season"]
FEATS = [
    "l_n_modis", "l_frp_modis", "l_n_aqua", "l_n_terra",
    "l_frp_aqua", "l_frp_terra", "aqua_share", "day_frac",
    "solar_aqua", "solar_terra", "woy_sin", "woy_cos",
]
SEASON_FEATS = ["l_n_modis", "l_frp_modis", "day_frac", "woy_sin", "woy_cos"]

# target -> (VIIRS column, MODIS column, primary log feature)
TARGETS = {
    "count": ("n_viirs", "n_modis", "l_n_modis"),
    "frp": ("frp_viirs", "frp_modis", "l_frp_modis"),
}


# ----------------------------------------------------------------- loading
def read_any(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def find_detections(arg):
    if arg:
        return Path(arg)
    cands = [p for p in sorted(PROCESSED.glob("*"))
             if p.suffix in (".parquet", ".csv") and "detect" in p.name.lower()]
    if not cands:
        files = [p.name for p in PROCESSED.glob("*")]
        sys.exit(f"No detections file found in {PROCESSED}. Files there: {files}\n"
                 "Pass it with --detections <path>.")
    print(f"Using detections file: {cands[0]}")
    return cands[0]


def pick(df, options, what):
    for o in options:
        if o in df.columns:
            return o
    sys.exit(f"Could not find a '{what}' column. Tried {options}. Columns: {list(df.columns)}")


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    s_col = pick(df, ["sensor", "instrument"], "sensor")
    p_col = pick(df, ["platform", "satellite"], "platform")
    lat_col = pick(df, ["latitude", "lat"], "latitude")
    lon_col = pick(df, ["longitude", "lon"], "longitude")

    out = pd.DataFrame({
        "sensor": df[s_col].astype(str).str.upper().str.extract(r"(MODIS|VIIRS)")[0],
        "plat_raw": df[p_col].astype(str).str.lower(),
        "frp": pd.to_numeric(df["frp"], errors="coerce").fillna(0.0),
        "lat": df[lat_col], "lon": df[lon_col],
    })
    out["platform"] = np.where(out.plat_raw.str.startswith("a"), "Aqua",
                        np.where(out.plat_raw.str.startswith("t"), "Terra", "Other"))

    dt_col = next((c for c in ["datetime", "acq_datetime", "timestamp", "acq_dt", "utc"]
                   if c in df.columns), None)
    if dt_col:
        dt = pd.to_datetime(df[dt_col])
    else:
        t = df["acq_time"].astype(int).astype(str).str.zfill(4)
        dt = pd.to_datetime(df["acq_date"].astype(str) + " " + t.str[:2] + ":" + t.str[2:])
    out["dt"] = dt
    out["week"] = dt.dt.normalize() - pd.to_timedelta(dt.dt.day_of_week, unit="D")
    utc_h = dt.dt.hour + dt.dt.minute / 60
    out["solar"] = (utc_h + out.lon / 15) % 24
    out["is_day"] = ((out.solar >= 6) & (out.solar <= 18)).astype(float)
    return out.dropna(subset=["sensor"])


# ----------------------------------------------------------- weekly table
def weekly_table(det: pd.DataFrame) -> pd.DataFrame:
    weeks = pd.date_range(det.week.min(), det.week.max(), freq="7D")
    w = pd.DataFrame(index=weeks)
    w.index.name = "week_start"
    mod = det[det.sensor == "MODIS"]
    vii = det[det.sensor == "VIIRS"]

    w["n_viirs"] = vii.groupby("week").size()
    w["frp_viirs"] = vii.groupby("week").frp.sum()
    w["n_modis"] = mod.groupby("week").size()
    w["frp_modis"] = mod.groupby("week").frp.sum()
    for plat in ["Aqua", "Terra"]:
        m = mod[mod.platform == plat]
        k = plat.lower()
        w[f"n_{k}"] = m.groupby("week").size()
        w[f"frp_{k}"] = m.groupby("week").frp.sum()
        w[f"solar_{k}"] = m[m.is_day == 1].groupby("week").solar.median()
    w["day_frac"] = mod.groupby("week").is_day.mean()

    count_cols = ["n_viirs", "frp_viirs", "n_modis", "frp_modis",
                  "n_aqua", "frp_aqua", "n_terra", "frp_terra"]
    w[count_cols] = w[count_cols].fillna(0)
    for c in ["solar_aqua", "solar_terra", "day_frac"]:
        w[c] = w[c].interpolate(limit_direction="both")
        w[c] = w[c].fillna(w[c].median())
    return w


def build_features(w: pd.DataFrame) -> pd.DataFrame:
    d = w.copy()
    d["l_n_modis"] = np.log1p(d.n_modis)
    d["l_frp_modis"] = np.log1p(d.frp_modis)
    d["l_n_aqua"] = np.log1p(d.n_aqua)
    d["l_n_terra"] = np.log1p(d.n_terra)
    d["l_frp_aqua"] = np.log1p(d.frp_aqua)
    d["l_frp_terra"] = np.log1p(d.frp_terra)
    d["aqua_share"] = d.n_aqua / d.n_modis.clip(lower=1)
    iso = d.index.isocalendar()
    woy = iso.week.astype(int).to_numpy()
    d["woy_sin"] = np.sin(2 * np.pi * woy / 52)
    d["woy_cos"] = np.cos(2 * np.pi * woy / 52)
    d["year"] = iso.year.astype(int).to_numpy()
    return d


def terra_view(d: pd.DataFrame) -> pd.DataFrame:
    """Copy of the weekly table in which the 'MODIS' columns hold Terra only.
    Lets every model in fit_predict run unchanged as a Terra-only calibration."""
    t = d.copy()
    t["n_modis"] = d["n_terra"]
    t["frp_modis"] = d["frp_terra"]
    t["l_n_modis"] = d["l_n_terra"]
    t["l_frp_modis"] = d["l_frp_terra"]
    return t


# ----------------------------------------------------------------- models
def back_transform(fit_tr, fit_te, y_raw_tr):
    """Mass-preserving back-transform from log1p space."""
    p_tr = np.clip(np.expm1(fit_tr), 0, None)
    s = y_raw_tr.sum() / max(p_tr.sum(), 1e-9)
    return np.clip(np.expm1(fit_te), 0, None) * s


def fit_predict(model, tr, te, target):
    y_col, m_col, prim = TARGETS[target]
    if model == "constant":
        r = tr[y_col].sum() / max(tr[m_col].sum(), 1e-9)
        return r * te[m_col].to_numpy()

    y_raw = tr[y_col].to_numpy()
    y_tr = np.log1p(y_raw)
    lin = LinearRegression().fit(tr[[prim]], y_tr)

    if model == "loglinear":
        return back_transform(lin.predict(tr[[prim]]), lin.predict(te[[prim]]), y_raw)

    if model == "ridge_drift":
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=3.0)).fit(tr[FEATS], y_tr)
        return back_transform(pipe.predict(tr[FEATS]), pipe.predict(te[FEATS]), y_raw)

    if model in ("gbm_resid", "gbm_season"):
        feats = FEATS if model == "gbm_resid" else SEASON_FEATS
        base_tr = lin.predict(tr[[prim]])
        gb = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=250,
            min_samples_leaf=10, random_state=0,
        ).fit(tr[feats], y_tr - base_tr)
        fit_tr = base_tr + gb.predict(tr[feats])
        fit_te = lin.predict(te[[prim]]) + gb.predict(te[feats])
        return back_transform(fit_tr, fit_te, y_raw)

    raise ValueError(model)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    rho = spearmanr(y, p).correlation if len(y) > 2 else np.nan
    return {
        "wape_%": 100 * np.abs(y - p).sum() / max(y.sum(), 1e-9),
        "mae_log": float(np.abs(np.log1p(y) - np.log1p(p)).mean()),
        "spearman": rho,
        "bias_%": 100 * (p.sum() / max(y.sum(), 1e-9) - 1),
    }


# ------------------------------------------------------------- experiments
def make_folds(d, mode, a):
    years = sorted(d.year.unique())
    folds = []
    if mode == "backcast":
        cal = d[(d.year >= a.cal_start) & (d.year <= a.cal_end)]
        folds = [(ty, cal, d[d.year == ty]) for ty in years if ty < a.cal_start]
    elif mode == "frozen":
        tr = d[d.year <= a.cal_end]
        folds = [(ty, tr, d[d.year == ty]) for ty in years if ty > a.cal_end]
    elif mode == "rolling":
        folds = [(ty, d[d.year < ty], d[d.year == ty]) for ty in years[a.min_train_years:]]
    return [f for f in folds if len(f[2]) >= 20 and len(f[1]) >= 52]


def run_experiment(d, mode, a, models=MODELS):
    folds = make_folds(d, mode, a)
    if not folds:
        print(f"\n[{mode}] no valid folds with this data (check --cal-start/--cal-end).")
        return pd.DataFrame(), pd.DataFrame()
    rows, oof = [], []
    for target, (y_col, _, _) in TARGETS.items():
        for ty, tr, te in folds:
            for m in models:
                p = fit_predict(m, tr, te, target)
                rows.append({"mode": mode, "target": target, "model": m,
                             "test_year": ty, **metrics(te[y_col], p)})
                oof.append(pd.DataFrame({
                    "mode": mode, "target": target, "model": m, "year": ty,
                    "week_start": te.index, "actual": te[y_col].to_numpy(), "pred": p}))
    return pd.DataFrame(rows), pd.concat(oof)


def pooled_table(oof, models=MODELS):
    recs = {}
    for m, g in oof.groupby("model"):
        recs[m] = metrics(g.actual, g.pred)
    return pd.DataFrame(recs).T.loc[models]


def report(res, oof, mode, models=MODELS):
    for target in TARGETS:
        r, o = res[res.target == target], oof[oof.target == target]
        if r.empty:
            continue
        yrs = sorted(int(y) for y in r.test_year.unique())
        print(f"\n=== [{mode}] {target.upper()}: pooled results, test years {yrs} ===")
        print(pooled_table(o, models).round(3).to_string())
        print(f"\n[{mode}] {target}: WAPE % by test year")
        print(r.pivot(index="model", columns="test_year", values="wape_%")
                .loc[models].round(1).to_string())
        print(f"\n[{mode}] {target}: bias % by test year (negative = undercount)")
        print(r.pivot(index="model", columns="test_year", values="bias_%")
                .loc[models].round(1).to_string())


def best_model(oof, target, allowed=None):
    o = oof[oof.target == target]
    scores = {m: float(np.abs(np.log1p(g.actual) - np.log1p(g.pred)).mean())
              for m, g in o.groupby("model") if m != "constant"
              and (allowed is None or m in allowed)}
    m = min(scores, key=scores.get)
    return m, scores[m]


# ------------------------------------------------------------------- plot
def make_plot(oof, mode, allowed=None):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed; skipping plot (pip install plotly).")
        return

    o = oof[oof.target == "count"]
    if o.empty:
        return
    best, _ = best_model(oof, "count", allowed)
    act = o[o.model == "constant"].set_index("week_start")
    drift = o[o.model == best].set_index("week_start").pred

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Linear scale", "Log scale (shows quiet-season fit)"))
    for r in (1, 2):
        s = r == 1
        fig.add_trace(go.Scatter(x=act.index, y=act.actual, name="VIIRS (target)",
                                 line=dict(color="black", width=2), showlegend=s), r, 1)
        fig.add_trace(go.Scatter(x=act.index, y=act.pred, name="MODIS x constant ratio",
                                 line=dict(color="crimson", dash="dot"), showlegend=s), r, 1)
        fig.add_trace(go.Scatter(x=drift.index, y=drift, name=f"Harmonized ({best})",
                                 line=dict(color="seagreen"), showlegend=s), r, 1)
    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_layout(title=f"[{mode}] Held-out weeks: VIIRS vs constant-ratio MODIS vs harmonized",
                      height=750, hovermode="x unified")
    out = REPORTS / f"harmonization_{mode}.html"
    fig.write_html(out)
    print(f"Saved {out}")


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", default=None)
    ap.add_argument("--cal-start", type=int, default=2015)
    ap.add_argument("--cal-end", type=int, default=2019)
    ap.add_argument("--min-train-years", type=int, default=3)
    ap.add_argument("--modes", default="backcast,frozen,rolling")
    ap.add_argument("--final-train-end", type=int, default=2021,
                    help="last year used to fit the final model (pre-drift era)")
    a = ap.parse_args()

    det = normalise(read_any(find_detections(a.detections)))
    d = build_features(weekly_table(det))

    first_mod = d.index[d.n_modis > 0].min()
    first_vii = d.index[d.n_viirs > 0].min()
    last_mod = d.index[d.n_modis > 0].max()
    last_vii = d.index[d.n_viirs > 0].max()
    start = max(first_mod, first_vii) + pd.Timedelta(days=7)
    end = min(last_mod, last_vii) - pd.Timedelta(days=7)
    ov = d.loc[start:end]
    print(f"\nOverlap: {start.date()} to {end.date()} ({len(ov)} weeks, "
          f"years {ov.year.min()}-{ov.year.max()})")
    print("Weeks per year in overlap:", ov.groupby("year").size().to_dict())

    l = np.log1p
    print("Raw weekly agreement (count): "
          f"Spearman={spearmanr(ov.n_modis, ov.n_viirs).correlation:.3f}, "
          f"log-scale Pearson={np.corrcoef(l(ov.n_modis), l(ov.n_viirs))[0, 1]:.3f}")
    ratio = ov.groupby("year").apply(lambda g: g.n_viirs.sum() / max(g.n_modis.sum(), 1),
                                     include_groups=False)
    print("VIIRS/MODIS count ratio by year:", ratio.round(2).to_dict())
    tratio = ov.groupby("year").apply(lambda g: g.n_viirs.sum() / max(g.n_terra.sum(), 1),
                                      include_groups=False)
    print("VIIRS/Terra count ratio by year:", tratio.round(1).to_dict())

    results, oofs = {}, {}
    for mode in [m.strip() for m in a.modes.split(",") if m.strip()]:
        res, oof = run_experiment(ov, mode, a)
        if res.empty:
            continue
        results[mode], oofs[mode] = res, oof
        report(res, oof, mode)
        make_plot(oof, mode)

    if not results:
        sys.exit("No experiment produced results. Need more overlap years.")
    for mode in results:
        results[mode].to_csv(REPORTS / f"harmonization_results_{mode}.csv", index=False)
        oofs[mode].to_csv(REPORTS / f"harmonization_oof_{mode}.csv", index=False)

    # ---- final fit (Terra+Aqua), applied to every week with MODIS data
    apply_df = d.loc[first_mod:].copy()
    allowed = {"loglinear", "gbm_season"}  # platform-split models are invalid pre-Aqua
    sel_mode = "backcast" if "backcast" in oofs else next(iter(oofs))
    ov_final = ov[ov.year <= a.final_train_end]
    print(f"\nFinal model fitted on {len(ov_final)} weeks "
          f"({ov_final.year.min()}-{ov_final.year.max()}).")
    out = pd.DataFrame(index=apply_df.index)
    for target, (y_col, _, _) in TARGETS.items():
        m, s = best_model(oofs[sel_mode], target, allowed)
        print(f"Final model for {target} (Terra+Aqua): {m} "
              f"(selected on [{sel_mode}] log-MAE {s:.3f})")
        out[f"{target}_viirs_equiv"] = fit_predict(m, ov_final, apply_df, target)
        out[f"{target}_viirs_observed"] = apply_df[y_col].where(apply_df.index.isin(ov.index))

    # ---- Terra-only calibration: needed for weeks before Aqua exists
    print("\n" + "=" * 70)
    print("TERRA-ONLY CALIBRATION (used where Aqua is not yet flying)")
    print("=" * 70)
    tv_ov, tv_apply, tv_final = terra_view(ov), terra_view(apply_df), terra_view(ov_final)
    tres, toof = run_experiment(tv_ov, "backcast", a, TERRA_MODELS)
    if tres.empty:
        sys.exit("Terra-only backcast produced no folds; check --cal-start/--cal-end.")
    report(tres, toof, "terra_backcast", TERRA_MODELS)
    make_plot(toof, "terra_backcast")
    tres.to_csv(REPORTS / "harmonization_results_terra_backcast.csv", index=False)
    toof.to_csv(REPORTS / "harmonization_oof_terra_backcast.csv", index=False)
    for target, (y_col, _, _) in TARGETS.items():
        m, s = best_model(toof, target)
        print(f"Final Terra-only model for {target}: {m} (backcast log-MAE {s:.3f})")
        out[f"{target}_terra_equiv"] = fit_predict(m, tv_final, tv_apply, target)

    first_aqua_week = det.loc[det.platform == "Aqua", "week"].min()
    t_only = out.index <= first_aqua_week
    out["calib_source"] = np.where(t_only, "terra_only", "terra+aqua")
    for target in TARGETS:
        out.loc[t_only, f"{target}_viirs_equiv"] = out.loc[t_only, f"{target}_terra_equiv"]
    print(f"\nFirst Aqua detection week: {first_aqua_week.date()}. "
          f"{int(t_only.sum())} weeks ({out.index[t_only].min().date()} to "
          f"{out.index[t_only].max().date()}) use the Terra-only calibration.")

    # cross-check: where both calibrations are valid and VIIRS does not exist yet
    chk = out.loc[(~t_only) & (out.index < start)]
    if len(chk) > 20:
        print("\nCross-check on Aqua-era weeks before VIIRS "
              f"({chk.index.min().date()} to {chk.index.max().date()}, {len(chk)} weeks):")
        c, t = chk["count_viirs_equiv"], chk["count_terra_equiv"]
        mon = chk.index.month
        print(f"  Terra-only / Terra+Aqua estimate, ratio of sums: all {t.sum()/max(c.sum(),1):.2f}"
              f" | Oct-Nov {t[np.isin(mon,[10,11])].sum()/max(c[np.isin(mon,[10,11])].sum(),1):.2f}"
              f" | Apr-May {t[np.isin(mon,[4,5])].sum()/max(c[np.isin(mon,[4,5])].sum(),1):.2f}")
        print(f"  weekly log-MAE between the two estimates: "
              f"{np.abs(np.log1p(c) - np.log1p(t)).mean():.3f}")
        print("  (ratio near 1.0 = the Terra-only model is consistent with the combined one;\n"
              "   far from 1.0 in a season = expect that bias in the 2000-2002 estimates)")

    out["in_overlap"] = out.index.isin(ov.index)
    out["drift_era"] = apply_df.year.to_numpy() > a.final_train_end
    out.to_csv(PROCESSED / "harmonized_weekly.csv")
    print(f"\nSaved {PROCESSED / 'harmonized_weekly.csv'}")


if __name__ == "__main__":
    main()