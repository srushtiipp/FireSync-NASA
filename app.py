"""
app.py - FireSync: harmonized MODIS/VIIRS fire-detection calendar.

Run from the project root (D:\\NASA\\FireSync) with the venv active:
    pip install streamlit plotly
    streamlit run app.py

Also put config.toml in D:\\NASA\\FireSync\\.streamlit\\ (theme colours).

Needs (created by the pipeline scripts):
    data/processed/detections_clean.parquet   (preprocessing.py)
    data/processed/harmonized_weekly.csv      (harmonize.py)
    src/anomalies.py, src/seasons.py

What is and is not valid:
    - The harmonized series is calibrated at REGION level only, so it is offered
      only when the whole region is selected.
    - Raw MODIS is not corrected for the overpass drift that starts in 2022.
    - Terra-only weeks (before Aqua, to Jul 2002) are low confidence and never scored.
    - A hotspot is a detection, not a fire or burned area.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if not (ROOT / "data").exists():      # app.py was placed inside src/
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from anomalies import climatology, describe, detect_anomalies, load_series, peak_window, woy_label
from seasons import add_baselines, describe_season, season_table

st.set_page_config(page_title="FireSync", layout="wide")

DET_PATH = ROOT / "data" / "processed" / "detections_clean.parquet"
HARM_PATH = ROOT / "data" / "processed" / "harmonized_weekly.csv"
HARMONIZED = "Harmonized (VIIRS-equivalent)"
VIIRS = "VIIRS (observed)"
MODIS = "MODIS (raw)"

# ------------------------------------------------------------------ design
ASH, INK, SMOKE, GRID = "#ECEEEC", "#1B1F1E", "#5A6560", "#CDD3CF"
EMBER, TEAL, MUTED, UNSCORED = "#C8421B", "#2F6F7E", "#AEB7B1", "#DADFDB"
HEAT = [[0, "#F3F1EA"], [0.25, "#F7D9A6"], [0.5, "#EE9A4B"], [0.75, "#C0391B"], [1, "#4A0E1B"]]
FONT = "Public Sans, sans-serif"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=Public+Sans:wght@400;500;600&display=swap');
html, body, .stApp, [data-testid="stSidebar"] {{ font-family: {FONT}; color: {INK}; }}
.stApp {{ background: {ASH}; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"], .stAppDeployButton, footer {{ display: none !important; }}
[data-testid="stSidebar"] {{ background: #E1E5E2; border-right: 1px solid {GRID}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
h1, h2, h3, [data-testid="stSidebar"] h2 {{ font-family: 'Newsreader', Georgia, serif; font-weight: 500; letter-spacing: -0.01em; }}
.fs-mast {{ display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap; }}
.fs-mark {{ font-family: 'Newsreader', Georgia, serif; font-size: 1.5rem; font-weight: 500; }}
.fs-tag {{ color: {SMOKE}; font-size: 0.95rem; }}
.fs-headline {{ font-family: 'Newsreader', Georgia, serif; font-weight: 500; line-height: 1.12;
  font-size: clamp(2rem, 3.6vw, 3.1rem); max-width: 22em; margin: 1.6rem 0 0.9rem; letter-spacing: -0.015em; }}
.fs-sub {{ color: {SMOKE}; font-size: 1.02rem; line-height: 1.55; max-width: 62ch; margin: 0 0 1.6rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {GRID}; }}
.stTabs [data-baseweb="tab"] {{ padding: 0.5rem 0; background: transparent; font-weight: 500; }}
.stTabs [aria-selected="true"] {{ color: {EMBER}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {EMBER}; }}
[data-testid="stMetric"] {{ background: transparent; padding: 0; }}
[data-testid="stMetricLabel"] {{ color: {SMOKE}; }}
[data-testid="stMetricValue"] {{ font-family: 'Newsreader', Georgia, serif; font-weight: 500; }}
[data-testid="stCaptionContainer"] {{ color: {SMOKE}; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def style(fig, height=None, heatmap=False):
    fig.update_layout(font=dict(family=FONT, color=INK, size=13), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=44, b=10),
                      hoverlabel=dict(font_family=FONT), **({"height": height} if height else {}))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, showgrid=not heatmap)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, showgrid=not heatmap)
    return fig


def show(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def table(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading detections...")
def load_detections():
    if not DET_PATH.exists():
        return None
    raw = pd.read_parquet(DET_PATH)
    lower = {c.lower(): c for c in raw.columns}

    def pick(*names):
        for n in names:
            if n in lower:
                return lower[n]
        return None

    c_lat, c_lon = pick("latitude", "lat"), pick("longitude", "lon", "lng")
    c_frp, c_sen = pick("frp"), pick("sensor")
    c_plat = pick("platform", "satellite")
    c_dt, c_date = pick("datetime", "acq_datetime"), pick("acq_date", "date")
    if None in (c_lat, c_lon, c_frp, c_sen) or (c_dt is None and c_date is None):
        return {"error": list(raw.columns)}

    day = pd.to_datetime(raw[c_dt] if c_dt else raw[c_date])
    if getattr(day.dt, "tz", None) is not None:
        day = day.dt.tz_localize(None)
    day = day.dt.normalize()

    txt = raw[c_sen].astype(str).str.upper()
    sensor = np.where(txt.str.contains("VIIRS"), "VIIRS",
                      np.where(txt.str.contains("MODIS"), "MODIS", txt))
    plat = raw[c_plat].astype(str).str.upper() if c_plat else pd.Series("", index=raw.index)

    det = pd.DataFrame({
        "lat": raw[c_lat].astype("float32").to_numpy(),
        "lon": raw[c_lon].astype("float32").to_numpy(),
        "frp": raw[c_frp].astype("float32").to_numpy(),
        "sensor": pd.Categorical(sensor),
        "aqua": plat.str.contains("AQUA").to_numpy(),
        "week_start": (day - pd.to_timedelta(day.dt.day_of_week, unit="D")).to_numpy(),
    })
    week = pd.Timedelta(days=7)
    ranges = {}
    for sen in ("MODIS", "VIIRS"):
        w = det.loc[det.sensor == sen, "week_start"]
        if len(w):  # drop first and last week: they are partial
            ranges[sen] = (w.min() + week, w.max() - week)
    aq = det.loc[(det.sensor == "MODIS") & det.aqua, "week_start"]
    first_aqua = aq.min() if len(aq) else pd.NaT
    extent = dict(lat_min=float(det.lat.min()), lat_max=float(det.lat.max()),
                  lon_min=float(det.lon.min()), lon_max=float(det.lon.max()))
    return {"det": det, "ranges": ranges, "first_aqua": first_aqua, "extent": extent}


@st.cache_data(show_spinner="Aggregating weekly series...")
def aoi_series(lat_lo, lat_hi, lon_lo, lon_hi, sensor, metric):
    d = load_detections()
    det = d["det"]
    m = det[(det.sensor == sensor) & (det.lat >= lat_lo) & (det.lat <= lat_hi)
            & (det.lon >= lon_lo) & (det.lon <= lon_hi)]
    lo, hi = d["ranges"][sensor]
    idx = pd.date_range(lo, hi, freq="7D")
    g = m.groupby("week_start").size() if metric == "count" else m.groupby("week_start")["frp"].sum()
    return g.reindex(idx, fill_value=0).astype(float).rename_axis("week_start")


@st.cache_data(show_spinner=False)
def harmonized(metric):
    if not HARM_PATH.exists():
        return None
    return load_series(HARM_PATH, metric)


@st.cache_data(show_spinner=False)
def run_anomalies(s, z_thresh, contamination, min_count):
    return detect_anomalies(s, z_thresh, contamination, min_count=min_count)


def aoi_sample(lat_lo, lat_hi, lon_lo, lon_hi, n=4000):
    det = load_detections()["det"]
    m = det[(det.lat >= lat_lo) & (det.lat <= lat_hi) & (det.lon >= lon_lo) & (det.lon <= lon_hi)]
    k = len(m)
    pts = m.sample(min(n, k), random_state=0)[["lat", "lon"]] if k else m[["lat", "lon"]]
    return k, pts.astype("float64")   # float32 is not JSON serializable for st.map


# ---------------------------------------------------------------- figures
def calendar_fig(s, res, source, log):
    iso = s.index.isocalendar()
    df = pd.DataFrame({"year": iso.year.astype(int).to_numpy(),
                       "week": iso.week.astype(int).to_numpy(), "v": s.to_numpy()})
    pv = df.pivot_table(index="year", columns="week", values="v", aggfunc="first")
    pv = pv.reindex(columns=range(1, 54))
    z = np.log10(1 + pv) if log else pv
    fig = go.Figure(go.Heatmap(
        z=z.to_numpy(), x=list(pv.columns), y=list(pv.index), customdata=pv.to_numpy(),
        colorscale=HEAT, xgap=1, ygap=1,
        hovertemplate="Year %{y}, ISO week %{x}<br>%{customdata:,.0f}<extra></extra>",
        colorbar=dict(title="log10(1+x)" if log else "", thickness=12, len=0.8)))

    t = source[source == "terra_only"].index
    if len(t):
        ti = t.isocalendar()
        fig.add_trace(go.Scatter(x=ti.week.astype(int), y=ti.year.astype(int), mode="markers",
                                 marker=dict(symbol="x", size=5, color=SMOKE),
                                 name="Terra-only (low confidence)", hoverinfo="skip"))
    f = res[res.flagged]
    for direction, symbol, fill in (("high", "triangle-up", "#FFFFFF"), ("low", "triangle-down", TEAL)):
        g = f[f.direction == direction]
        if len(g):
            gi = g.index.isocalendar()
            fig.add_trace(go.Scatter(
                x=gi.week.astype(int), y=gi.year.astype(int), mode="markers",
                marker=dict(symbol=symbol, size=9, color=fill,
                            line=dict(color=INK if direction == "high" else "#FFFFFF", width=1)),
                name=f"unusually {direction}",
                text=[describe(ts, r) for ts, r in g.iterrows()],
                hovertemplate="%{text}<extra></extra>"))
    # dotted brackets mark the two burning seasons (rabi Apr-May, kharif Oct-Nov)
    for name, (x0, x1) in {"rabi": (13.5, 22.5), "kharif": (39.5, 48.5)}.items():
        fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(0,0,0,0)", layer="above",
                      line=dict(color=INK, width=1, dash="dot"))
        fig.add_annotation(x=(x0 + x1) / 2, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                           text=name, font=dict(size=12, color=SMOKE))
    fig.update_yaxes(range=[int(pv.index.max()) + 0.5, int(pv.index.min()) - 0.5],
                     dtick=1, title="Year")
    fig.update_xaxes(title="ISO week of year", range=[0.5, 53.5])
    style(fig, height=max(480, 22 * len(pv) + 190), heatmap=True)
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", y=-0.1, yanchor="top"))
    return fig


def weekly_fig(res, label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res.index, y=res.value, name=label, line=dict(color=INK, width=1.3)))
    fig.add_trace(go.Scatter(x=res.index, y=res.baseline, name="seasonal baseline",
                             line=dict(color=TEAL, dash="dot")))
    for trig, color in (("z", EMBER), ("iforest", "#7A6FA0"), ("both", "#7A1B12")):
        g = res[res.trigger == trig]
        fig.add_trace(go.Scatter(x=g.index, y=g.value, mode="markers", name=f"flag: {trig}",
                                 marker=dict(color=color, size=8)))
    fig.update_yaxes(type="log")
    fig.update_layout(hovermode="x unified")
    return style(fig, height=450)


def season_fig(res):
    names = list(res["season"].unique())
    fig = make_subplots(rows=len(names), cols=2, horizontal_spacing=0.08,
                        subplot_titles=[t for n in names for t in
                                        (f"{n.capitalize()}: season total",
                                         f"{n.capitalize()}: centroid (days after season start)")])
    for r, name in enumerate(names, start=1):
        g = res[res.season == name]
        colors = np.where(~g.valid, UNSCORED,
                  np.where(g.level_flag == "high", EMBER,
                  np.where(g.level_flag == "low", TEAL, MUTED)))
        fig.add_trace(go.Bar(x=g.year, y=g.total, marker_color=list(colors), showlegend=False), row=r, col=1)
        fig.add_trace(go.Scatter(x=g.year, y=g.baseline_trail, mode="lines",
                                 line=dict(color=INK, dash="dot"),
                                 name="median of previous seasons", showlegend=(r == 1)), row=r, col=1)
        fig.add_trace(go.Scatter(x=g.year, y=g.centroid_off, mode="lines+markers",
                                 line=dict(color=INK, width=1.5), marker=dict(size=5),
                                 showlegend=False), row=r, col=2)
    fig.update_layout(hovermode="x unified")
    return style(fig, height=330 * len(names) + 60)


# --------------------------------------------------------------- headline
ORD = {2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}


def headline_text(sres):
    """One sentence about the latest autumn (kharif) season, computed from the data."""
    if sres is None:
        return "Fire detections in the selected area.", ""
    k = sres[(sres.season == "kharif") & sres.valid].sort_values("year")
    if k.empty:
        return "Fire detections in the selected area.", ""
    last = k.iloc[-1]
    yr = int(last.year)
    pct = f"{last.ratio_all * 100:,.0f}%" if pd.notna(last.ratio_all) else None
    run = int(last.run)
    if last.level_flag == "low" and pct:
        streak = (f", the {ORD[run]} low year in a row" if run in ORD
                  else (f", {run} low years in a row" if run >= 2 else ""))
        head = f"Autumn fire detections fell to {pct} of a typical season in {yr}{streak}."
    elif last.level_flag == "high" and pct:
        head = f"Autumn fire detections reached {pct} of a typical season in {yr}."
    elif pct:
        head = f"Autumn fire detections in {yr} were within their usual range, at {pct} of a typical season."
    else:
        head = f"Autumn {yr} recorded {last.total:,.0f} fire detections."
    sub = f"{last.total:,.0f} detections in October and November {yr}"
    if pd.notna(last.baseline_all):
        sub += f", against a typical {last.baseline_all:,.0f}"
    sub += ". "
    r = sres[(sres.season == "rabi") & sres.valid].sort_values("year")
    if len(r) and pd.notna(r.iloc[-1].ratio_all):
        sub += (f"Spring (April and May) {int(r.iloc[-1].year)} was "
                f"{r.iloc[-1].ratio_all * 100:,.0f}% of typical. ")
    sub += "Counts are satellite hotspots, not burned area."
    return head, sub


# ------------------------------------------------------------------- main
st.markdown('<div class="fs-mast"><span class="fs-mark">FireSync</span>'
            '<span class="fs-tag">MODIS and VIIRS fire detections, harmonized into one 25-year record'
            '</span></div>', unsafe_allow_html=True)

D = load_detections()
if D is None:
    st.error(f"Missing {DET_PATH}. Run `python src/preprocessing.py` first.")
    st.stop()
if "error" in D:
    st.error("Could not find the expected columns in the detections file. Columns present: "
             f"{D['error']}")
    st.stop()

ext = D["extent"]
LAT0, LAT1 = float(np.floor(ext["lat_min"] * 10) / 10), float(np.ceil(ext["lat_max"] * 10) / 10)
LON0, LON1 = float(np.floor(ext["lon_min"] * 10) / 10), float(np.ceil(ext["lon_max"] * 10) / 10)

# ---- sidebar: area of interest and options
st.sidebar.header("Area of interest")
lat_lo, lat_hi = st.sidebar.slider("Latitude", LAT0, LAT1, (LAT0, LAT1), 0.1)
lon_lo, lon_hi = st.sidebar.slider("Longitude", LON0, LON1, (LON0, LON1), 0.1)
full_region = (lat_lo <= ext["lat_min"] and lat_hi >= ext["lat_max"]
               and lon_lo <= ext["lon_min"] and lon_hi >= ext["lon_max"])
bbox = (lat_lo, lat_hi, lon_lo, lon_hi)

n_det, sample = aoi_sample(*bbox)
st.sidebar.caption(f"{n_det:,} detections in the selected area "
                   f"({'whole region' if full_region else 'sub-area'}).")
if n_det:
    st.sidebar.map(sample.rename(columns={"lat": "latitude", "lon": "longitude"}), size=60)

st.sidebar.header("Series")
harm_ok = full_region and HARM_PATH.exists()
options = ([HARMONIZED] if harm_ok else []) + [VIIRS, MODIS]
choice = st.sidebar.radio("Data", options)
if not full_region:
    st.sidebar.caption("The harmonized series is calibrated for the whole region only. "
                       "Select the full extent to use it.")
metric_label = st.sidebar.radio("Measure", ["Detection count", "Fire radiative power (sum)"])
metric = "count" if metric_label == "Detection count" else "frp"

with st.sidebar.expander("Advanced"):
    z_thresh = st.slider("Weekly flag threshold (sigma)", 2.0, 4.0, 2.5, 0.1)
    min_count = st.number_input("Ignore weeks below this level", min_value=0.0,
                                value=20.0 if metric == "count" else 100.0, step=10.0)
    contamination = 0.05
    log_scale = st.checkbox("Log colour scale in calendar", value=True)
    trailing = st.slider("Trailing baseline (seasons)", 3, 10, 5)
    s_z = st.slider("Season flag threshold (sigma)", 1.5, 3.5, 2.0, 0.1)
    timing_days = st.slider("Timing shift flag (days)", 3, 21, 10)

# ---- build the selected series
if choice == HARMONIZED:
    s, source = harmonized(metric)
else:
    sensor = "VIIRS" if choice == VIIRS else "MODIS"
    if sensor not in D["ranges"]:
        st.error(f"No {sensor} detections found.")
        st.stop()
    s = aoi_series(*bbox, sensor, metric)
    source = pd.Series("observed", index=s.index)
    if sensor == "MODIS":
        if pd.notna(D["first_aqua"]):
            source[s.index < D["first_aqua"]] = "terra_only"

if s.sum() == 0:
    st.warning("No detections in this area and series. Widen the area of interest.")
    st.stop()

s_scored = s.where(source != "terra_only")
res = run_anomalies(s_scored, z_thresh, contamination, min_count)
res["source"] = source

tbl = season_table(s, source)
sres = None
if not tbl.empty and "centroid_off" in tbl.columns and int(tbl.valid.sum()) >= 6:
    sres = add_baselines(tbl, trailing=trailing, z_thresh=s_z, timing_days=float(timing_days))

head, sub = headline_text(sres)
st.markdown(f'<h1 class="fs-headline">{head}</h1><p class="fs-sub">{sub}</p>', unsafe_allow_html=True)

if choice == MODIS:
    st.warning("Raw MODIS is not corrected: counts are about 4-5x lower than VIIRS, and from 2022 "
               "the Aqua and Terra overpass times drift, which changes what MODIS sees.")
if len(s) < 104:
    st.warning("Fewer than two years of data: anomaly and season baselines will be unreliable.")

tab_cal, tab_seasons, tab_weeks, tab_cmp, tab_about = st.tabs(
    ["Calendar", "Seasons", "Unusual weeks", "Sensors", "Method"])

# ---- calendar
with tab_cal:
    st.subheader(f"{choice}: weekly {metric_label.lower()}")
    show(calendar_fig(s, res, source, log_scale))
    st.caption("Rows are years and columns are ISO weeks. Dotted brackets mark the rabi (April and May) "
               "and kharif (October and November) seasons. White triangles are unusually high weeks, teal "
               "triangles unusually low weeks, grey crosses Terra-only weeks (low confidence).")
    if choice == HARMONIZED:
        st.caption("Observed VIIRS is used from January 2012. Before that the series is MODIS calibrated "
                   "to VIIRS, with a separate, weaker Terra-only calibration until July 2002.")

# ---- seasons
with tab_seasons:
    st.subheader("Seasons: rabi (April and May, wheat) and kharif (October and November, paddy)")
    if sres is None:
        st.info("Not enough valid seasons in this area and series to build baselines.")
    else:
        show(season_fig(sres))
        st.caption("Teal bars are unusually low seasons and ember bars unusually high ones, compared with "
                   "all other years. Grey bars are not scored (incomplete or Terra-only). The dotted line "
                   "is the median of the previous seasons, and 'run' counts consecutive flagged seasons.")
        sflag = sres[(sres.level_flag != "") | (sres.timing_flag != "")].copy()
        if len(sflag):
            st.markdown("**Unusual seasons**")
            sflag["_a"] = sflag.z_all.abs().fillna(0)
            for _, row in sflag.sort_values("_a", ascending=False).iterrows():
                st.markdown("- " + describe_season(row))
        for name in sres["season"].unique():
            d = sres[sres.season == name].copy()
            for c in ("centroid", "peak_date", "onset", "end"):
                d[c] = pd.to_datetime(d[c]).dt.strftime("%d %b").fillna("")
            d["total"] = d["total"].round(0)
            st.markdown(f"**{name.capitalize()}** (rank 1 = highest)")
            table(d[["year", "total", "rank_high", "ratio_all", "ratio_trail", "onset", "peak_date",
                     "centroid", "end", "level_flag", "run", "timing_flag", "confidence"]]
                  .round({"ratio_all": 2, "ratio_trail": 2}))

# ---- unusual weeks
with tab_weeks:
    flagged = res[res.flagged]
    c1, c2, c3 = st.columns(3)
    c1.metric("Weeks scored", f"{int(res.z.notna().sum()):,}")
    c2.metric("Flagged weeks", int(len(flagged)))
    c3.metric("High / low", f"{int((flagged.direction == 'high').sum())} / "
                            f"{int((flagged.direction == 'low').sum())}")
    show(weekly_fig(res, choice))
    st.caption("Low flags mean reduced detections, which is not always less burning: cloud and fog can hide "
               "fires. A shift that lasts for years, such as the autumn decline since 2022, is better "
               "read in the Seasons tab than as separate weekly flags.")
    if len(flagged):
        out = flagged.copy()
        out["week"] = out.index.strftime("%Y-%m-%d")
        out["reading"] = np.where(out.direction == "high", "elevated detections", "reduced detections")
        out["summary"] = [describe(ts, r) for ts, r in flagged.iterrows()]
        shown = out.reindex(out.z.abs().sort_values(ascending=False).index)
        table(shown[["week", "value", "baseline", "z", "reading", "trigger", "source", "summary"]]
              .round({"value": 0, "baseline": 0, "z": 1}))
        st.download_button("Download flagged weeks (CSV)",
                           shown.drop(columns=["summary"]).to_csv().encode(), "flagged_weeks.csv")
    sc = s_scored.dropna()
    if len(sc) > 156:
        lo, hi, pk = peak_window(climatology(sc))
        st.markdown(f"**Climatological peak:** weeks {lo}-{hi} (about {woy_label(lo)} to {woy_label(hi)}), "
                    f"peak week {pk} (about {woy_label(pk)}). This is the all-years median pattern and can be "
                    "out of date after a regime change.")

# ---- sensors
with tab_cmp:
    st.subheader("Raw sensors in the selected area (monthly totals)")
    modis_s = aoi_series(*bbox, "MODIS", metric) if "MODIS" in D["ranges"] else None
    viirs_s = aoi_series(*bbox, "VIIRS", metric) if "VIIRS" in D["ranges"] else None
    fig = go.Figure()
    if modis_s is not None:
        mm = modis_s.resample("MS").sum()
        fig.add_trace(go.Scatter(x=mm.index, y=mm, name="MODIS raw", line=dict(color=EMBER, width=1.5)))
    if viirs_s is not None:
        vv = viirs_s.resample("MS").sum()
        fig.add_trace(go.Scatter(x=vv.index, y=vv, name="VIIRS observed", line=dict(color=TEAL, width=1.5)))
    if full_region and harm_ok:
        hs, _ = harmonized(metric)
        hh = hs.resample("MS").sum()
        fig.add_trace(go.Scatter(x=hh.index, y=hh, name="Harmonized",
                                 line=dict(color=INK, width=1.5, dash="dot")))
    fig.update_yaxes(type="log", title="monthly total")
    fig.update_layout(hovermode="x unified")
    show(style(fig, height=450))
    if modis_s is not None and viirs_s is not None:
        common = viirs_s.index.intersection(modis_s.index)
        df = pd.DataFrame({"viirs": viirs_s[common], "modis": modis_s[common]})
        g = df.groupby(df.index.year).sum()
        g["ratio"] = g.viirs / g.modis.replace(0, np.nan)
        bar = go.Figure(go.Bar(x=g.index, y=g.ratio, marker_color=SMOKE))
        bar.update_layout(title="VIIRS / MODIS ratio by year (sum over weeks with both)", yaxis_title="ratio")
        show(style(bar, height=320))
        st.caption("In this region the ratio is roughly stable until 2021 and then falls, while the Aqua "
                   "overpass moves later and the Terra overpass earlier. The cause of the fall is not "
                   "separated from changes in fire size and burning timing.")

# ---- method
with tab_about:
    st.markdown("""
**Method.** MODIS (Terra and Aqua, 1 km) and VIIRS S-NPP (375 m) detections are filtered
(vegetation-fire type, medium or higher confidence) and aggregated to weekly counts and summed
fire radiative power. MODIS is mapped to VIIRS-equivalent activity with a gradient-boosted
calibration fitted on the stable overpass years 2012-2021, with a separate Terra-only model for
the period before Aqua launched. The final record uses observed VIIRS from 2012 and calibrated
MODIS before that.

| Period | Source | Confidence |
|---|---|---|
| Nov 2000 to Jul 2002 | MODIS Terra only, Terra-only calibration | Low: timing is reliable, magnitude is not, and these weeks are never scored |
| Jul 2002 to Dec 2011 | MODIS Terra and Aqua, calibrated to VIIRS | Estimated: about 19% weekly error in stable-year tests |
| Jan 2012 to Dec 2025 | Observed VIIRS S-NPP | Observed |

**Validation (this region).** Calibrating on 2015-2019 and predicting 2012-2014, a constant
MODIS-to-VIIRS ratio has about 19% weighted absolute error on weekly counts; the calibration has
a similar total error but cuts log-scale error from 0.59 to 0.40, mostly in quieter weeks. In the
overlap years 2012-2021 the calibrated estimate reproduces observed season timing to within about
a day and season totals to within about 15%. From 2022, overpass drift pushes the constant-ratio
error from about 19% to about 70% by 2025, and a stable-era calibration does not correct it.

**Anomalies.** Weekly flags use a robust z-score against the same weeks in other years, plus an
Isolation Forest check. Season flags compare each rabi and kharif season with all other years
(level) and with the previous few seasons (timing and trend). Flags describe detections, not
burned area.

**Limits.** Calibration is region level; sub-areas use raw sensor counts. Weeks hidden by cloud or
fog look like low activity.
""")