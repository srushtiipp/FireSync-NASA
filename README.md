# FireSync

**MODIS and VIIRS fire detections, harmonized into one 25-year record, with a calendar that shows when burning is unusual.**

Built for the NASA 2026 hackathon challenge *Harmonization of MODIS and VIIRS Hot Spots*, using Punjab and Haryana (India) as the study region.

![Calendar view](docs/figures/calendar.png)

Demo video: TODO (add link)

## What it found

Autumn fire detections in the study region have fallen to a small fraction of their usual level. The three most recent kharif (October and November) seasons, 2023, 2024 and 2025, are the three lowest of 24 scored seasons, at 0.48x, 0.16x and 0.08x of a typical season. In 2025 there were 4,861 detections against a typical 58,083. Spring (rabi, April and May) detections stayed within their normal range over the same period.

Counts are satellite hotspot detections, not burned area. A lower count means fewer detections, and this project does not explain why.

![Season view](docs/figures/seasons.png)

## The problem

Satellites have tracked active fire hotspots for over 20 years, but the record is split across two sensors that cannot be compared directly. MODIS (Terra and Aqua, 1 km pixels) covers 2000 onward. VIIRS S-NPP (375 m pixels) covers 2012 onward and records about 4.5 times as many detections for the same activity in this region. Fire managers therefore have no consistent picture of when and where burning has happened.

## What FireSync does

| Challenge deliverable | Where it lives |
|---|---|
| A harmonized dataset or method | `src/harmonize.py` writes `data/processed/harmonized_weekly.csv`, a continuous VIIRS-equivalent weekly series from November 2000 to December 2025 |
| A calendar-style visualization | The Calendar tab in `app.py`: year by ISO week, with the rabi and kharif seasons bracketed |
| An area-of-interest selector in a working web app | Latitude and longitude sliders with a live map of the detections inside the selected area |
| A way of flagging unusual periods | Weekly flags (`src/anomalies.py`) and season-level flags (`src/seasons.py`) |

## Results

| Question | Result |
|---|---|
| How well does one constant MODIS-to-VIIRS ratio work in stable years? | About 19% weighted error on weekly counts (2012 to 2014, calibrated on 2015 to 2019) |
| What does the gradient-boosted calibration add? | Log-scale error falls from 0.59 to 0.40 and rank correlation rises from 0.94 to 0.98. It does not improve peak-week totals (18.8% vs 18.6% error) |
| Does the estimate reproduce observed VIIRS where both exist? | In 2012 to 2021 the season timing agrees to within 0.4 days (kharif) and 0.7 days (rabi) on average, and the median ratio of season totals is 1.03 and 1.01. Individual seasons differ by up to 14% (kharif) and 18% (rabi). This is an in-sample check |
| Why can't the same ratio be used after 2021? | From 2022 the satellite overpass times drift. Aqua moves from about 13.3 h to 14.6 h local solar time and Terra from 10.9 h to 9.7 h, while VIIRS stays near 13.2 h. A constant ratio's error rises from 18.6% (2022) to 70.4% (2025), and a calibration fitted on the stable years does not fix it |
| How reliable is the pre-Aqua period? | November 2000 to July 2002 uses a separate Terra-only calibration. Its rank correlation is 0.96 but its magnitude error is about 74%, so those weeks are never scored |

The series uses observed VIIRS from January 2012 and calibrated MODIS before that. The overpass drift therefore only affects raw MODIS, not the harmonized record.

## How it works

1. **Preprocessing** (`src/preprocessing.py`): keep vegetation-fire detections (type 0), MODIS confidence of 30 or more and VIIRS confidence nominal or high, then aggregate to weekly counts and summed fire radiative power. 1.26 million detections remain in the region.
2. **Diagnostics** (`src/diagnostics.py`): sensor agreement, the VIIRS to MODIS ratio by year, and overpass time by satellite and year.
3. **Harmonization** (`src/harmonize.py`): compares a constant ratio, a log-linear fit, a gradient-boosted calibration and a drift-feature model. It uses three validation modes: backcast (predict earlier years from later ones, the real use case), frozen (calibrate once, test each later year) and rolling. The drift-feature model failed (it extrapolates badly outside the training range) and is reported as a negative result.
4. **Anomalies** (`src/anomalies.py`): weekly robust z-score against the same weeks in other years, plus an Isolation Forest check, with a minimum-count floor.
5. **Seasons** (`src/seasons.py`): per rabi and kharif season, the total, onset, peak, centroid and end, ranked against all other years and the previous five. A run counter shows consecutive unusual seasons.
6. **Checks** (`src/splice_check.py`): compares the calibrated estimate with observed VIIRS in the overlap years.

## Data confidence

| Period | Source | Confidence |
|---|---|---|
| Nov 2000 to Jul 2002 | MODIS Terra only, Terra-only calibration | Low: timing is reliable, magnitude is not. Never scored |
| Jul 2002 to Dec 2011 | MODIS Terra and Aqua, calibrated to VIIRS | Estimated: about 19% weekly error in stable-year tests |
| Jan 2012 to Dec 2025 | Observed VIIRS S-NPP | Observed |

## Run it

```
git clone https://github.com/<your-username>/FireSync-NASA.git
cd FireSync-NASA
python -m venv .venv
.venv\Scripts\activate          # macOS or Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Data.** Raw data is not in this repository. Request the archive from [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) as CSV for the box 74.0, 27.5, 77.7, 31.0 (west, south, east, north):

- MODIS C6.1, 2000-11-01 to 2025-12-31, into `data/raw/modis/`
- VIIRS S-NPP, 2012-01-01 to 2025-12-31, into `data/raw/viirs/`

If the archive rejects a long date range, split it into several requests and put all the files in the same folder.

**Pipeline, from the project root:**

```
python src/preprocessing.py
python src/diagnostics.py
python src/harmonize.py
python src/anomalies.py
python src/seasons.py
python src/splice_check.py
streamlit run app.py
```

To reproduce the drift test on its own: `python src/harmonize.py --cal-end 2021 --modes frozen`.

## Repository layout

```
app.py                 Streamlit web app
.streamlit/config.toml App theme
src/                   Pipeline scripts (see above)
docs/figures/          Figures used in this README
data/                  Not tracked, except small result tables
reports/               Generated plots, not tracked
```

## Limitations

- **Region-level calibration.** The harmonized series is valid for the whole study region only. For a sub-area the app shows raw VIIRS or raw MODIS counts for that area.
- **One region.** Only Punjab and Haryana have been tested. Other regions need their own calibration.
- **Detections, not burned area.** Cloud and fog hide fires and look like low activity, which matters most for the winter weeks.
- **The pre-2012 estimate cannot be tested directly.** VIIRS does not exist before 2012. Stable overpass times from 2002 to 2021 support applying the same calibration earlier, and the 2012 to 2021 check only catches bias, since the final model was fitted on those years.
- **Weekly flags are weak at the peak.** The autumn season is wide and variable, so a weekly z-score rarely flags it. That is why season-level flags are the headline.
- **The autumn decline is unexplained here.** The project does not attribute it to any cause, and the later kharif timing over the record (centroid about 24 October in the 2000s, 2 November in the 2010s, 6 November in the 2020s) is likewise reported without explanation.
- **Calibration choices.** Confidence filters and the Terra-only window affect the absolute numbers.

## Next steps

Per-cell harmonization so any selected area is calibrated, uncertainty bands on the estimate, other regions, NOAA-20 and NOAA-21 VIIRS, and a draw-rectangle area selector.

## Data and credits

Active fire data from NASA FIRMS: MODIS Collection 6.1 (Terra and Aqua) and VIIRS S-NPP 375 m. The data is courtesy of NASA and used under its open data policy.

Built by TODO (your name).