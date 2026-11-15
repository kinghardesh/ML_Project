# RAKSHA — Road Risk Intelligence

Crash maps show where people **were** hurt. RAKSHA also looks for the **near-misses** that come before crashes:
a GIS risk map built from real crash records, plus a machine-learning model that predicts, up to 3 seconds
ahead, when two road users are about to come within 1 metre of each other.

## Quick start (app only)
The repo includes the processed map data and the experiment results, so the app runs straight away.
```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app/raksha.py
```
Open http://localhost:8501. The app is laid out as a story for presenting:
Home → Risk map → Near-miss prediction → How it works → Report a hazard → Authority triage → Crash analytics.

## Full pipeline
Run from the project folder with `set PYTHONPATH=src`.

| Step | Command | Output |
|---|---|---|
| 1. Crash data (NYC, Chicago, GB, India demo) | `.venv\Scripts\python src\fetch_cities.py` | `data/processed/crashes.csv` |
| 2. Simulated trajectories | `.venv\Scripts\python src\simulate.py` | `data/processed/trajectories.csv.gz` (not in git) |
| 3. Experiments on simulated data | `.venv\Scripts\python src\experiments.py` | `reports/`, model in `data/processed/` |
| 4. GIS layers | `.venv\Scripts\python src\build_gis.py` | blackspots, emerging zones, demo reports |
| 5. Real trajectories (Argoverse 2) | `.venv\Scripts\python src\load_av2.py --download 5000` | `data/processed/av2/` (not in git, ~1.8 GB) |
| 6. Experiments on real data | `.venv\Scripts\python src\experiments.py --data data/processed/av2 --out reports/av2` | `reports/av2/` |

Step 1 reads `data/raw/nyc_collisions.csv` (included) and downloads Chicago and GB data.
Step 4 regenerates `hazard_reports.csv`, which resets any reports made in the app.
Steps 5–6 take about 1.5 h on a laptop; `features.extract_file` streams the large file in chunks.

## Data provenance
| Data | Source | Real? |
|---|---|---|
| Crash records / blackspots | NYC Open Data, City of Chicago, UK DfT STATS19 | Real |
| India, 8 cities | OpenStreetMap junction locations, synthetic crash counts | **Simulated counts** |
| Trajectories for ML (main result) | Argoverse 2 Motion Forecasting, val split, 5,000 scenarios | Real (CC BY-NC-SA 4.0, non-commercial) |
| Trajectories for ML (method development) | `src/simulate.py` | **Simulated** |
| Map alerts, community reports | `src/build_gis.py`, placed at real intersections | **Simulated demo** |
| DoTA / IDD / highD | not used (55 GB / registration / access request) | n/a |

## Method
At an anchor time t (every 0.5 s), using only the past 1.5 s of two road users' movement, predict whether a
near-miss starts within the next 3 s. Features: time-to-collision (TTC), post-encroachment time (PET),
deceleration rate to avoid crash (DRAC), gaps, speeds, accelerations and road-user types. Anchors at or after
onset are dropped and the window is never aligned to the closest approach. Train/test splits are by source
scenario; the alert threshold is tuned on a validation split.

On Argoverse 2, a near-miss is an oriented-box clearance < 1 m at > 1.5 m/s relative speed, with one road user
in the other's path. AV2 has no object sizes, so nominal sizes per class are assumed.

## Results
**Real traffic, Argoverse 2** (305k road-user pairs, ~2M anchors, near-miss rate 0.12%, 652 positive test rows):

| Model | AUC | AP | F1 |
|---|---|---|---|
| TTC threshold rule | 0.745 | 0.016 | 0.064 |
| XGBoost (all features) | 0.966 | 0.067 | 0.122 |

Lead time (AUC, XGBoost vs TTC rule): 0–1 s 0.986 vs 0.814 · 1–2 s 0.969 vs 0.715 · 2–3 s 0.938 vs 0.693.
Top SHAP features: clearance, DRAC, minimum TTC.

**Simulated traffic** (6,000 scenarios, near-miss rate 5%): TTC rule AUC 0.834 / AP 0.625;
XGBoost AUC 0.993 / AP 0.924 / F1 0.84. These scores reflect the simulator's structure and are flattering.

## Limitations
- Near-misses are rare, so precision is low: an alert is a prompt to inspect, not a confirmed conflict.
- Argoverse 2 covers six US cities. Indian mixed traffic still needs its own trajectories, for example
  from roadside video with `src/detect_track.py` (written, not yet run).
- AV2 object sizes are assumed, which adds label noise.
