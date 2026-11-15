"""Build GIS layers: blackspots (real NYC crashes), ML observations, emerging zones, hazard reports.

NOTE: NYC crash records are REAL. ML near-miss observations and community reports are
SIMULATED DEMO DATA anchored to real intersections, because no georeferenced video
corpus is available. They are flagged `simulated=True` and never merged into blackspots.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
CELL = 0.0015  # ~150 m grid

c = pd.read_csv("data/processed/crashes.csv", low_memory=False, parse_dates=["crash_date"])
c = c.dropna(subset=["latitude", "longitude"])
c["gy"] = (c.latitude / CELL).round().astype(int); c["gx"] = (c.longitude / CELL).round().astype(int)
c["severity_w"] = 1 + 2 * c.injured + 10 * c.killed
cells = c.groupby(["city", "country", "gy", "gx"]).agg(
    crashes=("collision_id", "count"), injured=("injured", "sum"), killed=("killed", "sum"),
    score=("severity_w", "sum"), lat=("latitude", "mean"), lon=("longitude", "mean"),
    real=("real", "first"), borough=("area", lambda s: s.mode().iat[0])).reset_index()
# blackspot threshold is PER CITY (top 0.5% of cells, minimum 8 crashes) so small datasets are comparable
thr = cells.groupby("city").crashes.transform(lambda s: max(8, s.quantile(0.995)))
cells["blackspot"] = cells.crashes >= thr
cells["cell_id"] = cells.city + "_" + cells.gy.astype(str) + "_" + cells.gx.astype(str)
cells.to_csv("data/processed/crash_cells.csv", index=False)
# ---- ML observations (simulated, anchored to real locations)
pred = pd.read_csv("data/processed/test_predictions.csv")
pos = pred[pred.p > 0.5].reset_index(drop=True)
black = cells[cells.blackspot]
# 60% of observations near existing blackspots (should correlate with risk), 40% at 'new' mid-risk cells
mid = cells[(~cells.blackspot) & (cells.crashes >= 3)]
n = 4000
loc_pool = pd.concat([g.sample(min(len(g), 6), random_state=1) for _, g in black.groupby("city")] +
                     [g.sample(min(len(g), 8), random_state=2) for _, g in mid.groupby("city")])
w = np.where(loc_pool.blackspot, 3.0, 1.0); w = w / w.sum()
pick = loc_pool.iloc[rng.choice(len(loc_pool), n, p=w)].reset_index(drop=True)
obs = pos.sample(n, replace=True, random_state=3).reset_index(drop=True)
obs["city"] = pick.city.values; obs["country"] = pick.country.values
obs["lat"] = pick.lat.values + rng.normal(0, 0.0003, n)
obs["lon"] = pick.lon.values + rng.normal(0, 0.0003, n)
obs["date"] = pd.Timestamp("2026-11-12") - pd.to_timedelta(rng.integers(0, 90, n), unit="D")
obs["status"] = rng.choice(["unverified", "verified", "rejected"], n, p=[0.6, 0.3, 0.1])
obs["simulated"] = True
obs["gy"] = (obs.lat / CELL).round().astype(int); obs["gx"] = (obs.lon / CELL).round().astype(int)
obs.to_csv("data/processed/ml_observations.csv", index=False)

# ---- Emerging zones: >=5 non-rejected observations in last 60 days, not an existing blackspot
recent = obs[(obs.date >= pd.Timestamp("2026-11-12") - pd.Timedelta(days=60)) & (obs.status != "rejected")]
z = recent.groupby(["city", "gy", "gx"]).agg(obs=("p", "count"), mean_p=("p", "mean"), lat=("lat", "mean"),
                                      lon=("lon", "mean")).reset_index()
z = z.merge(cells[["city", "gy", "gx", "blackspot", "crashes"]], on=["city", "gy", "gx"], how="left")
z["blackspot"] = z.blackspot.fillna(False).astype(bool)
z["emerging"] = (z.obs >= 5) & (~z.blackspot)
z.to_csv("data/processed/risk_zones.csv", index=False)

# ---- community reports (simulated)
m = 500
base = cells.sample(m, weights=cells.crashes, random_state=5, replace=True).reset_index(drop=True)
rep = pd.DataFrame({"city": base.city, "country": base.country,
    "id": np.arange(1, m + 1), "category": rng.choice(["Pothole", "Damaged road", "Dangerous intersection", "Missing signage"], m),
    "lat": base.lat + rng.normal(0, 0.0004, m), "lon": base.lon + rng.normal(0, 0.0004, m),
    "severity": rng.choice(["Low", "Medium", "High"], m, p=[.3, .45, .25]),
    "status": rng.choice(["Submitted", "Under review", "Verified", "Resolved", "Rejected"], m, p=[.3, .2, .25, .15, .1]),
    "date": pd.Timestamp("2026-11-12") - pd.to_timedelta(rng.integers(0, 120, m), unit="D"),
    "simulated": True})
rep.to_csv("data/processed/hazard_reports.csv", index=False)
print(f"crashes={len(c)} cells={len(cells)} blackspots={int(cells.blackspot.sum())}"); print(cells.groupby("city").blackspot.sum())
print(f"ml_obs={len(obs)} emerging_zones={int(z.emerging.sum())} reports={len(rep)}")




