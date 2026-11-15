"""Fetch multi-country crash data into data/processed/crashes.csv (unified schema).

REAL:      NYC (USA), Chicago (USA), Great Britain STATS19 (UK, latest full year in the 5-year file)
SIMULATED: Indian cities - real junction coordinates from OpenStreetMap (traffic signals) where Overpass
           responds; crash counts are synthetic (no open GPS-level Indian crash data is accessible).
           Everything India-derived is flagged real=False.
"""
import io, time
import numpy as np, pandas as pd, requests

H = {"User-Agent": "raksha-research/0.1"}
rng = np.random.default_rng(11)
frames = []

# --- NYC (already downloaded)
n = pd.read_csv("data/raw/nyc_collisions.csv", low_memory=False).dropna(subset=["latitude", "longitude"])
n = n[n.latitude.between(40.4, 41.0) & n.longitude.between(-74.3, -73.6)]
frames.append(pd.DataFrame(dict(
    collision_id=n.collision_id.astype(str), crash_date=pd.to_datetime(n.crash_date), latitude=n.latitude,
    longitude=n.longitude, area=n.borough.fillna("Unknown").str.title(),
    injured=pd.to_numeric(n.number_of_persons_injured, errors="coerce").fillna(0),
    killed=pd.to_numeric(n.number_of_persons_killed, errors="coerce").fillna(0),
    vehicle=n.vehicle_type_code1.fillna("Unknown").str.title(), city="New York", country="USA", real=True)))

# --- Chicago
try:
    u = ("https://data.cityofchicago.org/resource/85ca-t3if.csv?$limit=60000&$where=latitude IS NOT NULL "
         "AND crash_date>'2023-01-01'&$select=crash_record_id,crash_date,latitude,longitude,injuries_total,injuries_fatal,"
         "trafficway_type,beat_of_occurrence")
    c = pd.read_csv(io.StringIO(requests.get(u, headers=H, timeout=240).text))
    c = c[(c.latitude > 41) & (c.longitude < -87)]
    frames.append(pd.DataFrame(dict(
        collision_id=c.crash_record_id.astype(str), crash_date=pd.to_datetime(c.crash_date), latitude=c.latitude,
        longitude=c.longitude, area="Chicago", injured=pd.to_numeric(c.injuries_total, errors="coerce").fillna(0),
        killed=pd.to_numeric(c.injuries_fatal, errors="coerce").fillna(0), vehicle="Unknown", city="Chicago",
        country="USA", real=True)))
    print("chicago", len(c))
except Exception as e:
    print("chicago FAILED", e)

# --- Great Britain STATS19 (large file)
try:
    r = requests.get("https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-last-5-years.csv",
                     headers=H, timeout=600)
    g = pd.read_csv(io.StringIO(r.text), usecols=["collision_index", "collision_year", "longitude", "latitude",
                    "collision_severity", "number_of_casualties", "date", "police_force"], low_memory=False)
    g = g.dropna(subset=["latitude", "longitude"])
    g = g[g.collision_year == g.collision_year.max()]
    g["date"] = pd.to_datetime(g.date, dayfirst=True, errors="coerce")
    fatal = g.collision_severity == 1
    # coarse region labels via lat bands (kept simple; police_force numeric codes are not human-readable)
    area = np.where(g.latitude > 55.3, "Scotland/North", np.where(g.latitude > 53.0, "Northern England",
            np.where((g.latitude < 51.75) & (g.longitude > -0.6) & (g.longitude < 0.4), "London & SE", "Midlands/South")))
    frames.append(pd.DataFrame(dict(
        collision_id=g.collision_index.astype(str), crash_date=g.date, latitude=g.latitude, longitude=g.longitude,
        area=area, injured=np.where(fatal, 0, g.number_of_casualties), killed=np.where(fatal, g.number_of_casualties, 0),
        vehicle="Unknown", city="Great Britain", country="United Kingdom", real=True)))
    print("uk", len(g), "year", int(g.collision_year.max()))
except Exception as e:
    print("uk FAILED", e)

# --- India (SIMULATED crash counts on real OSM junction locations)
CITIES = {"Delhi": (28.50, 77.10, 28.72, 77.32), "Mumbai": (18.90, 72.80, 19.20, 72.99),
          "Bengaluru": (12.88, 77.52, 13.05, 77.70), "Chennai": (12.92, 80.15, 13.15, 80.30),
          "Kolkata": (22.48, 88.30, 22.65, 88.45), "Hyderabad": (17.33, 78.38, 17.50, 78.55),
          "Pune": (18.46, 73.80, 18.60, 73.95), "Ahmedabad": (22.97, 72.50, 23.10, 72.66)}
MIRRORS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
           "https://overpass.private.coffee/api/interpreter"]
india = []
for city, (s, w, nn, e) in CITIES.items():
    q = f'[out:json][timeout:60];node["highway"="traffic_signals"]({s},{w},{nn},{e});out;'
    pts, src = None, "OSM junctions"
    for attempt in range(1):
        for m in MIRRORS:
            try:
                r = requests.post(m, data={"data": q}, headers=H, timeout=25)
                if r.status_code == 200:
                    pts = [(x["lat"], x["lon"]) for x in r.json()["elements"]]; break
            except Exception:
                pass
        if pts is not None: break
        time.sleep(4)
    if not pts or len(pts) < 15:
        # fallback: scatter around city centre (NOT real junctions)
        cy, cx = (s + nn) / 2, (w + e) / 2
        pts = [(cy + rng.normal(0, .035), cx + rng.normal(0, .035)) for _ in range(150)]; src = "scattered (OSM unavailable)"
    pts = np.array(pts)
    print("india", city, len(pts), src)
    # synthetic: heavy-tailed crash intensity per junction
    lam = rng.lognormal(1.2, 1.0, len(pts))
    for (la, lo), k in zip(pts, rng.poisson(lam)):
        for _ in range(k):
            sev = rng.random()
            india.append((la + rng.normal(0, .0003), lo + rng.normal(0, .0003), city,
                          pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0, 1050))),
                          int(rng.poisson(1.1)) if sev > .15 else 0, 1 if sev > .985 else 0,
                          rng.choice(["Motorcycle", "Car", "Bus", "Truck", "Auto-rickshaw"], p=[.45, .25, .08, .1, .12])))
d = pd.DataFrame(india, columns=["latitude", "longitude", "area", "crash_date", "injured", "killed", "vehicle"])
d["collision_id"] = ["IN-%d" % i for i in range(len(d))]; d["city"] = d.area; d["country"] = "India"; d["real"] = False
frames.append(d[["collision_id", "crash_date", "latitude", "longitude", "area", "injured", "killed", "vehicle", "city", "country", "real"]])

allc = pd.concat(frames, ignore_index=True)
allc.to_csv("data/processed/crashes.csv", index=False)
print(allc.groupby(["country", "city"]).agg(n=("collision_id", "count"), real=("real", "first")))


