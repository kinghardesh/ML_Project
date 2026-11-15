"""Argoverse 2 Motion Forecasting -> RAKSHA pairwise trajectories (real data).

Downloads AV2 *val* scenarios from the public bucket (no account needed; licence CC BY-NC-SA 4.0,
non-commercial) and converts each scenario into road-user PAIRS in the same schema simulate.py
writes, so features.py / experiments.py run unchanged:

    trajectories.csv.gz : scenario, frame, t, ax, ay, avx, avy, bx, by, bvx, bvy, ra, rb, clr, near
    scenarios.csv       : scenario, group, kind, type_a, type_b, event, t_event, min_clearance, city, av2_id

Differences from the simulator, all deliberate:
  * clr   = true distance between ORIENTED BOUNDING BOXES (heading-aware), not circles. AV2 has no object
            sizes, so nominal sizes per class are assumed (SIZES below) - this is a source of label error.
  * near  = clr < 1.0 m AND relative speed > MIN_REL_SPEED AND one road user is in the other's path
            (_in_path). Without these, queued/parked cars and side-by-side passes in adjacent lanes count as
            "near-misses" (in a first run, 74% of car-car events were exactly that).
  * group = AV2 scenario id. Pairs from one scenario share agents, so experiments.py splits on group.

Usage:
    python src/load_av2.py --download 500          # fetch first 500 val scenarios (~100 MB) then convert
    python src/load_av2.py                         # convert whatever is already in data/raw/av2/val
"""
import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import shapely

BUCKET = "https://s3.amazonaws.com/argoverse"
PREFIX = "datasets/av2/motion-forecasting/val/"
RAW = Path("data/raw/av2/val")
OUT = Path("data/processed/av2")
DT = 0.1

# AV2 class -> (RAKSHA type, length m, width m). AV2 "vehicle" covers cars AND trucks (no separate truck class);
# cyclists are folded into "motorcycle" (two-wheeler) so the model's context columns still apply.
SIZES = {"vehicle": ("car", 4.6, 1.9), "bus": ("bus", 12.0, 2.6), "motorcyclist": ("motorcycle", 2.2, 0.8),
         "cyclist": ("motorcycle", 1.8, 0.6), "pedestrian": ("pedestrian", 0.5, 0.5)}
MIN_OVERLAP = 70       # frames both tracks must be present (contiguous); 66 is the minimum for one anchor
MAX_START_DIST = 40.0  # m; pairs further apart than this at every frame never interact
MIN_REL_SPEED = 1.5    # m/s; a close pass has to happen at speed to count as a near-miss
NEAR = 1.0
PATH_MARGIN = 0.5      # m; extra lateral width of the swept path used by _in_path


# ----------------------------------------------------------------------------- download
def list_scenarios(n):
    ids, marker = [], ""
    while len(ids) < n:
        r = requests.get(BUCKET, params={"prefix": PREFIX, "max-keys": 1000, "marker": marker}, timeout=60)
        r.raise_for_status()
        keys = re.findall(r"<Key>([^<]+)</Key>", r.text)
        ids += [k.split("/")[-2] for k in keys if k.endswith(".parquet")]
        if "<IsTruncated>true</IsTruncated>" not in r.text or not keys:
            break
        marker = keys[-1]
    return ids[:n]


def _fetch(sid):
    p = RAW / sid / f"scenario_{sid}.parquet"
    if p.exists() and p.stat().st_size > 0:
        return 0
    p.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(f"{BUCKET}/{PREFIX}{sid}/scenario_{sid}.parquet", timeout=120)
    r.raise_for_status()
    p.write_bytes(r.content)
    return len(r.content)


def download(n):
    ids = list_scenarios(n)
    with ThreadPoolExecutor(8) as ex:
        got = sum(ex.map(_fetch, ids))
    print(f"scenarios listed={len(ids)} downloaded {got / 1e6:.1f} MB into {RAW}")


# ----------------------------------------------------------------------------- geometry
def _boxes(x, y, h, L, W):
    """Array of oriented rectangles (shapely polygons), one per frame."""
    c, s = np.cos(h), np.sin(h)
    corners = np.array([[L / 2, W / 2], [L / 2, -W / 2], [-L / 2, -W / 2], [-L / 2, W / 2]])
    px = x[:, None] + corners[:, 0] * c[:, None] - corners[:, 1] * s[:, None]
    py = y[:, None] + corners[:, 0] * s[:, None] + corners[:, 1] * c[:, None]
    return shapely.polygons(np.stack([px, py], -1))


def _in_path(a, b, Wa, Wb, margin=PATH_MARGIN):
    """True where either road user is AHEAD of the other and inside its swept path (lateral offset in the other's
    heading frame < half-widths + margin). Excludes side-by-side passes of queued/parked cars in adjacent lanes."""
    rx, ry = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
    out = np.zeros(len(a), bool)
    for h, sign in ((a[:, 2], 1), (b[:, 2], -1)):
        c, s = np.cos(h), np.sin(h)
        lon, lat = sign * (rx * c + ry * s), sign * (-rx * s + ry * c)
        out |= (lon > 0) & (np.abs(lat) < (Wa + Wb) / 2 + margin)
    return out


def _longest_run(mask):
    """(start, stop) of the longest run of True."""
    best, cur, start = (0, 0), 0, 0
    for i, v in enumerate(np.append(mask, False)):
        if v:
            if cur == 0: start = i
            cur += 1
        else:
            if cur > best[1] - best[0]: best = (start, i)
            cur = 0
    return best


# ----------------------------------------------------------------------------- conversion
def convert_scenario(path, next_id):
    d = pd.read_parquet(path)
    d = d[d.object_type.isin(SIZES)]
    T = int(d.num_timestamps.iloc[0]) if len(d) else 110
    tracks = {}
    for tid, g in d.groupby("track_id"):
        if len(g) < MIN_OVERLAP:
            continue
        arr = np.full((T, 5), np.nan)
        arr[g.timestep.values] = g[["position_x", "position_y", "heading", "velocity_x", "velocity_y"]].values
        tracks[tid] = (g.object_type.iloc[0], arr)
    trajs, metas = [], []
    for (ia, (ca, A)), (ib, (cb, B)) in combinations(tracks.items(), 2):
        if ca == "pedestrian" and cb == "pedestrian":
            continue  # pedestrian-pedestrian contact is not a road-safety conflict
        s0, s1 = _longest_run(~np.isnan(A[:, 0]) & ~np.isnan(B[:, 0]))
        if s1 - s0 < MIN_OVERLAP:
            continue
        a, b = A[s0:s1], B[s0:s1]
        cdist = np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1])
        if cdist.min() > MAX_START_DIST:
            continue
        spa, spb = np.hypot(a[:, 3], a[:, 4]), np.hypot(b[:, 3], b[:, 4])
        if max(spa.max(), spb.max()) < 1.0:
            continue  # both stationary throughout (parked)
        if ca == "pedestrian":  # keep the vehicle as A, like the simulator
            ca, cb, a, b = cb, ca, b, a
        ta, La, Wa = SIZES[ca]; tb, Lb, Wb = SIZES[cb]
        clr = shapely.distance(_boxes(a[:, 0], a[:, 1], a[:, 2], La, Wa), _boxes(b[:, 0], b[:, 1], b[:, 2], Lb, Wb))
        rel = np.hypot(a[:, 3] - b[:, 3], a[:, 4] - b[:, 4])
        near = (clr < NEAR) & (rel > MIN_REL_SPEED) & _in_path(a, b, Wa, Wb)
        n = s1 - s0
        sid = next_id + len(metas)
        hd = np.abs((np.arctan2(a[:, 4], a[:, 3]) - np.arctan2(b[:, 4], b[:, 3]) + np.pi) % (2 * np.pi) - np.pi)
        moving = (spa > 1) & (spb > 1)
        ang = np.median(hd[moving]) if moving.any() else 0.0
        te = int(np.argmax(near)) if near.any() else int(np.argmin(clr))
        trajs.append(pd.DataFrame(dict(
            scenario=sid, frame=np.arange(n), t=np.arange(n) * DT,
            ax=a[:, 0], ay=a[:, 1], avx=a[:, 3], avy=a[:, 4], bx=b[:, 0], by=b[:, 1], bvx=b[:, 3], bvy=b[:, 4],
            ra=(La + Wa) / 4, rb=(Lb + Wb) / 4, clr=clr, near=near.astype(int))))
        metas.append(dict(scenario=sid, group=path.parent.name, kind="crossing" if ang > np.radians(30) else "following",
                          type_a=ta, type_b=tb, event=int(near.any()), t_event=te * DT,
                          min_clearance=float(clr.min()), city=d.city.iloc[0], av2_id=path.parent.name,
                          track_a=ia, track_b=ib))
    return trajs, metas


def convert():
    files = sorted(RAW.glob("*/scenario_*.parquet"))
    trajs, metas = [], []
    for i, p in enumerate(files):
        t, m = convert_scenario(p, len(metas))
        trajs += t; metas += m
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)} scenarios -> {len(metas)} pairs")
    OUT.mkdir(parents=True, exist_ok=True)
    traj, meta = pd.concat(trajs, ignore_index=True), pd.DataFrame(metas)
    traj.to_csv(OUT / "trajectories.csv.gz", index=False)
    meta.to_csv(OUT / "scenarios.csv", index=False)
    print(f"av2 scenarios={len(files)} pairs={len(meta)} near-miss pairs={int(meta.event.sum())} "
          f"({meta.event.mean():.2%})")
    print(meta.groupby(["type_a", "type_b"]).event.agg(["sum", "count"]))
    return traj, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", type=int, default=0, help="fetch the first N val scenarios first")
    a = ap.parse_args()
    if a.download:
        download(a.download)
    convert()
