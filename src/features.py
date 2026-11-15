"""Surrogate safety measures (TTC, PET, DRAC) + causal sliding-window features.

Deployment-realistic formulation: at an anchor time t we only see the last WINDOW seconds
of the trajectory and ask "will clearance drop below NEAR within the next L seconds?".
Anchors are taken every STEP seconds across the whole scenario, so the window is NOT
aligned to the (unknowable) moment of closest approach. Anchors at or after the near-miss
onset are dropped (the conflict is already happening). Each row carries `tte`, the time from
t to onset (inf if no near-miss), so any lookahead L can be labelled as tte <= L.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

DT = 0.1
WINDOW = 1.5    # seconds of history summarised
CAP = 30.0      # TTC / PET cap (s)
NEAR = 1.0      # near-miss clearance threshold (m) - same as simulate.py ground truth
STEP = 0.5      # anchor spacing (s)
MAX_LOOK = 5.0  # every anchor has at least this much future, so any L <= MAX_LOOK is labelled fully


def _pair_measures(g):
    ax, ay, bx, by = g.ax.values, g.ay.values, g.bx.values, g.by.values
    avx, avy, bvx, bvy = g.avx.values, g.avy.values, g.bvx.values, g.bvy.values
    rad = g.ra.values + g.rb.values
    rx, ry = bx - ax, by - ay
    rvx, rvy = avx - bvx, avy - bvy  # closing velocity of A toward B
    dist = np.hypot(rx, ry)
    raw_clr = dist - rad
    clr = np.maximum(raw_clr, 0.01)
    closing = (rx * rvx + ry * rvy) / np.maximum(dist, 1e-3)  # >0 means approaching
    # 2D TTC: time until separation < rad assuming constant velocities
    vv = rvx ** 2 + rvy ** 2
    b = rx * rvx + ry * rvy
    c = rx ** 2 + ry ** 2 - rad ** 2
    disc = b ** 2 - vv * c
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where((vv > 1e-6) & (disc >= 0) & (b > 0), (b - np.sqrt(np.maximum(disc, 0))) / vv, CAP)
    ttc = np.clip(np.where(c < 0, 0.0, ttc), 0, CAP)
    drac = np.where(closing > 0, closing ** 2 / (2 * clr), 0.0)
    return dist, clr, raw_clr, closing, ttc, drac


def _pet(g):
    """Projected PET from current kinematics only (causal), for any 2D geometry.
    Paths crossing (angle > ~15 deg): |ETA_A - ETA_B| to where the two velocity rays intersect.
    Near-parallel or one road user stopped: headway = gap / follower speed (follower = the one moving toward
    the other). Reduces to the original axis-aligned formulas on simulator data."""
    rx, ry = g.bx.values - g.ax.values, g.by.values - g.ay.values
    avx, avy, bvx, bvy = g.avx.values, g.avy.values, g.bvx.values, g.bvy.values
    rad = g.ra.values + g.rb.values
    sa, sb = np.hypot(avx, avy), np.hypot(bvx, bvy)
    det = bvx * avy - avx * bvy                      # [vA, -vB] [tA, tB]^T = r
    crossing = np.abs(det) > 0.26 * sa * sb          # |sin(angle)| > sin(15 deg)
    with np.errstate(divide="ignore", invalid="ignore"):
        tA = (bvx * ry - bvy * rx) / det
        tB = (avx * ry - avy * rx) / det
    pet_x = np.where(crossing & (tA > 0) & (tB > 0), np.abs(tA - tB), CAP)
    gap = np.maximum(np.hypot(rx, ry) - rad, 0.01)
    a_follows = (avx * rx + avy * ry) > 0
    b_follows = (bvx * rx + bvy * ry) < 0
    spd = np.where(a_follows, sa, np.where(b_follows, sb, 0.0))
    pet_f = np.where(spd > 0.1, gap / np.maximum(spd, 0.5), CAP)
    return np.clip(np.where(crossing, pet_x, pet_f), 0, CAP)


def _back_diff(a):
    """Backward difference / DT: uses only past frames (np.gradient would peek one frame ahead)."""
    return np.diff(a, prepend=a[0]) / DT


def extract_file(path, meta: pd.DataFrame, chunksize: int = 2_000_000, **kw) -> pd.DataFrame:
    """Stream a trajectories CSV in chunks (low memory). Rows of one scenario are contiguous on disk, so the
    last (possibly partial) scenario of each chunk is carried over into the next."""
    parts, carry = [], None
    for ch in pd.read_csv(path, chunksize=chunksize):
        fl = ch.select_dtypes("float64").columns
        ch[fl] = ch[fl].astype("float32")
        if carry is not None:
            ch = pd.concat([carry, ch], ignore_index=True)
        last = ch.scenario.iat[-1]
        carry = ch[ch.scenario == last]
        done = ch[ch.scenario != last]
        if len(done):
            parts.append(extract(done, meta, **kw))
            print(f"  features: {sum(len(p) for p in parts):,} rows", flush=True)
    if carry is not None and len(carry):
        parts.append(extract(carry, meta, **kw))
    return pd.concat(parts, ignore_index=True)


def extract(traj: pd.DataFrame, meta: pd.DataFrame, step: float = STEP, max_look: float = MAX_LOOK) -> pd.DataFrame:
    W, S, L = int(round(WINDOW / DT)), int(round(step / DT)), int(round(max_look / DT))
    meta_i = meta.set_index("scenario")
    parts = []
    for sid, g in traj.groupby("scenario", sort=False):
        n = len(g)
        dist, clr, raw_clr, closing, ttc, drac = _pair_measures(g)
        if "clr" in g:  # real data (load_av2.py): oriented-box clearance + speed-gated near-miss flag
            clr = np.maximum(g.clr.values, 0.01)
            drac = np.where(closing > 0, closing ** 2 / (2 * clr), 0.0)
            hit = np.flatnonzero(g.near.values > 0)
        else:
            hit = np.flatnonzero(raw_clr < NEAR)
        onset = int(hit[0]) if len(hit) else n
        idx = np.arange(W, n - L, S)
        idx = idx[idx < onset]
        if not len(idx):
            continue
        pet = _pet(g)
        spA = np.hypot(g.avx.values, g.avy.values); spB = np.hypot(g.bvx.values, g.bvy.values)
        accA, accB = _back_diff(spA), _back_diff(spB)
        jerkA = np.abs(_back_diff(accA))
        win = lambda a: swv(a, W + 1)[idx - W]  # frames idx-W .. idx (inclusive), all in the past
        parts.append(pd.DataFrame(dict(
            scenario=sid, t=idx * DT,
            tte=(onset - idx) * DT if onset < n else np.inf,
            ttc_min=win(ttc).min(1), ttc_last=ttc[idx], pet_min=win(pet).min(1), pet_last=pet[idx],
            drac_max=win(drac).max(1), drac_last=drac[idx],
            dist_last=dist[idx], clr_last=clr[idx], closing_last=closing[idx],
            speed_a=spA[idx], speed_b=spB[idx], rel_speed=np.abs(spA[idx] - spB[idx]),
            acc_a=accA[idx], acc_b=accB[idx], acc_a_min=win(accA).min(1), acc_b_min=win(accB).min(1),
            jerk_a=swv(jerkA, W)[idx - W + 1].mean(1),
            crossing=int(meta_i.at[sid, "kind"] == "crossing"),
        )))
    f = pd.concat(parts, ignore_index=True)
    cols = ["type_a", "type_b", "event"] + (["group"] if "group" in meta_i else [])
    f = f.join(meta_i[cols].rename(columns={"event": "scenario_event"}), on="scenario")
    if "group" not in f:
        f["group"] = f.scenario  # simulator: every scenario is independent
    return f
