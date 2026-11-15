"""Mixed-traffic interaction simulator (stand-in for DoTA/IDD/highD trajectories).

Generates pairwise interactions (car-following and crossing conflicts) at 10 Hz
between heterogeneous road users. Ground truth ("near-miss") is defined from the
OUTCOME (minimum physical clearance < 1.0 m), NOT from TTC, so a TTC-only
baseline is not trivially perfect.

The output schema (x, y, vx, vy per agent per frame) matches what
detect_track.py produces from real video, so real trajectories can be dropped in.
"""
import numpy as np
import pandas as pd

DT = 0.1
T_TOTAL = 16.0
# type: (length m, cruise speed mean m/s, std, max brake m/s2, reaction mean s)
TYPES = {
    "car": (4.2, 11.0, 2.5, 7.0, 1.0),
    "motorcycle": (2.0, 10.0, 3.0, 6.0, 0.8),
    "bus": (11.0, 8.5, 1.8, 4.5, 1.3),
    "truck": (8.0, 8.5, 2.0, 4.5, 1.3),
    "pedestrian": (0.6, 1.4, 0.3, 2.0, 0.9),
}
VEH = ["car", "motorcycle", "bus", "truck"]


def _sim_following(rng, ta, tb):
    """tb = leader, ta = follower, same lane along +x."""
    la, va_m, va_s, ba, ra = TYPES[ta]
    lb, vb_m, vb_s, bb, _ = TYPES[tb]
    n = int(T_TOTAL / DT)
    v_b0 = max(3.0, rng.normal(vb_m, vb_s))
    v_a0 = max(3.0, rng.normal(va_m, va_s) * rng.uniform(0.9, 1.25))
    hw = rng.lognormal(np.log(1.4), 0.45)  # initial time headway
    gap0 = max(4.0, hw * v_a0 + (la + lb) / 2)
    xa = np.zeros(n); xb = np.zeros(n); va = np.zeros(n); vb = np.zeros(n)
    xa[0] = 0.0; xb[0] = gap0 + (la + lb) / 2; va[0] = v_a0; vb[0] = v_b0
    t_b = rng.uniform(4, 9)
    brake = rng.uniform(1.5, bb)
    brake_dur = rng.uniform(1.0, 3.5)
    drop = rng.uniform(0.25, 1.0)  # fraction of speed shed (1.0 = stop)
    tau = max(0.3, rng.normal(ra, 0.35))
    distract = rng.random() < 0.22
    d0 = rng.uniform(5, 11); d1 = d0 + rng.uniform(1.2, 3.0)
    # IDM params
    T = rng.lognormal(np.log(1.3), 0.3); s0 = 2.0; a_max = 1.8; b_c = rng.uniform(2.0, 3.5)
    v_des = v_a0 * 1.05
    tgt = v_b0 * (1 - drop)
    acc_a = np.zeros(n)
    for i in range(1, n):
        t = i * DT
        if t_b <= t < t_b + brake_dur and vb[i - 1] > tgt:
            ab = -brake
        elif t >= t_b + brake_dur and vb[i - 1] < v_b0:
            ab = 1.2
        else:
            ab = 0.0
        vb[i] = max(0.0, vb[i - 1] + ab * DT)
        xb[i] = xb[i - 1] + vb[i] * DT
        j = max(0, i - 1 - int(tau / DT))
        gap = xb[j] - xa[j] - (la + lb) / 2
        dv = va[j] - vb[j]
        if distract and d0 <= t < d1:
            a = 0.0
        else:
            s_star = s0 + max(0.0, va[j] * T + va[j] * dv / (2 * np.sqrt(a_max * b_c)))
            a = a_max * (1 - (va[j] / v_des) ** 4 - (s_star / max(gap, 0.1)) ** 2)
        a = float(np.clip(a, -ba, a_max))
        acc_a[i] = a
        va[i] = max(0.0, va[i - 1] + a * DT)
        xa[i] = xa[i - 1] + va[i] * DT
    zeros = np.zeros(n)
    A = dict(x=xa, y=zeros, vx=va, vy=zeros, r=la / 2)
    B = dict(x=xb, y=zeros, vx=vb, vy=zeros, r=lb / 2)
    clr = (xb - xa) - (la + lb) / 2
    return A, B, clr, "following"


def _sim_crossing(rng, ta, tb):
    """A travels +x, B travels +y, conflict point at origin."""
    la, va_m, va_s, ba, ra = TYPES[ta]
    lb, vb_m, vb_s, bb, rb = TYPES[tb]
    n = int(T_TOTAL / DT)
    va0 = max(1.0, rng.normal(va_m, va_s)); vb0 = max(0.8, rng.normal(vb_m, vb_s))
    t_arr_a = rng.uniform(6, 10)
    offset = rng.normal(0, 1.8) if rng.random() < 0.6 else rng.uniform(-5, 5)
    t_arr_b = t_arr_a + offset
    xa = -va0 * t_arr_a; yb = -vb0 * max(1.0, t_arr_b)
    xs = np.zeros(n); ys_a = np.zeros(n); xs_b = np.zeros(n); ys = np.zeros(n)
    va = np.zeros(n); vb = np.zeros(n); va[0] = va0; vb[0] = vb0
    xs[0] = xa; ys[0] = yb
    # reactive braking
    notice_a = rng.normal(ra, 0.35); notice_b = rng.normal(rb, 0.35)
    react_t = rng.uniform(2.0, 5.0)  # moment each starts paying attention to conflict
    yield_a = rng.random() < 0.5
    dec_a = rng.uniform(1.0, ba); dec_b = rng.uniform(1.0, bb)
    inattentive_b = rng.random() < 0.25
    for i in range(1, n):
        t = i * DT
        # who yields? the one that would arrive later, if attentive
        ta_eta = (0 - xs[i - 1]) / max(va[i - 1], 0.1)
        tb_eta = (0 - ys[i - 1]) / max(vb[i - 1], 0.1)
        a_a = a_b = 0.0
        if 0 < ta_eta < 7 and 0 < tb_eta < 7 and abs(ta_eta - tb_eta) < 2.5:
            if yield_a and t > react_t + notice_a:
                a_a = -dec_a
            elif (not yield_a) and not inattentive_b and t > react_t + notice_b:
                a_b = -dec_b
        elif a_a == 0 and va[i - 1] < va0 * 0.99:
            a_a = 0.8
        if a_b == 0 and vb[i - 1] < vb0 * 0.99 and not (0 < tb_eta < 7):
            a_b = 0.8
        va[i] = max(0.0, va[i - 1] + a_a * DT); vb[i] = max(0.0, vb[i - 1] + a_b * DT)
        xs[i] = xs[i - 1] + va[i] * DT; ys[i] = ys[i - 1] + vb[i] * DT
    A = dict(x=xs, y=ys_a, vx=va, vy=np.zeros(n), r=la / 2 * 0.7)
    B = dict(x=xs_b, y=ys, vx=np.zeros(n), vy=vb, r=lb / 2 * 0.7)
    dist = np.hypot(A["x"] - B["x"], A["y"] - B["y"])
    clr = dist - (A["r"] + B["r"])
    return A, B, clr, "crossing"


def simulate_scenario(rng, sid):
    kind = "following" if rng.random() < 0.5 else "crossing"
    if kind == "following":
        ta, tb = rng.choice(VEH), rng.choice(VEH)
        A, B, clr, k = _sim_following(rng, ta, tb)
    else:
        ta = rng.choice(VEH)
        tb = rng.choice(VEH + ["pedestrian"] * 2)
        A, B, clr, k = _sim_crossing(rng, ta, tb)
    n = len(clr)
    t_e = int(np.argmin(clr))
    event = bool(clr[t_e] < 1.0)
    traj = pd.DataFrame({
        "scenario": sid, "frame": np.arange(n), "t": np.arange(n) * DT,
        "ax": A["x"], "ay": A["y"], "avx": A["vx"], "avy": A["vy"],
        "bx": B["x"], "by": B["y"], "bvx": B["vx"], "bvy": B["vy"],
        "ra": A["r"], "rb": B["r"],
    })
    meta = dict(scenario=sid, kind=k, type_a=ta, type_b=tb, event=int(event),
                t_event=t_e * DT, min_clearance=float(clr[t_e]))
    return traj, meta


def simulate(n_scenarios=6000, seed=0):
    rng = np.random.default_rng(seed)
    trajs, metas = [], []
    for s in range(n_scenarios):
        t, m = simulate_scenario(rng, s)
        trajs.append(t); metas.append(m)
    return pd.concat(trajs, ignore_index=True), pd.DataFrame(metas)


if __name__ == "__main__":
    traj, meta = simulate(6000)
    traj.to_parquet("data/processed/trajectories.parquet") if False else traj.to_csv(
        "data/processed/trajectories.csv.gz", index=False)
    meta.to_csv("data/processed/scenarios.csv", index=False)
    print(meta.groupby("kind").event.agg(["mean", "count"]))
    print(meta.groupby("type_a").event.mean())

