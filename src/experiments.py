"""Experiments 1-4 for RAKSHA. Outputs figures + results.json to reports/.

Leakage controls:
  * features come from a causal sliding window at anchors every 0.5 s (not aligned to closest approach)
  * label = near-miss onset within the next LOOK seconds; anchors after onset are dropped
  * train/val/test split by SCENARIO (GroupShuffleSplit) - no scenario appears in two splits
  * F1 threshold chosen on the validation split, then applied unchanged to test
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from features import extract_file

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed", help="dir with trajectories.csv.gz + scenarios.csv")
ap.add_argument("--out", default="reports", help="dir for figures + results.json")
args = ap.parse_args()
D, RP = Path(args.data), Path(args.out)
RP.mkdir(parents=True, exist_ok=True)
meta = pd.read_csv(D / "scenarios.csv", low_memory=False)
OUT = {"data": str(D)}

TTC_ONLY = ["ttc_last"]
KIN = ["dist_last", "clr_last", "closing_last", "speed_a", "speed_b", "rel_speed", "acc_a", "acc_b",
       "acc_a_min", "acc_b_min", "jerk_a"]
SSM = ["ttc_min", "ttc_last", "pet_min", "pet_last", "drac_max", "drac_last"]
CTX = ["crossing", "ta_bus", "ta_car", "ta_motorcycle", "ta_truck", "tb_bus", "tb_car", "tb_motorcycle",
       "tb_pedestrian", "tb_truck"]
FULL = SSM + KIN + CTX
LOOK = 3.0  # main task: will a near-miss start within the next 3 s?


def models():
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        # max_samples caps each tree's bootstrap so memory stays bounded on the 1M+ row AV2 set
        "RandomForest": (RandomForestClassifier(100, min_samples_leaf=10, max_samples=200_000, class_weight="balanced",
                                                n_jobs=4, random_state=0) if len(fit) > 200_000 else
                         RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=0)),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="logloss", random_state=0, n_jobs=-1),
    }


def gsplit(f, size, seed):
    a, b = next(GroupShuffleSplit(1, test_size=size, random_state=seed).split(f, groups=f.group))
    return f.iloc[a].copy(), f.iloc[b].copy()


def best_thr(y, p):
    return max(np.quantile(p, np.linspace(0.5, 0.995, 100)), key=lambda t: f1_score(y, p > t))


def score(y, p, thr):
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p), f1=f1_score(y, p > thr))


def run(model, cols, y_col="y"):
    """fit on FIT, pick threshold on VAL, report on TEST."""
    if model is None:  # TTC rule: smaller TTC => riskier
        pv, pt = 1 - va.ttc_last / 30, 1 - te.ttc_last / 30
    else:
        model.fit(fit[cols], fit[y_col])
        pv, pt = model.predict_proba(va[cols])[:, 1], model.predict_proba(te[cols])[:, 1]
    thr = best_thr(va[y_col], pv)
    return score(te[y_col], pt, thr), pt, thr


f = extract_file(D / "trajectories.csv.gz", meta)
fl = f.select_dtypes("float64").columns
f[fl] = f[fl].astype("float32")
f = pd.concat([f, pd.get_dummies(f.type_a, prefix="ta").astype(int),
               pd.get_dummies(f.type_b, prefix="tb").astype(int)], axis=1)
for c in CTX:
    if c not in f: f[c] = 0
f["y"] = (f.tte <= LOOK).astype(int)
tr, te = gsplit(f, 0.3, 0)
fit, va = gsplit(tr, 0.2, 1)
assert not set(tr.group) & set(te.group) and not set(fit.group) & set(va.group)
print(f"rows={len(f)} pairs={f.scenario.nunique()} groups={f.group.nunique()} positive rate={f.y.mean():.3f}")
print(f"positives: fit={int(fit.y.sum())} val={int(va.y.sum())} test={int(te.y.sum())}")
OUT["setup"] = dict(rows=len(f), pairs=int(f.scenario.nunique()), groups=int(f.group.nunique()), lookahead_s=LOOK,
                    anchor_step_s=0.5, window_s=1.5, split="by group (source scenario) 70/30, val 20% of train",
                    positive_rate=float(f.y.mean()), test_positives=int(te.y.sum()))
OUT["event_rate"] = float(meta.event.mean())  # scenario-level near-miss rate

# ---- Exp 1: TTC-only vs ML
exp1 = {"TTC threshold rule": run(None, TTC_ONLY)[0]}
for name, m in models().items():
    exp1[f"{name} (TTC only)"] = run(m, TTC_ONLY)[0]
fitted, preds = {}, {}
for name, m in models().items():
    exp1[f"{name} (full)"], preds[name], thr = run(m, FULL)
    fitted[name] = (m, thr)
OUT["exp1"] = exp1

# ---- Exp 2: feature-group ablation with XGBoost
groups = {"TTC": ["ttc_min", "ttc_last"], "TTC+PET": ["ttc_min", "ttc_last", "pet_min", "pet_last"],
          "TTC+PET+DRAC": SSM, "TTC+PET+DRAC+kinematics": SSM + KIN, "All + context": FULL}
OUT["exp2"] = {g: run(models()["XGBoost"], cols)[0] for g, cols in groups.items()}

# ---- Exp 3a: lookahead L (label = onset within next L s), same anchors and splits
exp3 = {}
for L in [1, 2, 3, 4, 5]:
    for d in (fit, va, te):
        d.loc[:, "yL"] = (d.tte <= L).astype(int)
    m = models()["XGBoost"].fit(fit[FULL], fit.yL)
    exp3[L] = dict(xgb=roc_auc_score(te.yL, m.predict_proba(te[FULL])[:, 1]),
                   ttc_rule=roc_auc_score(te.yL, 1 - te.ttc_last / 30))
OUT["exp3"] = exp3

# ---- Exp 3b: lead time for the main (L=3) model - positives bucketed by time to onset vs all negatives
p_main, ttc_s = preds["XGBoost"], 1 - te.ttc_last.values / 30
neg = te.y.values == 0
lead = {}
for lo, hi in [(0, 1), (1, 2), (2, 3)]:
    pos = (te.tte.values > lo) & (te.tte.values <= hi)
    msk = pos | neg; yy = pos[msk].astype(int)
    lead[f"{lo}-{hi}s"] = dict(n_pos=int(pos.sum()), xgb=roc_auc_score(yy, p_main[msk]), ttc_rule=roc_auc_score(yy, ttc_s[msk]))
OUT["exp3_leadtime"] = lead

# ---- Exp 4: vehicle-combination risk (scenario level, from the simulator)
combo = meta.assign(combo=meta.type_a + " vs " + meta.type_b)
c = combo.groupby("combo").event.agg(["mean", "count"]).query("count>=40").sort_values("mean", ascending=False)
OUT["exp4"] = c.round(3).to_dict("index")
m, thr = fitted["XGBoost"]
te_c = te.assign(p=p_main)
OUT["exp4_model_auc_by_kind"] = {str(k): float(roc_auc_score(g.y, g.p)) for k, g in te_c.groupby("crossing") if g.y.nunique() > 1}

# ---- figures
fig, ax = plt.subplots(figsize=(8, 4.5))
labels = list(exp1); ax.barh(labels, [exp1[k]["auc"] for k in labels], color="#c0392b")
ax.set_xlabel("ROC-AUC (test, scenario-held-out)"); ax.set_title(f"Exp 1: TTC-only vs ML (near-miss within {LOOK:.0f} s)")
ax.set_xlim(0.5, 1); plt.tight_layout(); plt.savefig(RP / "exp1.png", dpi=130); plt.close()

fig, axs = plt.subplots(1, 2, figsize=(12, 4)); hs = list(exp3)
axs[0].plot(hs, [exp3[h]["xgb"] for h in hs], "o-", label="XGBoost (full features)")
axs[0].plot(hs, [exp3[h]["ttc_rule"] for h in hs], "s--", label="TTC-only rule")
axs[0].set_xlabel("Lookahead window L (s)"); axs[0].set_ylabel("ROC-AUC"); axs[0].legend(); axs[0].grid(alpha=.3)
axs[0].set_title("Exp 3a: near-miss within next L s")
lk = list(lead)
axs[1].plot(lk, [lead[k]["xgb"] for k in lk], "o-", label="XGBoost (L=3)")
axs[1].plot(lk, [lead[k]["ttc_rule"] for k in lk], "s--", label="TTC-only rule")
axs[1].set_xlabel("Time until near-miss onset"); axs[1].legend(); axs[1].grid(alpha=.3)
axs[1].set_title("Exp 3b: how early is it caught?")
plt.tight_layout(); plt.savefig(RP / "exp3.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(8, 5)); c2 = c.head(15)[::-1]
ax.barh(c2.index, c2["mean"], color="#e67e22"); ax.set_xlabel("Near-miss rate (simulated scenarios)")
ax.set_title("Exp 4: risk by vehicle pair"); plt.tight_layout(); plt.savefig(RP / "exp4.png", dpi=130); plt.close()

fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for a_, (n, (mm, t_)) in zip(axs, fitted.items()):
    ConfusionMatrixDisplay(confusion_matrix(te.y, preds[n] > t_)).plot(ax=a_, colorbar=False)
    a_.set_title(f"{n} (thr={t_:.2f} from val)")
plt.tight_layout(); plt.savefig(RP / "confusion.png", dpi=130); plt.close()

sample = te[FULL].sample(min(1500, len(te)), random_state=0)
sv = shap.TreeExplainer(m).shap_values(sample)
plt.figure(); shap.summary_plot(sv, sample, show=False); plt.tight_layout()
plt.savefig(RP / "shap.png", dpi=130); plt.close()
OUT["top_shap"] = pd.Series(np.abs(sv).mean(0), index=FULL).sort_values(ascending=False).head(8).round(3).to_dict()

# one row per test scenario (peak risk) for the GIS demo layer
per = te_c.groupby("scenario").agg(type_a=("type_a", "first"), type_b=("type_b", "first"),
                                   crossing=("crossing", "first"), event=("scenario_event", "first"), p=("p", "max")).reset_index()
per.to_csv(D / "test_predictions.csv", index=False)
joblib.dump(dict(model=m, features=FULL, threshold=float(thr), lookahead_s=LOOK), D / "xgb_model.joblib")
json.dump(OUT, open(RP / "results.json", "w"), indent=2, default=float)
print(json.dumps(OUT, indent=1, default=float))

