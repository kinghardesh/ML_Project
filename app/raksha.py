"""RAKSHA - ML x GIS road-safety project. Run: streamlit run app/raksha.py

Presentation front end: top navigation, a landing page laid out like a project report (problem, objectives,
methodology, datasets, results, interactive demo, limitations, about, references), and the working tools
(risk map, prediction engine, citizen reporting, authority console, analytics).
Every number shown comes from the data files and reports/ results; nothing is hard-coded.
Edit PROJECT below to add team names, guide and institution.
"""
import json
from pathlib import Path

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent.parent
APP = Path(__file__).resolve().parent
P = ROOT / "data" / "processed"
R = ROOT / "reports"
st.set_page_config(page_title="RAKSHA — Predictive Road Safety", page_icon=str(APP / "assets" / "icon.svg"),
                   layout="wide")
st.logo(str(APP / "assets" / "logo.svg"), size="large", icon_image=str(APP / "assets" / "icon.svg"))

# ---- edit these for your presentation (empty fields are simply not shown)
PROJECT = {
    "title": "RAKSHA: GIS road-risk mapping and near-miss prediction",
    "summary": "A machine-learning and GIS project that maps where crashes have happened and predicts where "
               "dangerous interactions are about to happen.",
    "team": [],          # e.g. ["Name 1", "Name 2"]
    "guide": "",         # e.g. "Prof. Name"
    "institution": "",   # e.g. "College name, Department"
}

INK, BODY, SUBTLE, LINE = "#0F172A", "#334155", "#64748B", "#E2E8F0"
ACCENT, ACCENT2, TEAL, MUTED = "#FF6A3D", "#FFB547", "#14B8A6", "#CBD5E1"

# ============================================================================================ STYLE
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
.stApp {{background: #F8FAFC;}}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button, .stApp td, .stApp th
  {{font-family: 'Inter', system-ui, sans-serif;}}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{font-family: 'Plus Jakarta Sans', 'Inter', sans-serif; color: {INK};
  letter-spacing: -.02em;}}
.block-container {{max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem;}}
[data-testid="stHeader"] {{background: rgba(255,255,255,.85); backdrop-filter: blur(10px); border-bottom: 1px solid {LINE};}}
[data-testid="stAppDeployButton"], #MainMenu, footer {{display: none !important;}}

/* ---------- hero ---------- */
.st-key-hero {{position: relative; overflow: hidden; border-radius: 28px; padding: 64px 56px 48px;
  background: radial-gradient(900px 420px at 90% -20%, rgba(255,106,61,.42), transparent 60%),
              radial-gradient(700px 360px at -10% 120%, rgba(20,184,166,.28), transparent 60%),
              linear-gradient(180deg, #0B1220 0%, #111A2E 100%);
  box-shadow: 0 30px 60px -30px rgba(15,23,42,.55);}}
.st-key-hero::before {{content: ""; position: absolute; inset: 0; pointer-events: none; opacity: .5;
  background-image: linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px);
  background-size: 44px 44px; mask-image: radial-gradient(ellipse at 70% 30%, #000 30%, transparent 75%);}}
.hero-eyebrow {{display: inline-flex; align-items: center; gap: .5rem; padding: .35rem .8rem; border-radius: 999px;
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); color: #FED7C3; font-size: .82rem;
  font-weight: 600; letter-spacing: .02em;}}
.hero-eyebrow i {{width: .5rem; height: .5rem; border-radius: 50%; background: {TEAL}; box-shadow: 0 0 0 4px rgba(20,184,166,.25);}}
.hero-title {{font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; color: #fff; font-size: 3.6rem;
  line-height: 1.06; letter-spacing: -.035em; margin: 1.2rem 0 1rem;}}
.hero-title .muted {{background: linear-gradient(180deg, #E2E8F0 0%, #94A3B8 100%); -webkit-background-clip: text;
  background-clip: text; color: transparent;}}
.hero-title .dot-accent {{color: {ACCENT};}}
.hero-sub {{color: #CBD5E1; font-size: 1.18rem; line-height: 1.6; max-width: 58ch; margin: 0 0 1.6rem;}}
.st-key-hero [data-testid="stBaseButton-primary"] {{background: linear-gradient(90deg, {ACCENT}, #FF8A4C); border: 0;
  border-radius: 12px; padding: .7rem 1.2rem; font-weight: 700; box-shadow: 0 10px 24px -10px rgba(255,106,61,.8);}}
.st-key-hero [data-testid="stBaseButton-secondary"] {{background: rgba(255,255,255,.06); color: #fff;
  border: 1px solid rgba(255,255,255,.25); border-radius: 12px; padding: .7rem 1.2rem; font-weight: 600;}}
.st-key-hero [data-testid="stBaseButton-secondary"]:hover {{border-color: #fff; color: #fff;}}
.hero-proof {{display: flex; gap: 2.4rem; flex-wrap: wrap; margin-top: 2.2rem; padding-top: 1.6rem;
  border-top: 1px solid rgba(255,255,255,.12);}}
.hero-proof div {{color: #94A3B8; font-size: .86rem;}}
.hero-proof b {{display: block; color: #fff; font-family: 'Plus Jakarta Sans'; font-size: 1.6rem; font-weight: 800;}}

/* ---------- sections & cards ---------- */
.section {{margin: 4.2rem 0 1.4rem; text-align: center;}}
.section .eyebrow, .ph .eyebrow {{color: {ACCENT}; font-weight: 700; font-size: .8rem; letter-spacing: .12em;
  text-transform: uppercase;}}
.section h2 {{font-size: 2.3rem; margin: .4rem auto .6rem; max-width: 24ch; line-height: 1.15;}}
.section p {{color: {SUBTLE}; font-size: 1.08rem; max-width: 62ch; margin: 0 auto;}}
.card {{background: #fff; border: 1px solid {LINE}; border-radius: 20px; padding: 1.5rem 1.4rem; height: 100%;
  box-shadow: 0 1px 2px rgba(15,23,42,.04); transition: transform .15s ease, box-shadow .15s ease;}}
.card:hover {{transform: translateY(-2px); box-shadow: 0 16px 32px -18px rgba(15,23,42,.25);}}
.card h4 {{margin: .9rem 0 .4rem; font-size: 1.12rem;}}
[data-testid="stHorizontalBlock"] .card {{min-height: 15.5rem;}}
.card p {{margin: 0; color: {SUBTLE}; font-size: .95rem; line-height: 1.55;}}
.icon {{width: 44px; height: 44px; border-radius: 12px; display: grid; place-items: center; font-size: 1.3rem;
  background: linear-gradient(135deg, #FFF1EA, #FFE4D3);}}
.icon.teal {{background: linear-gradient(135deg, #E6FFFA, #CCFBF1);}}
.num {{width: 34px; height: 34px; border-radius: 50%; display: grid; place-items: center; font-weight: 800;
  color: #fff; background: linear-gradient(135deg, {ACCENT}, {ACCENT2}); font-family: 'Plus Jakarta Sans';}}
.stat {{font-family: 'Plus Jakarta Sans'; font-weight: 800; font-size: 3rem; color: {INK}; line-height: 1;
  letter-spacing: -.03em;}}
.stat span {{background: linear-gradient(90deg, {ACCENT}, {ACCENT2}); -webkit-background-clip: text;
  background-clip: text; color: transparent;}}
.vs {{margin-top: 1rem;}}
.vs .row {{display: flex; align-items: center; gap: .7rem; margin: .45rem 0; font-size: .88rem; color: {BODY};}}
.vs .lbl {{width: 9.5rem; flex: none;}}
.vs .bar {{height: 10px; border-radius: 99px; background: #F1F5F9; flex: 1; overflow: hidden;}}
.vs .bar i {{display: block; height: 100%; border-radius: 99px;}}
.vs .v {{width: 3rem; text-align: right; font-weight: 700; color: {INK};}}
.kicker {{color: {ACCENT}; font-weight: 700; font-size: .76rem; letter-spacing: .1em;}}
.st-key-calc {{background: #fff; border: 1px solid {LINE}; border-radius: 22px; padding: 1.8rem 1.8rem 1.2rem;
  box-shadow: 0 20px 40px -30px rgba(15,23,42,.35);}}
.result {{background: #0B1220; border-radius: 18px; padding: 1.5rem 1.6rem; color: #CBD5E1;}}
.result .lvl {{display: inline-block; color: #fff; font-weight: 800; font-size: .78rem; letter-spacing: .1em;
  padding: .3rem .7rem; border-radius: 8px;}}
.result .rgrid {{display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1.1rem 0 .6rem;}}
.result .rnum {{font-family: 'Plus Jakarta Sans'; font-weight: 800; font-size: 2.6rem; color: #fff; line-height: 1.1;}}
.result .rnum small {{font-size: 1rem; color: #94A3B8; font-weight: 600;}}
.result p {{margin: .4rem 0 0; color: #E2E8F0;}}
.result .note {{color: #94A3B8; font-size: .85rem; margin-top: .9rem;}}
.list {{margin: .6rem 0 0; padding-left: 1.1rem; color: {SUBTLE};}}
.list li {{margin: .35rem 0; font-size: .95rem;}}
.about {{display: flex; flex-wrap: wrap; gap: 2.5rem; justify-content: center; text-align: center;}}
.about span {{display: block; color: {SUBTLE}; font-size: .8rem; font-weight: 600; letter-spacing: .06em;
  text-transform: uppercase;}}
.about b {{color: {INK}; font-size: 1.05rem;}}
.chips {{display: flex; flex-wrap: wrap; gap: .5rem; justify-content: center;}}
.chips span {{background: #fff; border: 1px solid {LINE}; border-radius: 999px; padding: .35rem .9rem; font-size: .88rem;
  font-weight: 600; color: {BODY};}}
.refs {{margin: 0; padding-left: 1.3rem; color: {BODY};}}
.refs li {{margin: .45rem 0; font-size: .93rem; line-height: 1.5;}}
.foot {{margin-top: 3.5rem; padding-top: 1.6rem; border-top: 1px solid {LINE}; color: {SUBTLE}; font-size: .85rem;
  display: flex; justify-content: space-between; gap: 2rem; flex-wrap: wrap;}}

/* ---------- product pages ---------- */
.ph {{margin: .2rem 0 1.6rem;}}
.ph h1 {{font-size: 2.4rem; margin: .35rem 0 .45rem; line-height: 1.1;}}
.ph p {{color: {SUBTLE}; font-size: 1.06rem; max-width: 70ch; margin: 0;}}
.takeaway {{display: flex; gap: .8rem; align-items: flex-start; padding: 1rem 1.15rem; border-radius: 14px;
  margin: .4rem 0 1.1rem; color: {INK}; font-size: .98rem; line-height: 1.55;
  background: linear-gradient(90deg, #FFF4EE, #FFFFFF); border: 1px solid #FFE0D1;}}
.takeaway .k {{flex: none; font-size: .72rem; font-weight: 800; letter-spacing: .08em; color: #fff; background: {ACCENT};
  border-radius: 6px; padding: .2rem .45rem; margin-top: .15rem;}}
[data-testid="stMetric"] {{background: #fff; border: 1px solid {LINE}; border-radius: 16px; padding: 1rem 1.1rem;}}
[data-testid="stMetricLabel"] p {{color: {SUBTLE}; font-weight: 600; font-size: .85rem;}}
[data-testid="stMetricValue"] {{font-family: 'Plus Jakarta Sans'; font-weight: 800; color: {INK};}}
[data-testid="stVegaLiteChart"], [data-testid="stImage"] {{background: #fff; border: 1px solid {LINE};
  border-radius: 16px; padding: .8rem;}}
[data-testid="stExpander"] details {{border-radius: 14px; border-color: {LINE}; background: #fff;}}
[data-testid="stForm"] {{background: #fff; border: 1px solid {LINE}; border-radius: 20px; padding: 1.4rem;}}
iframe {{border-radius: 16px;}}
.badge {{display: inline-block; padding: .14rem .6rem; border-radius: 999px; font-size: .72rem; font-weight: 700;
  letter-spacing: .03em;}}
.real {{background: #DCFCE7; color: #15803D;}} .sim {{background: #FEF3C7; color: #B45309;}}
.chip {{display: inline-flex; align-items: center; margin: 0 1rem .5rem 0; font-size: .86rem; color: {BODY};}}
.dot {{display: inline-block; width: .75rem; height: .75rem; border-radius: 50%; margin-right: .4rem;}}
.next [data-testid="stBaseButton-secondary"] {{border-radius: 12px; font-weight: 700;}}
@media (max-width: 760px) {{ .hero-title {{font-size: 2.4rem;}} .st-key-hero {{padding: 40px 24px 32px;}}
  .section h2 {{font-size: 1.8rem;}} }}
</style>
""", unsafe_allow_html=True)

REAL = '<span class="badge real">REAL DATA</span>'
SIM = '<span class="badge sim">SIMULATED</span>'


def html(s, where=st):
    where.markdown(s, unsafe_allow_html=True)


def takeaway(s):
    html(f'<div class="takeaway"><span class="k">INSIGHT</span><div>{s}</div></div>')


def page_header(eyebrow, title, sub):
    html(f'<div class="ph"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{sub}</p></div>')


def section(eyebrow, title, sub=""):
    html(f'<div class="section"><div class="eyebrow">{eyebrow}</div><h2>{title}</h2>' + (f"<p>{sub}</p>" if sub else "")
         + "</div>")


def card(col, icon, title, body, teal=False):
    html(f'<div class="card"><div class="icon{" teal" if teal else ""}">{icon}</div><h4>{title}</h4><p>{body}</p></div>', col)


def show(ch, where=st):
    """Render an Altair chart in the product style (own fonts/colours, no Streamlit theme)."""
    ch = (ch.configure_view(strokeWidth=0)
          .configure_axis(labelFont="Inter", titleFont="Inter", labelColor=SUBTLE, titleColor=SUBTLE, labelFontSize=12,
                          titleFontWeight=600, gridColor="#EEF2F6", domainColor=LINE, tickColor=LINE)
          .configure_title(font="Plus Jakarta Sans", fontSize=15, fontWeight=700, color=INK, anchor="start", offset=12)
          .configure_legend(labelFont="Inter", labelColor=BODY, labelFontSize=12, symbolType="circle"))
    where.altair_chart(ch, use_container_width=True, theme=None)


# ============================================================================================ DATA
@st.cache_data
def load():
    d = {k: pd.read_csv(P / f"{k}.csv") for k in ["crash_cells", "risk_zones", "scenarios"]}
    d["hazards"] = pd.read_csv(P / "hazard_reports.csv", parse_dates=["date"])
    d["obs"] = pd.read_csv(P / "ml_observations.csv", parse_dates=["date"])
    d["crashes"] = pd.read_csv(P / "crashes.csv", parse_dates=["crash_date"], low_memory=False)
    return d


D = load()
_rj = lambda p: json.load(open(p)) if p.exists() else {}
RES, RES_AV2 = _rj(R / "results.json"), _rj(R / "av2" / "results.json")
SOURCES = {k: v for k, v in {
    "Real traffic (Argoverse 2)": (RES_AV2, R / "av2", P / "av2", True),
    "Simulated traffic": (RES, R, P, False)}.items() if v[0]}

MODEL_NAMES = {"TTC threshold rule": "Classic rule (time-to-collision only)",
               "LogReg (TTC only)": "Logistic regression · TTC only", "RandomForest (TTC only)": "Random forest · TTC only",
               "XGBoost (TTC only)": "XGBoost · TTC only", "LogReg (full)": "Logistic regression · all signals",
               "RandomForest (full)": "Random forest · all signals", "XGBoost (full)": "RAKSHA model (XGBoost · all signals)"}
SHORT_NAMES = {"TTC threshold rule": "Classic TTC rule", "LogReg (TTC only)": "LogReg · TTC only",
               "RandomForest (TTC only)": "Forest · TTC only", "XGBoost (TTC only)": "XGBoost · TTC only",
               "LogReg (full)": "LogReg · all signals", "RandomForest (full)": "Forest · all signals",
               "XGBoost (full)": "★ RAKSHA engine"}
FEATURE_NAMES = {
    "drac_last": "Braking needed to avoid contact (now)", "drac_max": "Peak braking needed (last 1.5 s)",
    "clr_last": "Gap between the two road users", "dist_last": "Centre-to-centre distance",
    "ttc_min": "Lowest time-to-collision (last 1.5 s)", "ttc_last": "Time-to-collision (now)",
    "pet_min": "Smallest time gap at the conflict point", "pet_last": "Time gap at the conflict point (now)",
    "closing_last": "Closing speed", "speed_a": "Speed of road user A", "speed_b": "Speed of road user B",
    "rel_speed": "Speed difference", "acc_a": "Acceleration of A", "acc_b": "Acceleration of B",
    "acc_a_min": "Hardest braking by A", "acc_b_min": "Hardest braking by B", "jerk_a": "Jerkiness of A",
    "crossing": "Paths cross (vs following)", "tb_pedestrian": "Other road user is a pedestrian"}
GROUP_NAMES = {"TTC": "1 · Time-to-collision", "TTC+PET": "2 · + time gap at conflict point",
               "TTC+PET+DRAC": "3 · + braking needed", "TTC+PET+DRAC+kinematics": "4 · + speeds, gaps, braking",
               "All + context": "5 · + road-user types"}


def next_button(i):
    if i + 1 < len(NAV):
        st.write("")
        _, r = st.columns([3, 2])
        with r.container(key=f"next_{i}"):
            if st.button(f"Continue to {NAV[i + 1].title}  →", use_container_width=True):
                st.switch_page(NAV[i + 1])


# ============================================================================================ OVERVIEW
def overview():
    cr = D["crashes"]
    s = RES_AV2.get("setup", {})
    e1 = RES_AV2.get("exp1", {})
    ml, ttc = e1.get("XGBoost (full)", {}), e1.get("TTC threshold rule", {})
    lead = RES_AV2.get("exp3_leadtime", {}).get("2-3s", {})

    # ---- hero
    with st.container(key="hero"):
        html("""
<div class="hero-eyebrow"><i></i>Machine Learning × GIS · Road-safety project</div>
<div class="hero-title">See the crash<br><span class="muted">before it happens</span><span class="dot-accent">.</span></div>
<p class="hero-sub"><b style="color:#fff">RAKSHA</b> (Hindi for <i>protection</i>) combines a GIS map of real crash
records with a machine-learning model that predicts near-misses between road users up to
<b style="color:#fff">3 seconds</b> before they happen.</p>""")
        a, b, _ = st.columns([1.2, 1.2, 2])
        if a.button("Open the risk map  →", type="primary", use_container_width=True):
            st.switch_page(NAV[1])
        if b.button("See the results", use_container_width=True):
            st.switch_page(NAV[2])
        html(f"""<div class="hero-proof">
<div><b>{int(cr.real.sum()):,}</b>real crash records mapped</div>
<div><b>{s.get('pairs', 0):,}</b>real road-user pairs analysed</div>
<div><b>{ml.get('auc', 0):.0%}</b>ranking accuracy (AUC)</div>
<div><b>3 s</b>prediction horizon</div></div>""")

    # ---- problem
    section("Problem statement", "Road safety reacts after people are hurt.",
            "Authorities find dangerous places from police crash records, so a junction only qualifies for action once "
            "crashes have piled up. Near-misses happen far more often, but nobody records them.")
    c = st.columns(3)
    card(c[0], "⏱️", "Reactive", "Blackspot lists need years of crash history before a location is flagged.")
    card(c[1], "🙈", "Near-misses unrecorded", "Close calls leave no police record, yet they point to where the next crash will be.")
    card(c[2], "📏", "Single-rule baselines", "The classic time-to-collision (TTC) rule ignores braking, gaps, speeds and road-user type.")

    # ---- objectives
    section("Objectives", "What this project set out to do.")
    c = st.columns(4)
    objs = [("Map the risk", "Find crash blackspots from real police records for New York, Chicago and Great Britain."),
            ("Predict near-misses", "Train a model that warns, from the past 1.5 s of movement, that two road users will "
                                    "come within 1 m in the next 3 s."),
            ("Beat the baseline", "Compare against the TTC rule and test on real traffic the model has never seen."),
            ("Close the loop", "Turn repeated alerts and citizen reports into zones that authorities verify.")]
    for i, (t, b_) in enumerate(objs):
        html(f'<div class="card"><div class="num">{i + 1}</div><h4>{t}</h4><p>{b_}</p></div>', c[i])

    # ---- methodology
    section("Methodology", "From raw movement to a verified risk zone.")
    c = st.columns(5)
    steps = [("📍", "Trajectories", "Positions and speeds of every road user, 10 times a second."),
             ("🔗", "Pairs", "Every two road users that come within 40 m of each other."),
             ("📐", "Safety signals", "TTC, time gap at the conflict point (PET), braking needed (DRAC), gap, speeds."),
             ("🤖", "Prediction", "XGBoost, random forest and logistic regression, tested on held-out recordings."),
             ("🗺️", "GIS & verification", "Alerts are mapped; repeated ones become emerging zones for review.")]
    for col, (ic, t, b_) in zip(c, steps):
        card(col, ic, t, b_)

    # ---- data
    section("Datasets", "Real data where it exists, clearly labelled simulation where it doesn't.")
    c = st.columns(4)
    data = [("🚓", "Crash records", "NYC Open Data, City of Chicago, UK DfT STATS19.", REAL),
            ("🚗", "Argoverse 2", "5,000 recordings of real traffic in six US cities, used to train and test the model.", REAL),
            ("🧪", "Traffic simulator", "Our own car-following and crossing simulator, used to develop the method.", SIM),
            ("🇮🇳", "India demo layer", "Real junctions from OpenStreetMap with simulated crash counts.", SIM)]
    for col, (ic, t, b_, badge) in zip(c, data):
        html(f'<div class="card"><div class="icon">{ic}</div><h4>{t}</h4><p>{b_}</p><div style="margin-top:.8rem">{badge}</div></div>', col)

    # ---- results
    if ml:
        section("Key results", "Better than the classic rule, on real traffic.",
                f"Tested on {s.get('pairs', 0):,} real road-user pairs from Argoverse 2, using recordings the model "
                "never saw during training.")
        a, b = st.columns(2)

        def vs(rows):
            return '<div class="vs">' + "".join(
                f'<div class="row"><span class="lbl">{l}</span><span class="bar"><i style="width:{v * 100:.0f}%;'
                f'background:{col}"></i></span><span class="v">{v:.0%}</span></div>' for l, v, col in rows) + "</div>"
        html(f"""<div class="card"><div class="kicker">RANKING ACCURACY (AUC)</div>
<div class="stat" style="margin-top:.6rem"><span>{ml['auc']:.0%}</span></div>
<p style="margin-top:.5rem">Shown a dangerous moment and a safe one, the model picks the dangerous one {ml['auc']:.0%} of
the time.</p>{vs([("XGBoost model", ml['auc'], f"linear-gradient(90deg,{ACCENT},{ACCENT2})"),
                  ("TTC rule (baseline)", ttc['auc'], MUTED)])}</div>""", a)
        if lead:
            html(f"""<div class="card"><div class="kicker">EARLY WARNING · 2–3 SECONDS AHEAD</div>
<div class="stat" style="margin-top:.6rem"><span>{lead['xgb']:.0%}</span></div>
<p style="margin-top:.5rem">Even with 2–3 seconds still to go, the model keeps {lead['xgb']:.0%} ranking accuracy;
the baseline drops to {lead['ttc_rule']:.0%}.</p>{vs([("XGBoost model", lead['xgb'], f"linear-gradient(90deg,{ACCENT},{ACCENT2})"),
                                                       ("TTC rule (baseline)", lead['ttc_rule'], MUTED)])}</div>""", b)
        st.caption(f"Near-misses are rare (about 1 in {1 / s['positive_rate']:,.0f} moments), so precision is low: "
                   f"average precision {ml['ap']:.3f} vs {ttc['ap']:.3f} for the baseline. Full results on the "
                   "Prediction Engine page.")

    # ---- interactive demo
    section("Try it", "How close is too close?",
            "Move the sliders to see the two classic safety measures the model starts from.")
    with st.container(key="calc"):
        a, b = st.columns([1, 1], gap="large")
        with a:
            v1 = st.slider("Your speed (km/h)", 0, 120, 50)
            v2 = st.slider("Speed of the vehicle ahead (km/h)", 0, 120, 30)
            gap = st.slider("Gap between you (m)", 1, 60, 15)
        closing = (v1 - v2) / 3.6
        t = gap / closing if closing > 0 else float("inf")
        drac = closing ** 2 / (2 * gap) if closing > 0 else 0.0
        if t < 1.5 or drac > 3.4:
            lvl, col, msg = "CRITICAL", "#DC2626", "A collision is imminent unless someone brakes hard right now."
        elif t < 3:
            lvl, col, msg = "WARNING", "#D97706", "Closing fast: a near-miss is likely if nobody reacts."
        else:
            lvl, col, msg = "SAFE", "#0D9488", "Enough time and distance to react comfortably."
        ttc_txt = "∞" if t == float("inf") else f"{t:.1f} s"
        html(f"""<div class="result"><span class="lvl" style="background:{col}">{lvl}</span>
<div class="rgrid"><div><div class="kicker">TIME-TO-COLLISION</div><div class="rnum">{ttc_txt}</div></div>
<div><div class="kicker">BRAKING NEEDED</div><div class="rnum">{drac:.1f}<small> m/s²</small></div></div></div>
<p>{msg}</p><p class="note">Common thresholds in traffic-conflict studies: TTC below 1.5 s is critical; braking above
about 3.4 m/s² is dangerous. RAKSHA's model goes further, adding gaps, speeds, recent braking and road-user type.</p></div>""", b)

    # ---- limitations & future work
    section("Limitations & future work", "What we'd do next.")
    a, b = st.columns(2)
    html("""<div class="card"><div class="icon">⚠️</div><h4>Limitations</h4><ul class="list">
<li>Real trajectories are from US cities; Indian mixed traffic is not yet covered.</li>
<li>Argoverse 2 has no vehicle sizes, so standard sizes were assumed.</li>
<li>Near-misses are rare, so many alerts are false alarms.</li>
<li>Map alerts and citizen reports in this demo are simulated.</li></ul></div>""", a)
    html("""<div class="card"><div class="icon teal">🚀</div><h4>Future work</h4><ul class="list">
<li>Extract trajectories from Indian roadside video (YOLO + ByteTrack pipeline already written).</li>
<li>Store citizen reports in a real database (e.g. Supabase / PostGIS).</li>
<li>Deploy online (Streamlit Community Cloud).</li>
<li>Real-time alerts from live camera feeds.</li></ul></div>""", b)

    # ---- about
    section("About the project", PROJECT["title"], PROJECT["summary"])
    people = [p for p in PROJECT["team"] if p]
    if people or PROJECT["guide"] or PROJECT["institution"]:
        rows = f"<div><span>Team</span><b>{', '.join(people)}</b></div>" if people else ""
        rows += f"<div><span>Guide</span><b>{PROJECT['guide']}</b></div>" if PROJECT["guide"] else ""
        rows += f"<div><span>Institution</span><b>{PROJECT['institution']}</b></div>" if PROJECT["institution"] else ""
        html(f'<div class="card about">{rows}</div>')
        st.write("")
    html('<div class="chips">' + "".join(f"<span>{t}</span>" for t in [
        "Python", "pandas", "NumPy", "scikit-learn", "XGBoost", "SHAP", "Shapely", "Streamlit", "Folium",
        "Altair"]) + "</div>")

    # ---- references
    section("References", "Data and methods we built on.")
    html("""<div class="card"><ol class="refs">
<li>NYC Open Data, <i>Motor Vehicle Collisions – Crashes</i>. data.cityofnewyork.us</li>
<li>City of Chicago, <i>Traffic Crashes – Crashes</i>. data.cityofchicago.org</li>
<li>UK Department for Transport, <i>Road Safety Data (STATS19)</i>. data.gov.uk</li>
<li>Wilson, B. et al. (2021). <i>Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting</i>. NeurIPS Datasets and Benchmarks.</li>
<li>Hayward, J. C. (1972). <i>Near-miss determination through use of a scale of danger</i>. Highway Research Record 384.</li>
<li>Treiber, M., Hennecke, A. &amp; Helbing, D. (2000). <i>Congested traffic states in empirical observations and microscopic simulations</i>. Physical Review E 62.</li>
<li>Chen, T. &amp; Guestrin, C. (2016). <i>XGBoost: A Scalable Tree Boosting System</i>. KDD.</li>
<li>Lundberg, S. &amp; Lee, S.-I. (2017). <i>A Unified Approach to Interpreting Model Predictions</i>. NeurIPS.</li>
<li>OpenStreetMap contributors. openstreetmap.org</li></ol></div>""")

    html(f"""<div class="foot"><div><b style="color:{INK}">RAKSHA</b> · ML × GIS road-safety project</div>
<div>{REAL} crash records &amp; Argoverse 2 trajectories · {SIM} India crash counts, map alerts &amp; citizen reports.</div></div>""")
    next_button(0)


# ============================================================================================ RISK MAP
def risk_map():
    page_header("Live risk map", "Where is the risk building up?",
                "Red circles are proven crash blackspots. Orange zones are places where near-misses keep repeating "
                "before crashes have piled up: the places to fix first.")
    places = D["crash_cells"].groupby(["country", "city"]).agg(lat=("lat", "median"), lon=("lon", "median")).reset_index()
    places["label"] = places.city + " (" + places.country + ")"
    real_city = D["crash_cells"].groupby("city").real.first()

    a, b = st.columns([1, 2.2])
    pl = a.selectbox("Place", ["All places"] + places.label.tolist())
    layer_opts = {"Crash blackspots": True, "Emerging risk zones": True, "Near-miss heatmap": True,
                  "Individual model alerts": False, "Verified hazard reports": False}
    on = b.pills("Layers", list(layer_opts), selection_mode="multi", default=[k for k, v in layer_opts.items() if v]) or []
    layers = {k: k in on for k in layer_opts}
    with st.expander("More filters"):
        f1, f2, f3 = st.columns(3)
        vt = f1.multiselect("Vehicle type", sorted(D["obs"].type_a.unique()), default=list(D["obs"].type_a.unique()))
        dmin, dmax = D["obs"].date.min().date(), D["obs"].date.max().date()
        dr = f2.slider("Date range", dmin, dmax, (dmin, dmax))
        show_status = f3.multiselect("Alert status", ["verified", "unverified"], default=["verified"])
        sev = f1.multiselect("Hazard severity", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
        cat = f2.multiselect("Hazard category", sorted(D["hazards"].category.unique()),
                             default=list(D["hazards"].category.unique()))

    obs = D["obs"][D["obs"].type_a.isin(vt) & D["obs"].date.dt.date.between(*dr) & D["obs"].status.isin(show_status)]
    cells_v, zones_v, hz_v = D["crash_cells"], D["risk_zones"], D["hazards"]
    if pl == "All places":
        loc, zoom, sel = [30.0, 0.0], 2, None
    else:
        row = places[places.label == pl].iloc[0]
        loc, zoom, sel = [row.lat, row.lon], (6 if row.city == "Great Britain" else 11), row.city
        obs, cells_v = obs[obs.city == sel], cells_v[cells_v.city == sel]
        zones_v, hz_v = zones_v[zones_v.city == sel], hz_v[hz_v.city == sel]

    k = st.columns(4)
    k[0].metric("Crash blackspots", f"{int(cells_v.blackspot.sum()):,}")
    k[1].metric("Emerging risk zones", f"{int(zones_v.emerging.sum()):,}")
    k[2].metric("Near-miss alerts shown", f"{len(obs):,}")
    k[3].metric("Crash data", "Real" if real_city.get(sel, True) or sel is None else "Simulated")

    html("""<div style="margin:.6rem 0 .2rem">
<span class="chip"><span class="dot" style="background:#9F1239"></span>Crash blackspot (size = crashes)</span>
<span class="chip"><span class="dot" style="background:#FF6A3D"></span>Emerging risk zone</span>
<span class="chip"><span class="dot" style="background:linear-gradient(90deg,#FDE047,#EF4444)"></span>Near-miss heat</span>
<span class="chip"><span class="dot" style="background:#2563EB"></span>Model alert</span>
<span class="chip"><span class="dot" style="background:#16A34A"></span>Verified hazard report</span></div>""")
    m = folium.Map(location=loc, zoom_start=zoom, tiles="OpenStreetMap")
    if layers["Near-miss heatmap"]:
        hv = obs[obs.status == "verified"]
        HeatMap(hv[["lat", "lon"]].values.tolist(), radius=18, blur=22, min_opacity=.3,
                gradient={0.3: "#FDE047", 0.6: "#FB923C", 1.0: "#EF4444"}).add_to(m)
    if layers["Crash blackspots"]:
        for _, r in cells_v[cells_v.blackspot].iterrows():
            tag = "" if r.real else " (simulated counts)"
            folium.CircleMarker([r.lat, r.lon], radius=4 + r.crashes ** .5 / (2 if sel is None else 1), color="#9F1239",
                                fill=True, fill_opacity=.7, weight=1.5, tooltip=f"<b>{r.city}</b><br>{r.crashes} crashes · "
                                f"{int(r.injured)} injured · {int(r.killed)} killed{tag}").add_to(m)
    if layers["Emerging risk zones"]:
        for _, r in zones_v[zones_v.emerging].iterrows():
            folium.Circle([r.lat, r.lon], radius=140, color="#FF6A3D", fill=True, fill_opacity=.25, weight=2,
                          tooltip=f"<b>Emerging risk zone</b><br>{r.obs} near-miss alerts · avg risk {r.mean_p:.0%}<br>"
                                  f"past crashes here: {int(r.crashes) if pd.notna(r.crashes) else 0}").add_to(m)
    if layers["Individual model alerts"]:
        for _, r in obs.head(600).iterrows():
            folium.CircleMarker([r.lat, r.lon], radius=3, color="#2563EB", fill=True,
                                tooltip=f"{r.type_a} vs {r.type_b} · risk {r.p:.0%} · {r.status} (demo)").add_to(m)
    if layers["Verified hazard reports"]:
        h = hz_v[hz_v.status.isin(["Verified"]) & hz_v.severity.isin(sev) & hz_v.category.isin(cat)]
        for _, r in h.iterrows():
            folium.Marker([r.lat, r.lon], icon=folium.Icon(color="green", icon="warning-sign"),
                          tooltip=f"{r.category} ({r.severity})").add_to(m)
    st_folium(m, height=560, use_container_width=True, returned_objects=[])
    takeaway(f"A hot spot means <b>inspect here</b>, not <b>this is a blackspot</b>. In this demo, map alerts are "
             f"{SIM} data placed at real intersections; blackspots come from {REAL} police crash records.")

    st.subheader("Top places to inspect")
    cz = (D["risk_zones"] if sel is None else D["risk_zones"][D["risk_zones"].city == sel])
    cz = cz.sort_values("obs", ascending=False).head(10)
    st.dataframe(cz[["city", "obs", "mean_p", "crashes", "blackspot", "emerging", "lat", "lon"]], hide_index=True,
                 use_container_width=True, column_config={
                     "city": "Place", "obs": st.column_config.NumberColumn("Near-miss alerts"),
                     "mean_p": st.column_config.ProgressColumn("Average risk", min_value=0, max_value=1, format="%.2f"),
                     "crashes": st.column_config.NumberColumn("Past crashes", format="%d"),
                     "blackspot": "Already a blackspot?", "emerging": "Emerging zone?",
                     "lat": st.column_config.NumberColumn("Lat", format="%.4f"),
                     "lon": st.column_config.NumberColumn("Lon", format="%.4f")})
    next_button(1)


# ============================================================================================ EVIDENCE
def evidence():
    page_header("Prediction engine", "Can we see a near-miss coming?",
                "Given the last 1.5 seconds of two road users' movement, will they come within 1 metre of each other "
                "in the next 3 seconds?")
    if not SOURCES:
        st.error("No results yet. Run `src/experiments.py` first.")
        return
    src = st.segmented_control("Dataset", list(SOURCES), default=list(SOURCES)[0]) or list(SOURCES)[0]
    res, fig, dd, is_real = SOURCES[src]
    s, e1 = res.get("setup", {}), res["exp1"]
    if is_real:
        html(f"{REAL}&nbsp; About 5,000 recordings of real traffic in six US cities (Argoverse 2). Every pair of nearby "
             "road users is checked twice a second.")
    else:
        html(f"{SIM}&nbsp; Computer-generated traffic, used to develop the method. Scores here are flattering; quote "
             "the real-traffic results.")
    st.write("")
    if s:
        m = st.columns(4)
        m[0].metric("Road-user pairs", f"{s.get('pairs', s.get('scenarios', 0)):,}")
        m[1].metric("Moments checked", f"{s['rows'] / 1e6:.1f} M" if s["rows"] >= 1e6 else f"{s['rows'] / 1e3:.0f} k")
        m[2].metric("How rare is a near-miss?", f"1 in {1 / s['positive_rate']:,.0f}")
        m[3].metric("Near-miss moments tested", f"{s.get('test_positives', 0):,}")

    t1, t2, t3, t4, t5 = st.tabs(["Beats the classic rule", "Early warning", "What it looks at",
                                  "Who is most at risk", "Technical details"])
    with t1:
        ttc, ml = e1["TTC threshold rule"], e1["XGBoost (full)"]
        takeaway(f"The RAKSHA engine ranks danger correctly <b>{ml['auc']:.0%}</b> of the time, against "
                 f"<b>{ttc['auc']:.0%}</b> for the classic time-to-collision rule, and its alerts are "
                 f"<b>{ml['ap'] / max(ttc['ap'], 1e-9):.1f}×</b> more precise.")
        rows = [dict(model=SHORT_NAMES.get(k, k), full=MODEL_NAMES.get(k, k), metric=lbl, value=v[key],
                     kind="All signals" if "(full)" in k else ("Classic rule" if k == "TTC threshold rule" else "ML, TTC only"))
                for k, v in e1.items() for key, lbl in [("auc", "Ranking accuracy (AUC)"), ("ap", "Alert precision (AP)")]]
        df = pd.DataFrame(rows)
        a, b = st.columns(2)
        for col, (metric, dom) in zip([a, b], [("Ranking accuracy (AUC)", [0.5, 1]), ("Alert precision (AP)", None)]):
            d = df[df.metric == metric]
            show(alt.Chart(d).mark_bar(cornerRadiusEnd=6, clip=True, height=18).encode(
                x=alt.X("value:Q", title=None, scale=alt.Scale(domain=dom) if dom else alt.Undefined),
                y=alt.Y("model:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200, ticks=False, domain=False)),
                color=alt.Color("kind:N", scale=alt.Scale(domain=["Classic rule", "ML, TTC only", "All signals"],
                                                          range=[MUTED, ACCENT2, ACCENT]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=[alt.Tooltip("full:N", title="model"), alt.Tooltip("value:Q", format=".3f")]
            ).properties(height=260, title=metric), col)
        with st.expander("What do these two scores mean?"):
            st.markdown(f"""
- **Ranking accuracy (AUC):** show the model one dangerous moment and one safe moment. AUC is how often it
  rates the dangerous one higher. 50% is a coin flip and 100% is perfect.
- **Alert precision (AP):** of the moments the model flags, how many really were near-misses, averaged across
  alert thresholds. Random guessing scores the near-miss rate ({s.get('positive_rate', 0):.2%}), so even small numbers
  can be many times better than chance.
""")
    with t2:
        lead = res.get("exp3_leadtime", {})
        if lead:
            l2 = lead.get("2-3s", {})
            takeaway(f"With <b>2–3 seconds</b> still to go, RAKSHA picks the dangerous moment <b>{l2.get('xgb', 0):.0%}</b> "
                     f"of the time; the classic rule manages <b>{l2.get('ttc_rule', 0):.0%}</b>. It stays ahead the whole way.")
            ld = pd.DataFrame([dict(when=k.replace("s", " s before"), who=w, auc=v[c]) for k, v in lead.items()
                               for c, w in [("xgb", "RAKSHA engine"), ("ttc_rule", "Classic rule")]])
            order = [k.replace("s", " s before") for k in lead]
            show(alt.Chart(ld).mark_line(point=alt.OverlayMarkDef(size=110, filled=True), strokeWidth=3.5).encode(
                x=alt.X("when:N", sort=order[::-1], title="Time left until the near-miss", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("auc:Q", scale=alt.Scale(domain=[0.5, 1]), title="Ranking accuracy (AUC)"),
                color=alt.Color("who:N", scale=alt.Scale(domain=["RAKSHA engine", "Classic rule"], range=[ACCENT, "#94A3B8"]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["who", "when", alt.Tooltip("auc:Q", format=".3f")]).properties(height=340))
        with st.expander("Same idea, different question: \"will it happen within the next L seconds?\""):
            e3 = pd.DataFrame([dict(L=int(k), who=w, auc=v[c]) for k, v in res["exp3"].items()
                               for c, w in [("xgb", "RAKSHA engine"), ("ttc_rule", "Classic rule")]])
            show(alt.Chart(e3).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X("L:O", title="Look-ahead window L (seconds)", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("auc:Q", scale=alt.Scale(domain=[0.5, 1]), title="AUC"),
                color=alt.Color("who:N", scale=alt.Scale(domain=["RAKSHA engine", "Classic rule"], range=[ACCENT, "#94A3B8"]),
                                legend=alt.Legend(orient="bottom", title=None))).properties(height=260))
    with t3:
        e2 = res["exp2"]
        first, last = list(e2.values())[0], list(e2.values())[-1]
        takeaway(f"Time-to-collision alone reaches <b>{first['auc']:.0%}</b>. Adding braking needed, gaps and speeds "
                 f"lifts it to <b>{last['auc']:.0%}</b>. The engine leans most on the <b>gap between road users</b> and "
                 "the <b>braking needed to avoid contact</b>.")
        a, b = st.columns(2)
        g = pd.DataFrame([dict(group=GROUP_NAMES.get(k, k), auc=v["auc"]) for k, v in e2.items()])
        show(alt.Chart(g).mark_bar(cornerRadiusEnd=6, height=20).encode(
            x=alt.X("auc:Q", scale=alt.Scale(domain=[0.5, 1]), title="AUC"),
            y=alt.Y("group:N", sort=None, title=None, axis=alt.Axis(labelLimit=240, ticks=False, domain=False)),
            color=alt.Color("auc:Q", scale=alt.Scale(range=["#FFD3BF", ACCENT]), legend=None),
            tooltip=["group", alt.Tooltip("auc:Q", format=".3f")]).properties(height=280, title="Adding signals step by step"), a)
        shp = pd.DataFrame([dict(feature=FEATURE_NAMES.get(k, k), importance=v) for k, v in res.get("top_shap", {}).items()])
        if len(shp):
            show(alt.Chart(shp).mark_bar(cornerRadiusEnd=6, height=16, color=INK).encode(
                x=alt.X("importance:Q", title="Average influence (SHAP)"),
                y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240, ticks=False, domain=False)),
                tooltip=["feature", alt.Tooltip("importance:Q", format=".2f")]
            ).properties(height=280, title="What the engine pays attention to"), b)
    with t4:
        e4 = pd.DataFrame(res["exp4"]).T.reset_index().rename(columns={"index": "pair", "mean": "rate", "count": "n"})
        e4 = e4.sort_values("rate", ascending=False)
        common = e4[e4.n >= 1000]
        top = e4.iloc[0]
        big = common.iloc[0] if len(common) else top
        takeaway(f"The riskiest pairing is <b>{top.pair}</b>: {top.rate:.1%} of {int(top.n):,} encounters became a "
                 f"near-miss. Among common pairings (1,000+ encounters) it is <b>{big.pair}</b> at {big.rate:.1%}. "
                 "Pairings with few encounters are less certain; hover a bar for the count.")
        show(alt.Chart(e4.head(12)).mark_bar(cornerRadiusEnd=6, height=18).encode(
            x=alt.X("rate:Q", axis=alt.Axis(format="%"), title="Share of encounters that became a near-miss"),
            y=alt.Y("pair:N", sort="-x", title=None, axis=alt.Axis(ticks=False, domain=False)),
            color=alt.Color("rate:Q", scale=alt.Scale(range=["#FFD3BF", ACCENT]), legend=None),
            tooltip=["pair", alt.Tooltip("rate:Q", format=".2%"), alt.Tooltip("n:Q", title="encounters")]
        ).properties(height=400))
        if is_real:
            st.caption("Argoverse 2 labels trucks as 'vehicle' (shown as car) and cyclists are grouped with motorcycles.")
    with t5:
        if s:
            st.markdown(f"**Setup:** {s['window_s']} s of history → near-miss within {s['lookahead_s']:.0f} s · "
                        f"checks every {s['anchor_step_s']} s · split {s['split']}.")
        st.markdown("**All models**")
        st.dataframe(pd.DataFrame(e1).T.rename(index=MODEL_NAMES).round(3), use_container_width=True)
        st.markdown("**Feature groups**")
        st.dataframe(pd.DataFrame(res["exp2"]).T.round(3), use_container_width=True)
        a, b = st.columns(2)
        a.image(str(fig / "confusion.png"), caption="Confusion matrices (threshold chosen on validation data)")
        b.image(str(fig / "shap.png"), caption="SHAP summary")
        st.markdown("**Try the engine on your own data**")
        up = st.file_uploader("CSV with the model's feature columns", type="csv")
        if up:
            import joblib
            bundle = joblib.load(dd / "xgb_model.joblib"); df = pd.read_csv(up)
            miss = [c for c in bundle["features"] if c not in df]
            if miss:
                st.error(f"Missing columns: {miss}")
            else:
                df["risk_probability"] = bundle["model"].predict_proba(df[bundle["features"]])[:, 1]
                st.dataframe(df); st.download_button("Download predictions", df.to_csv(index=False), "predictions.csv")
    next_button(2)


# ============================================================================================ TECHNOLOGY
def technology():
    page_header("Technology", "How the engine works",
                "Plain-language explanation of the signals, the model and the data behind RAKSHA.")
    c = st.columns(5)
    for i, (t, b) in enumerate([("Trajectories", "Where each road user is and how fast it moves, 10× per second."),
                                ("Pairs", "Every two road users that come within 40 m of each other."),
                                ("Safety signals", "Time-to-collision, time gap at the conflict point, braking needed, gap, speeds."),
                                ("Prediction", "Every 0.5 s, using only the past 1.5 s: near-miss in the next 3 s?"),
                                ("Map & verify", "Repeated alerts become emerging zones for authorities to inspect.")]):
        html(f'<div class="card"><div class="num">{i + 1}</div><h4>{t}</h4><p>{b}</p></div>', c[i])

    section("Glossary", "The words behind the numbers.")
    g = st.columns(3)
    terms = [("🎯", "Near-miss", "Two road users come within 1 metre while moving relative to each other (> 1.5 m/s), "
                                 "with one in the other's path. Nothing touched, but it nearly did."),
             ("⏳", "Time-to-collision (TTC)", "If nobody changes speed or direction, how many seconds until they touch. "
                                              "The classic safety measure, and our baseline."),
             ("🔀", "Post-encroachment time (PET)", "The time gap between the two arriving at the point where their paths cross."),
             ("🛑", "Braking needed (DRAC)", "How hard someone would have to brake to avoid contact. Higher means more dangerous."),
             ("📈", "AUC · ranking accuracy", "How often the model rates a dangerous moment above a safe one. 50% is a coin flip."),
             ("🔔", "AP · alert precision", "How many of the model's alerts are real near-misses. Chance level equals how rare they are.")]
    for i, (ic, t, b) in enumerate(terms):
        with g[i % 3]:
            card(st, ic, t, b, teal=i % 2 == 1)
            st.write("")

    section("Data", "Open, documented sources.")
    ds = [("NYC Motor Vehicle Collisions", "Crash blackspots", REAL), ("Chicago Traffic Crashes", "Crash blackspots", REAL),
          ("Great Britain STATS19", "Crash blackspots", REAL),
          ("Argoverse 2 Motion Forecasting · 5,000 scenarios", "Train & test the near-miss engine", REAL),
          ("RAKSHA traffic simulator", "Method development", SIM),
          ("India, 8 cities (OpenStreetMap junctions)", "Demo map coverage (crash counts simulated)", SIM),
          ("Map alerts & citizen reports", "Demo of the verification workflow", SIM)]
    html("<div class='card' style='padding:.6rem 1.2rem'><table style='width:100%;border-collapse:collapse'>" + "".join(
        f"<tr style='border-bottom:1px solid {LINE}'><td style='padding:.75rem .2rem;font-weight:600;color:{INK}'>{n}</td>"
        f"<td style='padding:.75rem .2rem;color:{SUBTLE}'>{u}</td><td style='padding:.75rem .2rem;text-align:right'>{b}</td></tr>"
        for n, u, b in ds) + "</table></div>")

    section("Transparency", "What we're upfront about.")
    st.markdown("""
- Argoverse 2 has no vehicle sizes, so standard sizes are assumed. That adds some error to the near-miss labels.
- The real trajectories are from **US cities**. Indian mixed traffic needs its own data, for example from roadside
  video using `src/detect_track.py`.
- Near-misses are rare, so many alerts are false alarms. Alerts are a prompt to inspect, not a verdict.
- Test data is kept fully separate: no recording appears in both training and testing.
""")
    next_button(3)


# ============================================================================================ REPORT
def report():
    page_header("Citizen reporting", "Report a road hazard",
                "Anyone can flag a pothole, missing sign or dangerous junction. Reports reach the map only after an "
                "authority verifies them.")
    places = D["crash_cells"].groupby(["country", "city"]).agg(lat=("lat", "median"), lon=("lon", "median")).reset_index()
    left, right = st.columns([3, 2], gap="large")
    with left, st.form("rep", clear_on_submit=True):
        a, b = st.columns(2)
        cat = a.selectbox("What's the problem?", ["Pothole", "Damaged road", "Dangerous intersection", "Missing signage"])
        sev = b.select_slider("How serious?", ["Low", "Medium", "High"], value="Medium")
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude", value=40.7580, format="%.5f")
        lon = c2.number_input("Longitude", value=-73.9855, format="%.5f")
        desc = st.text_area("Describe it (optional)")
        img = st.file_uploader("Add a photo (optional)", type=["jpg", "png"])
        if st.form_submit_button("Submit report", type="primary", use_container_width=True):
            path = P / "hazard_reports.csv"; h = pd.read_csv(path)
            near = places.iloc[int(np.argmin(np.hypot(places.lat - lat, places.lon - lon)))]
            rid = int(h.id.max()) + 1 if len(h) else 1
            if img:
                (P / "uploads").mkdir(exist_ok=True)
                (P / "uploads" / f"{rid}_{Path(img.name).name}").write_bytes(img.read())
            new = dict(id=rid, city=near.city, country=near.country, category=cat, lat=lat, lon=lon, severity=sev,
                       status="Submitted", date=pd.Timestamp.now().normalize(), simulated=False, description=desc)
            pd.concat([h, pd.DataFrame([new])]).to_csv(path, index=False); load.clear()
            st.success(f"Thanks! Report #{rid} ({near.city}) was submitted and is waiting for review.")
    with right:
        for i, (t, b) in enumerate([("Submit", "Describe the hazard, set how serious it is and add a photo."),
                                    ("Review", "A road authority checks it in the authority console."),
                                    ("Map", "Once verified, it appears on the live risk map for everyone.")]):
            html(f'<div class="card" style="margin-bottom:.8rem"><div class="num">{i + 1}</div><h4>{t}</h4><p>{b}</p></div>')
    next_button(4)


# ============================================================================================ CONSOLE
def console():
    page_header("Authority console", "Review, verify, resolve",
                "Officials review citizen reports and model-flagged zones before anything becomes official.")
    t1, t2 = st.tabs(["Citizen hazard reports", "Model-flagged emerging zones"])
    with t1:
        path = P / "hazard_reports.csv"; h = pd.read_csv(path)
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Waiting for review", int(h.status.isin(["Submitted", "Under review"]).sum()))
        q2.metric("High severity open", int((h.status.isin(["Submitted", "Under review"]) & (h.severity == "High")).sum()))
        q3.metric("Verified", int((h.status == "Verified").sum()))
        q4.metric("Resolved", int((h.status == "Resolved").sum()))
        flt = st.pills("Show status", sorted(h.status.unique()), selection_mode="multi",
                       default=["Submitted", "Under review"]) or []
        rank = {"High": 0, "Medium": 1, "Low": 2}
        q = h[h.status.isin(flt)].sort_values(["severity", "date"], key=lambda c: c.map(rank) if c.name == "severity" else c)
        cols = [c for c in ["id", "severity", "category", "city", "status", "date", "description"] if c in q]
        st.dataframe(q[cols], hide_index=True, use_container_width=True, height=320,
                     column_config={"id": st.column_config.NumberColumn("Report #", format="%d")})
        with st.form("triage"):
            c1, c2, c3 = st.columns([1, 1, 1])
            rid = c1.number_input("Report #", int(h.id.min()), int(h.id.max()), int(q.id.iloc[0]) if len(q) else int(h.id.min()))
            new = c2.selectbox("Set status to", ["Under review", "Verified", "Resolved", "Rejected"])
            c3.write(""); c3.write("")
            if c3.form_submit_button("Update status", type="primary", use_container_width=True):
                h.loc[h.id == rid, "status"] = new; h.to_csv(path, index=False); load.clear()
                st.success(f"Report #{rid} → {new}")
    with t2:
        z = D["risk_zones"][D["risk_zones"].emerging].sort_values("obs", ascending=False)
        html(f"Places with 5+ recent model alerts that are <b>not</b> yet crash blackspots. {SIM} demo alerts.")
        st.dataframe(z[["city", "obs", "mean_p", "crashes", "lat", "lon"]], hide_index=True, use_container_width=True,
                     column_config={"city": "Place", "obs": "Alerts (60 days)",
                                    "mean_p": st.column_config.ProgressColumn("Average risk", min_value=0, max_value=1, format="%.2f"),
                                    "crashes": st.column_config.NumberColumn("Past crashes", format="%d")})
    next_button(5)


# ============================================================================================ ANALYTICS
def analytics():
    page_header("Analytics", "Crash history at a glance", "Real police crash records for the selected places.")
    c = D["crashes"].dropna(subset=["crash_date"]).copy()
    c["crash_date"] = pd.to_datetime(c.crash_date, errors="coerce")
    places = sorted(c.city.unique())
    pick = st.pills("Places", places, selection_mode="multi",
                    default=[p for p in ["New York", "Chicago", "Great Britain"] if p in places]) or []
    c = c[c.city.isin(pick)]
    if c.empty:
        st.info("Pick at least one place.")
        return
    k = st.columns(3)
    k[0].metric("Crashes", f"{len(c):,}"); k[1].metric("People injured", f"{int(c.injured.sum()):,}")
    k[2].metric("People killed", f"{int(c.killed.sum()):,}")
    a, b = st.columns(2)
    byc = c.groupby("city").size().reset_index(name="crashes")
    show(alt.Chart(byc).mark_bar(cornerRadiusEnd=6, height=20, color=ACCENT).encode(
        x=alt.X("crashes:Q", title="Crash records"), y=alt.Y("city:N", sort="-x", title=None,
                                                             axis=alt.Axis(ticks=False, domain=False)),
        tooltip=["city", "crashes"]).properties(height=300, title="Crashes by place"), a)
    bym = c.groupby(["city", pd.Grouper(key="crash_date", freq="MS")]).size().reset_index(name="crashes")
    show(alt.Chart(bym).mark_line(strokeWidth=2.5).encode(
        x=alt.X("crash_date:T", title=None), y=alt.Y("crashes:Q", title="Crashes per month"),
        color=alt.Color("city:N", scale=alt.Scale(range=[ACCENT, TEAL, INK, ACCENT2, "#8B5CF6", "#64748B"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["city", alt.Tooltip("crash_date:T", format="%b %Y"), "crashes"]
    ).properties(height=300, title="Crashes per month"), b)
    sev = c.assign(outcome=np.where(c.killed > 0, "Fatal", np.where(c.injured > 0, "Injury", "Damage only")))
    sv = sev.groupby(["city", "outcome"]).size().reset_index(name="n")
    show(alt.Chart(sv).mark_bar(height=22).encode(
        x=alt.X("n:Q", stack="normalize", axis=alt.Axis(format="%"), title="Share of crashes"),
        y=alt.Y("city:N", title=None, axis=alt.Axis(ticks=False, domain=False)),
        color=alt.Color("outcome:N", scale=alt.Scale(domain=["Damage only", "Injury", "Fatal"],
                                                     range=["#CBD5E1", ACCENT2, "#BE123C"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["city", "outcome", "n"]).properties(height=240, title="How severe were the crashes?"))
    st.caption("Great Britain records only injury collisions, so it has no 'damage only' share.")


# ============================================================================================ NAV
NAV = [st.Page(overview, title="Overview", url_path="overview", default=True),
       st.Page(risk_map, title="Risk Map", url_path="risk-map"),
       st.Page(evidence, title="Prediction Engine", url_path="engine"),
       st.Page(technology, title="Technology", url_path="technology"),
       st.Page(report, title="Report a Hazard", url_path="report"),
       st.Page(console, title="Authority Console", url_path="console"),
       st.Page(analytics, title="Analytics", url_path="analytics")]
st.navigation(NAV, position="top").run()
