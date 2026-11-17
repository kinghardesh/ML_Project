"""RAKSHA - road risk mapping and near-miss prediction (GIS + machine learning).
Run: streamlit run app/raksha.py

Plain, presentation-friendly layout built from standard Streamlit components:
  Overview · Risk Map · Near-Miss Prediction · How it Works · Report a Hazard · Authority Triage · Crash Analytics
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
P = ROOT / "data" / "processed"
R = ROOT / "reports"
st.set_page_config(page_title="RAKSHA · Road Risk Mapping", layout="wide")

# ---- edit these for your presentation (empty fields are simply not shown)
PROJECT = {
    "title": "RAKSHA",
    "subtitle": "Road risk mapping and near-miss prediction using GIS and machine learning",
    "team": [],          # e.g. ["Name 1", "Name 2"]
    "guide": "",         # e.g. "Prof. Name"
    "institution": "",   # e.g. "College name, Department"
}

NAVY, GREY, LIGHT, RED, ORANGE = "#1F4E79", "#9CA3AF", "#7FA7CC", "#B91C1C", "#D97706"

st.markdown("""
<style>
.block-container {max-width: 1250px; padding-top: 2rem;}
[data-testid="stAppDeployButton"], #MainMenu, footer {display: none !important;}
.small {color: #6B7280; font-size: .9rem;}
[data-testid="stSidebar"] {background: #F3F4F6;}
[data-testid="stSidebarNavLink"] {border-radius: 10px; padding: .55rem .7rem; margin: .12rem 0;}
[data-testid="stSidebarNavLink"][aria-current="page"] {background: #FDE2E2;}
[data-testid="stSidebarNavLink"][aria-current="page"] span {color: #B91C1C; font-weight: 600;}
.hero-t {font-size: 3rem; font-weight: 800; color: #111827; line-height: 1; margin: 0;}
.hero-s {font-size: 1.7rem; color: #374151; margin: .35rem 0 .9rem;}
.hero-p {color: #4B5563; font-size: 1.05rem; max-width: 46ch;}
.kpis {display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.2rem 0;}
.kpi {display: flex; gap: .9rem; align-items: center; border: 1px solid #E5E7EB; border-radius: 12px;
  padding: 1rem 1.1rem; background: #fff;}
.kpi b {display: block; font-size: 1.5rem; color: #111827;}
.kpi span {color: #4B5563; font-size: .9rem;}
.mbig {font-size: 2.4rem; font-weight: 800; color: #DC2626; line-height: 1.1;}
.meter {height: 8px; border-radius: 9px; background: #E5E7EB; overflow: hidden; margin: .4rem 0 .8rem;}
.meter i {display: block; height: 100%; background: #DC2626;}
.mrow {display: flex; justify-content: space-between; color: #4B5563; font-size: .9rem; margin: .35rem 0;}
.mrow b {color: #111827;}
.st-key-hero [data-testid="stBaseButton-primary"] {background: #DC2626; border-color: #DC2626;}
@media (max-width: 900px) {.kpis {grid-template-columns: repeat(2, 1fr);}}
.stApp {background: #F5F6F8;}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {background: #fff;}
.kpi {box-shadow: 0 1px 3px rgba(0,0,0,.06);}
.st-key-hero {background: #fff; border: 1px solid #E5E7EB; border-radius: 16px; padding: 1.8rem 2rem;
  box-shadow: 0 1px 3px rgba(0,0,0,.06);}
.sec {font-size: 1.5rem; font-weight: 800; color: #111827; margin: 2.2rem 0 .9rem; padding-left: .7rem;
  border-left: 5px solid #DC2626;}
.wcard {background: #fff; border: 1px solid #E5E7EB; border-radius: 14px; padding: 1.2rem 1.3rem; height: 100%;
  box-shadow: 0 1px 3px rgba(0,0,0,.06);}
.wcard h3 {margin: .5rem 0 .2rem; font-size: 1.35rem;} .wcard h4 {margin: .8rem 0 .3rem; font-size: 1.05rem; color: #111827;}
.wcard p {color: #4B5563; font-size: .93rem; margin: 0;} .wcard ul {color: #374151; font-size: .93rem; margin: .6rem 0 0;}
.tint-red {background: linear-gradient(180deg, #FEF2F2, #fff 60%); border-color: #FECACA;}
.tint-blue {background: linear-gradient(180deg, #EFF6FF, #fff 60%); border-color: #BFDBFE;}
.tag {display: inline-block; color: #fff; font-size: .7rem; font-weight: 700; letter-spacing: .06em; padding: .2rem .55rem; border-radius: 6px;}
.ibox {width: 52px; height: 52px; border-radius: 12px; display: grid; place-items: center;}
.grid4 {display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;}
.grid3 {display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem;} .grid2 {display: grid; grid-template-columns: 1fr 1fr; gap: .8rem;}
.mini {border-radius: 10px; padding: .8rem;} .mini b {display: block; font-size: 1.15rem;} .mini span {color: #374151; font-size: .85rem;}
.lbl {color: #6B7280; font-size: .85rem;} .num {font-size: 2.1rem; font-weight: 800; line-height: 1.2;}
.note {color: #6B7280; font-size: .82rem; margin-top: .6rem !important;}
.stat p {margin-top: .3rem;}
.chips {display: flex; flex-wrap: wrap; gap: .6rem;}
.chip {background: #fff; border: 1px solid #E5E7EB; border-radius: 999px; padding: .4rem .5rem .4rem .9rem; font-size: .9rem; color: #1F2937;}
.chip i {font-style: normal; font-size: .68rem; font-weight: 700; margin-left: .5rem; padding: .15rem .5rem; border-radius: 999px;}
.chip .real {background: #DCFCE7; color: #15803D;} .chip .sim {background: #FEF3C7; color: #B45309;}
@media (max-width: 900px) {.grid4, .grid3 {grid-template-columns: 1fr 1fr;}}
</style>""", unsafe_allow_html=True)

_I = {"layers": '<path d="m12 2 10 5-10 5L2 7l10-5z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
      "car": '<path d="M5 17h14v-5l-2-5H7l-2 5v5z"/><path d="M5 12h14"/><circle cx="8" cy="17" r="1.6"/><circle cx="16" cy="17" r="1.6"/>',
      "bolt": '<path d="M13 2 3 14h9l-1 8 10-12h-9z"/>',
      "bars": '<path d="M18 20V10M12 20V4M6 20v-6"/>',
      "map": '<path d="M3 6l6-3 6 3 6-3v15l-6 3-6-3-6 3z"/><path d="M9 3v15M15 6v15"/>',
      "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'}


def icon(name, color):
    return (f'<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round">{_I[name]}</svg>')


SCENE = """<svg viewBox="0 0 520 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">
<rect width="520" height="280" rx="14" fill="#EEF2F6"/>
<rect x="0" y="100" width="520" height="80" fill="#9CA3AF"/><rect x="220" y="0" width="80" height="280" fill="#9CA3AF"/>
<path d="M0 140H210M310 140H520M260 0V90M260 190V280" stroke="#F9FAFB" stroke-width="3" stroke-dasharray="14 12"/>
<path d="M60 160H242" stroke="#D97706" stroke-width="2.5" stroke-dasharray="4 7"/>
<path d="M282 270V162" stroke="#1F4E79" stroke-width="2.5" stroke-dasharray="4 7"/>
<circle cx="270" cy="158" r="22" fill="none" stroke="#DC2626" stroke-width="2.5"/>
<rect x="150" y="149" width="44" height="22" rx="5" fill="#D97706"/>
<rect x="273" y="214" width="18" height="32" rx="7" fill="#1F4E79"/>
<rect x="320" y="30" width="170" height="50" rx="8" fill="#fff" stroke="#DC2626"/>
<text x="334" y="51" font-family="Arial" font-size="12" font-weight="700" fill="#B91C1C">Possible conflict</text>
<text x="334" y="69" font-family="Arial" font-size="12" fill="#374151">paths cross in about 2–3 s</text>
<path d="M330 80 L285 140" stroke="#DC2626" stroke-width="1.2"/>
<text x="12" y="270" font-family="Arial" font-size="11" fill="#6B7280">Illustration</text></svg>"""


def show(ch, where=st):
    """Altair chart with plain, report-style formatting."""
    ch = (ch.configure_view(strokeWidth=0)
          .configure_axis(labelColor="#4B5563", titleColor="#4B5563", gridColor="#E5E7EB", domainColor="#D1D5DB")
          .configure_title(fontSize=14, color="#1F2937", anchor="start")
          .configure_legend(labelColor="#374151"))
    where.altair_chart(ch, use_container_width=True, theme=None)


# ============================================================================================ DATA
@st.cache_data
def load():
    d = {k: pd.read_csv(P / f"{k}.csv") for k in ["crash_cells", "risk_zones", "scenarios"]}
    d["hazards"] = pd.read_csv(P / "hazard_reports.csv", parse_dates=["date"])
    d["obs"] = pd.read_csv(P / "ml_observations.csv", parse_dates=["date"])
    c = pd.read_csv(P / "crashes.csv", low_memory=False)
    # cities store dates differently (date only vs date + time), so parse each value on its own
    c["crash_date"] = pd.to_datetime(c.crash_date, format="mixed", errors="coerce")
    d["crashes"] = c
    return d


D = load()
_rj = lambda p: json.load(open(p)) if p.exists() else {}
RES, RES_AV2 = _rj(R / "results.json"), _rj(R / "av2" / "results.json")
SOURCES = {k: v for k, v in {
    "Real traffic (Argoverse 2)": (RES_AV2, R / "av2", P / "av2", True),
    "Simulated traffic": (RES, R, P, False)}.items() if v[0]}

MODEL_NAMES = {"TTC threshold rule": "TTC rule (baseline)", "LogReg (TTC only)": "Logistic regression (TTC only)",
               "RandomForest (TTC only)": "Random forest (TTC only)", "XGBoost (TTC only)": "XGBoost (TTC only)",
               "LogReg (full)": "Logistic regression (all features)", "RandomForest (full)": "Random forest (all features)",
               "XGBoost (full)": "XGBoost (all features)"}
FEATURE_NAMES = {
    "drac_last": "Braking needed to avoid contact (now)", "drac_max": "Peak braking needed (last 1.5 s)",
    "clr_last": "Gap between the two road users", "dist_last": "Centre-to-centre distance",
    "ttc_min": "Lowest time-to-collision (last 1.5 s)", "ttc_last": "Time-to-collision (now)",
    "pet_min": "Smallest time gap at the conflict point", "pet_last": "Time gap at the conflict point (now)",
    "closing_last": "Closing speed", "speed_a": "Speed of road user A", "speed_b": "Speed of road user B",
    "rel_speed": "Speed difference", "acc_a": "Acceleration of A", "acc_b": "Acceleration of B",
    "acc_a_min": "Hardest braking by A", "acc_b_min": "Hardest braking by B", "jerk_a": "Jerkiness of A",
    "crossing": "Paths cross (vs following)", "tb_pedestrian": "Other road user is a pedestrian"}
GROUP_NAMES = {"TTC": "1. Time-to-collision", "TTC+PET": "2. + time gap at conflict point",
               "TTC+PET+DRAC": "3. + braking needed", "TTC+PET+DRAC+kinematics": "4. + speeds, gaps, braking",
               "All + context": "5. + road-user types"}


def next_button(i):
    if i + 1 < len(NAV):
        st.divider()
        _, r = st.columns([3, 1])
        if r.button(f"Next: {NAV[i + 1].title} →", use_container_width=True, key=f"next_{i}"):
            st.switch_page(NAV[i + 1])


PIPELINE = f"""digraph G {{
  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
  node [shape=box, style="rounded", fontname="Helvetica", fontsize=11, color="#6B7280", fontcolor="#1F2937",
        margin="0.18,0.08"];
  edge [color="#9CA3AF", arrowsize=0.7];
  subgraph cluster_a {{ label="Part A · Crash map (GIS)"; fontname="Helvetica"; fontsize=11; fontcolor="{NAVY}";
    color="#D1D5DB"; style="rounded";
    crash [label="Police crash records\\n(NYC, Chicago, GB)"]; grid [label="150 m grid cells"];
    black [label="Blackspots\\n(top 0.5% of cells)"]; crash -> grid -> black; }}
  subgraph cluster_b {{ label="Part B · Near-miss prediction (ML)"; fontname="Helvetica"; fontsize=11; fontcolor="{NAVY}";
    color="#D1D5DB"; style="rounded";
    traj [label="Vehicle trajectories\\n(Argoverse 2)"]; feat [label="Safety measures\\nTTC · PET · DRAC"];
    model [label="ML model\\n(XGBoost)"]; alert [label="Near-miss alert\\n(next 3 s)"];
    traj -> feat -> model -> alert; }}
  map [label="Risk map", color="{NAVY}", fontcolor="{NAVY}", penwidth=1.5];
  verify [label="Authority\\nverification"]; reports [label="Citizen hazard\\nreports"];
  black -> map; alert -> map; map -> verify; reports -> verify;
}}"""


# ============================================================================================ OVERVIEW
def overview():
    cr, cells = D["crashes"], D["crash_cells"]
    s = RES_AV2.get("setup", {})
    e1 = RES_AV2.get("exp1", {})
    ml, ttc = e1.get("XGBoost (full)", {}), e1.get("TTC threshold rule", {})
    lead = RES_AV2.get("exp3_leadtime", {})

    people = [p for p in PROJECT["team"] if p]
    meta = " · ".join(x for x in [", ".join(people), PROJECT["guide"] and f"Guide: {PROJECT['guide']}",
                                  PROJECT["institution"]] if x)
    with st.container(key="hero"):
        a, b = st.columns([1.1, 1], gap="large", vertical_alignment="center")
        with a:
            st.markdown('<div class="hero-t">RAKSHA</div><div class="hero-s">Road Risk Intelligence</div>'
                        '<div class="hero-p">A data-driven project to identify high-risk road locations and predict '
                        'near-miss events before they turn into crashes.</div><br>', unsafe_allow_html=True)
            if meta:
                st.caption(meta)
            c1, c2 = st.columns(2)
            if c1.button("Explore Risk Map →", type="primary", use_container_width=True):
                st.switch_page(NAV[1])
            if c2.button("See Near-Miss Prediction", use_container_width=True):
                st.switch_page(NAV[2])
        b.markdown(SCENE, unsafe_allow_html=True)

    real_places = cells[cells.real].city.nunique()
    st.markdown(f"""<div class="kpis">
<div class="kpi">{icon('layers', '#16A34A')}<div><b>{real_places}</b><span>Regions · NYC, Chicago, GB</span></div></div>
<div class="kpi">{icon('car', '#1F4E79')}<div><b>{s.get('pairs', 0) / 1000:,.0f}K</b><span>Road-user pairs</span></div></div>
<div class="kpi">{icon('bolt', '#D97706')}<div><b>{s.get('positive_rate', 0):.2%}</b><span>Near-miss rate</span></div></div>
<div class="kpi">{icon('bars', '#16A34A')}<div><b>{ml.get('auc', 0):.3f}</b><span>AUC (XGBoost)</span></div></div>
</div>""", unsafe_allow_html=True)

    a, b = st.columns([1.15, 1], gap="medium")
    with a, st.container(border=True):
        st.markdown("**Road Risk Map** · New York")
        ny = cells[cells.city == "New York"]
        m = folium.Map(location=[40.72, -73.94], zoom_start=10, tiles="OpenStreetMap", zoom_control=False)
        HeatMap(ny.nlargest(400, "crashes")[["lat", "lon", "crashes"]].values.tolist(), radius=14, blur=18,
                min_opacity=.35).add_to(m)
        for _, r in ny[ny.blackspot].iterrows():
            folium.CircleMarker([r.lat, r.lon], radius=4, color=RED, fill=True, fill_opacity=.8, weight=1,
                                tooltip=f"Blackspot · {r.crashes} crashes").add_to(m)
        st_folium(m, height=280, use_container_width=True, returned_objects=[], key="home_map")
        st.caption("Heat = crash density from real police records · red dots = blackspots")
    with b, st.container(border=True):
        st.markdown("**Near-Miss Prediction** · real traffic (Argoverse 2)")
        if ml:
            l23 = lead.get("2-3s", {})
            st.markdown(f"""<div class="mrow"><span>Ranking accuracy (AUC)</span></div>
<div class="mbig">{ml['auc']:.3f}</div><div class="meter"><i style="width:{ml['auc'] * 100:.0f}%"></i></div>
<div class="mrow"><span>TTC rule (baseline)</span><b>{ttc['auc']:.3f}</b></div>
<div class="mrow"><span>Accuracy with 2–3 s still to go</span><b>{l23.get('xgb', 0):.3f}</b></div>
<div class="mrow"><span>Prediction horizon</span><b>3 s ahead</b></div>""", unsafe_allow_html=True)
            ld = pd.DataFrame([dict(when=k.replace("s", " s"), who=w, auc=v[c]) for k, v in lead.items()
                               for c, w in [("xgb", "XGBoost"), ("ttc_rule", "TTC rule")]])
            show(alt.Chart(ld).mark_line(point=True, strokeWidth=2.5).encode(
                x=alt.X("when:N", sort=[k.replace("s", " s") for k in lead][::-1], title="Time before near-miss",
                        axis=alt.Axis(labelAngle=0)),
                y=alt.Y("auc:Q", scale=alt.Scale(domain=[0.6, 1]), title=None),
                color=alt.Color("who:N", scale=alt.Scale(domain=["XGBoost", "TTC rule"], range=[RED, GREY]),
                                legend=alt.Legend(orient="top", title=None))).properties(height=140))
    # ---- how it works: 4 coloured step cards
    st.markdown('<div class="sec">How RAKSHA works</div>', unsafe_allow_html=True)
    steps = [("map", "#DC2626", "#FEE2E2", "1. Map crashes", "Real police crash records are grouped into 150 m cells; the worst cells become blackspots."),
             ("car", "#1F4E79", "#DBEAFE", "2. Track road users", "Positions and speeds of cars, bikes and pedestrians, 10 times a second."),
             ("bolt", "#D97706", "#FEF3C7", "3. Predict near-misses", "The model warns when two road users will come within 1 m in the next 3 s."),
             ("shield", "#16A34A", "#DCFCE7", "4. Verify & act", "Repeated alerts become risk zones that authorities check before acting.")]
    st.markdown('<div class="grid4">' + "".join(
        f'<div class="wcard"><div class="ibox" style="background:{bg}">{icon(ic, col)}</div><h4>{t}</h4><p>{b}</p></div>'
        for ic, col, bg, t, b in steps) + "</div>", unsafe_allow_html=True)

    # ---- two parts
    a, b = st.columns(2, gap="medium")
    with a:
        st.markdown(f"""<div class="wcard tint-red"><div class="tag" style="background:#DC2626">PART A · GIS</div>
<h3>Crash risk map</h3><p>Where have crashes already happened?</p><ul>
<li>{int(cr.real.sum()):,} real crash records from New York, Chicago and Great Britain</li>
<li>{int(cells.blackspot.sum())} blackspots found on a 150 m grid</li>
<li>Emerging risk zones where near-misses repeat before crashes pile up</li></ul></div>""", unsafe_allow_html=True)
    with b:
        st.markdown(f"""<div class="wcard tint-blue"><div class="tag" style="background:#1F4E79">PART B · ML</div>
<h3>Near-miss prediction</h3><p>Where is a crash about to happen?</p><ul>
<li>Trained and tested on {s.get('pairs', 0):,} real road-user pairs (Argoverse 2)</li>
<li>Uses time-to-collision, braking needed, gaps and speeds</li>
<li>XGBoost reaches AUC {ml.get('auc', 0):.3f} vs {ttc.get('auc', 0):.3f} for the classic TTC rule</li></ul></div>""",
                    unsafe_allow_html=True)

    # ---- near-miss explained
    st.markdown('<div class="sec">What is a near-miss?</div>', unsafe_allow_html=True)
    a, b = st.columns([1.4, 1], gap="medium")
    a.markdown("""<div class="wcard"><p style="margin-bottom:1rem">Two road users come <b>within 1 metre</b> of each
other while moving: nothing touched, but it nearly did. The model looks at three classic safety measures:</p>
<div class="grid3">
<div class="mini" style="background:#FEE2E2"><b style="color:#B91C1C">TTC</b><span>Seconds until contact if nobody changes speed</span></div>
<div class="mini" style="background:#FEF3C7"><b style="color:#B45309">PET</b><span>Time gap at the point where paths cross</span></div>
<div class="mini" style="background:#DBEAFE"><b style="color:#1F4E79">DRAC</b><span>How hard someone must brake to avoid contact</span></div>
</div></div>""", unsafe_allow_html=True)
    closing = (50 - 30) / 3.6
    b.markdown(f"""<div class="wcard"><div class="tag" style="background:#D97706">EXAMPLE</div>
<p style="margin:.6rem 0 .8rem">A car at <b>50 km/h</b> is <b>15 m</b> behind a car at <b>30 km/h</b>.</p>
<div class="grid2"><div><span class="lbl">Time-to-collision</span><div class="num" style="color:#D97706">{15 / closing:.1f} s</div></div>
<div><span class="lbl">Braking needed</span><div class="num" style="color:#1F4E79">{closing ** 2 / 30:.1f} m/s²</div></div></div>
<p class="note">Under 3 s is a warning; under 1.5 s is critical.</p></div>""", unsafe_allow_html=True)

    # ---- results
    if ml:
        l23 = lead.get("2-3s", {})
        st.markdown('<div class="sec">Results on real traffic</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="grid3">
<div class="wcard stat" style="border-top:4px solid #DC2626"><span class="lbl">Ranking accuracy (AUC)</span>
<div class="num" style="color:#DC2626">{ml['auc']:.0%}</div><p>XGBoost vs <b>{ttc['auc']:.0%}</b> for the TTC rule</p></div>
<div class="wcard stat" style="border-top:4px solid #D97706"><span class="lbl">Early warning, 2–3 s ahead</span>
<div class="num" style="color:#D97706">{l23.get('xgb', 0):.0%}</div><p>still accurate before the event (TTC rule: {l23.get('ttc_rule', 0):.0%})</p></div>
<div class="wcard stat" style="border-top:4px solid #1F4E79"><span class="lbl">Near-miss rate</span>
<div class="num" style="color:#1F4E79">1 in {1 / s['positive_rate']:,.0f}</div><p>moments, so alerts mean <i>check this place</i>, not certainty</p></div>
</div>""", unsafe_allow_html=True)

    # ---- data
    st.markdown('<div class="sec">Data used</div>', unsafe_allow_html=True)
    data = [("NYC crash records", True), ("Chicago crash records", True), ("UK STATS19 (GB)", True),
            ("Argoverse 2 trajectories", True), ("Own traffic simulator", False), ("India demo layer", False),
            ("Map alerts & citizen reports", False)]
    st.markdown('<div class="chips">' + "".join(
        f'<span class="chip">{n}<i class="{"real" if r else "sim"}">{"REAL" if r else "SIMULATED"}</i></span>'
        for n, r in data) + "</div>", unsafe_allow_html=True)
    next_button(0)


# ============================================================================================ RISK MAP
def risk_map():
    st.title("Risk Map")
    st.markdown("Red circles are **crash blackspots** from real police records. Orange circles are **emerging risk "
                "zones**: places where near-miss alerts repeat before crashes have piled up.")
    places = D["crash_cells"].groupby(["country", "city"]).agg(lat=("lat", "median"), lon=("lon", "median")).reset_index()
    places["label"] = places.city + " (" + places.country + ")"
    real_city = D["crash_cells"].groupby("city").real.first()

    a, b = st.columns([1, 2])
    pl = a.selectbox("Place", ["All places"] + places.label.tolist())
    layer_opts = {"Crash blackspots": True, "Emerging risk zones": True, "Near-miss heatmap": True,
                  "Individual model alerts": False, "Verified hazard reports": False}
    on = b.multiselect("Layers", list(layer_opts), default=[k for k, v in layer_opts.items() if v])
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
    k[3].metric("Crash data", "Real" if sel is None or real_city.get(sel, True) else "Simulated")

    m = folium.Map(location=loc, zoom_start=zoom, tiles="OpenStreetMap")
    if layers["Near-miss heatmap"]:
        hv = obs[obs.status == "verified"]
        HeatMap(hv[["lat", "lon"]].values.tolist(), radius=18, blur=22, min_opacity=.3).add_to(m)
    if layers["Crash blackspots"]:
        for _, r in cells_v[cells_v.blackspot].iterrows():
            tag = "" if r.real else " (simulated counts)"
            folium.CircleMarker([r.lat, r.lon], radius=4 + r.crashes ** .5 / (2 if sel is None else 1), color=RED,
                                fill=True, fill_opacity=.6, weight=1, tooltip=f"<b>{r.city}</b><br>{r.crashes} crashes · "
                                f"{int(r.injured)} injured · {int(r.killed)} killed{tag}").add_to(m)
    if layers["Emerging risk zones"]:
        for _, r in zones_v[zones_v.emerging].iterrows():
            folium.Circle([r.lat, r.lon], radius=140, color=ORANGE, fill=True, fill_opacity=.25, weight=2,
                          tooltip=f"<b>Emerging risk zone</b><br>{r.obs} near-miss alerts · avg risk {r.mean_p:.0%}<br>"
                                  f"past crashes here: {int(r.crashes) if pd.notna(r.crashes) else 0}").add_to(m)
    if layers["Individual model alerts"]:
        for _, r in obs.head(600).iterrows():
            folium.CircleMarker([r.lat, r.lon], radius=3, color=NAVY, fill=True,
                                tooltip=f"{r.type_a} vs {r.type_b} · risk {r.p:.0%} · {r.status} (demo)").add_to(m)
    if layers["Verified hazard reports"]:
        h = hz_v[hz_v.status.isin(["Verified"]) & hz_v.severity.isin(sev) & hz_v.category.isin(cat)]
        for _, r in h.iterrows():
            folium.Marker([r.lat, r.lon], icon=folium.Icon(color="green", icon="warning-sign"),
                          tooltip=f"{r.category} ({r.severity})").add_to(m)
    st_folium(m, height=540, use_container_width=True, returned_objects=[])
    st.caption("Legend: red = crash blackspot (size = number of crashes) · orange = emerging risk zone · "
               "heat = near-miss alerts · blue = individual alert · green marker = verified hazard report. "
               "Map alerts and hazard reports are simulated demo data placed at real intersections.")

    st.subheader("Top places to inspect")
    cz = (D["risk_zones"] if sel is None else D["risk_zones"][D["risk_zones"].city == sel])
    cz = cz.sort_values("obs", ascending=False).head(10)
    st.dataframe(cz[["city", "obs", "mean_p", "crashes", "blackspot", "emerging", "lat", "lon"]], hide_index=True,
                 use_container_width=True, column_config={
                     "city": "Place", "obs": st.column_config.NumberColumn("Near-miss alerts"),
                     "mean_p": st.column_config.NumberColumn("Average risk", format="%.2f"),
                     "crashes": st.column_config.NumberColumn("Past crashes", format="%d"),
                     "blackspot": "Already a blackspot?", "emerging": "Emerging zone?",
                     "lat": st.column_config.NumberColumn("Lat", format="%.4f"),
                     "lon": st.column_config.NumberColumn("Lon", format="%.4f")})
    next_button(1)


# ============================================================================================ PREDICTION
def prediction():
    st.title("Near-Miss Prediction")
    st.markdown("**Task:** using the last 1.5 seconds of two road users' movement, predict whether they will come "
                "within 1 metre of each other in the next 3 seconds.")
    if not SOURCES:
        st.error("No results yet. Run `src/experiments.py` first.")
        return
    src = st.radio("Dataset", list(SOURCES), horizontal=True)
    res, fig, dd, is_real = SOURCES[src]
    s, e1 = res.get("setup", {}), res["exp1"]
    if is_real:
        st.info("Real traffic: about 5,000 recordings from six US cities (Argoverse 2). Every pair of nearby road users "
                "is checked twice a second.")
    else:
        st.warning("Simulated traffic, used to develop the method. These scores are optimistic; use the real-traffic "
                   "results when presenting.")
    if s:
        m = st.columns(4)
        m[0].metric("Road-user pairs", f"{s.get('pairs', s.get('scenarios', 0)):,}")
        m[1].metric("Moments checked", f"{s['rows']:,}")
        m[2].metric("Near-miss rate", f"1 in {1 / s['positive_rate']:,.0f}")
        m[3].metric("Near-miss moments in test set", f"{s.get('test_positives', 0):,}")

    t1, t2, t3, t4, t5 = st.tabs(["1. Model vs TTC rule", "2. How early", "3. Important features",
                                  "4. Road-user pairs", "5. Technical details"])
    with t1:
        ttc, ml = e1["TTC threshold rule"], e1["XGBoost (full)"]
        st.markdown(f"XGBoost ranks danger correctly **{ml['auc']:.0%}** of the time vs **{ttc['auc']:.0%}** for the "
                    f"TTC rule, and its alert precision is **{ml['ap'] / max(ttc['ap'], 1e-9):.1f}×** higher.")
        df = pd.DataFrame([dict(model=MODEL_NAMES.get(k, k), metric=lbl, value=v[key],
                                kind="Baseline" if k == "TTC threshold rule" else ("All features" if "(full)" in k else "TTC only"))
                           for k, v in e1.items() for key, lbl in [("auc", "AUC (ranking accuracy)"),
                                                                    ("ap", "AP (alert precision)")]])
        a, b = st.columns(2)
        for col, (metric, dom) in zip([a, b], [("AUC (ranking accuracy)", [0.5, 1]), ("AP (alert precision)", None)]):
            d = df[df.metric == metric]
            show(alt.Chart(d).mark_bar(height=16).encode(
                x=alt.X("value:Q", title=None, scale=alt.Scale(domain=dom, clamp=True) if dom else alt.Undefined),
                y=alt.Y("model:N", sort="-x", title=None, axis=alt.Axis(labelLimit=230)),
                color=alt.Color("kind:N", scale=alt.Scale(domain=["Baseline", "TTC only", "All features"],
                                                          range=[GREY, LIGHT, NAVY]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["model", alt.Tooltip("value:Q", format=".3f")]).properties(height=250, title=metric), col)
        with st.expander("What do AUC and AP mean?"):
            st.markdown(f"""
- **AUC (ranking accuracy):** show the model one dangerous moment and one safe moment; AUC is how often it rates the
  dangerous one higher. 0.5 is a coin flip, 1.0 is perfect.
- **AP (average precision):** of the moments the model flags, how many really were near-misses, averaged over alert
  thresholds. Random guessing scores the near-miss rate ({s.get('positive_rate', 0):.2%}).
""")
    with t2:
        lead = res.get("exp3_leadtime", {})
        if lead:
            l2 = lead.get("2-3s", {})
            st.markdown(f"With **2–3 seconds** still to go, XGBoost still ranks danger correctly **{l2.get('xgb', 0):.0%}** "
                        f"of the time (TTC rule: {l2.get('ttc_rule', 0):.0%}). Predictions get easier closer to the event.")
            ld = pd.DataFrame([dict(when=k.replace("s", " s before"), who=w, auc=v[c]) for k, v in lead.items()
                               for c, w in [("xgb", "XGBoost"), ("ttc_rule", "TTC rule")]])
            order = [k.replace("s", " s before") for k in lead]
            show(alt.Chart(ld).mark_line(point=True, strokeWidth=2.5).encode(
                x=alt.X("when:N", sort=order[::-1], title="Time left until the near-miss", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("auc:Q", scale=alt.Scale(domain=[0.5, 1]), title="AUC"),
                color=alt.Color("who:N", scale=alt.Scale(domain=["XGBoost", "TTC rule"], range=[NAVY, GREY]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["who", "when", alt.Tooltip("auc:Q", format=".3f")]).properties(height=320))
        with st.expander("AUC for different look-ahead windows (1–5 s)"):
            e3 = pd.DataFrame([dict(L=int(k), who=w, auc=v[c]) for k, v in res["exp3"].items()
                               for c, w in [("xgb", "XGBoost"), ("ttc_rule", "TTC rule")]])
            show(alt.Chart(e3).mark_line(point=True, strokeWidth=2.5).encode(
                x=alt.X("L:O", title="Look-ahead window (seconds)", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("auc:Q", scale=alt.Scale(domain=[0.5, 1]), title="AUC"),
                color=alt.Color("who:N", scale=alt.Scale(domain=["XGBoost", "TTC rule"], range=[NAVY, GREY]),
                                legend=alt.Legend(orient="bottom", title=None))).properties(height=250))
    with t3:
        e2 = res["exp2"]
        first, last = list(e2.values())[0], list(e2.values())[-1]
        st.markdown(f"TTC alone reaches AUC **{first['auc']:.3f}**. Adding braking needed, gaps and speeds raises it to "
                    f"**{last['auc']:.3f}**. The most important features are the gap between road users and the "
                    "braking needed to avoid contact.")
        a, b = st.columns(2)
        g = pd.DataFrame([dict(group=GROUP_NAMES.get(k, k), auc=v["auc"]) for k, v in e2.items()])
        show(alt.Chart(g).mark_bar(height=18, color=NAVY).encode(
            x=alt.X("auc:Q", scale=alt.Scale(domain=[0.5, 1], clamp=True), title="AUC"),
            y=alt.Y("group:N", sort=None, title=None, axis=alt.Axis(labelLimit=230)),
            tooltip=["group", alt.Tooltip("auc:Q", format=".3f")]).properties(height=260, title="Adding features step by step"), a)
        shp = pd.DataFrame([dict(feature=FEATURE_NAMES.get(k, k), importance=v) for k, v in res.get("top_shap", {}).items()])
        if len(shp):
            show(alt.Chart(shp).mark_bar(height=14, color=LIGHT).encode(
                x=alt.X("importance:Q", title="Mean |SHAP value|"),
                y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=230)),
                tooltip=["feature", alt.Tooltip("importance:Q", format=".2f")]).properties(height=260, title="Feature importance (SHAP)"), b)
    with t4:
        e4 = pd.DataFrame(res["exp4"]).T.reset_index().rename(columns={"index": "pair", "mean": "rate", "count": "n"})
        e4 = e4.sort_values("rate", ascending=False)
        common = e4[e4.n >= 1000]
        top = e4.iloc[0]
        big = common.iloc[0] if len(common) else top
        st.markdown(f"Highest near-miss rate: **{top.pair}** ({top.rate:.1%} of {int(top.n):,} encounters). Among common "
                    f"pairings (1,000+ encounters): **{big.pair}** ({big.rate:.1%}). Pairings with few encounters are less certain.")
        show(alt.Chart(e4.head(12)).mark_bar(height=16, color=NAVY).encode(
            x=alt.X("rate:Q", axis=alt.Axis(format="%"), title="Share of encounters that became a near-miss"),
            y=alt.Y("pair:N", sort="-x", title=None),
            tooltip=["pair", alt.Tooltip("rate:Q", format=".2%"), alt.Tooltip("n:Q", title="encounters")]).properties(height=380))
        if is_real:
            st.caption("Argoverse 2 labels trucks as 'vehicle' (shown as car); cyclists are grouped with motorcycles.")
    with t5:
        if s:
            st.markdown(f"**Setup:** {s['window_s']} s of history → near-miss within {s['lookahead_s']:.0f} s · "
                        f"checks every {s['anchor_step_s']} s · split {s['split']}.")
        st.markdown("**All models**")
        st.dataframe(pd.DataFrame(e1).T.rename(index=MODEL_NAMES).round(3), use_container_width=True)
        st.markdown("**Feature groups (XGBoost)**")
        st.dataframe(pd.DataFrame(res["exp2"]).T.round(3), use_container_width=True)
        a, b = st.columns(2)
        a.image(str(fig / "confusion.png"), caption="Confusion matrices (threshold chosen on validation data)")
        b.image(str(fig / "shap.png"), caption="SHAP summary")
        st.markdown("**Try the model on your own data**")
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


# ============================================================================================ HOW IT WORKS
def how_it_works():
    st.title("How it Works")
    st.markdown("Method, definitions and an interactive example.")

    st.subheader("1. Problem statement")
    st.markdown("""
- Authorities find dangerous places from police crash records, so a location is only flagged after crashes pile up.
- Near-misses happen far more often but are not recorded.
- The usual near-miss measure, a single **time-to-collision (TTC)** threshold, ignores braking, gaps, speeds and the
  type of road user.
""")
    st.subheader("2. Objectives")
    st.markdown("""
1. Map crash blackspots from real police records (New York, Chicago, Great Britain).
2. Predict, from the past 1.5 s of movement, whether two road users will come within 1 m in the next 3 s.
3. Compare the model against the TTC rule on real traffic it has never seen.
4. Combine alerts and citizen reports into risk zones that authorities verify.
""")
    st.subheader("3. Method")
    st.markdown("""
| Step | What happens |
|---|---|
| Trajectories | Position and speed of every road user, 10 times per second (Argoverse 2) |
| Pairs | Every two road users that come within 40 m of each other |
| Features | TTC, PET, DRAC, gap, closing speed, speeds, accelerations, road-user types (over the last 1.5 s) |
| Label | 1 if the pair comes within 1 m in the next 3 s (at > 1.5 m/s relative speed, one in the other's path) |
| Models | Logistic regression, random forest, XGBoost; TTC rule as baseline |
| Testing | Train/test split by recording, so the model is tested on traffic it never saw |
| Mapping | Crashes are grouped into 150 m cells; the top 0.5% per city (≥ 8 crashes) are blackspots |
""")
    st.subheader("4. Try it: time-to-collision calculator")
    with st.container(border=True):
        a, b = st.columns(2, gap="large")
        with a:
            v1 = st.slider("Your speed (km/h)", 0, 120, 50)
            v2 = st.slider("Speed of the vehicle ahead (km/h)", 0, 120, 30)
            gap = st.slider("Gap between you (m)", 1, 60, 15)
        closing = (v1 - v2) / 3.6
        t = gap / closing if closing > 0 else float("inf")
        drac = closing ** 2 / (2 * gap) if closing > 0 else 0.0
        with b:
            c1, c2 = st.columns(2)
            c1.metric("Time-to-collision", "no collision" if t == float("inf") else f"{t:.1f} s")
            c2.metric("Braking needed", f"{drac:.1f} m/s²")
            if t < 1.5 or drac > 3.4:
                st.error("Critical: a collision is imminent unless someone brakes hard now.")
            elif t < 3:
                st.warning("Warning: closing fast, a near-miss is likely if nobody reacts.")
            else:
                st.success("Safe: enough time and distance to react.")
            st.caption("TTC = gap ÷ closing speed. Braking needed (DRAC) = closing speed² ÷ (2 × gap). "
                       "Common thresholds: TTC < 1.5 s critical; DRAC > about 3.4 m/s² dangerous.")

    st.subheader("5. Glossary")
    st.markdown("""
| Term | Meaning |
|---|---|
| Near-miss | Two road users come within 1 m of each other while moving; nothing touches |
| TTC | Time-to-collision: seconds until contact if nobody changes speed or direction |
| PET | Post-encroachment time: time gap between the two reaching the point where their paths cross |
| DRAC | Deceleration rate to avoid crash: how hard someone must brake to avoid contact |
| AUC | How often the model rates a dangerous moment above a safe one (0.5 = coin flip) |
| AP | Average precision: how many alerts are real near-misses (chance = near-miss rate) |
| Blackspot | A 150 m map cell among the worst 0.5% for crashes in its city |
| Emerging risk zone | A cell with repeated near-miss alerts that is not yet a blackspot |
""")
    st.subheader("6. Limitations")
    st.markdown("""
- The real trajectories come from US cities; Indian mixed traffic is not yet covered.
- Argoverse 2 has no vehicle sizes, so standard sizes were assumed.
- Near-misses are rare (about 0.1% of moments), so many alerts are false alarms.
- Map alerts and citizen reports in this demo are simulated.
""")
    st.subheader("7. Future work")
    st.markdown("""
- Extract trajectories from Indian roadside video (a YOLO + ByteTrack script is already written).
- Store citizen reports in a database instead of CSV files.
- Deploy the app online and add live camera feeds.
""")
    st.subheader("8. Tools used")
    st.markdown("Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · Shapely · Streamlit · Folium · Altair")
    st.subheader("9. References")
    st.markdown("""
1. NYC Open Data, *Motor Vehicle Collisions – Crashes*. data.cityofnewyork.us
2. City of Chicago, *Traffic Crashes – Crashes*. data.cityofchicago.org
3. UK Department for Transport, *Road Safety Data (STATS19)*. data.gov.uk
4. Wilson, B. et al. (2021). *Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting*. NeurIPS Datasets and Benchmarks.
5. Hayward, J. C. (1972). *Near-miss determination through use of a scale of danger*. Highway Research Record 384.
6. Treiber, M., Hennecke, A. & Helbing, D. (2000). *Congested traffic states in empirical observations and microscopic simulations*. Physical Review E 62.
7. Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
8. Lundberg, S. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
9. OpenStreetMap contributors. openstreetmap.org
""")
    next_button(3)


# ============================================================================================ REPORT
def report():
    st.title("Report a Hazard")
    st.markdown("Citizens can report potholes, missing signs or dangerous junctions. A report appears on the map only "
                "after an authority verifies it (see Authority Triage).")
    places = D["crash_cells"].groupby(["country", "city"]).agg(lat=("lat", "median"), lon=("lon", "median")).reset_index()
    with st.form("rep", clear_on_submit=True):
        a, b = st.columns(2)
        cat = a.selectbox("Problem", ["Pothole", "Damaged road", "Dangerous intersection", "Missing signage"])
        sev = b.select_slider("Severity", ["Low", "Medium", "High"], value="Medium")
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude", value=40.7580, format="%.5f")
        lon = c2.number_input("Longitude", value=-73.9855, format="%.5f")
        desc = st.text_area("Description (optional)")
        img = st.file_uploader("Photo (optional)", type=["jpg", "png"])
        if st.form_submit_button("Submit report", type="primary"):
            path = P / "hazard_reports.csv"; h = pd.read_csv(path)
            near = places.iloc[int(np.argmin(np.hypot(places.lat - lat, places.lon - lon)))]
            rid = int(h.id.max()) + 1 if len(h) else 1
            if img:
                (P / "uploads").mkdir(exist_ok=True)
                (P / "uploads" / f"{rid}_{Path(img.name).name}").write_bytes(img.read())
            new = dict(id=rid, city=near.city, country=near.country, category=cat, lat=lat, lon=lon, severity=sev,
                       status="Submitted", date=pd.Timestamp.now().normalize(), simulated=False, description=desc)
            pd.concat([h, pd.DataFrame([new])]).to_csv(path, index=False); load.clear()
            st.success(f"Report #{rid} ({near.city}) submitted. It is now waiting for review.")
    next_button(4)


# ============================================================================================ TRIAGE
def triage():
    st.title("Authority Triage")
    st.markdown("Officials review citizen reports and model-flagged zones before anything becomes official.")
    t1, t2 = st.tabs(["Citizen hazard reports", "Model-flagged emerging zones"])
    with t1:
        path = P / "hazard_reports.csv"; h = pd.read_csv(path)
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Waiting for review", int(h.status.isin(["Submitted", "Under review"]).sum()))
        q2.metric("High severity open", int((h.status.isin(["Submitted", "Under review"]) & (h.severity == "High")).sum()))
        q3.metric("Verified", int((h.status == "Verified").sum()))
        q4.metric("Resolved", int((h.status == "Resolved").sum()))
        flt = st.multiselect("Show status", sorted(h.status.unique()), default=["Submitted", "Under review"])
        rank = {"High": 0, "Medium": 1, "Low": 2}
        q = h[h.status.isin(flt)].sort_values(["severity", "date"], key=lambda c: c.map(rank) if c.name == "severity" else c)
        cols = [c for c in ["id", "severity", "category", "city", "status", "date", "description"] if c in q]
        st.dataframe(q[cols], hide_index=True, use_container_width=True, height=320,
                     column_config={"id": st.column_config.NumberColumn("Report #", format="%d")})
        with st.form("triage"):
            c1, c2, c3 = st.columns(3)
            rid = c1.number_input("Report #", int(h.id.min()), int(h.id.max()), int(q.id.iloc[0]) if len(q) else int(h.id.min()))
            new = c2.selectbox("Set status to", ["Under review", "Verified", "Resolved", "Rejected"])
            c3.write(""); c3.write("")
            if c3.form_submit_button("Update status", type="primary", use_container_width=True):
                h.loc[h.id == rid, "status"] = new; h.to_csv(path, index=False); load.clear()
                st.success(f"Report #{rid} → {new}")
    with t2:
        z = D["risk_zones"][D["risk_zones"].emerging].sort_values("obs", ascending=False)
        st.markdown("Places with 5 or more recent model alerts that are **not** yet crash blackspots (simulated demo alerts).")
        st.dataframe(z[["city", "obs", "mean_p", "crashes", "lat", "lon"]], hide_index=True, use_container_width=True,
                     column_config={"city": "Place", "obs": "Alerts (60 days)",
                                    "mean_p": st.column_config.NumberColumn("Average risk", format="%.2f"),
                                    "crashes": st.column_config.NumberColumn("Past crashes", format="%d")})
    next_button(5)


# ============================================================================================ ANALYTICS
def analytics():
    st.title("Crash Analytics")
    st.markdown("Real police crash records for the selected places.")
    c = D["crashes"].dropna(subset=["crash_date"])
    places = sorted(c.city.unique())
    pick = st.multiselect("Places", places, default=[p for p in ["New York", "Chicago", "Great Britain"] if p in places])
    c = c[c.city.isin(pick)]
    if c.empty:
        st.info("Pick at least one place.")
        return
    k = st.columns(3)
    k[0].metric("Crashes", f"{len(c):,}"); k[1].metric("People injured", f"{int(c.injured.sum()):,}")
    k[2].metric("People killed", f"{int(c.killed.sum()):,}")
    a, b = st.columns(2)
    byc = c.groupby("city").size().reset_index(name="crashes")
    show(alt.Chart(byc).mark_bar(height=18, color=NAVY).encode(
        x=alt.X("crashes:Q", title="Crash records"), y=alt.Y("city:N", sort="-x", title=None),
        tooltip=["city", "crashes"]).properties(height=280, title="Crashes by place"), a)
    bym = c.groupby(["city", pd.Grouper(key="crash_date", freq="MS")]).size().reset_index(name="crashes")
    show(alt.Chart(bym).mark_line(strokeWidth=2).encode(
        x=alt.X("crash_date:T", title=None), y=alt.Y("crashes:Q", title="Crashes per month"),
        color=alt.Color("city:N", scale=alt.Scale(range=[NAVY, ORANGE, GREY, LIGHT, RED, "#6B7280"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["city", alt.Tooltip("crash_date:T", format="%b %Y"), "crashes"]).properties(height=280, title="Crashes per month"), b)
    sev = c.assign(outcome=np.where(c.killed > 0, "Fatal", np.where(c.injured > 0, "Injury", "Damage only")))
    sv = sev.groupby(["city", "outcome"]).size().reset_index(name="n")
    show(alt.Chart(sv).mark_bar(height=20).encode(
        x=alt.X("n:Q", stack="normalize", axis=alt.Axis(format="%"), title="Share of crashes"),
        y=alt.Y("city:N", title=None),
        color=alt.Color("outcome:N", scale=alt.Scale(domain=["Damage only", "Injury", "Fatal"], range=["#D1D5DB", LIGHT, NAVY]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["city", "outcome", "n"]).properties(height=220, title="Crash severity"))
    st.caption("Great Britain records only injury collisions, so it has no 'damage only' share.")


# ============================================================================================ NAV
NAV = [st.Page(overview, title="Home", icon=":material/home:", url_path="home", default=True),
       st.Page(risk_map, title="Risk Map", icon=":material/map:", url_path="risk-map"),
       st.Page(prediction, title="Near-Miss Prediction", icon=":material/play_arrow:", url_path="prediction"),
       st.Page(how_it_works, title="How it Works", icon=":material/layers:", url_path="how-it-works"),
       st.Page(report, title="Report a Hazard", icon=":material/warning:", url_path="report"),
       st.Page(triage, title="Authority Triage", icon=":material/shield:", url_path="triage"),
       st.Page(analytics, title="Crash Analytics", icon=":material/bar_chart:", url_path="analytics")]
nav = st.navigation(NAV, position="sidebar")
with st.sidebar:
    st.markdown("**RAKSHA**")
    st.caption("Road risk mapping and near-miss prediction")
    st.divider()
    st.caption("Real data: crash records (NYC, Chicago, GB), Argoverse 2 trajectories. "
               "Simulated: India crash counts, map alerts, citizen reports.")
nav.run()
