"""RAKSHA platform - run: streamlit run app/raksha.py

Laid out as a guided story for presenting: Home -> Risk map -> Prediction results -> How it works ->
Community -> Authority triage -> Crash analytics. Every chart has a plain-English takeaway above it;
technical tables live under "Technical details".
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
st.set_page_config(page_title="RAKSHA · Road Risk Intelligence", page_icon="🛡️", layout="wide")

ACCENT, MUTED, GOOD = "#D9480F", "#9AA5B1", "#2F9E44"

# ------------------------------------------------------------------------------------------ style
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {padding: 1.6rem 1.8rem; border-radius: 14px; background: linear-gradient(135deg, #FFF4E6 0%, #FFE8CC 100%);
       border: 1px solid #FFD8A8; margin-bottom: 1.2rem;}
.hero h1 {margin: 0 0 .3rem 0; font-size: 2.3rem; color: #1F2933;}
.hero p {margin: 0; font-size: 1.1rem; color: #3E4C59;}
.card {padding: 1rem 1.1rem; border-radius: 12px; border: 1px solid #E4E7EB; background: #FFFFFF; height: 100%;
       box-shadow: 0 1px 2px rgba(0,0,0,.04);}
.card h4 {margin: .2rem 0 .4rem 0; font-size: 1.02rem; color: #1F2933;}
.card p {margin: 0; font-size: .92rem; color: #52606D; line-height: 1.45;}
.step {font-size: .78rem; font-weight: 700; letter-spacing: .06em; color: #D9480F; text-transform: uppercase;}
.big {font-size: 2.2rem; font-weight: 700; color: #1F2933; line-height: 1.1;}
.big small {font-size: 1rem; font-weight: 500; color: #7B8794;}
.takeaway {padding: .85rem 1rem; border-left: 4px solid #D9480F; background: #FFF8F3; border-radius: 6px;
           margin: .4rem 0 1rem 0; color: #1F2933; font-size: .98rem;}
.badge {display: inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .75rem; font-weight: 700;}
.real {background: #D3F9D8; color: #2B8A3E;} .sim {background: #FFF3BF; color: #A65E00;}
.chip {display: inline-block; margin: 0 .5rem .4rem 0; font-size: .85rem; color: #3E4C59;}
.dot {display: inline-block; width: .8rem; height: .8rem; border-radius: 50%; margin-right: .35rem; vertical-align: -1px;}
</style>
""", unsafe_allow_html=True)

REAL = '<span class="badge real">REAL DATA</span>'
SIM = '<span class="badge sim">SIMULATED</span>'


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def takeaway(s):
    html(f'<div class="takeaway">💡 {s}</div>')


def card(col, step, title, body):
    col.markdown(f'<div class="card"><div class="step">{step}</div><h4>{title}</h4><p>{body}</p></div>',
                 unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------ data
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
# dataset label -> (results, figure dir, data dir, is_real)
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
               "XGBoost (full)": "★ RAKSHA (XGBoost)"}
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

PAGES = ["🏠 Home", "🗺️ Risk map", "🤖 Near-miss prediction", "📖 How it works", "📣 Report a hazard",
         "🏛️ Authority triage", "📊 Crash analytics"]


def goto(p):
    st.session_state.page = p


def next_button(current):
    i = PAGES.index(current)
    if i + 1 < len(PAGES):
        st.divider()
        _, r = st.columns([3, 2])
        r.button(f"Next: {PAGES[i + 1]} →", on_click=goto, args=(PAGES[i + 1],), use_container_width=True)


# ------------------------------------------------------------------------------------------ sidebar
with st.sidebar:
    html("<h2 style='margin-bottom:0'>🛡️ RAKSHA</h2><p style='color:#7B8794;margin-top:0'>Road Risk Intelligence</p>")
    page = st.radio("Go to", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    html(f"""<div style="font-size:.85rem;color:#52606D;line-height:1.6">
<b>What's real, what's simulated</b><br>
{REAL} Crash records — New York, Chicago, Great Britain<br>
{REAL} Road-user trajectories — Argoverse 2<br>
{SIM} India crash counts (real junction locations)<br>
{SIM} Map observations &amp; community reports (demo)
</div>""")

# ================================================================================== HOME
if page == PAGES[0]:
    html("""<div class="hero"><h1>🛡️ RAKSHA — Road Risk Intelligence</h1>
<p>Crash maps tell us where people <i>were</i> hurt. RAKSHA also looks for the <b>near-misses</b> that come before
crashes, so risky places can be fixed <b>before</b> someone is hurt.</p></div>""")

    st.subheader("The idea in four steps")
    c = st.columns(4)
    card(c[0], "Step 1 · Watch", "Track road users",
         "Positions and speeds of cars, buses, bikes and pedestrians, ten times a second.")
    card(c[1], "Step 2 · Measure", "Compute safety signals",
         "How soon they would collide, how hard someone must brake, how small the gap is.")
    card(c[2], "Step 3 · Predict", "Flag near-misses early",
         "A machine-learning model warns up to 3 seconds before two road users come within 1 metre.")
    card(c[3], "Step 4 · Act", "Map it and verify it",
         "Repeated warnings become candidate risk zones that authorities check on the ground.")

    st.subheader("Key numbers")
    cr, cells = D["crashes"], D["crash_cells"]
    k = st.columns(4)
    k[0].metric("Real crash records mapped", f"{int(cr.real.sum()):,}")
    k[1].metric("Crash blackspots found", f"{int(cells.blackspot.sum())}", help="Grid cells (~150 m) in the top 0.5% "
                "for crashes in their city, with at least 8 crashes.")
    if RES_AV2:
        s = RES_AV2["setup"]
        k[2].metric("Real road-user pairs analysed", f"{s['pairs']:,}")
        k[3].metric("Moments checked by the model", f"{s['rows'] / 1e6:.1f} M")

    if RES_AV2:
        e1 = RES_AV2["exp1"]; ttc, ml = e1["TTC threshold rule"], e1["XGBoost (full)"]
        lead = RES_AV2.get("exp3_leadtime", {}).get("2-3s")
        st.subheader("Headline result on real traffic")
        a, b = st.columns([1, 1])
        with a:
            html(f"""<div class="card"><div class="step">How well do we spot danger?</div>
<div class="big">{ml['auc'] * 100:.0f}% <small>vs {ttc['auc'] * 100:.0f}% for the classic rule</small></div>
<p style="margin-top:.5rem">Shown a dangerous moment and a safe one, the RAKSHA model picks the dangerous one
{ml['auc'] * 100:.0f}% of the time. The usual time-to-collision rule manages {ttc['auc'] * 100:.0f}%.</p></div>""")
        with b:
            if lead:
                html(f"""<div class="card"><div class="step">How early?</div>
<div class="big">{lead['xgb'] * 100:.0f}% <small>with 2–3 seconds still to go</small></div>
<p style="margin-top:.5rem">Even 2–3 seconds before the near-miss, the model still picks the dangerous moment
{lead['xgb'] * 100:.0f}% of the time (classic rule: {lead['ttc_rule'] * 100:.0f}%). That's enough time for a warning.</p></div>""")
        takeaway(f"Near-misses are rare (about 1 in {1 / RES_AV2['setup']['positive_rate']:,.0f} moments), so many "
                 "alerts are still false alarms. "
                 "RAKSHA is a tool to point inspectors at places worth checking, not an automatic verdict.")
    next_button(page)

# ================================================================================== MAP
elif page == PAGES[1]:
    st.title("🗺️ Where is the risk?")
    st.caption("Red circles are proven crash blackspots. Orange zones are places where the model keeps flagging "
               "near-misses but crashes haven't piled up yet: candidates for prevention.")
    places = D["crash_cells"].groupby(["country", "city"]).agg(lat=("lat", "median"), lon=("lon", "median")).reset_index()
    places["label"] = places.city + " (" + places.country + ")"
    real_city = D["crash_cells"].groupby("city").real.first()

    pl = st.selectbox("Place", ["All places"] + places.label.tolist())
    layer_opts = {"Crash blackspots": True, "Emerging risk zones": True, "Near-miss heatmap": True,
                  "Individual model alerts": False, "Verified hazard reports": False}
    on = st.pills("Layers (click to show / hide)", list(layer_opts), selection_mode="multi",
                  default=[k for k, v in layer_opts.items() if v]) or []
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

    html("""<div>
<span class="chip"><span class="dot" style="background:#7b0000"></span>Crash blackspot (size = crashes)</span>
<span class="chip"><span class="dot" style="background:#e67e22"></span>Emerging risk zone</span>
<span class="chip"><span class="dot" style="background:linear-gradient(90deg,yellow,red)"></span>Near-miss heat</span>
<span class="chip"><span class="dot" style="background:#2980b9"></span>Model alert</span>
<span class="chip"><span class="dot" style="background:#37b24d"></span>Verified hazard report</span></div>""")

    obs = D["obs"][D["obs"].type_a.isin(vt) & D["obs"].date.dt.date.between(*dr) & D["obs"].status.isin(show_status)]
    cells_v, zones_v, hz_v = D["crash_cells"], D["risk_zones"], D["hazards"]
    if pl == "All places":
        loc, zoom, sel = [30.0, 0.0], 2, None
    else:
        row = places[places.label == pl].iloc[0]
        loc, zoom, sel = [row.lat, row.lon], (6 if row.city == "Great Britain" else 11), row.city
        obs, cells_v = obs[obs.city == sel], cells_v[cells_v.city == sel]
        zones_v, hz_v = zones_v[zones_v.city == sel], hz_v[hz_v.city == sel]
        html(f"Crash data for {sel}: {REAL if real_city.get(sel, True) else SIM}")
    m = folium.Map(location=loc, zoom_start=zoom, tiles="OpenStreetMap")
    if layers["Near-miss heatmap"]:
        hv = obs[obs.status == "verified"]
        HeatMap(hv[["lat", "lon"]].values.tolist(), radius=18, blur=22, min_opacity=.3,
                gradient={0.3: "yellow", 0.6: "orange", 1.0: "red"}).add_to(m)
    if layers["Crash blackspots"]:
        for _, r in cells_v[cells_v.blackspot].iterrows():
            tag = "" if r.real else " (simulated counts)"
            folium.CircleMarker([r.lat, r.lon], radius=4 + r.crashes ** .5 / (2 if sel is None else 1), color="#7b0000",
                                fill=True, fill_opacity=.7, tooltip=f"<b>{r.city}</b><br>{r.crashes} crashes · "
                                f"{int(r.injured)} injured · {int(r.killed)} killed{tag}").add_to(m)
    if layers["Emerging risk zones"]:
        for _, r in zones_v[zones_v.emerging].iterrows():
            folium.Circle([r.lat, r.lon], radius=140, color="#e67e22", fill=True, fill_opacity=.25, weight=2,
                          tooltip=f"<b>Emerging risk zone</b><br>{r.obs} near-miss alerts · avg risk {r.mean_p:.0%}<br>"
                                  f"past crashes here: {int(r.crashes) if pd.notna(r.crashes) else 0}").add_to(m)
    if layers["Individual model alerts"]:
        for _, r in obs.head(600).iterrows():
            folium.CircleMarker([r.lat, r.lon], radius=3, color="#2980b9", fill=True,
                                tooltip=f"{r.type_a} vs {r.type_b} · risk {r.p:.0%} · {r.status} (demo)").add_to(m)
    if layers["Verified hazard reports"]:
        h = hz_v[hz_v.status.isin(["Verified"]) & hz_v.severity.isin(sev) & hz_v.category.isin(cat)]
        for _, r in h.iterrows():
            folium.Marker([r.lat, r.lon], icon=folium.Icon(color="green", icon="warning-sign"),
                          tooltip=f"{r.category} ({r.severity})").add_to(m)
    st_folium(m, height=560, use_container_width=True, returned_objects=[])
    takeaway("A hot spot on the heatmap means <b>look here</b>, not <b>this is a blackspot</b>. Map alerts are "
             f"demo data {SIM} placed at real intersections. Blackspots come from real police crash records.")

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
    next_button(page)

# ================================================================================== PREDICTION
elif page == PAGES[2]:
    st.title("🤖 Can we see a near-miss coming?")
    st.caption("The question: given the last 1.5 seconds of two road users' movement, will they come within 1 metre "
               "of each other in the next 3 seconds?")
    if not SOURCES:
        st.error("No results yet. Run `src/experiments.py` first.")
        st.stop()
    src = st.segmented_control("Dataset", list(SOURCES), default=list(SOURCES)[0]) or list(SOURCES)[0]
    res, fig, dd, is_real = SOURCES[src]
    s, e1 = res.get("setup", {}), res["exp1"]
    if is_real:
        html(f"{REAL} About 5,000 short recordings of real traffic in six US cities (Argoverse 2). "
             "Every pair of nearby road users is checked twice a second.")
    else:
        html(f"{SIM} Computer-generated traffic. Useful to check the method works, but scores here are "
             "flattering. Use the real-traffic results when presenting.")
    if s:
        m = st.columns(4)
        m[0].metric("Road-user pairs", f"{s.get('pairs', s.get('scenarios', 0)):,}")
        m[1].metric("Moments checked", f"{s['rows'] / 1e6:.1f} M" if s["rows"] >= 1e6 else f"{s['rows'] / 1e3:.0f} k")
        m[2].metric("How rare is a near-miss?", f"1 in {1 / s['positive_rate']:,.0f}")
        m[3].metric("Near-miss moments in test set", f"{s.get('test_positives', 0):,}")

    t1, t2, t3, t4, t5 = st.tabs(["① Does it beat the classic rule?", "② How early?", "③ What does it look at?",
                                  "④ Who is most at risk?", "⑤ Technical details"])

    with t1:
        ttc, ml = e1["TTC threshold rule"], e1["XGBoost (full)"]
        takeaway(f"The RAKSHA model ranks danger correctly <b>{ml['auc']:.0%}</b> of the time, against "
                 f"<b>{ttc['auc']:.0%}</b> for the classic time-to-collision rule. Its alerts are also "
                 f"<b>{ml['ap'] / max(ttc['ap'], 1e-9):.1f}×</b> more precise.")
        rows = [dict(model=SHORT_NAMES.get(k, k), full=MODEL_NAMES.get(k, k), metric=lbl, value=v[key],
                     kind="All signals" if "(full)" in k else ("Classic rule" if k == "TTC threshold rule" else "ML, TTC only"))
                for k, v in e1.items() for key, lbl in [("auc", "Ranking accuracy (AUC)"), ("ap", "Alert precision (AP)")]]
        df = pd.DataFrame(rows)
        for metric, dom in [("Ranking accuracy (AUC)", [0.5, 1]), ("Alert precision (AP)", None)]:
            d = df[df.metric == metric]
            ch = alt.Chart(d).mark_bar(cornerRadiusEnd=4, clip=True).encode(
                x=alt.X("value:Q", title=None, scale=alt.Scale(domain=dom) if dom else alt.Undefined),
                y=alt.Y("model:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
                color=alt.Color("kind:N", scale=alt.Scale(domain=["Classic rule", "ML, TTC only", "All signals"],
                                                          range=[MUTED, "#F4A261", ACCENT]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=[alt.Tooltip("full:N", title="model"), alt.Tooltip("value:Q", format=".3f")]
            ).properties(height=230, title=metric)
            st.altair_chart(ch, use_container_width=True)
        with st.expander("What do these two scores mean?"):
            st.markdown(f"""
- **Ranking accuracy (AUC):** show the model one dangerous moment and one safe moment. AUC is how often it
  rates the dangerous one higher. 50% is a coin flip and 100% is perfect.
- **Alert precision (AP):** of the moments the model flags, how many really were near-misses, averaged across
  alert thresholds. Random guessing scores the near-miss rate
  (here {s.get('positive_rate', 0):.2%}), so even small numbers can be many times better than chance.
""")

    with t2:
        lead = res.get("exp3_leadtime", {})
        if lead:
            l2 = lead.get("2-3s", {})
            takeaway(f"With <b>2–3 seconds</b> still to go, the model picks the dangerous moment "
                     f"<b>{l2.get('xgb', 0):.0%}</b> of the time; the classic rule manages <b>{l2.get('ttc_rule', 0):.0%}</b>. "
                     "The closer the event, the easier it gets, but the model stays ahead the whole way.")
            ld = pd.DataFrame([dict(when=k.replace("s", " s before"), who=w, auc=v[c]) for k, v in lead.items()
                               for c, w in [("xgb", "RAKSHA model"), ("ttc_rule", "Classic rule")]])
            order = [k.replace("s", " s before") for k in lead]
            ch = alt.Chart(ld).mark_line(point=alt.OverlayMarkDef(size=90), strokeWidth=3).encode(
                x=alt.X("when:N", sort=order[::-1], title="Time left until the near-miss"),
                y=alt.Y("auc:Q", scale=alt.Scale(domain=[0.5, 1]), title="Ranking accuracy (AUC)"),
                color=alt.Color("who:N", scale=alt.Scale(domain=["RAKSHA model", "Classic rule"], range=[ACCENT, MUTED]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["who", "when", alt.Tooltip("auc:Q", format=".3f")]).properties(height=320)
            st.altair_chart(ch, use_container_width=True)
        with st.expander("Same idea, different question: \"will it happen within the next L seconds?\""):
            e3 = pd.DataFrame([dict(L=int(k), who=w, auc=v[c]) for k, v in res["exp3"].items()
                               for c, w in [("xgb", "RAKSHA model"), ("ttc_rule", "Classic rule")]])
            st.altair_chart(alt.Chart(e3).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X("L:O", title="Look-ahead window L (seconds)"),
                y=alt.Y("auc:Q", scale=alt.Scale(domain=[0.5, 1]), title="AUC"),
                color=alt.Color("who:N", scale=alt.Scale(domain=["RAKSHA model", "Classic rule"], range=[ACCENT, MUTED]),
                                legend=alt.Legend(orient="bottom", title=None))).properties(height=260),
                use_container_width=True)

    with t3:
        e2 = res["exp2"]
        first, last = list(e2.values())[0], list(e2.values())[-1]
        takeaway(f"Time-to-collision alone gets to <b>{first['auc']:.0%}</b>. Adding how hard someone must brake, "
                 f"the gap and the speeds lifts it to <b>{last['auc']:.0%}</b>. The model leans most on the "
                 "<b>gap between road users</b> and the <b>braking needed to avoid contact</b>.")
        a = b = st.container()
        g = pd.DataFrame([dict(group=GROUP_NAMES.get(k, k), auc=v["auc"]) for k, v in e2.items()])
        a.altair_chart(alt.Chart(g).mark_bar(color=ACCENT, cornerRadiusEnd=4).encode(
            x=alt.X("auc:Q", scale=alt.Scale(domain=[0.5, 1]), title="AUC"),
            y=alt.Y("group:N", sort=None, title=None, axis=alt.Axis(labelLimit=300)), tooltip=["group", alt.Tooltip("auc:Q", format=".3f")]
        ).properties(height=260, title="Adding signals step by step"), use_container_width=True)
        shp = pd.DataFrame([dict(feature=FEATURE_NAMES.get(k, k), importance=v) for k, v in res.get("top_shap", {}).items()])
        if len(shp):
            b.altair_chart(alt.Chart(shp).mark_bar(color="#495057", cornerRadiusEnd=4).encode(
                x=alt.X("importance:Q", title="Average influence on the prediction (SHAP)"),
                y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300)), tooltip=["feature", alt.Tooltip("importance:Q", format=".2f")]
            ).properties(height=260, title="What the model pays most attention to"), use_container_width=True)

    with t4:
        e4 = pd.DataFrame(res["exp4"]).T.reset_index().rename(columns={"index": "pair", "mean": "rate", "count": "n"})
        e4 = e4.sort_values("rate", ascending=False)
        common = e4[e4.n >= 1000]
        top = e4.iloc[0]
        big = common.iloc[0] if len(common) else top
        e4 = e4.head(12)
        takeaway(f"The riskiest pairing is <b>{top.pair}</b>: {top.rate:.1%} of {int(top.n):,} encounters became a "
                 f"near-miss. Among common pairings (1,000+ encounters) it is <b>{big.pair}</b> at {big.rate:.1%}. "
                 "Pairings with few encounters are less certain; hover a bar to see the count.")
        st.altair_chart(alt.Chart(e4).mark_bar(color=ACCENT, cornerRadiusEnd=4).encode(
            x=alt.X("rate:Q", axis=alt.Axis(format="%"), title="Share of encounters that became a near-miss"),
            y=alt.Y("pair:N", sort="-x", title=None),
            tooltip=["pair", alt.Tooltip("rate:Q", format=".2%"), alt.Tooltip("n:Q", title="encounters")]
        ).properties(height=380), use_container_width=True)
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
        b.image(str(fig / "shap.png"), caption="SHAP summary (XGBoost)")
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
    next_button(page)

# ================================================================================== HOW IT WORKS
elif page == PAGES[3]:
    st.title("📖 How it works")
    st.subheader("From movement to a warning")
    c = st.columns(5)
    card(c[0], "1", "Trajectories", "Where each road user is and how fast it moves, 10× per second.")
    card(c[1], "2", "Pairs", "Every two road users that come within 40 m of each other.")
    card(c[2], "3", "Safety signals", "Time-to-collision, time gap at the conflict point, braking needed, gap, speeds.")
    card(c[3], "4", "Prediction", "Every 0.5 s, using only the past 1.5 s: near-miss in the next 3 s?")
    card(c[4], "5", "Map & verify", "Repeated alerts become emerging zones for authorities to inspect.")

    st.subheader("Glossary")
    g = st.columns(2)
    g[0].markdown("""
**Near-miss**: two road users come within **1 metre** of each other while moving relative to each other
(> 1.5 m/s), with one in the other's path. Nothing touched, but it nearly did.

**Time-to-collision (TTC)**: if nobody changes speed or direction, how many seconds until they touch.
The classic safety measure, and our baseline.

**Post-encroachment time (PET)**: the time gap between the two arriving at the point where their paths cross.
""")
    g[1].markdown("""
**Braking needed (DRAC)**: how hard someone would have to brake to avoid contact. Higher means more dangerous.

**AUC (ranking accuracy)**: how often the model rates a dangerous moment above a safe one. 50% = coin flip.

**AP (alert precision)**: how many of the model's alerts are real near-misses. Chance level = how rare
near-misses are.
""")

    st.subheader("Data sources")
    ds = [("NYC Motor Vehicle Collisions", "Crash blackspots", REAL), ("Chicago Traffic Crashes", "Crash blackspots", REAL),
          ("Great Britain STATS19", "Crash blackspots", REAL),
          ("Argoverse 2 Motion Forecasting · 5,000 scenarios", "Train & test the near-miss model", REAL),
          ("RAKSHA traffic simulator", "Method development", SIM),
          ("India, 8 cities (OpenStreetMap junctions)", "Demo map coverage (crash counts simulated)", SIM),
          ("Map alerts & community reports", "Demo of the verification workflow", SIM)]
    html("<table style='width:100%;border-collapse:collapse'>" + "".join(
        f"<tr style='border-bottom:1px solid #E4E7EB'><td style='padding:.45rem'>{n}</td>"
        f"<td style='padding:.45rem;color:#52606D'>{u}</td><td style='padding:.45rem'>{b}</td></tr>" for n, u, b in ds)
         + "</table>")

    st.subheader("Being honest about limits")
    st.markdown("""
- Argoverse 2 has no vehicle sizes, so standard sizes are assumed. That adds some error to the near-miss labels.
- The real trajectories are from **US cities**. Indian mixed traffic still needs its own trajectories,
  for example from roadside video using `src/detect_track.py`.
- Near-misses are rare, so many alerts are false alarms. Alerts are a prompt to inspect, not a verdict.
- Test data is kept fully separate: no recording appears in both training and testing.
""")
    next_button(page)

# ================================================================================== COMMUNITY
elif page == PAGES[4]:
    st.title("📣 Report a road hazard")
    st.caption("Anyone can report a hazard. Reports only reach the map after an authority verifies them.")
    places = D["crash_cells"].groupby(["country", "city"]).agg(lat=("lat", "median"), lon=("lon", "median")).reset_index()
    with st.form("rep", clear_on_submit=True):
        a, b = st.columns(2)
        cat = a.selectbox("What's the problem?", ["Pothole", "Damaged road", "Dangerous intersection", "Missing signage"])
        sev = b.select_slider("How serious?", ["Low", "Medium", "High"], value="Medium")
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude", value=40.7580, format="%.5f")
        lon = c2.number_input("Longitude", value=-73.9855, format="%.5f")
        desc = st.text_area("Describe it (optional)")
        img = st.file_uploader("Add a photo (optional)", type=["jpg", "png"])
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
            st.success(f"Thanks! Report #{rid} ({near.city}) was submitted and is waiting for review.")
    next_button(page)

# ================================================================================== TRIAGE
elif page == PAGES[5]:
    st.title("🏛️ Authority triage")
    st.caption("Officials review citizen reports and model-flagged zones before anything becomes official.")
    t1, t2 = st.tabs(["Citizen hazard reports", "Model-flagged emerging zones"])
    with t1:
        path = P / "hazard_reports.csv"; h = pd.read_csv(path)
        q1, q2, q3 = st.columns(3)
        q1.metric("Waiting for review", int(h.status.isin(["Submitted", "Under review"]).sum()))
        q2.metric("Verified", int((h.status == "Verified").sum()))
        q3.metric("Resolved", int((h.status == "Resolved").sum()))
        flt = st.multiselect("Show status", sorted(h.status.unique()), default=["Submitted", "Under review"])
        rank = {"High": 0, "Medium": 1, "Low": 2}
        q = h[h.status.isin(flt)].sort_values(["severity", "date"], key=lambda c: c.map(rank) if c.name == "severity" else c)
        cols = [c for c in ["id", "severity", "category", "city", "status", "date", "description", "lat", "lon"] if c in q]
        st.dataframe(q[cols], hide_index=True, use_container_width=True)
        with st.form("triage"):
            c1, c2, c3 = st.columns([1, 1, 1])
            rid = c1.number_input("Report #", int(h.id.min()), int(h.id.max()), int(q.id.iloc[0]) if len(q) else int(h.id.min()))
            new = c2.selectbox("Set status to", ["Under review", "Verified", "Resolved", "Rejected"])
            c3.write(""); c3.write("")
            if c3.form_submit_button("Update", type="primary", use_container_width=True):
                h.loc[h.id == rid, "status"] = new; h.to_csv(path, index=False); load.clear()
                st.success(f"Report #{rid} → {new}")
    with t2:
        z = D["risk_zones"][D["risk_zones"].emerging].sort_values("obs", ascending=False)
        html(f"Places with 5+ recent model alerts that are <b>not</b> yet crash blackspots. {SIM} demo alerts.")
        st.dataframe(z[["city", "obs", "mean_p", "crashes", "lat", "lon"]], hide_index=True, use_container_width=True,
                     column_config={"city": "Place", "obs": "Alerts (60 days)",
                                    "mean_p": st.column_config.ProgressColumn("Average risk", min_value=0, max_value=1, format="%.2f"),
                                    "crashes": st.column_config.NumberColumn("Past crashes", format="%d")})
    next_button(page)

# ================================================================================== ANALYTICS
else:
    st.title("📊 Crash analytics")
    c = D["crashes"].dropna(subset=["crash_date"]).copy()
    c["crash_date"] = pd.to_datetime(c.crash_date, errors="coerce")
    places = sorted(c.city.unique())
    pick = st.multiselect("Places", places, default=[p for p in ["New York", "Chicago", "Great Britain"] if p in places])
    c = c[c.city.isin(pick)]
    if c.empty:
        st.info("Pick at least one place.")
        st.stop()
    k = st.columns(3)
    k[0].metric("Crashes", f"{len(c):,}"); k[1].metric("People injured", f"{int(c.injured.sum()):,}")
    k[2].metric("People killed", f"{int(c.killed.sum()):,}")
    a, b = st.columns(2)
    byc = c.groupby("city").size().reset_index(name="crashes")
    a.altair_chart(alt.Chart(byc).mark_bar(color=ACCENT, cornerRadiusEnd=4).encode(
        x=alt.X("crashes:Q", title="Crash records"), y=alt.Y("city:N", sort="-x", title=None),
        tooltip=["city", "crashes"]).properties(height=280, title="Crashes by place"), use_container_width=True)
    bym = c.groupby(["city", pd.Grouper(key="crash_date", freq="MS")]).size().reset_index(name="crashes")
    b.altair_chart(alt.Chart(bym).mark_line(strokeWidth=2).encode(
        x=alt.X("crash_date:T", title=None), y=alt.Y("crashes:Q", title="Crashes per month"),
        color=alt.Color("city:N", legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["city", alt.Tooltip("crash_date:T", format="%b %Y"), "crashes"]
    ).properties(height=280, title="Crashes per month"), use_container_width=True)
    sev = c.assign(outcome=np.where(c.killed > 0, "Fatal", np.where(c.injured > 0, "Injury", "Damage only")))
    sv = sev.groupby(["city", "outcome"]).size().reset_index(name="n")
    st.altair_chart(alt.Chart(sv).mark_bar().encode(
        x=alt.X("n:Q", stack="normalize", axis=alt.Axis(format="%"), title="Share of crashes"),
        y=alt.Y("city:N", title=None),
        color=alt.Color("outcome:N", scale=alt.Scale(domain=["Damage only", "Injury", "Fatal"],
                                                     range=["#CED4DA", "#F59F00", "#C92A2A"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["city", "outcome", "n"]).properties(height=220, title="How bad were the crashes?"),
        use_container_width=True)
    st.caption("Great Britain records only injury collisions, so it has no 'damage only' share.")
