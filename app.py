"""
Orbital Industrial Pulse - Dashboard (Person 3)

Run:
    py -m streamlit run app.py
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Orbital Industrial Pulse", page_icon="🛰️", layout="wide")

# =====================================================================
# EDIT HERE 1: SITES  (keys of "file_name" must match site_name in the data)
# =====================================================================
SITES = {
    "jsw": {"name": "JSW Vijayanagar Steel Plant", "country": "India", "type": "Steel plant",
            "lat": 15.1890, "lon": 76.6562, "file_name": "JSW_Vijayanagar_Steel_India"},
    "jamnagar": {"name": "Jamnagar Refinery", "country": "India", "type": "Oil refinery",
                 "lat": 22.3395, "lon": 69.8550, "file_name": "Jamnagar_Refinery_India"},
    "gary": {"name": "Gary Works Steel Plant", "country": "USA", "type": "Steel plant",
             "lat": 41.6185, "lon": -87.3307, "file_name": "Gary_Works_Steel_USA"},
    "duisburg": {"name": "Thyssenkrupp Duisburg Steelworks", "country": "Germany", "type": "Steel plant",
                 "lat": 51.4823, "lon": 6.7199, "file_name": "Thyssenkrupp_Duisburg_Germany"},
}
MAIN_SITE = list(SITES.keys())[0]

SAT_FILE = "satellite_data.csv"   # from Person 1 (all sites in one file)
AI_FILE = "ai_alerts.csv"         # from Person 2 (optional; replaces the preliminary scoring)
MAX_CLOUD = 15                    # passes cloudier than this (%) are removed

# =====================================================================
# EDIT HERE 2: ECONOMICS  (EXAMPLE numbers - replace with Person 4's)
# =====================================================================
ECONOMICS = {
    "customers": ["Commodity traders", "Steel & power buyers", "Insurers & banks", "Governments"],
    "price": "$2,000 per site per month (example)",
    "value": "One early shutdown alert can save a buyer weeks of supply delay (example)",
    "market": "Thousands of major plants worldwide (example)",
}


# =====================================================================
# EDIT HERE 3: REAL-WORLD EVENTS (publicly reported, used to sanity-check the satellite signal)
# =====================================================================
EVENTS = {
    "jsw": {
        "title": "Blast Furnace 3 offline for upgrade",
        "start": "2025-09-25", "end": "2026-06-20",
        "story": ("JSW Steel reported that Blast Furnace 3 at Vijayanagar was shut from late September 2025 "
                  "for a capacity upgrade (3 to 4.5 MTPA) and restarted in June 2026."),
        "sources": ["https://themachinemaker.com/news/jsw-steel-posts-24-75-lakh-tonnes-of-consolidated-crude-steel-output-in-january-2026/",
                    "https://gmk.center/en/news/jsw-steel-increased-steel-output-by-3-y-y-in-april-june/"],
    },
    "jamnagar": {
        "title": "Reported crude-unit maintenance (approx. dates)",
        "start": "2026-05-15", "end": "2026-06-12",
        "story": ("Reliance reported a planned 3-4 week maintenance shutdown of a crude unit at the Jamnagar "
                  "complex starting around mid-May 2026; India's fuel exports fell in May."),
        "sources": ["https://hydrocarbonprocessing.com/news/2026/05/indias-reliance-to-shut-some-units-at-660-000-bpd-refinery/",
                    "https://www.outlookbusiness.com/news/indias-fuel-exports-fall-to-lowest-since-2022-amid-refinery-maintenance-domestic-demand"],
    },
}


# =====================================================================
# DATA LOADING
# =====================================================================
def preliminary_scores(d):
    """Simple statistical scoring, used until Person 2's AI output arrives."""
    base = d["thermal_delta"].median()
    spread = max((d["thermal_delta"] - base).abs().median() * 1.4826, 1.0)
    z = (d["thermal_delta"] - base) / spread
    d["activity_index"] = (100 * d["thermal_delta"] / base).clip(0, 100).round(0).astype(int) if base > 0 else 50
    d["risk_level"] = np.where(z <= -2.5, "Alert", np.where(z <= -1.5, "Watch", "Normal"))
    d["anomaly_flag"] = (d["risk_level"] == "Alert").astype(int)
    d["supply_chain_alert"] = [
        f"Heat {dt - base:+.1f} °C vs normal: possible shutdown or slowdown" if r == "Alert"
        else f"Heat {dt - base:+.1f} °C vs normal: monitor" if r == "Watch"
        else "Normal operating heat signature"
        for dt, r in zip(d["thermal_delta"], d["risk_level"])]
    return d


@st.cache_data
def load_site(site_id, file_mtime):
    if not os.path.exists(SAT_FILE):
        return None, {}
    raw = pd.read_csv(SAT_FILE, parse_dates=["date"])
    d = raw[raw["site_name"] == SITES[site_id]["file_name"]].copy()
    if d.empty:
        return None, {}
    d = d.rename(columns={"plant_temp_c": "plant_temp", "ambient_temp_c": "ambient_temp",
                          "relative_temp_c": "thermal_delta"})
    if "thermal_delta" not in d.columns:
        d["thermal_delta"] = d["plant_temp"] - d["ambient_temp"]
    info = {"cloudy_removed": 0, "ai_source": "preliminary"}
    if "cloud_cover_pct" in d.columns:
        before = len(d)
        d = d[d["cloud_cover_pct"] <= MAX_CLOUD]
        info["cloudy_removed"] = before - len(d)
    d = d.sort_values("date").reset_index(drop=True)

    if os.path.exists(AI_FILE):
        ai = pd.read_csv(AI_FILE, parse_dates=["date"])
        if "site_name" in ai.columns:
            ai = ai[ai["site_name"] == SITES[site_id]["file_name"]].drop(columns=["site_name"])
        keep = [c for c in ["date", "activity_index", "anomaly_flag", "risk_level", "supply_chain_alert"]
                if c in ai.columns]
        if len(ai) and {"activity_index", "risk_level"} <= set(keep):
            d = d.merge(ai[keep], on="date", how="inner")
            if "anomaly_flag" not in d.columns:
                d["anomaly_flag"] = d["risk_level"].astype(str).str.lower().str.contains("alert").astype(int)
            info["ai_source"] = "model"
            return d, info
    return preliminary_scores(d), info


def mtime(site_id):
    t = 0
    for f in (SAT_FILE, AI_FILE):
        if os.path.exists(f):
            t += os.path.getmtime(f)
    return t


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.header("🛰️ Choose a site")
    site_id = st.selectbox("Industrial site", list(SITES.keys()),
                           format_func=lambda k: f"{SITES[k]['country']} - {SITES[k]['name']}")
    site = SITES[site_id]
    st.write(f"**Type:** {site['type']}")
    st.write(f"**Location:** {site['lat']:.2f}, {site['lon']:.2f}")
    st.divider()
    st.caption("Works for any industrial site on Earth: satellites cover the whole planet.")

data, info = load_site(site_id, mtime(site_id))

# =====================================================================
# HEADER + HOW IT WORKS
# =====================================================================
st.title("🛰️ Orbital Industrial Pulse")
st.markdown("##### Tracking industrial activity anywhere on Earth from satellite heat + AI")

steps = st.columns(4)
for col, icon, title, text in [
    (steps[0], "🛰️", "1. Satellite", "Landsat measures heat from space"),
    (steps[1], "🔥", "2. Heat signal", "Plant heat minus surroundings"),
    (steps[2], "🤖", "3. AI", "Physics rule + Isolation Forest must agree"),
    (steps[3], "🚨", "4. Alert", "Customers warned of supply risk"),
]:
    with col:
        with st.container(border=True):
            st.markdown(f"### {icon}")
            st.markdown(f"**{title}**")
            st.caption(text)

st.divider()

# =====================================================================
# WORLD OVERVIEW (all sites, coloured by latest status)
# =====================================================================
STATUS_COLOR = {"alert": "#e02424", "watch": "#ff9f1c", "normal": "#1fa34a", "none": "#9e9e9e"}


def site_status(sid):
    d, _ = load_site(sid, mtime(sid))
    if d is None or d.empty:
        return "none", None
    lv = str(d.iloc[-1]["risk_level"]).lower()
    key = "alert" if "alert" in lv else "watch" if "watch" in lv else "normal"
    return key, int(d.iloc[-1]["activity_index"])


rows = []
for sid, s in SITES.items():
    key, act = site_status(sid)
    rows.append({"lat": s["lat"], "lon": s["lon"], "color": STATUS_COLOR[key],
                 "Site": s["name"], "Country": s["country"],
                 "Status": {"alert": "🔴 Alert", "watch": "🟠 Watch", "normal": "🟢 Normal",
                            "none": "⚪ No data yet"}[key],
                 "Activity": "-" if act is None else f"{act}/100"})
overview = pd.DataFrame(rows)

st.subheader("🌍 Global overview")
g1, g2 = st.columns([3, 2])
with g1:
    st.map(overview, latitude="lat", longitude="lon", color="color", size=80000, zoom=0, height=320)
with g2:
    for _, r in overview.iterrows():
        st.markdown(f"{r['Status'].split()[0]} **{r['Site']}**  \n{r['Country']} · Activity {r['Activity']}")
    st.caption("Pick a site in the left sidebar (») to see its details.")

st.divider()

if data is None or data.empty:
    st.warning(f"No data yet for **{site['name']}**. Waiting for files from the data team.")
    st.stop()

latest = data.iloc[-1]
prev = data.iloc[-2] if len(data) > 1 else latest

# =====================================================================
# ALERT BOX + KEY NUMBERS
# =====================================================================
msg = (f"**{site['name']}** ({site['country']}) - latest pass {latest['date']:%d %b %Y} - "
       f"Activity Index **{int(latest['activity_index'])}/100**")
note = f"  \n_{latest['supply_chain_alert']}_" if "supply_chain_alert" in data.columns else ""
level = str(latest["risk_level"]).lower()
if "alert" in level:
    st.error(f"🚨 SUPPLY-CHAIN ALERT: possible shutdown. {msg}{note}")
elif "watch" in level:
    st.warning(f"🟡 WATCH: activity below normal. {msg}{note}")
else:
    st.success(f"🟢 NORMAL operation. {msg}{note}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Activity Index", f"{int(latest['activity_index'])}/100",
          f"{int(latest['activity_index'] - prev['activity_index'])}")
c2.metric("Plant temperature", f"{latest['plant_temp']:.1f} °C")
c3.metric("Heat above surroundings", f"{latest['thermal_delta']:.1f} °C")
c4.metric("Shutdown alerts (period)", int(data["anomaly_flag"].sum()))
src = ("hybrid AI model (physics rule + Isolation Forest)" if info.get("ai_source") == "model"
       else "preliminary statistical baseline (AI model results coming)")
st.caption(f"Real Landsat 8/9 thermal data · {len(data)} clear passes · "
           f"{info.get('cloudy_removed', 0)} cloudy passes removed (> {MAX_CLOUD}% cloud) · Scoring: {src}")

# =====================================================================
# MAP + ACTIVITY GRAPH
# =====================================================================
left, right = st.columns([1, 2])
with left:
    st.subheader("📍 Site location")
    st.map(pd.DataFrame({"lat": [site["lat"]], "lon": [site["lon"]]}), zoom=8)

with right:
    st.subheader("📈 Activity Index over time")
    hover = data["supply_chain_alert"] if "supply_chain_alert" in data.columns else data["risk_level"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["date"], y=data["activity_index"], mode="lines+markers",
                             name="Activity Index", line=dict(width=3, color="#1f77b4"),
                             text=hover, hovertemplate="%{x|%d %b %Y}<br>Activity %{y}<br>%{text}<extra></extra>"))
    lv = data["risk_level"].astype(str).str.lower()
    for mask, label, color in [(lv.str.contains("alert"), "Shutdown alert", "red"),
                               (lv.str.contains("watch"), "Watch", "orange")]:
        pts = data[mask]
        if not pts.empty:
            fig.add_trace(go.Scatter(x=pts["date"], y=pts["activity_index"], mode="markers", name=label,
                                     marker=dict(color=color, size=14, symbol="x"), hoverinfo="skip"))
    fig.add_hrect(y0=0, y1=50, fillcolor="red", opacity=0.08, line_width=0)
    fig.update_layout(yaxis_title="Activity (0-100)", yaxis_range=[0, 105], height=360,
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

# =====================================================================
# ALERT LOG
# =====================================================================
normal_delta = data.loc[data["risk_level"].astype(str).str.lower() == "normal", "thermal_delta"].median()
if pd.isna(normal_delta):
    normal_delta = data["thermal_delta"].median()
data["heat_vs_normal"] = (data["thermal_delta"] - normal_delta).round(1)

events = data[data["risk_level"].astype(str).str.lower() != "normal"]
if not events.empty:
    st.subheader("🚨 Alert log")
    cols = ["date", "risk_level", "activity_index", "heat_vs_normal"]
    if "supply_chain_alert" in data.columns:
        cols.append("supply_chain_alert")
    log = events[cols].copy()
    log["date"] = log["date"].dt.strftime("%d %b %Y")
    log = log.rename(columns={"date": "Date", "risk_level": "Risk level", "activity_index": "Activity",
                              "heat_vs_normal": "Heat vs normal (°C)",
                              "supply_chain_alert": "What it means"})
    st.dataframe(log, width="stretch", hide_index=True)
    st.download_button("📥 Download alert report (CSV)", log.to_csv(index=False).encode("utf-8"),
                       file_name=f"{site_id}_alert_report.csv", mime="text/csv")

    # ---------- WHY WAS THIS FLAGGED ----------
    st.subheader("🔍 Why did the AI flag these dates?")
    st.caption(f"Normally this plant runs about {normal_delta:.1f} °C hotter than its surroundings. "
               "Bars show how far below that normal level the heat fell on each date.")
    wf = go.Figure(go.Bar(
        x=data["date"], y=data["heat_vs_normal"],
        marker_color=["#e02424" if "alert" in str(r).lower() else "#ff9f1c" if "watch" in str(r).lower()
                      else "#9ecae1" for r in data["risk_level"]],
        hovertemplate="%{x|%d %b %Y}<br>%{y:+.1f} °C vs normal<extra></extra>"))
    wf.add_hline(y=0, line_color="#555", line_width=1)
    wf.update_layout(yaxis_title="Heat vs normal (°C)", height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(wf, width="stretch")

# =====================================================================
# REAL-WORLD CHECK
# =====================================================================
ev = EVENTS.get(site_id)
if ev:
    st.subheader("📰 Real-world check")
    st.markdown(f"**{ev['title']}.** {ev['story']}")
    s0, s1 = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"])
    inside = data[(data["date"] >= s0) & (data["date"] <= s1)]["thermal_delta"]
    outside = data[(data["date"] < s0) | (data["date"] > s1)]["thermal_delta"]
    rf = go.Figure()
    rf.add_vrect(x0=max(s0, data["date"].min()), x1=min(s1, data["date"].max()),
                 fillcolor="orange", opacity=0.15, line_width=0,
                 annotation_text="reported event window", annotation_position="top left")
    rf.add_trace(go.Scatter(x=data["date"], y=data["thermal_delta"], mode="lines+markers",
                            name="Heat above surroundings", line=dict(color="#d62728", width=3)))
    rf.update_layout(yaxis_title="Heat above surroundings (°C)", height=300,
                     margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    st.plotly_chart(rf, width="stretch")
    if len(inside) and len(outside):
        k1, k2, k3 = st.columns(3)
        k1.metric("During reported event (median)", f"{inside.median():.1f} °C", help=f"{len(inside)} clear passes")
        k2.metric("Outside event (median)", f"{outside.median():.1f} °C", help=f"{len(outside)} clear passes")
        k3.metric("Difference", f"{100 * (inside.median() / outside.median() - 1):+.0f}%")
    st.caption("Honest note: the satellite signal is consistent with these reports, but a small number of passes "
               "and a 300 m measuring spot cannot prove which unit changed. Sources: "
               + " · ".join(f"[{i + 1}]({u})" for i, u in enumerate(ev["sources"])))

# =====================================================================
# MODEL VALIDATION
# =====================================================================
if info.get("ai_source") == "model" and os.path.exists("model_report.json"):
    import json
    with open("model_report.json") as f:
        rep = json.load(f)
    v = rep.get("validation", {})
    st.subheader("🧪 How reliable is the AI?")
    st.caption("We took the real satellite data and inserted simulated shutdowns (heat cut to 0-30 % of normal) "
               "and slowdowns (40-70 %), then measured how many the model catches. "
               "A full alert needs BOTH the physics rule and the Isolation Forest to agree, which keeps false alarms low.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Shutdowns caught (all plants)", f"{v.get('shutdown_detection_pct', 0):.0f}%")
    m2.metric(f"Shutdowns caught ({site['name'].split()[0]})",
              f"{v.get('shutdown_detection_by_plant_pct', {}).get(site['file_name'], 0):.0f}%")
    m3.metric("Slowdowns flagged", f"{v.get('slowdown_detection_pct', 0):.0f}%")
    m4.metric("Alert rate on real data", f"{v.get('alert_rate_on_real_data_pct', 0):.1f}%")
    st.caption("Detection is strongest at plants with a strong heat signature (e.g. Jamnagar) and weaker where "
               "the plant is only slightly warmer than its surroundings - better targeting of hot zones improves this.")

# =====================================================================
# TEMPERATURE GRAPH (thermal delta idea)
# =====================================================================
st.subheader("🌡️ Plant heat vs. surroundings")
st.caption("Comparing with the surroundings removes summer/winter weather, so only real plant activity shows.")
tfig = go.Figure()
tfig.add_trace(go.Scatter(x=data["date"], y=data["plant_temp"], name="Plant", mode="lines+markers",
                          line=dict(color="#d62728", width=3)))
tfig.add_trace(go.Scatter(x=data["date"], y=data["ambient_temp"], name="Surroundings (10 km)",
                          mode="lines+markers", line=dict(color="#7f7f7f", width=2, dash="dash")))
tfig.update_layout(yaxis_title="Temperature (°C)", height=320, margin=dict(l=10, r=10, t=10, b=10),
                   legend=dict(orientation="h", y=1.12))
st.plotly_chart(tfig, width="stretch")

# =====================================================================
# HEAT PICTURES
# =====================================================================
st.subheader("🔥 Satellite heat view: normal vs. shutdown")
h1, h2 = st.columns(2)
imgs = [(h1, "Normal operation", f"{site_id}_heat_normal.png"),
        (h2, "Shutdown / low activity", f"{site_id}_heat_shutdown.png")]
if not any(os.path.exists(i[2]) for i in imgs):
    st.info("Thermal images for this site will appear here once the data team adds "
            f"`{site_id}_heat_normal.png` and `{site_id}_heat_shutdown.png`.")
else:
    for col, title, img in imgs:
        with col:
            st.markdown(f"**{title}**")
            if os.path.exists(img):
                st.image(img, width="stretch")

# =====================================================================
# ECONOMIC IMPACT
# =====================================================================
st.divider()
st.subheader("💰 Who pays for this, and how much is it worth?")
e1, e2 = st.columns([1, 2])
with e1:
    with st.container(border=True):
        st.markdown("**Customers**")
        for c in ECONOMICS["customers"]:
            st.markdown(f"- {c}")
with e2:
    with st.container(border=True):
        st.markdown(f"**Pricing:** {ECONOMICS['price']}")
        st.markdown(f"**Value:** {ECONOMICS['value']}")
        st.markdown(f"**Market:** {ECONOMICS['market']}")

# ---------- VALUE AT RISK CALCULATOR ----------
st.subheader("🧮 Value-at-risk calculator")
st.caption("Estimate how much production was at risk during the alerts. Change the numbers to fit any plant.")
v1, v2, v3 = st.columns(3)
capacity = v1.number_input("Plant output (tonnes per day)", min_value=0, value=30000, step=1000)
price = v2.number_input("Price per tonne (US$)", min_value=0, value=500, step=50)
days = v3.number_input("Days of disruption per alert", min_value=0, value=3, step=1)
n_alerts = int(data["risk_level"].astype(str).str.lower().str.contains("alert").sum())
per_alert = capacity * price * days
r1, r2 = st.columns(2)
r1.metric("Output at risk per alert", f"${per_alert:,.0f}")
r2.metric(f"Total across {n_alerts} alert(s)", f"${per_alert * n_alerts:,.0f}")
st.caption("Example inputs only: replace with real plant capacity and market prices.")

with st.expander("📄 See the data"):
    st.dataframe(data, width="stretch")

st.caption("Data: Landsat 8/9 Collection 2 surface temperature (Band 10) via Microsoft Planetary Computer. "
           "AI: hybrid physics rule + Isolation Forest, validated with simulated shutdowns.")
