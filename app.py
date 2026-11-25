"""
Orbital Industrial Pulse - Dashboard (Person 3)

Run:
    py make_fake_data.py          (only until real data arrives)
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
# EDIT HERE 1: SITES  (add / change plants; first one = main site)
# =====================================================================
SITES = {
    "jsw": {"name": "JSW Vijayanagar Steel Plant", "country": "India",
            "type": "Steel plant", "lat": 15.18, "lon": 76.66},
    "scherer": {"name": "Plant Scherer Power Station", "country": "USA",
                "type": "Coal power plant", "lat": 33.06, "lon": -83.81},
    "pohang": {"name": "POSCO Pohang Steelworks", "country": "South Korea",
               "type": "Steel plant", "lat": 36.00, "lon": 129.38},
}
MAIN_SITE = list(SITES.keys())[0]

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
# DATA LOADING
# =====================================================================
def files_for(site_id):
    if os.path.exists(f"{site_id}_temperatures.csv"):
        return f"{site_id}_temperatures.csv", f"{site_id}_alerts.csv"
    if site_id == MAIN_SITE:
        return "temperatures.csv", "alerts.csv"
    return None, None


@st.cache_data
def load_site(site_id, file_mtime):
    t_file, a_file = files_for(site_id)
    if not t_file or not (os.path.exists(t_file) and os.path.exists(a_file)):
        return None
    temps = pd.read_csv(t_file, parse_dates=["date"])
    alerts = pd.read_csv(a_file, parse_dates=["date"])
    data = temps.merge(alerts, on="date", how="inner").sort_values("date")
    data["thermal_delta"] = data["plant_temp"] - data["ambient_temp"]
    return data


def mtime(site_id):
    t, a = files_for(site_id)
    try:
        return os.path.getmtime(t) + os.path.getmtime(a)
    except (TypeError, OSError):
        return 0


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

data = load_site(site_id, mtime(site_id))

# =====================================================================
# HEADER + HOW IT WORKS
# =====================================================================
st.title("🛰️ Orbital Industrial Pulse")
st.markdown("##### Tracking industrial activity anywhere on Earth from satellite heat + AI")

steps = st.columns(4)
for col, icon, title, text in [
    (steps[0], "🛰️", "1. Satellite", "Landsat measures heat from space"),
    (steps[1], "🔥", "2. Heat signal", "Plant heat minus surroundings"),
    (steps[2], "🤖", "3. AI", "Isolation Forest spots unusual drops"),
    (steps[3], "🚨", "4. Alert", "Customers warned of supply risk"),
]:
    with col:
        with st.container(border=True):
            st.markdown(f"### {icon}")
            st.markdown(f"**{title}**")
            st.caption(text)

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
events = data[data["risk_level"].astype(str).str.lower() != "normal"]
if not events.empty:
    st.subheader("🚨 Alert log")
    cols = ["date", "risk_level", "activity_index"]
    if "supply_chain_alert" in data.columns:
        cols.append("supply_chain_alert")
    log = events[cols].copy()
    log["date"] = log["date"].dt.strftime("%d %b %Y")
    log.columns = [c.replace("_", " ").title() for c in cols]
    st.dataframe(log, width="stretch", hide_index=True)

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


def fake_heatmap(intensity, seed):
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.linspace(-1, 1, 60), np.linspace(-1, 1, 60))
    return 28 + rng.normal(0, 0.6, x.shape) + intensity * np.exp(-(x**2 + y**2) * 8)


h1, h2 = st.columns(2)
for col, title, img, intensity, seed in [
    (h1, "Normal operation", f"{site_id}_heat_normal.png", 18, 1),
    (h2, "Shutdown detected", f"{site_id}_heat_shutdown.png", 4, 2),
]:
    if site_id == MAIN_SITE and not os.path.exists(img):
        img = img.replace(f"{site_id}_", "")
    with col:
        st.markdown(f"**{title}**")
        if os.path.exists(img):
            st.image(img, width="stretch")
        else:
            hm = px.imshow(fake_heatmap(intensity, seed), color_continuous_scale="inferno",
                           zmin=25, zmax=48, labels=dict(color="°C"))
            hm.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
            hm.update_xaxes(showticklabels=False)
            hm.update_yaxes(showticklabels=False)
            st.plotly_chart(hm, width="stretch")

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

with st.expander("📄 See the data"):
    st.dataframe(data, width="stretch")

st.caption("Data: Landsat 8/9 thermal (Band 10). AI: Isolation Forest anomaly detection.")
