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
    "customers": [
        "Commodity traders & hedge funds - early signal before official output reports",
        "Steel / fuel buyers - supplier risk alerts to re-plan sourcing",
        "Insurers & banks - monitor plants they insure or lend to",
        "Governments - industrial activity where official data is slow or missing",
    ],
    "pricing": [
        ("Starter", "US$500 per site / month", "a buyer watching its key suppliers"),
        ("Portfolio", "US$15,000 / month for 50 sites", "a trading desk or insurer"),
        ("Enterprise / API", "US$250,000 / year, global coverage", "hedge funds, governments"),
    ],
    "cost": "Near-zero data cost: Landsat is free, and processing one site costs cents of cloud compute.",
    "market": ("Global Energy Monitor tracks 550+ steel plants of 1 Mt/yr or more, plus refineries and power "
               "plants: 1,000+ large sites to monitor. Hedge funds alone spent about US$2.8 billion on "
               "alternative data in 2025 (Neudata)."),
    "goal": "Target: 50 customers x ~US$100k/yr = ~US$5M annual revenue.",
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
# EDIT HERE 4: VALUE-AT-RISK DEFAULTS (real capacities and recent market prices)
# =====================================================================
CALC = {
    "jsw": {"unit": "tonnes", "per_year": 17_500_000, "price": 650, "price_unit": "tonne",
            "note": "Capacity 17.5 MTPA (JSW). Price ~US$650/t: Indian domestic HRC, Mumbai, Oct 2026 (SMM)."},
    "jamnagar": {"unit": "barrels", "per_year": 1_400_000 * 365, "price": 80, "price_unit": "barrel",
                 "note": "Capacity ~1.4 million barrels/day (Reliance complex). Price: Brent crude per barrel - "
                         "check the live price before presenting."},
    "gary": {"unit": "short tons", "per_year": 7_500_000, "price": 1350, "price_unit": "short ton",
             "note": "Capacity 7.5 million net tons/yr (U.S. Steel). Price ~US$1,350/short ton: US HRC spot, "
                     "mid-Nov 2026 (Kallanish via GMK Center)."},
    "duisburg": {"unit": "tonnes", "per_year": 11_000_000, "price": 900, "price_unit": "tonne",
                 "note": "~11 million t crude steel/yr (thyssenkrupp Steel). Price ~US$900/t: EU HRC offers "
                         "for Nov 2026 deliveries (BigMint via GMK Center)."},
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

# =====================================================================
# LIVE CHECK: type any latitude / longitude
# =====================================================================
st.subheader("🔎 Check any plant on Earth (live)")
st.caption("Type the coordinates of any steel plant, refinery, cement or power plant. The app pulls real "
           "Landsat 8/9 thermal images for that point right now and runs the same AI on them.")


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def run_live(lat, lon, months):
    import live_check
    return live_check.scan_plant(lat, lon, months)


with st.form("live_check"):
    f1, f2, f3, f4 = st.columns([2, 2, 2, 3])
    live_lat = f1.number_input("Latitude", -80.0, 80.0, 15.1890, step=0.0001, format="%.4f")
    live_lon = f2.number_input("Longitude", -180.0, 180.0, 76.6562, step=0.0001, format="%.4f")
    live_months = f3.selectbox("Period", [6, 12], index=1, format_func=lambda m: f"Last {m} months")
    live_name = f4.text_input("Plant name (optional)", "")
    g1, g2, g3 = st.columns([2, 2, 3])
    live_cap = g1.number_input("Capacity (units per year, optional)", 0.0, value=0.0, step=100000.0,
                               help="e.g. tonnes of steel per year, or barrels per year")
    live_price = g2.number_input("Price per unit in US$ (optional)", 0.0, value=0.0, step=10.0)
    go_live = f4.form_submit_button("🛰️ Analyse this plant", type="primary")
    g3.caption("Tip: right-click the plant in Google Maps to copy its coordinates. "
               "The app searches 2 km around your point for the hottest spot. Takes about 20-60 s.")

if go_live:
    st.session_state["live_req"] = (round(live_lat, 4), round(live_lon, 4), live_months,
                                    live_name, live_cap, live_price)

if "live_req" in st.session_state:
    lat_q, lon_q, months_q, name_q, cap_q, price_q = st.session_state["live_req"]
    label = name_q or f"Plant at {lat_q:.4f}, {lon_q:.4f}"
    try:
        with st.spinner(f"Fetching real Landsat thermal images for {label}... (20-60 s)"):
            out = run_live(lat_q, lon_q, months_q)
    except ImportError:
        out = {"error": "The live check needs the satellite libraries (rasterio, pystac-client, "
                        "planetary-computer). They are installed on the online version of this site."}
    except Exception as e:  # noqa: BLE001
        out = {"error": f"Satellite service did not answer ({type(e).__name__}). Please try again."}

    if "error" in out:
        st.warning(out["error"])
    else:
        ld = out["data"]
        last = ld.iloc[-1]
        lvl = str(last["risk_level"])
        median_heat = float(ld["delta_t"].median())
        head = (f"**{label}** - latest clear pass {last['date']:%d %b %Y} - "
                f"Activity Index **{int(last['activity_index'])}/100**  \n_{last['explanation']}_")
        if median_heat < 2.0:
            st.info(f"ℹ️ Weak heat signal ({median_heat:.1f} °C above surroundings on a typical pass). "
                    "This point may not be a hot industrial site; check the coordinates.  \n" + head)
        elif "alert" in lvl.lower():
            st.error("🚨 ALERT: possible shutdown or major slowdown. " + head)
        elif "watch" in lvl.lower():
            st.warning("🟡 WATCH: activity below normal. " + head)
        else:
            st.success("🟢 NORMAL operation. " + head)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Activity Index", f"{int(last['activity_index'])}/100")
        m2.metric("Heat above surroundings", f"{last['delta_t']:.1f} °C",
                  f"{last['delta_t'] - last['expected_delta_t']:+.1f} vs usual")
        m3.metric("Typical heat signal", f"{median_heat:.1f} °C")
        m4.metric("Clear passes analysed", f"{len(ld)} of {out['passes_found']}")

        l1, l2 = st.columns([1, 2])
        with l1:
            st.map(pd.DataFrame({"lat": [lat_q, out["hot_lat"]], "lon": [lon_q, out["hot_lon"]],
                                 "color": ["#9e9e9e", "#e02424"]}),
                   latitude="lat", longitude="lon", color="color", size=120, zoom=12, height=300)
            st.caption(f"Grey = your point · Red = hot spot found by the AI "
                       f"({out['moved_m']:.0f} m away, {out['hot_lat']:.4f}, {out['hot_lon']:.4f})")
        with l2:
            lf = go.Figure()
            lf.add_trace(go.Scatter(x=ld["date"], y=ld["delta_t"], mode="lines+markers",
                                    name="Plant minus surroundings", line=dict(width=3, color="#e4572e"),
                                    text=ld["risk_level"],
                                    hovertemplate="%{x|%d %b %Y}<br>%{y:.1f} °C<br>%{text}<extra></extra>"))
            lf.add_trace(go.Scatter(x=ld["date"], y=ld["expected_delta_t"], mode="lines",
                                    name="AI's expected heat", line=dict(dash="dash", color="#1f77b4")))
            flag = ld[ld["risk_level"].str.lower().str.contains("alert|watch")]
            if len(flag):
                lf.add_trace(go.Scatter(x=flag["date"], y=flag["delta_t"], mode="markers", name="Flagged",
                                        marker=dict(size=14, color="#ff9f1c", symbol="x")))
            lf.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                             yaxis_title="°C above surroundings", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(lf, width="stretch")

        frames = out["frames"]
        hi_d = ld.loc[(ld["delta_t"] - ld["delta_t"].quantile(0.75)).abs().idxmin(), "date"]
        lo_d = ld.loc[ld["delta_t"].idxmin(), "date"]
        imgs = [(k, frames.get(f"{k:%Y-%m-%d}")) for k in (hi_d, lo_d)]
        imgs = [(k, v) for k, v in imgs if v is not None]
        if len(imgs) == 2:
            vmin = float(min(np.nanpercentile(v, 2) for _, v in imgs))
            vmax = float(max(np.nanpercentile(v, 99.5) for _, v in imgs))
            h1, h2 = st.columns(2)
            for col, (k, v), title in zip((h1, h2), imgs, ("Typical busy pass", "Coolest pass")):
                fig_h = px.imshow(v, color_continuous_scale="inferno", zmin=vmin, zmax=vmax,
                                  labels=dict(color="°C"))
                fig_h.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0),
                                    title=f"{title}: {k:%d %b %Y}")
                fig_h.update_xaxes(visible=False)
                fig_h.update_yaxes(visible=False)
                col.plotly_chart(fig_h, width="stretch")
            st.caption("Real Landsat surface temperature, 3 km x 3 km around the hot spot. Cloudy pixels are blank.")

        if cap_q > 0 and price_q > 0:
            per_day = cap_q * price_q / 365
            drop = max(0.0, 1 - float(last["heat_ratio"])) if "watch" in lvl.lower() or "alert" in lvl.lower() else 0.0
            v1, v2 = st.columns(2)
            v1.metric("Output value per day", f"${per_day / 1e6:,.1f}M")
            v2.metric("Estimated value at risk per day (latest pass)", f"${per_day * min(drop, 1) / 1e6:,.1f}M",
                      help="Value per day x how far heat is below the AI's expected level. 0 if normal.")

        with st.expander("See every pass (table) and download"):
            show = ld[["date", "plant_temp", "ambient_temp", "delta_t", "expected_delta_t",
                       "activity_index", "risk_level", "explanation"]].copy()
            show["date"] = show["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(show, width="stretch", hide_index=True)
            st.download_button("Download CSV", show.to_csv(index=False).encode(),
                               file_name=f"live_{lat_q:.4f}_{lon_q:.4f}.csv", mime="text/csv")
        st.caption("Honest limits: Landsat passes about every 8 days and clouds hide some passes. "
                   "The first 3 passes are used to learn the plant's normal heat, so they are never flagged.")

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
e1, e2 = st.columns([1, 1])
with e1:
    with st.container(border=True):
        st.markdown("**Who pays**")
        for c in ECONOMICS["customers"]:
            st.markdown(f"- {c}")
with e2:
    with st.container(border=True):
        st.markdown("**Proposed pricing**")
        for tier, price_txt, who in ECONOMICS["pricing"]:
            st.markdown(f"- **{tier}:** {price_txt} - _{who}_")
        st.markdown(f"**Costs:** {ECONOMICS['cost']}")
with st.container(border=True):
    st.markdown(f"**Market:** {ECONOMICS['market']}  \n**{ECONOMICS['goal']}**")

# ---------- VALUE AT RISK CALCULATOR ----------
st.subheader("🧮 Value-at-risk calculator")
cd = CALC.get(site_id, {"unit": "tonnes", "per_year": 10_000_000, "price": 600, "price_unit": "tonne", "note": ""})
st.caption(f"How much output is at stake when this plant slows down. Pre-filled with real figures for "
           f"{site['name']}; change any number to test scenarios.")
v1, v2, v3, v4 = st.columns(4)
per_day = v1.number_input(f"Plant output ({cd['unit']}/day)", min_value=0,
                          value=int(round(cd["per_year"] / 365, -2)), step=1000, key=f"cap_{site_id}")
price = v2.number_input(f"Price per {cd['price_unit']} (US$)", min_value=0, value=int(cd["price"]),
                        step=10, key=f"price_{site_id}")
share = v3.number_input("Share of plant affected (%)", min_value=0, max_value=100, value=20, step=5,
                        key=f"share_{site_id}", help="e.g. one blast furnace out of several")
days = v4.number_input("Days of disruption", min_value=0, value=8, step=1, key=f"days_{site_id}",
                       help="8 days = time between two Landsat passes")
daily_value = per_day * price
at_risk = daily_value * share / 100 * days
r1, r2, r3 = st.columns(3)
r1.metric("Plant output value per day", f"${daily_value / 1e6:,.1f}M")
r2.metric("Output at risk in this scenario", f"${at_risk / 1e6:,.1f}M")
r3.metric("If an alert gives buyers 1 week earlier warning",
          f"${daily_value * share / 100 * 7 / 1e6:,.1f}M", help="value of the disrupted supply buyers can re-plan for")
st.caption("Sources: " + cd["note"] + " Scenario values are estimates, not forecasts.")

with st.expander("📄 See the data"):
    st.dataframe(data, width="stretch")

st.caption("Data: Landsat 8/9 Collection 2 surface temperature (Band 10) via Microsoft Planetary Computer. "
           "AI: hybrid physics rule + Isolation Forest, validated with simulated shutdowns.")
