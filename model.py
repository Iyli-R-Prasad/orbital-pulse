"""
Orbital Industrial Pulse - AI model (v2)

Reads Person 1's satellite file and writes:
  ai_alerts.csv      -> used by the website (app.py)
  model_report.json  -> validation results shown on the website

How it works (per plant):
  1. Remove cloudy passes (> 15 % cloud).
  2. Heat signal  = plant temp - surrounding temp (removes weather/seasons).
  3. Expected heat = median of the plant's own last 6 clear passes
     (only PAST data is used, like a real live system).
  4. Two independent checks:
       a) Physics rule  : how far below its own normal is the plant? (robust z-score)
       b) Isolation Forest: is this pass statistically unusual compared with
          normal behaviour learned across all plants?
     A full ALERT needs BOTH checks to agree. Otherwise it is at most a WATCH.
  5. Only COOLING is treated as a supply risk (a hotter plant is not a shortage).
  6. Validation: we inject simulated shutdowns/slowdowns into the real data and
     measure how many the model catches and how often it raises false alarms.

Run:  py model.py            (needs: pandas numpy scikit-learn)
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

INPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "satellite_data.csv"
OUTPUT_CSV = "ai_alerts.csv"
REPORT_JSON = "model_report.json"

MAX_CLOUD = 15.0        # % cloud allowed
WINDOW = 6              # passes used for the expected baseline
MIN_HISTORY = 3         # passes needed before the rolling baseline is used
MIN_SPREAD = 1.0        # °C, floor for natural variation
SEED = 42

Z_WATCH, Z_ALERT = -1.5, -2.5          # how many "normal variations" below expected
RATIO_WATCH, RATIO_ALERT = 0.80, 0.60  # heat as a fraction of expected
COLD_DELTA = 0.5                       # °C above surroundings = basically no heat


# ------------------------------------------------------------------ data ---
def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"plant_temp_c": "plant_temp", "ambient_temp_c": "ambient_temp",
                            "cloud_cover_pct": "cloud"})
    df["date"] = pd.to_datetime(df["date"])
    if "cloud" not in df:
        df["cloud"] = 0.0
    df["delta_t"] = df["plant_temp"] - df["ambient_temp"]
    df = df[df["cloud"] <= MAX_CLOUD].dropna(subset=["delta_t"])
    return df.sort_values(["site_name", "date"]).reset_index(drop=True)


def add_features(g):
    g = g.sort_values("date").reset_index(drop=True).copy()
    dt = g["delta_t"].to_numpy()
    site_med = float(np.median(dt))
    expected = np.array([np.median(dt[max(0, i - WINDOW):i]) if i >= MIN_HISTORY else site_med
                         for i in range(len(dt))])
    expected = np.where(expected < 1.0, max(site_med, 1.0), expected)
    # natural variation = how much passes usually differ from their own baseline
    # (measured on residuals, so seasonal swings are not counted as noise)
    resid = (dt - expected)[MIN_HISTORY:] if len(dt) > MIN_HISTORY + 2 else dt - site_med
    spread = max(float(np.median(np.abs(resid - np.median(resid))) * 1.4826), MIN_SPREAD)
    g["expected_delta_t"] = expected.round(2)
    g["heat_ratio"] = (dt / expected).round(3)
    g["z_score"] = ((dt - expected) / spread).round(2)
    g["change_vs_prev"] = np.r_[0.0, np.diff(dt)] / spread
    g["activity_index"] = np.clip(100 * dt / expected, 0, 100).round(0).astype(int)
    g["spread"] = spread
    g["baseline_ready"] = np.arange(len(g)) >= MIN_HISTORY
    return g


FEATURES = ["z_score", "heat_ratio", "change_vs_prev"]


def model_inputs(df):
    X = df[FEATURES].copy()
    X["heat_ratio"] = X["heat_ratio"].clip(-1, 3)
    return X


# ------------------------------------------------------------ decisions ---
def decide(delta, z, ratio, iso_anomaly, baseline_ready=True):
    if not baseline_ready:
        return "Normal"                     # still learning this plant's normal
    rule_alert = z <= Z_ALERT and ratio <= RATIO_ALERT
    rule_watch = z <= Z_WATCH and ratio <= RATIO_WATCH
    severe_drop = ratio <= 0.35 and z <= -1.5
    if delta <= COLD_DELTA and z <= Z_WATCH and (iso_anomaly or ratio <= 0.2):
        return "Cold / Shutdown Alert"
    if (rule_alert or severe_drop) and iso_anomaly:
        return "Bottleneck Alert"
    if rule_alert or rule_watch or severe_drop or (iso_anomaly and z <= -1.0):
        return "Watch (Low)"
    return "Normal"


def explain(r):
    diff = r["delta_t"] - r["expected_delta_t"]
    base = (f"Plant {r['delta_t']:.1f} °C above surroundings vs usual {r['expected_delta_t']:.1f} °C "
            f"({diff:+.1f} °C)")
    lvl = r["risk_level"]
    if not r["baseline_ready"]:
        return f"Learning this plant's normal heat level (first passes). {base}."
    if lvl == "Cold / Shutdown Alert":
        return f"Almost no waste heat - looks shut down or idle. {base}. Both checks agree."
    if lvl == "Bottleneck Alert":
        return f"Much cooler than usual - running far below capacity. {base}. Both checks agree."
    if lvl == "Watch (Low)":
        return f"Cooler than usual - possible slowdown, monitor next pass. {base}."
    if diff > 2 * r["spread"]:
        return f"Hotter than usual - high activity. {base}."
    return f"Normal operating heat signature. {base}."


def score(df, iso, threshold):
    s = -iso.score_samples(model_inputs(df))
    df = df.copy()
    df["anomaly_score"] = s.round(3)
    df["iso_anomaly"] = s >= threshold
    df["risk_level"] = [decide(d, z, r, a, b) for d, z, r, a, b in
                        zip(df["delta_t"], df["z_score"], df["heat_ratio"], df["iso_anomaly"],
                            df["baseline_ready"])]
    df["anomaly_flag"] = df["risk_level"].str.contains("Alert").astype(int)
    return df


# ------------------------------------------------------------ validation ---
def validate(feat, iso, threshold, trials=400):
    """Inject simulated events into real data and measure detection."""
    rng = np.random.default_rng(SEED)
    feat = feat[feat["baseline_ready"]]
    out = {}
    for name, lo, hi, hit in [("shutdown", 0.0, 0.3, "Alert"),
                              ("slowdown", 0.4, 0.7, "Watch|Alert")]:
        caught = 0
        for _ in range(trials):
            row = feat.sample(1, random_state=int(rng.integers(1e9))).iloc[0].copy()
            new = rng.uniform(lo, hi) * row["expected_delta_t"]
            row["change_vs_prev"] += (new - row["delta_t"]) / row["spread"]
            row["delta_t"] = new
            row["heat_ratio"] = new / row["expected_delta_t"]
            row["z_score"] = (new - row["expected_delta_t"]) / row["spread"]
            lvl = score(pd.DataFrame([row]), iso, threshold)["risk_level"].iloc[0]
            caught += bool(pd.Series([lvl]).str.contains(hit).iloc[0])
        out[f"{name}_detection_pct"] = round(100 * caught / trials, 1)
    return out


# ------------------------------------------------------------------ main ---
def main():
    raw = load(INPUT_CSV)
    feat = pd.concat([add_features(g) for _, g in raw.groupby("site_name")], ignore_index=True)

    iso = IsolationForest(n_estimators=300, contamination=0.08, random_state=SEED)
    iso.fit(model_inputs(feat))
    threshold = float(np.quantile(-iso.score_samples(model_inputs(feat)), 0.92))

    res = score(feat, iso, threshold)
    res["supply_chain_alert"] = res.apply(explain, axis=1)

    out = res[["date", "site_name", "activity_index", "anomaly_flag", "risk_level",
               "supply_chain_alert", "anomaly_score", "expected_delta_t", "heat_ratio", "z_score"]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(OUTPUT_CSV, index=False)

    val = validate(feat, iso, threshold)
    val["alert_rate_on_real_data_pct"] = round(100 * res["anomaly_flag"].mean(), 1)
    val["shutdown_detection_by_plant_pct"] = {
        s: validate(feat[feat["site_name"] == s], iso, threshold, trials=200)["shutdown_detection_pct"]
        for s in feat["site_name"].unique()}
    report = {
        "model": "Hybrid: physics rule (robust z-score vs plant's own baseline) + Isolation Forest",
        "passes_analysed": int(len(res)),
        "plants": int(res["site_name"].nunique()),
        "validation": val,
        "per_plant": {s: {"passes": int(len(g)),
                          "alerts": int(g["anomaly_flag"].sum()),
                          "watch": int((g["risk_level"] == "Watch (Low)").sum()),
                          "latest": g.iloc[-1]["risk_level"]}
                      for s, g in res.groupby("site_name")},
    }
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved {OUTPUT_CSV} and {REPORT_JSON}")


if __name__ == "__main__":
    main()
