"""
Creates FAKE practice files for every site in the dashboard.
Real files from Person 1 / Person 2 replace these later.

Main site (first in the list) uses:   temperatures.csv, alerts.csv
Other sites use:                      <site_id>_temperatures.csv, <site_id>_alerts.csv

Run:  py make_fake_data.py
"""
import numpy as np
import pandas as pd

SITE_IDS = ["jsw", "scherer", "pohang"]   # must match SITES in app.py


def make_site(site_id, seed, shutdowns, first):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=24, freq="16D")
    n = len(dates)
    ambient = 26 + 7 * np.sin(np.arange(n) / n * 2 * np.pi) + rng.normal(0, 1, n)
    delta = 15 + rng.normal(0, 1.2, n)
    for i, d in shutdowns.items():
        delta[i] = d
    plant = ambient + delta

    temps = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "plant_temp": plant.round(1),
        "ambient_temp": ambient.round(1),
    })
    activity = np.clip((delta / 16) * 100, 0, 100).round(0).astype(int)
    risk = np.where(activity < 50, "Alert", np.where(activity < 75, "Watch", "Normal"))
    alerts = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "activity_index": activity,
        "anomaly_flag": (risk == "Alert").astype(int),
        "risk_level": risk,
    })

    prefix = "" if first else f"{site_id}_"
    temps.to_csv(f"{prefix}temperatures.csv", index=False)
    alerts.to_csv(f"{prefix}alerts.csv", index=False)
    print(f"Created {prefix}temperatures.csv and {prefix}alerts.csv")


make_site("jsw", 1, {9: 4, 10: 6, 18: 7}, first=True)
make_site("scherer", 2, {5: 9, 15: 3}, first=False)
make_site("pohang", 3, {21: 10, 22: 5}, first=False)
