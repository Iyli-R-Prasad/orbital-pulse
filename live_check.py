"""
Orbital Industrial Pulse - LIVE check for any plant on Earth.

The website calls scan_plant(lat, lon, months) when someone types a latitude/longitude.
It does the same steps as fetch_data.py + model.py, but live, for one point:
  1. Search Microsoft Planetary Computer for Landsat 8/9 passes over the point.
  2. Read only a small 24 km window of the thermal band + quality band (fast).
  3. Find the plant's persistent hot spot within 2 km, remove clouds/water.
  4. Heat signal = plant temperature minus its 10 km surroundings.
  5. Score every pass with the same hybrid AI (physics rules + Isolation Forest).
"""
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd

import fetch_data as fd

SCENE_CLOUD_MAX = 60
MAX_PASSES = 60          # safety limit so one check never runs for too long
WORKERS = 8              # passes read in parallel

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")


def search(lat, lon, start, end):
    import planetary_computer
    import pystac_client
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)
    s = cat.search(collections=["landsat-c2-l2"],
                   intersects={"type": "Point", "coordinates": [lon, lat]},
                   datetime=f"{start}/{end}",
                   query={"eo:cloud_cover": {"lt": SCENE_CLOUD_MAX},
                          "platform": {"in": ["landsat-8", "landsat-9"]}})
    items = sorted(s.items(), key=lambda i: i.datetime)
    # keep one pass per day (overlapping paths), newest MAX_PASSES
    seen, out = set(), []
    for it in items:
        d = f"{it.datetime:%Y-%m-%d}"
        if d not in seen:
            seen.add(d)
            out.append(it)
    return out[-MAX_PASSES:]


def _read(it, lat, lon):
    try:
        st_raw, qa_raw, xy, to_ll = fd.read_window(it, lat, lon, fd.AMB_HALF_M + fd.SEARCH_RADIUS_M)
    except Exception:  # noqa: BLE001  (a broken/unreachable scene is just skipped)
        return None
    temp = fd.to_celsius(st_raw).astype(np.float32)
    bad, water = fd.masks(qa_raw, st_raw)
    return it, temp, bad, water, xy, to_ll


def measure_scenes(scenes):
    """scenes: list of (item, temp, bad, water, xy, to_ll). Same logic as fetch_data.process_site."""
    s_half = int(fd.SEARCH_RADIUS_M / fd.PIX) + 3
    anomalies = []
    for _, temp, bad, water, _, _ in scenes:
        cy0, cx0 = temp.shape[0] // 2, temp.shape[1] // 2
        sub_t, sub_b = fd.box(temp, cy0, cx0, s_half), fd.box(bad, cy0, cx0, s_half)
        if sub_t.shape != (2 * s_half + 1, 2 * s_half + 1):
            continue
        if sub_b.mean() < 0.1:
            a = sub_t - np.nanmedian(np.where(sub_b, np.nan, sub_t))
            anomalies.append(np.where(sub_b, np.nan, a))
    dy, dx = fd.find_hotspot(anomalies, int(fd.SEARCH_RADIUS_M / fd.PIX)) if anomalies else (0, 0)
    x0, y0 = scenes[0][4]
    hot_lon, hot_lat = scenes[0][5].transform(x0 + dx * fd.PIX, y0 - dy * fd.PIX)

    rows, frames = [], []
    for it, temp, bad, water, _, _ in scenes:
        hy, hx = temp.shape[0] // 2 + dy, temp.shape[1] // 2 + dx
        m = fd.measure(temp, bad, water, hy, hx)
        if m is None:
            continue
        plant, amb, p_cloud, _ = m
        rows.append({"date": pd.Timestamp(f"{it.datetime:%Y-%m-%d}"),
                     "plant_temp": round(plant, 2), "ambient_temp": round(amb, 2),
                     "delta_t": round(plant - amb, 2), "cloud": round(p_cloud, 1)})
        img = np.where(fd.box(bad, hy, hx, 50), np.nan, fd.box(temp, hy, hx, 50))
        frames.append((f"{it.datetime:%Y-%m-%d}", img))
    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df, (float(hot_lat), float(hot_lon)), float(np.hypot(dy, dx) * fd.PIX), dict(frames), len(anomalies)


def score(df, reference_csv="satellite_data.csv"):
    """Run the same hybrid AI as model.py on one new plant."""
    import model as ai
    from sklearn.ensemble import IsolationForest

    df = df[df["cloud"] <= ai.MAX_CLOUD].copy()
    df["site_name"] = "live"
    feat = ai.add_features(df)
    # learn "normal behaviour" from our 4 reference plants + this plant
    train = [feat]
    if os.path.exists(reference_csv):
        ref = ai.load(reference_csv)
        train += [ai.add_features(g) for _, g in ref.groupby("site_name")]
    allf = pd.concat(train, ignore_index=True)
    iso = IsolationForest(n_estimators=300, contamination=0.08, random_state=ai.SEED)
    iso.fit(ai.model_inputs(allf))
    threshold = float(np.quantile(-iso.score_samples(ai.model_inputs(allf)), 0.92))
    res = ai.score(feat, iso, threshold)
    res["explanation"] = res.apply(ai.explain, axis=1)
    return res


def scan_plant(lat, lon, months=12, progress=None):
    """Returns dict with results, or {'error': text}."""
    end = date.today()
    start = end - timedelta(days=int(months * 30.5))
    if progress:
        progress(0.05, "Searching Landsat archive...")
    items = search(lat, lon, start.isoformat(), end.isoformat())
    if len(items) < 4:
        return {"error": f"Only {len(items)} Landsat passes found here in the last {months} months."}

    scenes, done = [], 0
    with ThreadPoolExecutor(WORKERS) as pool:
        for r in pool.map(lambda it: _read(it, lat, lon), items):
            done += 1
            if r is not None:
                scenes.append(r)
            if progress:
                progress(0.05 + 0.85 * done / len(items), f"Reading thermal images {done}/{len(items)}...")
    if len(scenes) < 4:
        return {"error": "Could not read enough Landsat images for this point. Try again in a minute."}

    if progress:
        progress(0.92, "Finding the hot spot and running the AI...")
    df, hot, moved_m, frames, n_clear = measure_scenes(scenes)
    if len(df) < 4:
        return {"error": f"Only {len(df)} cloud-free passes here. Try 12 months or another point."}
    res = score(df)
    if res.empty:
        return {"error": "All passes were too cloudy for a reliable result."}
    if progress:
        progress(1.0, "Done")
    return {"data": res, "hot_lat": hot[0], "hot_lon": hot[1], "moved_m": moved_m,
            "passes_found": len(items), "frames": frames, "clear_for_hotspot": n_clear}
