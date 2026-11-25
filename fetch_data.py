"""
Orbital Industrial Pulse - satellite data pipeline (v2, Person 1)

Improvements over v1:
  1. HOT-SPOT FINDER: instead of trusting one coordinate, it averages many clear
     Landsat passes and finds the plant's persistent hottest spot (the furnaces)
     within 2 km of the given point. Fixes weak signals like JSW.
  2. PIXEL CLOUD MASK: uses Landsat's QA_PIXEL band to remove cloud, cloud shadow
     and cirrus pixels at the plant itself (not the whole 185 km scene's cloud %).
  3. CLEANER BACKGROUND: "surroundings" = 10 km box, excluding the plant area,
     water and clouds.
  4. HEAT MAPS: saves real thermal images (normal vs lowest-activity pass) per plant.

Outputs:
  satellite_data.csv               -> for model.py and the website
  <site>_heat_normal.png / <site>_heat_shutdown.png  -> for the website

Run in Google Colab:
  !pip install -q pystac-client planetary-computer rasterio pyproj matplotlib
  !python fetch_data.py
"""
import numpy as np
import pandas as pd

SITES = [
    # id (used in file names), name in data, lat, lon
    ("jsw", "JSW_Vijayanagar_Steel_India", 15.1764, 76.6433),
    ("jamnagar", "Jamnagar_Refinery_India", 22.3511, 69.8597),
    ("gary", "Gary_Works_Steel_USA", 41.6139, -87.3364),
    ("duisburg", "Thyssenkrupp_Duisburg_Germany", 51.4981, 6.7278),
]
START, END = "2025-11-01", "2026-11-01"
SCENE_CLOUD_MAX = 60        # scene-level filter is loose; pixel mask does the real work
SEARCH_RADIUS_M = 2000      # how far from the given point to look for the hot spot
PLANT_HALF_M = 150          # 300 m box around the hot spot
AMB_HALF_M = 10000          # 10 km box for surroundings
AMB_EXCLUDE_M = 1500        # ignore pixels this close to the plant when measuring surroundings
MAX_PLANT_MASKED = 0.2      # skip pass if >20 % of plant pixels are cloudy/invalid
MAX_AMB_MASKED = 0.5        # skip pass if >50 % of surrounding pixels are cloudy/invalid
PIX = 30.0                  # Landsat thermal grid spacing (m)

ST_SCALE, ST_OFFSET = 0.00341802, 149.0
# QA_PIXEL bits: 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 7 water
BAD_BITS = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
WATER_BIT = 1 << 7


# --------------------------------------------------------------- helpers ---
def to_celsius(raw):
    return raw.astype(float) * ST_SCALE + ST_OFFSET - 273.15


def masks(qa, st_raw):
    bad = ((qa & BAD_BITS) != 0) | (st_raw == 0)
    water = (qa & WATER_BIT) != 0
    return bad, water


def box(arr, cy, cx, half_px):
    return arr[max(0, cy - half_px):cy + half_px + 1, max(0, cx - half_px):cx + half_px + 1]


def find_hotspot(anomaly_stack, half_search_px, smooth=3):
    """anomaly_stack: list of 2-D arrays (temp minus scene median, NaN where bad),
    all centred on the given point. Returns (dy, dx) offset of the persistent hot spot."""
    stack = np.stack(anomaly_stack)
    mean = np.nanmean(stack, axis=0)
    mean = np.where(np.isnan(mean), -np.inf, mean)
    k = smooth // 2
    padded = np.pad(mean, k, constant_values=-np.inf)
    sm = np.full_like(mean, -np.inf)
    h, w = mean.shape
    for y in range(h):
        for x in range(w):
            win = padded[y:y + smooth, x:x + smooth]
            if np.isfinite(win).all():
                sm[y, x] = win.mean()
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    sm[(yy - cy) ** 2 + (xx - cx) ** 2 > half_search_px ** 2] = -np.inf
    if not np.isfinite(sm).any():
        return 0, 0
    y, x = np.unravel_index(np.argmax(sm), sm.shape)
    return int(y - cy), int(x - cx)


def measure(temp_c, bad, water, cy, cx):
    """Plant = 95th percentile in a small box at the hot spot.
    Surroundings = median of the 10 km box, excluding plant area, water and clouds."""
    p_half = int(PLANT_HALF_M / PIX)
    a_half = int(AMB_HALF_M / PIX)
    ex = int(AMB_EXCLUDE_M / PIX)

    p_t, p_bad = box(temp_c, cy, cx, p_half), box(bad, cy, cx, p_half)
    if p_bad.size == 0 or p_bad.mean() > MAX_PLANT_MASKED:
        return None
    a_t, a_bad, a_w = box(temp_c, cy, cx, a_half), box(bad, cy, cx, a_half), box(water, cy, cx, a_half)
    oy, ox = min(cy, a_half), min(cx, a_half)
    yy, xx = np.mgrid[0:a_t.shape[0], 0:a_t.shape[1]]
    near = (np.abs(yy - oy) <= ex) & (np.abs(xx - ox) <= ex)
    if a_bad.mean() > MAX_AMB_MASKED:
        return None
    amb_ok = ~a_bad & ~a_w & ~near
    if amb_ok.sum() < 100:
        return None
    plant = float(np.percentile(p_t[~p_bad], 95))
    amb = float(np.median(a_t[amb_ok]))
    return plant, amb, float(p_bad.mean() * 100), float(a_bad.mean() * 100)


# ----------------------------------------------------------- satellite IO ---
def search(lat, lon):
    import planetary_computer
    import pystac_client
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)
    s = cat.search(collections=["landsat-c2-l2"], intersects={"type": "Point", "coordinates": [lon, lat]},
                   datetime=f"{START}/{END}",
                   query={"eo:cloud_cover": {"lt": SCENE_CLOUD_MAX},
                          "platform": {"in": ["landsat-8", "landsat-9"]}})
    return sorted(s.items(), key=lambda i: i.datetime)


def read_window(item, lat, lon, half_m):
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import from_bounds
    with rasterio.open(item.assets["lwir11"].href) as st, rasterio.open(item.assets["qa_pixel"].href) as qa:
        x, y = Transformer.from_crs("EPSG:4326", st.crs, always_xy=True).transform(lon, lat)
        win = from_bounds(x - half_m, y - half_m, x + half_m, y + half_m, st.transform)
        st_raw = st.read(1, window=win, boundless=True, fill_value=0)
        qa_raw = qa.read(1, window=win, boundless=True, fill_value=1)
        to_ll = Transformer.from_crs(st.crs, "EPSG:4326", always_xy=True)
    return st_raw, qa_raw, (x, y), to_ll


def process_site(site_id, name, lat, lon):
    items = search(lat, lon)
    print(f"\n{name}: {len(items)} candidate passes")
    scenes = []
    for it in items:
        try:
            st_raw, qa_raw, xy, to_ll = read_window(it, lat, lon, AMB_HALF_M + SEARCH_RADIUS_M)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {it.datetime:%Y-%m-%d}: {e}")
            continue
        temp = to_celsius(st_raw)
        bad, water = masks(qa_raw, st_raw)
        scenes.append((it, temp, bad, water, xy, to_ll))

    # 1) find the persistent hot spot from mostly-clear passes
    s_half = int(SEARCH_RADIUS_M / PIX) + 3
    anomalies = []
    for _, temp, bad, water, _, _ in scenes:
        cy0, cx0 = temp.shape[0] // 2, temp.shape[1] // 2
        sub_t, sub_b = box(temp, cy0, cx0, s_half), box(bad, cy0, cx0, s_half)  # water allowed (lakeside plants)
        if sub_t.shape != (2 * s_half + 1, 2 * s_half + 1):
            continue
        if sub_b.mean() < 0.1:
            a = sub_t - np.nanmedian(np.where(sub_b, np.nan, sub_t))
            anomalies.append(np.where(sub_b, np.nan, a))
    dy, dx = find_hotspot(anomalies, int(SEARCH_RADIUS_M / PIX)) if anomalies else (0, 0)
    if scenes:
        x0, y0 = scenes[0][4]
        new_lon, new_lat = scenes[0][5].transform(x0 + dx * PIX, y0 - dy * PIX)
    else:
        new_lat, new_lon = lat, lon
    print(f"  hot spot moved {np.hypot(dy, dx) * PIX:.0f} m -> {new_lat:.4f}, {new_lon:.4f} "
          f"(from {len(anomalies)} clear passes)")

    # 2) measure every pass at the hot spot
    rows, frames = [], []
    for it, temp, bad, water, _, _ in scenes:
        hy, hx = temp.shape[0] // 2 + dy, temp.shape[1] // 2 + dx
        m = measure(temp, bad, water, hy, hx)
        if m is None:
            continue
        plant, amb, p_cloud, a_cloud = m
        rows.append({"date": f"{it.datetime:%Y-%m-%d}", "site_name": name,
                     "lat": round(new_lat, 4), "lon": round(new_lon, 4),
                     "plant_temp_c": round(plant, 2), "ambient_temp_c": round(amb, 2),
                     "relative_temp_c": round(plant - amb, 2),
                     "cloud_cover_pct": round(p_cloud, 1), "ambient_cloud_pct": round(a_cloud, 1),
                     "scene_cloud_pct": round(float(it.properties.get("eo:cloud_cover", 0)), 1)})
        frames.append((plant - amb, box(temp, hy, hx, 50), box(bad, hy, hx, 50), f"{it.datetime:%d %b %Y}"))
    df = pd.DataFrame(rows).drop_duplicates("date")
    print(f"  kept {len(df)} clean passes")
    save_heatmaps(site_id, name, frames)
    return df


def save_heatmaps(site_id, name, frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if len(frames) < 3:
        return
    deltas = np.array([f[0] for f in frames])
    picks = {"normal": int(np.argsort(np.abs(deltas - np.percentile(deltas, 75)))[0]),
             "shutdown": int(np.argmin(deltas))}
    vmin = min(np.nanpercentile(frames[i][1], 2) for i in picks.values())
    vmax = max(np.nanpercentile(frames[i][1], 99.5) for i in picks.values())
    for label, i in picks.items():
        d, img, bad, when = frames[i]
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        im = ax.imshow(np.where(bad, np.nan, img), cmap="inferno", vmin=vmin, vmax=vmax)
        ax.plot(img.shape[1] // 2, img.shape[0] // 2, marker="+", color="cyan", ms=14, mew=2)
        ax.set_title(f"{name.replace('_', ' ')}\n{when} - {d:+.1f} °C vs surroundings", fontsize=8)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="°C")
        fig.tight_layout()
        fig.savefig(f"{site_id}_heat_{label}.png")
        plt.close(fig)
    print(f"  saved {site_id}_heat_normal.png and {site_id}_heat_shutdown.png")


def main():
    all_df = [process_site(*s) for s in SITES]
    out = pd.concat([d for d in all_df if len(d)], ignore_index=True)
    out.to_csv("satellite_data.csv", index=False)
    print(f"\nSaved satellite_data.csv with {len(out)} rows")
    print(out.groupby("site_name")["relative_temp_c"].describe()[["count", "mean", "50%", "min"]])


if __name__ == "__main__":
    main()
