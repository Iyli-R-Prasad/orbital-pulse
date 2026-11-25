# 🛰️ Orbital Industrial Pulse

**Tracking how hard steel plants and refineries are working, anywhere on Earth, from satellite heat and AI.**
Mach-A-Thon, Aeromania '26 (RVCE × Boeing), Track A.

**Live demo:** https://orbital-industrial-pulse.streamlit.app

## The idea
A working blast furnace or refinery gives off a lot of waste heat. Landsat 8 and 9 measure ground temperature
worldwide about every 8 days, for free. When a plant's heat drops compared with its own normal level,
production has probably dropped too. That shows up weeks before company reports or official statistics.

## How it works
1. **Collect:** Landsat 8/9 Collection 2 Level-2 surface temperature (band `lwir11`) from Microsoft Planetary Computer.
2. **Clean:** the satellite's own `qa_pixel` quality band removes cloud, cirrus, cloud shadow and water pixels.
3. **Find the plant:** the hot-spot finder averages clear passes and finds the hottest persistent spot within 2 km of the given point.
4. **Measure:** heat signal = plant temperature (300 m box) minus surroundings (10 km box, with the plant and water left out),
   so season and weather cancel out.
5. **Decide:** a hybrid AI (`model.py`):
   - physics rules compare each pass with the plant's own last 6 passes and its long-term baseline (robust z-score, heat ratio);
   - an Isolation Forest flags statistically unusual passes;
   - a full **Alert** needs both to agree; otherwise at most a **Watch**.

## Results (4 plants, 3 countries, 98 clear passes)
| Test | Result |
|---|---|
| Simulated shutdowns caught (heat injected at 0–30 % of normal) | **95.8 %** |
| Simulated slowdowns caught (40–70 % of normal) | **63.5 %** |
| Alert rate on normal real data | **2.0 %** |
| Real event: JSW Vijayanagar BF3 outage (Sep 2025 – Jun 2026) | heat signal **17 % lower** in that window |
| Real event: Reliance Jamnagar maintenance (May 2026) | flagged **"Watch (Sustained low)"** in spring 2026 |

Plants: JSW Vijayanagar and Reliance Jamnagar (India), US Steel Gary Works (USA), Thyssenkrupp Duisburg (Germany).

## Any plant on Earth
In the website's **"Check any plant on Earth (live)"** section, type a latitude and longitude. The app fetches
real Landsat images for that point live, finds the hot spot, and runs the same AI (`live_check.py`).

## Who pays
Commodity traders and hedge funds, insurers (business interruption), industrial buyers and governments.
Plans: $500 per site per month, $15,000 per month for 50 sites, $250,000 per year enterprise.
Hedge funds spent about $2.8B on alternative data in 2025 (Neudata). Example: JSW Vijayanagar makes about $31M of steel a day,
so an 8-day, 20 % slowdown puts about $44–50M of output at risk.

## Files
| File | What it does |
|---|---|
| `fetch_data.py` | Downloads and processes Landsat data, writes `satellite_data.csv` and heat images |
| `model.py` | Hybrid AI + validation, writes `ai_alerts.csv` and `model_report.json` |
| `live_check.py` | Same pipeline, live, for any latitude/longitude |
| `app.py` | Streamlit website |

Run locally: `pip install -r requirements.txt` then `streamlit run app.py`.
To refresh the data (e.g. in Google Colab): `python fetch_data.py` then `python model.py satellite_data.csv`.

## Honest limits
- About one Landsat pass every 8 days, and clouds hide some passes, so this tracks weeks-long changes, not daily ones.
- Partial slowdowns are harder to catch than full shutdowns.
- Heat is a proxy for output, not a direct measurement.
