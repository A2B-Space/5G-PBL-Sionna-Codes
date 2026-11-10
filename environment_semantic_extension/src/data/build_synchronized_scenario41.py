
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = ROOT / "reference" / "base-paper" / "scenario41.csv"
RESOURCE = (
    ROOT / "data" / "raw" / "scenario41" /
    "resources" / "static_pos_unit1.txt"
)
OUT = ROOT / "results" / "synchronized_scenario41"
OUT.mkdir(parents=True, exist_ok=True)

MAX_OFFSET_SEC = 2.0
MAX_GPS_GAP_SEC = 0.5

def parse_time(value):
    match = re.search(
        r"(\d{2})-(\d{2})-(\d{2}\.\d+)",
        str(value)
    )
    if not match:
        return np.nan
    h, m, s = match.groups()
    return float(h) * 3600 + float(m) * 60 + float(s)

def interpolate_position(group, target):
    times = group["gps_time_sec"].to_numpy()
    lat = group["unit2_gps1_lat"].to_numpy()
    lon = group["unit2_gps1_lon"].to_numpy()

    order = np.argsort(times)
    times, lat, lon = times[order], lat[order], lon[order]

    if len(times) < 2:
        return np.nan, np.nan, "insufficient_gps"

    right = np.searchsorted(times, target, side="left")

    if right < len(times) and np.isclose(times[right], target):
        return lat[right], lon[right], "exact"

    if right == 0 or right == len(times):
        return np.nan, np.nan, "outside_gps_interval"

    left = right - 1
    gap = times[right] - times[left]

    if gap <= 0 or gap > MAX_GPS_GAP_SEC:
        return np.nan, np.nan, "gps_gap_too_large"

    fraction = (target - times[left]) / gap

    # Local latitude/longitude interpolation.
    # This is suitable for these short local intervals.
    out_lat = lat[left] + fraction * (lat[right] - lat[left])
    out_lon = lon[left] + fraction * (lon[right] - lon[left])

    return out_lat, out_lon, "interpolated"

df = pd.read_csv(CSV_PATH)

unit1_lat, unit1_lon = map(
    float, RESOURCE.read_text().strip().split()
)

df["gps_time_sec"] = df["unit2_gps1"].apply(parse_time)
df["pwr_time_sec"] = df["unit1_pwr3"].apply(parse_time)

df["offset_sec"] = df["gps_time_sec"] - df["pwr_time_sec"]

# Remove midnight wrap if present.
df["offset_sec"] = (
    (df["offset_sec"] + 43200) % 86400
) - 43200

df["geometry_status"] = "pending"
df["corrected_lat"] = np.nan
df["corrected_lon"] = np.nan

# Exclude known asynchronous records from primary analysis.
df.loc[
    df["offset_sec"].abs() > MAX_OFFSET_SEC,
    "geometry_status"
] = "offset_outlier"

for seq, indices in df.groupby("seq_index").groups.items():
    group = df.loc[indices].sort_values("gps_time_sec")

    for idx in group.index:
        if df.at[idx, "geometry_status"] == "offset_outlier":
            continue

        target = df.at[idx, "pwr_time_sec"]

        lat, lon, status = interpolate_position(group, target)

        df.at[idx, "corrected_lat"] = lat
        df.at[idx, "corrected_lon"] = lon
        df.at[idx, "geometry_status"] = status

# Geographic bearing from Unit 1 to corrected Unit 2 position.
lat1 = np.radians(unit1_lat)
lon1 = np.radians(unit1_lon)

lat2 = np.radians(df["corrected_lat"].to_numpy())
lon2 = np.radians(df["corrected_lon"].to_numpy())

dlon = lon2 - lon1

x = np.sin(dlon) * np.cos(lat2)
y = (
    np.cos(lat1) * np.sin(lat2)
    - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
)

bearing = (np.degrees(np.arctan2(x, y)) + 360) % 360

df["corrected_bearing_deg"] = bearing

usable = df[
    df["geometry_status"].isin(["exact", "interpolated"])
    & df["corrected_bearing_deg"].notna()
].copy()

# Per-beam circular statistics.
def circular_mean(values):
    r = np.radians(np.asarray(values))
    return (
        np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))
        + 360
    ) % 360

beam_summary = (
    usable.groupby("unit1_pwr3_best-beam")
    .agg(
        samples=("abs_index", "size"),
        mean_bearing_deg=("corrected_bearing_deg", circular_mean),
        bearing_std_linear_deg=("corrected_bearing_deg", "std"),
        sequences=("seq_index", "nunique"),
    )
    .reset_index()
)

status_summary = (
    df.groupby(["seq_index", "geometry_status"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

df.to_csv(OUT / "synchronized_master.csv", index=False)
usable.to_csv(OUT / "synchronized_geometry_samples.csv", index=False)

df.loc[
    df["geometry_status"] == "offset_outlier"
].to_csv(OUT / "excluded_offset_records.csv", index=False)

beam_summary.to_csv(OUT / "corrected_beam_bearing_summary.csv", index=False)
status_summary.to_csv(OUT / "geometry_status_summary.csv", index=False)

print("\nSYNCHRONIZED SCENARIO 41 AUDIT")
print("Total samples:", len(df))
print("Usable geometry samples:", len(usable))
print("Excluded offset outliers:", (df["geometry_status"] == "offset_outlier").sum())
print("\nGEOMETRY STATUS")
print(status_summary.to_string(index=False))
print("\nCORRECTED BEAM SUMMARY")
print(beam_summary.to_string(index=False))
print("\nSaved to:", OUT)
