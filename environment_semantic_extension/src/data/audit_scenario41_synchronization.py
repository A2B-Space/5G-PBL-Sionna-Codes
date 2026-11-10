
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "reference" / "base-paper" / "scenario41.csv"
OUT = ROOT / "results" / "synchronization_audit"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

def extract_time(path):
    match = re.search(r"(\d{2}-\d{2}-\d{2}\.\d+)", str(path))
    if not match:
        return pd.NaT
    return pd.to_timedelta(match.group(1).replace("-", ":"))

# Extract timestamps from the actual referenced measurement files.
df["pwr3_time"] = df["unit1_pwr3"].apply(extract_time)
df["gps_file_time"] = df["unit2_gps1"].apply(extract_time)
df["csv_time"] = pd.to_timedelta(df["timestamp"].str.replace("-", ":", regex=False))

# Positive offset means GPS is later than the PWR3 measurement.
df["gps_minus_pwr3_sec"] = (
    df["gps_file_time"] - df["pwr3_time"]
).dt.total_seconds()

# Verify the CSV timestamp against the GPS filename timestamp.
df["csv_minus_gps_sec"] = (
    df["csv_time"] - df["gps_file_time"]
).dt.total_seconds()

offset_summary = df.groupby("seq_index").agg(
    samples=("abs_index", "size"),
    median_gps_pwr_offset_sec=("gps_minus_pwr3_sec", "median"),
    mean_gps_pwr_offset_sec=("gps_minus_pwr3_sec", "mean"),
    min_gps_pwr_offset_sec=("gps_minus_pwr3_sec", "min"),
    max_gps_pwr_offset_sec=("gps_minus_pwr3_sec", "max"),
    median_csv_gps_offset_sec=("csv_minus_gps_sec", "median"),
    invalid_pwr_timestamps=("pwr3_time", lambda x: x.isna().sum()),
    invalid_gps_timestamps=("gps_file_time", lambda x: x.isna().sum()),
).reset_index()

# Haversine distance between consecutive GPS samples.
lat = np.radians(df["unit2_gps1_lat"].to_numpy())
lon = np.radians(df["unit2_gps1_lon"].to_numpy())

lat_prev = np.roll(lat, 1)
lon_prev = np.roll(lon, 1)

dlat = lat - lat_prev
dlon = lon - lon_prev

a = (
    np.sin(dlat / 2) ** 2
    + np.cos(lat_prev) * np.cos(lat) * np.sin(dlon / 2) ** 2
)
distance_m = 2 * 6371000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

df["gps_step_distance_m"] = distance_m
df["gps_time_step_sec"] = df.groupby("seq_index")["csv_time"].diff().dt.total_seconds()

# Do not calculate speeds across sequence boundaries.
same_sequence = df["seq_index"].eq(df["seq_index"].shift())
valid_steps = (
    same_sequence
    & df["gps_time_step_sec"].gt(0)
    & df["gps_step_distance_m"].notna()
)

df["gps_speed_mps"] = np.nan
df.loc[valid_steps, "gps_speed_mps"] = (
    df.loc[valid_steps, "gps_step_distance_m"]
    / df.loc[valid_steps, "gps_time_step_sec"]
)

movement_summary = df.groupby("seq_index").agg(
    samples=("abs_index", "size"),
    unique_gps_positions=("unit2_gps1", "nunique"),
    median_step_distance_m=("gps_step_distance_m", "median"),
    median_speed_mps=("gps_speed_mps", "median"),
    max_speed_mps=("gps_speed_mps", "max"),
    valid_speed_steps=("gps_speed_mps", "count"),
).reset_index()

df.to_csv(OUT / "sample_synchronization_audit.csv", index=False)
offset_summary.to_csv(OUT / "timestamp_offset_by_sequence.csv", index=False)
movement_summary.to_csv(OUT / "gps_movement_by_sequence.csv", index=False)

report = f"""
SCENARIO 41 SYNCHRONIZATION AUDIT

Rows: {len(df)}

PWR3 timestamp valid: {df['pwr3_time'].notna().sum()}
GPS timestamp valid: {df['gps_file_time'].notna().sum()}

GPS minus PWR3 offset:
Median: {df['gps_minus_pwr3_sec'].median():.6f} seconds
Mean:   {df['gps_minus_pwr3_sec'].mean():.6f} seconds
Min:    {df['gps_minus_pwr3_sec'].min():.6f} seconds
Max:    {df['gps_minus_pwr3_sec'].max():.6f} seconds

CSV timestamp minus GPS filename timestamp:
Median: {df['csv_minus_gps_sec'].median():.6f} seconds
Max absolute: {df['csv_minus_gps_sec'].abs().max():.6f} seconds

SEQUENCE TIMESTAMP OFFSETS
{offset_summary.to_string(index=False)}

GPS MOVEMENT SUMMARY
{movement_summary.to_string(index=False)}
"""

(OUT / "synchronization_report.txt").write_text(
    report, encoding="utf-8"
)

print(report)
print(f"\nOutputs saved to: {OUT}")
