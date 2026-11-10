
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = ROOT / "reference" / "base-paper" / "scenario41.csv"
RESOURCE_PATH = (
    ROOT / "data" / "raw" / "scenario41" /
    "resources" / "static_pos_unit1.txt"
)
OUT_DIR = ROOT / "results" / "geometry_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Read Unit 1's static GPS coordinates.
unit1_lat, unit1_lon = map(
    float, RESOURCE_PATH.read_text().strip().split()
)

df = pd.read_csv(CSV_PATH)

required = [
    "unit2_gps1_lat",
    "unit2_gps1_lon",
    "seq_index",
    "unit1_pwr3_best-beam",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Calculate bearing from Unit 1 to Unit 2.
lat1 = np.radians(unit1_lat)
lon1 = np.radians(unit1_lon)

lat2 = np.radians(df["unit2_gps1_lat"].to_numpy())
lon2 = np.radians(df["unit2_gps1_lon"].to_numpy())

dlon = lon2 - lon1

x = np.sin(dlon) * np.cos(lat2)
y = (
    np.cos(lat1) * np.sin(lat2)
    - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
)

df["gps_bearing_deg"] = (
    np.degrees(np.arctan2(x, y)) + 360
) % 360

df["pwr3_beam"] = pd.to_numeric(
    df["unit1_pwr3_best-beam"], errors="coerce"
)

valid = df.dropna(
    subset=[
        "gps_bearing_deg",
        "pwr3_beam",
        "seq_index",
    ]
).copy()

valid["pwr3_beam"] = valid["pwr3_beam"].astype(int)
valid = valid[valid["pwr3_beam"].between(0, 63)]

def circular_mean_deg(values):
    r = np.radians(np.asarray(values, dtype=float))
    return (
        np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))
        + 360
    ) % 360

# Per-beam bearing statistics.
beam_summary = (
    valid.groupby("pwr3_beam")
    .agg(
        sample_count=("gps_bearing_deg", "size"),
        mean_bearing_deg=("gps_bearing_deg", circular_mean_deg),
        bearing_std_deg=("gps_bearing_deg", "std"),
        sequence_count=("seq_index", "nunique"),
    )
    .reset_index()
    .sort_values("pwr3_beam")
)

# Per-sequence movement and beam statistics.
sequence_summary = (
    valid.groupby("seq_index")
    .agg(
        samples=("pwr3_beam", "size"),
        unique_beams=("pwr3_beam", "nunique"),
        first_bearing=("gps_bearing_deg", "first"),
        last_bearing=("gps_bearing_deg", "last"),
        first_beam=("pwr3_beam", "first"),
        last_beam=("pwr3_beam", "last"),
    )
    .reset_index()
)

valid.to_csv(
    OUT_DIR / "sample_bearings_and_beams.csv",
    index=False,
)
beam_summary.to_csv(
    OUT_DIR / "pwr3_beam_bearing_summary.csv",
    index=False,
)
sequence_summary.to_csv(
    OUT_DIR / "sequence_geometry_summary.csv",
    index=False,
)

report = f"""
SCENARIO 41 GPS / PWR3 GEOMETRY AUDIT

Unit 1 latitude: {unit1_lat}
Unit 1 longitude: {unit1_lon}

Total CSV rows: {len(df)}
Valid rows: {len(valid)}
Unique sequences: {valid['seq_index'].nunique()}
Unique PWR3 beams: {valid['pwr3_beam'].nunique()}

GPS bearing range:
{valid['gps_bearing_deg'].min():.4f} to
{valid['gps_bearing_deg'].max():.4f} degrees

PWR3 BEAM SUMMARY
{beam_summary.to_string(index=False)}

SEQUENCE SUMMARY
{sequence_summary.to_string(index=False)}
"""

(OUT_DIR / "geometry_audit_report.txt").write_text(
    report, encoding="utf-8"
)

print(report)
print(f"\nOutputs saved to: {OUT_DIR}")
