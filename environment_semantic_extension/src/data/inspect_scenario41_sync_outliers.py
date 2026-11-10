
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "reference" / "base-paper" / "scenario41.csv"
OUT = ROOT / "results" / "synchronization_audit"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

def parse_time(value):
    match = re.search(
        r"(\d{2})-(\d{2})-(\d{2}\.\d+)",
        str(value)
    )
    if not match:
        return pd.NaT
    h, m, s = match.groups()
    return pd.to_timedelta(f"{h}:{m}:{s}")

df["gps_time"] = df["unit2_gps1"].apply(parse_time)
df["pwr_time"] = df["unit1_pwr3"].apply(parse_time)

df["offset_sec"] = (
    df["gps_time"] - df["pwr_time"]
).dt.total_seconds()

# Preserve sequence order.
df = df.sort_values(["seq_index", "abs_index"]).copy()

# Identify unusually large offsets.
outliers = df[df["offset_sec"].abs() > 2].copy()

# Show neighbouring records for each outlier.
parts = []

for idx in outliers.index:
    seq = df.loc[idx, "seq_index"]
    pos = df.index.get_loc(idx)

    start = max(0, pos - 2)
    end = min(len(df), pos + 3)

    # Limit neighbours to the same sequence.
    neighbours = df.iloc[start:end]
    neighbours = neighbours[neighbours["seq_index"] == seq]

    parts.append(neighbours)

if parts:
    detail = pd.concat(parts).drop_duplicates()
else:
    detail = pd.DataFrame(columns=df.columns)

# Offset distribution by sequence.
distribution = df.groupby("seq_index").agg(
    rows=("offset_sec", "size"),
    median_sec=("offset_sec", "median"),
    over_1_sec=("offset_sec", lambda x: (x.abs() > 1).sum()),
    over_2_sec=("offset_sec", lambda x: (x.abs() > 2).sum()),
    over_5_sec=("offset_sec", lambda x: (x.abs() > 5).sum()),
    over_30_sec=("offset_sec", lambda x: (x.abs() > 30).sum()),
    minimum_sec=("offset_sec", "min"),
    maximum_sec=("offset_sec", "max"),
).reset_index()

columns = [
    "abs_index",
    "seq_index",
    "timestamp",
    "unit1_pwr3",
    "unit2_gps1",
    "offset_sec",
    "unit2_gps1_lat",
    "unit2_gps1_lon",
    "unit1_pwr3_best-beam",
]

outliers[columns].to_csv(
    OUT / "large_offset_records.csv", index=False
)

detail[columns].to_csv(
    OUT / "outlier_neighbour_records.csv", index=False
)

distribution.to_csv(
    OUT / "offset_distribution_by_sequence.csv", index=False
)

print("OFFSET DISTRIBUTION")
print(distribution.to_string(index=False))

print("\nNUMBER OF RECORDS WITH OFFSET > 2 SECONDS:")
print(len(outliers))

print("\nLARGEST OFFSET RECORDS:")
print(
    outliers.nlargest(10, "offset_sec")[columns]
    .to_string(index=False)
)

print(f"\nSaved to: {OUT}")
