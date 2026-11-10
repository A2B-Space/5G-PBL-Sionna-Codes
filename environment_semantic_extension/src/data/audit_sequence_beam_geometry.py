
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "synchronized_scenario41" / "synchronized_geometry_samples.csv"
OUT = ROOT / "results" / "sequence_geometry_analysis"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

def circular_stats(degrees):
    r = np.radians(np.asarray(degrees, dtype=float))
    s, c = np.sin(r).mean(), np.cos(r).mean()
    mean = (np.degrees(np.arctan2(s, c)) + 360) % 360
    resultant = np.hypot(s, c)
    circular_std = np.degrees(np.sqrt(-2 * np.log(max(resultant, 1e-12))))
    return mean, resultant, circular_std

rows = []

for (seq, beam), group in df.groupby(
    ["seq_index", "unit1_pwr3_best-beam"]
):
    mean, resultant, cstd = circular_stats(
        group["corrected_bearing_deg"]
    )
    rows.append({
        "seq_index": seq,
        "beam": beam,
        "samples": len(group),
        "mean_bearing_deg": mean,
        "resultant_length": resultant,
        "circular_std_deg": cstd,
    })

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "beam_bearing_by_sequence.csv", index=False)

# Per-sequence overall bearing and beam coverage.
sequence_rows = []

for seq, group in df.groupby("seq_index"):
    mean, resultant, cstd = circular_stats(
        group["corrected_bearing_deg"]
    )
    sequence_rows.append({
        "seq_index": seq,
        "samples": len(group),
        "unique_beams": group["unit1_pwr3_best-beam"].nunique(),
        "mean_bearing_deg": mean,
        "bearing_resultant_length": resultant,
        "bearing_circular_std_deg": cstd,
    })

sequence_summary = pd.DataFrame(sequence_rows)
sequence_summary.to_csv(
    OUT / "sequence_bearing_summary.csv", index=False
)

print("SEQUENCE SUMMARY")
print(sequence_summary.to_string(index=False))

print("\nMOST POPULATED BEAMS PER SEQUENCE")
for seq, group in df.groupby("seq_index"):
    print(f"\nSequence {seq}")
    print(
        group["unit1_pwr3_best-beam"]
        .value_counts()
        .head(10)
        .to_string()
    )

print("\nSaved to:", OUT)
