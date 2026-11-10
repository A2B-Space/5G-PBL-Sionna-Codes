import zipfile
import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument("--archive", type=Path, required=True,
                    help="Path to the DeepSense beam codebook ZIP file")
args = parser.parse_args()

archive = args.archive.resolve()

with zipfile.ZipFile(archive) as z:
    files = [
        n for n in z.namelist()
        if n.endswith(".mat")
        and "/codebook_beams/built_in_beam_pattern_" in n
    ]

    files.sort(key=lambda p: int(Path(p).stem.split("_")[-1]))

    patterns = []
    for f in files:
        mat = scipy.io.loadmat(z.open(f))
        p = np.asarray(mat["beam_pattern"]).squeeze().astype(float)
        p[0] = p[1]
        patterns.append(p)

P = np.asarray(patterns)
angles = np.linspace(-4, 176, 211)

rows = []

for i, p in enumerate(P):
    peak_idx = int(np.argmax(p))
    threshold = p[peak_idx] / 2

    above = p >= threshold

    # Find contiguous regions above half-power
    changes = np.diff(np.r_[False, above, False].astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1

    regions = list(zip(starts, ends))

    # Select the region containing the principal peak
    main_region = next(
        (s, e) for s, e in regions if s <= peak_idx <= e
    )

    start, end = main_region

    rows.append({
        "beam_id": i,
        "peak_angle_deg": angles[peak_idx],
        "peak_value": p[peak_idx],
        "mean_value": p.mean(),
        "peak_mean_ratio": p[peak_idx] / p.mean(),
        "mainlobe_width_deg": angles[end] - angles[start],
        "half_power_regions": len(regions)
    })

df = pd.DataFrame(rows)

# Normalized adjacent pattern correlations
corr = []

for i in range(63):
    a = P[i] - P[i].mean()
    b = P[i+1] - P[i+1].mean()
    corr.append(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)))

df["next_beam_correlation"] = corr + [np.nan]

out = PROJECT_ROOT / "results" / "analysis" / "deepsense_codebook_inspection"
out.mkdir(parents=True, exist_ok=True)

df.to_csv(out / "beam_pattern_metrics_refined.csv", index=False)

print(df.to_string(index=False))

print("\nSUMMARY FOR BEAMS 1-62")
print(df[df.beam_id.between(1,62)].describe().round(3))

print("\nPOTENTIAL ANOMALIES")
print(df[df.beam_id.isin([0,63])].to_string(index=False))
