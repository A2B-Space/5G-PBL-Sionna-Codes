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

peaks = np.argmax(P, axis=1)
peak_angles = angles[peaks]

rows = []

for i in range(64):
    p = P[i]
    peak = p.max()

    # Half-power beamwidth: -3 dB in power
    mask = p >= peak / 2

    bw = angles[mask][-1] - angles[mask][0]

    rows.append({
        "beam_id": i,
        "peak_angle_deg": peak_angles[i],
        "peak_value": peak,
        "mean_value": p.mean(),
        "beamwidth_3db_deg": bw
    })

df = pd.DataFrame(rows)

print("\nADJACENT BEAM PEAK SEPARATION")
print(np.round(np.diff(peak_angles), 3))

print("\nADJACENT PATTERN CORRELATION")
corr = [
    np.corrcoef(P[i], P[i+1])[0,1]
    for i in range(63)
]
print(np.round(corr, 3))

print("\nBEAMWIDTH AND POWER SUMMARY")
print(df.to_string(index=False))

print("\nWEAKEST BEAMS BY PEAK VALUE")
print(df.nsmallest(8, "peak_value").to_string(index=False))

out = PROJECT_ROOT / "results" / "analysis" / "deepsense_codebook_inspection"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "beam_pattern_metrics.csv", index=False)

