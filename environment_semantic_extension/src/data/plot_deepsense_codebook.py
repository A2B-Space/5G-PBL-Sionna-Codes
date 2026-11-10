import zipfile
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument("--archive", type=Path, required=True,
                    help="Path to the DeepSense beam codebook ZIP file")
args = parser.parse_args()

archive = args.archive.resolve()
out = PROJECT_ROOT / "results" / "analysis" / "deepsense_codebook_inspection"
out.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(archive) as z:
    names = z.namelist()

    mat_files = [
        n for n in names
        if n.endswith(".mat")
        and "/codebook_beams/built_in_beam_pattern_" in n
    ]

    def beam_id(path):
        return int(Path(path).stem.split("_")[-1])

    mat_files = sorted(mat_files, key=beam_id)

    patterns = []

    for name in mat_files:
        mat = scipy.io.loadmat(z.open(name))
        
        pattern = np.asarray(mat["beam_pattern"]).squeeze().astype(float)
        pattern[0] = pattern[1]
        patterns.append(pattern)

patterns = np.asarray(patterns)

# Angle convention from the supplied MATLAB script
angles = np.linspace(-4.0, 176.0, patterns.shape[1])

peak_indices = np.argmax(patterns, axis=1)
peak_angles = angles[peak_indices]
peak_values = patterns[np.arange(64), peak_indices]

# Save numerical results
df = pd.DataFrame({
    "beam_id": np.arange(64),
    "peak_index": peak_indices,
    "peak_angle_deg": peak_angles,
    "peak_value": peak_values,
})

df.to_csv(out / "beam_peak_angles.csv", index=False)

# Heatmap
plt.figure(figsize=(13, 7))
plt.imshow(
    patterns,
    aspect="auto",
    origin="lower",
    extent=[angles[0], angles[-1], -0.5, 63.5]
)
plt.colorbar(label="Measured beam-pattern value")
plt.xlabel("Measurement angle (degrees)")
plt.ylabel("Beam ID")
plt.title("DeepSense 64-Beam Measured Pattern Heatmap")
plt.tight_layout()
plt.savefig(out / "beam_pattern_heatmap.png", dpi=200)
plt.close()

# Peak-angle progression
plt.figure(figsize=(12, 5))
plt.plot(df["beam_id"], df["peak_angle_deg"], marker="o")
plt.xlabel("Beam ID")
plt.ylabel("Peak measurement angle (degrees)")
plt.title("Peak Angle versus Beam ID")
plt.grid(True)
plt.tight_layout()
plt.savefig(out / "beam_peak_angle_ordering.png", dpi=200)
plt.close()

print(df.to_string(index=False))
print("\nSaved results to:", out.resolve())
