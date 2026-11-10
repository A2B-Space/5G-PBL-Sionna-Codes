
import argparse
import zipfile
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--archive", type=Path, required=True)
args = parser.parse_args()

root = Path(__file__).resolve().parents[2]
out = root / "results" / "analysis" / "deepsense_codebook_inspection"
out.mkdir(parents=True, exist_ok=True)

selected = [0, 1, 31, 62, 63]
patterns = {}

with zipfile.ZipFile(args.archive) as z:
    files = [
        n for n in z.namelist()
        if n.endswith(".mat")
        and "/codebook_beams/built_in_beam_pattern_" in n
    ]

    for name in files:
        beam_id = int(Path(name).stem.split("_")[-1])

        if beam_id in selected:
            mat = scipy.io.loadmat(z.open(name))
            patterns[beam_id] = np.asarray(
                mat["beam_pattern"]
            ).squeeze().astype(float)

angles = np.linspace(-4, 176, len(next(iter(patterns.values()))))

rows = []

for beam_id in selected:
    p = patterns[beam_id]

    rows.append({
        "beam_id": beam_id,
        "samples": len(p),
        "minimum": p.min(),
        "maximum": p.max(),
        "mean": p.mean(),
        "std": p.std(),
        "peak_index": np.argmax(p),
        "peak_angle": angles[np.argmax(p)],
        "dynamic_range": p.max() - p.min()
    })

    plt.plot(angles, p, label=f"Beam {beam_id}")

df = pd.DataFrame(rows)
print(df.to_string(index=False))

df.to_csv(out / "boundary_beam_raw_metrics.csv", index=False)

plt.xlabel("Measurement angle (degrees)")
plt.ylabel("Raw measured pattern value")
plt.title("Raw Boundary and Representative Beam Patterns")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out / "boundary_beam_raw_patterns.png", dpi=200)
plt.show()
