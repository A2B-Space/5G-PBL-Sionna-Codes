import zipfile
import scipy.io
import numpy as np
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument("--archive", type=Path, required=True,
                    help="Path to the DeepSense beam codebook ZIP file")
args = parser.parse_args()

archive = args.archive.resolve()
out = PROJECT_ROOT / "data" / "processed" / "deepsense_beam_codebook"
out.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(archive) as z:
    z.extractall(out)

files = sorted(
    (out / "DeepSense_beam_codebook" / "codebook_beams").glob("*.mat"),
    key=lambda p: int(p.stem.split("_")[-1])
)

print("Number of beam files:", len(files))

patterns = []
for f in files:
    mat = scipy.io.loadmat(f)
    pattern = np.asarray(mat["beam_pattern"]).squeeze()
    patterns.append(pattern)

patterns = np.asarray(patterns)

print("Pattern matrix shape:", patterns.shape)
print("All finite:", np.isfinite(patterns).all())
print("Minimum:", patterns.min())
print("Maximum:", patterns.max())
print("First five beam maxima:", np.argmax(patterns, axis=1)[:5])
print("Peak measurement indices:", np.argmax(patterns, axis=1))
