import pandas as pd
from pathlib import Path

p = Path(
    "reference/sionna-teammate/recovered_results/"
    "s6_physical_layer_dataset/beam_codebook.csv"
)

df = pd.read_csv(p)

print("COLUMNS:")
print(df.columns.tolist())

print("\nSHAPE:", df.shape)
print("\nFIRST 15 BEAMS:")
print(df.head(15).to_string(index=False))

print("\nNUMERIC SUMMARY:")
print(df.describe().to_string())

print("\nDIRECTION / ANGLE COLUMNS:")
for c in df.columns:
    if any(k in c.lower() for k in ["angle", "azimuth", "elevation", "direction"]):
        print(f"\n{c}")
        print(df[c].describe())
