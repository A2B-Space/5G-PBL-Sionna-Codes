from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[2]
data = pd.read_csv(root / "reference/base-paper/scenario41.csv")
power = np.load(root / "results/data_validation/scenario41_unit1_power_192.npy")

arrays = [
    ("pwr3", 0, "unit1_pwr3_best-beam", "unit1_pwr3_max-pwr"),
    ("pwr1", 64, "unit1_pwr1_best-beam", "unit1_pwr1_max-pwr"),
    ("pwr2", 128, "unit1_pwr2_best-beam", "unit1_pwr2_max-pwr"),
]

print("BEAM LABEL CONSISTENCY AUDIT")
print("-" * 55)

for name, offset, label_col, max_col in arrays:
    measured = power[:, offset:offset + 64]
    measured_best = np.argmax(measured, axis=1)
    metadata_best = data[label_col].to_numpy(dtype=int)

    measured_max = np.max(measured, axis=1)
    metadata_max = data[max_col].to_numpy(dtype=float)

    label_mismatch = np.sum(measured_best != metadata_best)
    max_difference = np.abs(measured_max - metadata_max)

    print(f"\n{name}:")
    print(f"  Label mismatches: {label_mismatch}/{len(data)}")
    print(f"  Maximum power absolute difference: {max_difference.max():.8g}")
    print(f"  Mean power absolute difference: {max_difference.mean():.8g}")

print("\nAudit complete.")
