from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "reference" / "base-paper" / "scenario41.csv"
POWER_ROOT = ROOT / "data" / "raw" / "scenario41"
OUT = ROOT / "results" / "data_validation"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
assert len(df) == 22500, f"Unexpected row count: {len(df)}"

# Global beam order: Left, Front, Right
arrays = [
    ("unit1_pwr3", 0),
    ("unit1_pwr1", 64),
    ("unit1_pwr2", 128),
]

power_matrix = np.empty((len(df), 192), dtype=np.float32)
missing = []
invalid = []
checked = 0

for row_idx, row in df.iterrows():
    for column, offset in arrays:
        relative_path = Path(row[column])
        file_path = POWER_ROOT / relative_path

        if not file_path.is_file():
            missing.append((row_idx, column, str(file_path)))
            continue

        try:
            values = np.loadtxt(file_path, dtype=np.float32).reshape(-1)
            if values.size != 64:
                invalid.append((row_idx, column, str(file_path), values.size))
                continue
            if not np.all(np.isfinite(values)):
                invalid.append((row_idx, column, str(file_path), "non-finite"))
                continue

            power_matrix[row_idx, offset:offset + 64] = values
            checked += 1

        except Exception as exc:
            invalid.append((row_idx, column, str(file_path), str(exc)))

print("Rows:", len(df))
print("Power files successfully read:", checked)
print("Missing references:", len(missing))
print("Invalid power files:", len(invalid))

if missing:
    pd.DataFrame(missing, columns=["row", "column", "path"]).to_csv(
        OUT / "missing_power_files.csv", index=False
    )

if invalid:
    pd.DataFrame(invalid, columns=["row", "column", "path", "problem"]).to_csv(
        OUT / "invalid_power_files.csv", index=False
    )

if missing or invalid or checked != len(df) * 3:
    raise SystemExit("VALIDATION FAILED. See diagnostic CSV files.")

np.save(OUT / "scenario41_unit1_power_192.npy", power_matrix)

metadata_columns = [
    "abs_index", "timestamp", "seq_index",
    "unit1_pwr3_best-beam",
    "unit1_pwr1_best-beam",
    "unit1_pwr2_best-beam",
]
metadata = df[metadata_columns].copy()

metadata["unit1_pwr3_global_beam"] = metadata["unit1_pwr3_best-beam"]
metadata["unit1_pwr1_global_beam"] = metadata["unit1_pwr1_best-beam"] + 64
metadata["unit1_pwr2_global_beam"] = metadata["unit1_pwr2_best-beam"] + 128

metadata.to_csv(OUT / "scenario41_unit1_metadata.csv", index=False)

with open(OUT / "validation_report.txt", "w") as f:
    f.write("SCENARIO 41 UNIT 1 POWER VALIDATION\n")
    f.write(f"Rows: {len(df)}\n")
    f.write(f"Power vectors: {checked}\n")
    f.write(f"Global vector shape: {power_matrix.shape}\n")
    f.write("Beam order: pwr3 (0-63), pwr1 (64-127), pwr2 (128-191)\n")
    f.write("Missing files: 0\nInvalid files: 0\n")

print("\nVALIDATION PASSED")
print("192-beam matrix shape:", power_matrix.shape)
print("Saved outputs to:", OUT)
