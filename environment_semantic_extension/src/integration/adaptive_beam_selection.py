import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reference" / "base-paper"

PRED_FILE = ROOT / "results" / "integration" / "semantic_predictions_unit3_test.csv"
UTILITY_FILE = BASE / "research_extension" / "physical_utility_results" / "sample_level_physical_utility.csv"
POWER_DIR = BASE / "unit1" / "pwr3"

OUT_DIR = ROOT / "results" / "integration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

pred = pd.read_csv(PRED_FILE)
utility = pd.read_csv(UTILITY_FILE)

# Join by test row, not by sample_index or original ID.
df = pred.merge(
    utility,
    on="test_row",
    how="inner",
    validate="one_to_one",
    suffixes=("_pred", "_utility")
)

if len(df) != len(pred) or len(df) != len(utility):
    raise ValueError("Prediction and physical utility rows do not match.")

if not np.array_equal(
    df["true_beam_pred"].to_numpy(),
    df["true_beam_utility"].to_numpy()
):
    raise ValueError("Ground-truth beam mismatch after joining.")

def load_power(filename):
    path = POWER_DIR / str(filename)

    if not path.exists():
        raise FileNotFoundError(f"Missing power vector: {path}")

    values = np.asarray(
        np.loadtxt(path, dtype=np.float64).reshape(-1)
    )

    if len(values) != 64:
        raise ValueError(f"Expected 64 powers in {path}, got {len(values)}")

    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"Invalid power values in {path}")

    return values

records = []

for _, row in df.iterrows():
    power = load_power(row["power_file_pwr3"])

    oracle_beam = int(np.argmax(power))
    oracle_power = float(power[oracle_beam])

    candidates = json.loads(row["top_k_beams"])

    if len(candidates) != 5:
        raise ValueError("Expected five semantic candidates.")

    candidates = [int(x) for x in candidates]

    if len(set(candidates)) != len(candidates):
        raise ValueError("Duplicate semantic beam candidates.")

    if any(x < 0 or x >= 64 for x in candidates):
        raise ValueError(f"Beam outside PWR3 codebook: {candidates}")

    # All powers below are retrospective physical measurements.
    # Candidate probing cost is represented by the number of candidates.
    strategies = {
        "Exhaustive_64": list(range(64)),
        "Semantic_Top1": candidates[:1],
        "Semantic_Top3": candidates[:3],
        "Semantic_Top5": candidates[:5],
    }

    for name, beams in strategies.items():
        selected_beam = max(beams, key=lambda b: power[b])
        selected_power = float(power[selected_beam])

        regret = (
            10.0 * np.log10(oracle_power / selected_power)
            if selected_power > 0 and oracle_power > 0
            else np.nan
        )

        records.append({
            "test_row": int(row["test_row"]),
            "original_sample_id": int(row["original_sample_id"]),
            "sweep_id": row["sweep_id"],
            "strategy": name,
            "candidate_count": len(beams),
            "selected_beam": selected_beam,
            "oracle_beam": oracle_beam,
            "selected_power": selected_power,
            "oracle_power": oracle_power,
            "regret_db": regret,
            "within_1db": regret <= 1.0,
            "within_3db": regret <= 3.0,
            "oracle_in_candidates": oracle_beam in beams,
        })

results = pd.DataFrame(records)

summary = (
    results.groupby("strategy")
    .agg(
        samples=("test_row", "count"),
        mean_regret_db=("regret_db", "mean"),
        median_regret_db=("regret_db", "median"),
        p90_regret_db=("regret_db", lambda x: x.quantile(0.90)),
        within_1db=("within_1db", "mean"),
        within_3db=("within_3db", "mean"),
        oracle_in_candidates=("oracle_in_candidates", "mean"),
        mean_candidates_measured=("candidate_count", "mean"),
    )
    .reset_index()
)

summary["beam_measurement_reduction_pct"] = (
    1.0 - summary["mean_candidates_measured"] / 64.0
) * 100.0

results.to_csv(OUT_DIR / "adaptive_beam_sample_results.csv", index=False)
summary.to_csv(OUT_DIR / "adaptive_beam_summary.csv", index=False)

print("\nADAPTIVE BEAM BASELINE EXPERIMENT COMPLETE")
print("-------------------------------------------")
print("Samples:", len(df))
print("Unique sweeps:", df["sweep_id"].nunique())
print("\nSummary:")
print(summary.round(4).to_string(index=False))

print("\nSaved:")
print(OUT_DIR / "adaptive_beam_sample_results.csv")
print(OUT_DIR / "adaptive_beam_summary.csv")
