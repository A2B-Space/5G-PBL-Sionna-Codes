import pandas as pd
from pathlib import Path

p = Path("reference/sionna-teammate/recovered_results/s6_physical_layer_dataset/tracking_records.csv")
df = pd.read_csv(p)

a = df[df["method"] == "adaptive_tracking"].copy()

def contains_beam(row, beam):
    try:
        candidates = [int(x.strip()) for x in str(row["candidate_beam_ids"]).strip("[]").split(",") if x.strip()]
        return int(beam) in candidates
    except Exception:
        return False

a["oracle_in_candidates"] = a.apply(
    lambda r: contains_beam(r, r["oracle_beam_id"]), axis=1
)

cols = [
    "condition", "snapshot", "search_stage",
    "selected_beam_id", "oracle_beam_id",
    "regret_db", "beams_probed",
    "gain_drop_triggered", "same_beam_gain_drop_db",
    "expanded_search", "fallback_full_sweep",
    "oracle_in_candidates"
]

print("\n=== ADAPTIVE TRACKER DECISION LOG ===")
print(a[cols].to_string(index=False))

print("\n=== SEARCH STAGE COUNTS ===")
print(a.groupby(["condition", "search_stage"]).size().to_string())

print("\n=== EXPANSION / FALLBACK COUNTS ===")
print(a.groupby("condition")[[
    "gain_drop_triggered", "expanded_search", "fallback_full_sweep"
]].sum().to_string())

print("\n=== SAVED RESULTS ===")
out = Path("results/analysis")
out.mkdir(parents=True, exist_ok=True)
a[cols].to_csv(out / "adaptive_decision_audit.csv", index=False)
print(out / "adaptive_decision_audit.csv")
