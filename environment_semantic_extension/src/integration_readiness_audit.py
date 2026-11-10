
from pathlib import Path
import ast
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SEMANTIC = ROOT / "results/integration/semantic_predictions_unit3_test.csv"

SIONNA = (
    ROOT / "reference/sionna-teammate/recovered_results/"
    "s6_physical_layer_dataset"
)

pred = pd.read_csv(SEMANTIC)
codebook = pd.read_csv(SIONNA / "beam_codebook.csv")
snapshots = pd.read_csv(SIONNA / "snapshot_metadata.csv")

def parse_list(value):
    return ast.literal_eval(value) if isinstance(value, str) else value

pred["top_k_beams_parsed"] = pred["top_k_beams"].apply(parse_list)

print("\nSEMANTIC DATA")
print("Rows:", len(pred))
print("True beam range:", pred.true_beam.min(), pred.true_beam.max())
print("Predicted beam range:", pred.predicted_beam.min(), pred.predicted_beam.max())
print("Unique true beams:", pred.true_beam.nunique())
print("Unique predicted beams:", pred.predicted_beam.nunique())

print("\nSIONNA CODEBOOK")
print("Rows:", len(codebook))
print("Beam range:", codebook.beam_id.min(), codebook.beam_id.max())
print("Unique IDs:", codebook.beam_id.nunique())
print("All weights unit norm:",
      (codebook.weight_norm.sub(1).abs() < 1e-5).all())

print("\nSIONNA SNAPSHOTS")
print("Rows:", len(snapshots))
print("Conditions:", snapshots.condition.unique())
print("Unique snapshots:", snapshots.snapshot.nunique())
print("Unique sample IDs:", snapshots.sample_id.nunique())

semantic_beams = set(pred.true_beam.unique())
sionna_beams = set(codebook.beam_id.unique())

print("\nNUMERIC ID OVERLAP")
print("Overlapping numeric labels:", sorted(semantic_beams & sionna_beams))
print("This overlap does NOT establish physical correspondence.")

print("\nCLASS / CODEBOOK COMPATIBILITY")
print("Semantic true labels outside Sionna range:",
      sorted(semantic_beams - sionna_beams))

print("\nSAMPLE CORRESPONDENCE")
print("Semantic sample IDs:", pred.original_sample_id.nunique())
print("Sionna sample IDs:", snapshots.sample_id.nunique())
print("Shared sample identifiers: not established")

print("\nINTEGRATION STATUS")
print("Semantic predictions: AVAILABLE")
print("Sionna codebook: AVAILABLE")
print("Sionna physical-layer results: AVAILABLE")
print("Calibrated beam mapping: NOT ESTABLISHED")
print("Shared cross-domain sample alignment: NOT ESTABLISHED")
