import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

BASE = ROOT / "reference" / "base-paper"
LOGITS_FILE = BASE / "LSTM_bbox" / "unit3" / "LSTM_pred_beam_unit3_logits.csv"
TEST_FILE = BASE / "LSTM_bbox" / "unit3" / "bbox_test_unit3.csv"

OUTPUT = ROOT / "results" / "integration" / "semantic_predictions_unit3_test.csv"

TOP_K = 5

# Load saved LSTM logits
logits_df = pd.read_csv(LOGITS_FILE)

logit_columns = [f"logit_{i}" for i in range(192)]

required = ["sample_index", "true_beam"] + logit_columns
missing = [c for c in required if c not in logits_df.columns]

if missing:
    raise ValueError(f"Missing columns in logits CSV: {missing}")

# Load original headerless test CSV
test_df = pd.read_csv(TEST_FILE, header=None)

if len(logits_df) != len(test_df):
    raise ValueError(
        f"Row mismatch: logits={len(logits_df)}, test={len(test_df)}"
    )

scores = logits_df[logit_columns].to_numpy(dtype=np.float64)
true_beams = logits_df["true_beam"].to_numpy(dtype=int)
test_labels = test_df.iloc[:, 2].to_numpy(dtype=int)

# Verify row-wise label alignment
if not np.array_equal(true_beams, test_labels):
    mismatch = np.where(true_beams != test_labels)[0]
    raise ValueError(
        f"Label mismatch at rows: {mismatch[:10].tolist()}"
    )

# Numerically stable softmax
shifted = scores - scores.max(axis=1, keepdims=True)
exp_scores = np.exp(shifted)
probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)

# Rank beams from highest to lowest score
ranked = np.argsort(-scores, axis=1, kind="stable")
top_k_beams = ranked[:, :TOP_K]
top_k_probabilities = np.take_along_axis(
    probabilities, top_k_beams, axis=1
)

output = pd.DataFrame({
    "test_row": np.arange(len(test_df)),
    "original_sample_id": test_df.iloc[:, 0].to_numpy(),
    "sample_index": logits_df["sample_index"].to_numpy(),
    "true_beam": true_beams,
    "predicted_beam": top_k_beams[:, 0],
    "top1_probability": top_k_probabilities[:, 0],
    "top_k_beams": [
        json.dumps(row.tolist()) for row in top_k_beams
    ],
    "top_k_probabilities": [
        json.dumps(row.tolist()) for row in top_k_probabilities
    ],
})

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
output.to_csv(OUTPUT, index=False)

top1_accuracy = np.mean(output["predicted_beam"] == true_beams)
top5_accuracy = np.mean([
    true in candidates
    for true, candidates in zip(true_beams, top_k_beams)
])

print("SEMANTIC PREDICTION EXPORT SUCCESSFUL")
print("-------------------------------------")
print("Samples:", len(output))
print("Top-1 accuracy:", round(top1_accuracy * 100, 2), "%")
print("Top-5 accuracy:", round(top5_accuracy * 100, 2), "%")
print("Output:", OUTPUT)
print("\nFirst five predictions:")
print(output.head().to_string(index=False))

