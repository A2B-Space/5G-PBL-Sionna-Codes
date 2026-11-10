"""
Utility-aware adaptive semantic beam selector — aligned integration version.

Inputs:
  results/integration/semantic_predictions_unit3_test.csv
  results/integration/adaptive_beam_sample_results.csv

This version deliberately does NOT index the 22,500-row Unit 1 power matrix.
The existing adaptive_beam_sample_results.csv is the already-aligned physical
utility evaluation for the 449 Unit 3 semantic test samples. We join it to
predictions using BOTH test_row and original_sample_id.

Run:
  python src/data/run_utility_aware_selector.py

Interpretation:
  - Semantic probabilities are selector inputs.
  - Existing per-sample candidate-set regret is the supervised target.
  - Cross-validation is for algorithm development on this 449-sample
    evaluation set, not an independent final test.
  - Candidate count is a search-overhead proxy, not measured latency/energy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = ROOT / "results/integration/semantic_predictions_unit3_test.csv"
UTILITY_PATH = ROOT / "results/integration/adaptive_beam_sample_results.csv"
OUTPUT_DIR = ROOT / "results/analysis/utility_aware_adaptive_selector"

K_VALUES = (1,  3, 5)
LAMBDA_VALUES_DB = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
RIDGE_ALPHA = 3.0
N_FOLDS = 5
SEED = 41


def ridge_predict(x_train, y_train, x_test, alpha=3.0):
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    xtr = (x_train - mean) / scale
    xte = (x_test - mean) / scale
    xtr = np.column_stack([np.ones(len(xtr)), xtr])
    xte = np.column_stack([np.ones(len(xte)), xte])
    penalty = np.eye(xtr.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(xtr.T @ xtr + penalty, xtr.T @ y_train)
    return xte @ beta


def make_features(df):
    p = np.column_stack([
        df[f"prob_{i}"].to_numpy(dtype=float) for i in range(1, 6)
    ])
    p = np.clip(p, 1e-12, 1.0)
    p_sum = np.maximum(p.sum(axis=1), 1e-12)
    p_norm = p / p_sum[:, None]
    entropy = -np.sum(p_norm * np.log(p_norm), axis=1)
    sorted_p = np.sort(p, axis=1)[:, ::-1]
    margin12 = sorted_p[:, 0] - sorted_p[:, 1]
    margin13 = sorted_p[:, 0] - sorted_p[:, 2]
    return np.column_stack([
        p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4],
        margin12, margin13,
        p[:, :2].sum(axis=1),
        p[:, :3].sum(axis=1),
        p.sum(axis=1),
        entropy,
        entropy / np.log(5.0),
        np.log(p[:, 0]),
    ])


def metrics(name, regret, count, oracle_in):
    regret = np.asarray(regret, dtype=float)
    count = np.asarray(count, dtype=float)
    oracle_in = np.asarray(oracle_in, dtype=bool)
    return {
        "method": name,
        "mean_regret_db": regret.mean(),
        "median_regret_db": np.median(regret),
        "p95_regret_db": np.percentile(regret, 95),
        "within_1db_pct": 100 * np.mean(regret <= 1),
        "within_3db_pct": 100 * np.mean(regret <= 3),
        "oracle_in_candidate_set_pct": 100 * np.mean(oracle_in),
        "mean_candidates": count.mean(),
        "candidate_reduction_pct": 100 * (1 - count.mean() / 64),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(PREDICTIONS_PATH)
    util = pd.read_csv(UTILITY_PATH)

    pred_required = {
        "test_row", "original_sample_id", "top_k_probabilities"
    }
    util_required = {
        "test_row", "original_sample_id", "strategy", "candidate_count",
        "regret_db", "oracle_in_candidates"
    }
    if pred_required - set(pred.columns):
        raise ValueError(f"Prediction file missing: {sorted(pred_required-set(pred.columns))}")
    if util_required - set(util.columns):
        raise ValueError(f"Utility file missing: {sorted(util_required-set(util.columns))}")

    # The two identifiers must agree exactly. This is the alignment that the
    # previous script failed to establish by treating IDs as matrix positions.
    if pred.duplicated(["test_row", "original_sample_id"]).any():
        raise ValueError("Duplicate prediction keys.")
    if util.duplicated(["test_row", "original_sample_id", "strategy"]).any():
        raise ValueError("Duplicate utility keys within a strategy.")

    strategies = sorted(util["strategy"].dropna().unique().tolist())
    strategy_for_k = {1: "Semantic_Top1", 3: "Semantic_Top3", 5: "Semantic_Top5"}
    missing_strategies = [v for v in strategy_for_k.values() if v not in strategies]
    if missing_strategies:
        raise ValueError(
            f"Expected candidate-set strategies not found: {missing_strategies}. "
            f"Available strategies: {strategies}"
        )

    # Pivot the precomputed physical-utility labels to one row per test sample.
    value_cols = ["regret_db", "candidate_count", "oracle_in_candidates"]
    pivoted = util.pivot(
        index=["test_row", "original_sample_id"],
        columns="strategy",
        values=value_cols,
    )
    pivoted.columns = [f"{metric}__{strategy}" for metric, strategy in pivoted.columns]
    pivoted = pivoted.reset_index()

    data = pred.merge(
        pivoted,
        on=["test_row", "original_sample_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(data) != len(pred):
        missing = pred.merge(
            data[["test_row", "original_sample_id"]],
            on=["test_row", "original_sample_id"],
            how="left",
            indicator=True,
        )
        missing = missing[missing["_merge"] == "left_only"]
        raise ValueError(
            f"Only {len(data)}/{len(pred)} predictions matched utility labels. "
            f"Unmatched examples:\n{missing.head().to_string(index=False)}"
        )

    # Parse top-K probabilities, supporting the list representation in CSV.
    import ast
    probs = []
    for i, value in enumerate(data["top_k_probabilities"]):
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
        parsed = list(parsed)
        if len(parsed) < 5:
            raise ValueError(f"Prediction row {i} has fewer than five probabilities.")
        probs.append([float(x) for x in parsed[:5]])
    probs = np.asarray(probs, dtype=float)
    for j in range(5):
        data[f"prob_{j+1}"] = probs[:, j]

    # Verify the physical labels in the existing utility table are internally
    # consistent: exhaustive must have zero regret and 64 candidates.
    exhaustive = util[util["strategy"] == "Exhaustive_64"]
    if len(exhaustive) != len(pred):
        raise ValueError(
            f"Expected {len(pred)} Exhaustive_64 rows, found {len(exhaustive)}."
        )
    if not np.allclose(exhaustive["regret_db"].to_numpy(float), 0.0, atol=1e-8):
        raise ValueError("Exhaustive_64 rows contain nonzero regret.")

    x = make_features(data)
    y = {
        k: data[f"regret_db__{strategy_for_k[k]}"].to_numpy(dtype=float)
        for k in K_VALUES
    }
    candidate_counts = {
        k: data[f"candidate_count__{strategy_for_k[k]}"].to_numpy(dtype=float)
        for k in K_VALUES
    }
    oracle_in = {
        k: data[f"oracle_in_candidates__{strategy_for_k[k]}"].astype(bool).to_numpy()
        for k in K_VALUES
    }

    # Deterministic shuffled K-fold OOF estimates. Sequence-grouped CV is not
    # used because the Unit 3 test-row -> Scenario 41 sequence mapping has not
    # been established. No fabricated mapping is introduced here.
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(data))
    folds = np.array_split(indices, min(N_FOLDS, len(data)))
    oof = {k: np.full(len(data), np.nan) for k in K_VALUES}
    fold_id = np.full(len(data), -1, dtype=int)
    for fold, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(len(data)), test_idx)
        fold_id[test_idx] = fold
        for k in K_VALUES:
            oof[k][test_idx] = ridge_predict(
                x[train_idx], y[k][train_idx], x[test_idx], RIDGE_ALPHA
            )
    if any(np.isnan(v).any() for v in oof.values()):
        raise RuntimeError("OOF predictions incomplete.")

    output_rows = []
    for k in K_VALUES:
        output_rows.append(metrics(
            f"Fixed_Top{k}", y[k], candidate_counts[k], oracle_in[k]
        ))

    # Use existing exhaustive rows for a consistent baseline.
    output_rows.append(metrics(
        "Exhaustive_64",
        exhaustive["regret_db"].to_numpy(float),
        exhaustive["candidate_count"].to_numpy(float),
        exhaustive["oracle_in_candidates"].astype(bool).to_numpy(),
    ))

    tradeoff_rows = []
    chosen_outputs = {}
    for lam in LAMBDA_VALUES_DB:
        # Normalize candidate cost by 64 beams. Lambda is a dB-equivalent
        # penalty for a full 64-beam search.
        objective = np.column_stack([
            oof[k] + lam * (candidate_counts[k] / 64.0)
            for k in K_VALUES
        ])
        choice_idx = np.argmin(objective, axis=1)
        chosen_k = np.asarray(K_VALUES)[choice_idx]
        chosen_regret = np.array([y[int(k)][i] for i, k in enumerate(chosen_k)])
        chosen_count = np.array([
            candidate_counts[int(k)][i] for i, k in enumerate(chosen_k)
        ])
        chosen_oracle = np.array([
            oracle_in[int(k)][i] for i, k in enumerate(chosen_k)
        ])
        name = f"UtilityAware_lambda_{lam:g}dB"
        output_rows.append(metrics(
            name, chosen_regret, chosen_count, chosen_oracle
        ))
        chosen_outputs[lam] = (chosen_k, chosen_regret, chosen_count, chosen_oracle)
        tradeoff_rows.append({
            "lambda_db_per_full_64_beam_search": lam,
            "mean_regret_db": chosen_regret.mean(),
            "median_regret_db": np.median(chosen_regret),
            "mean_candidates": chosen_count.mean(),
            "candidate_reduction_pct": 100 * (1 - chosen_count.mean() / 64),
            "oracle_in_candidate_set_pct": 100 * chosen_oracle.mean(),
            "within_1db_pct": 100 * np.mean(chosen_regret <= 1),
            "within_3db_pct": 100 * np.mean(chosen_regret <= 3),
            "selected_top1_pct": 100 * np.mean(chosen_k == 1),
            "selected_top3_pct": 100 * np.mean(chosen_k == 3),
            "selected_top5_pct": 100 * np.mean(chosen_k == 5),
        })

    comparison = pd.DataFrame(output_rows)
    tradeoff = pd.DataFrame(tradeoff_rows)
    per_sample = data.copy()
    per_sample["cv_fold"] = fold_id
    for k in K_VALUES:
        per_sample[f"oof_predicted_regret_top{k}_db"] = oof[k]
    for lam, (chosen_k, chosen_regret, chosen_count, chosen_oracle) in chosen_outputs.items():
        tag = str(lam).replace(".", "p")
        per_sample[f"chosen_k_lambda_{tag}"] = chosen_k
        per_sample[f"chosen_regret_lambda_{tag}_db"] = chosen_regret

    comparison.to_csv(OUTPUT_DIR / "method_comparison.csv", index=False)
    tradeoff.to_csv(OUTPUT_DIR / "utility_cost_tradeoff.csv", index=False)
    per_sample.to_csv(OUTPUT_DIR / "per_sample_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in comparison.iterrows():
        ax.scatter(row["mean_candidates"], row["mean_regret_db"])
        ax.annotate(row["method"], (row["mean_candidates"], row["mean_regret_db"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean candidate beams")
    ax.set_ylabel("Mean candidate-set regret (dB)")
    ax.set_title("Utility-aware adaptive semantic beam selection")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "regret_vs_candidate_count.png", dpi=180)
    plt.close(fig)

    notes = [
        "UTILITY-AWARE ADAPTIVE SEMANTIC BEAM SELECTION",
        "==============================================",
        f"Matched prediction/utility samples: {len(data)}",
        "Join keys: test_row + original_sample_id",
        "Utility source: existing adaptive_beam_sample_results.csv",
        "Cross-validation: deterministic shuffled 5-fold OOF",
        "Model: standardized ridge regression",
        f"Ridge alpha: {RIDGE_ALPHA}",
        f"Cost penalties: {LAMBDA_VALUES_DB}",
        "",
        "The previous attempt incorrectly treated original_sample_id as a row",
        "index into the 22,500-row Unit 1 power matrix. This version does not.",
        "The existing per-sample utility table is joined using the two identifiers",
        "shared with the semantic prediction export.",
        "",
        "Caveats:",
        "- CV is for development on the existing 449-sample evaluation set.",
        "- It is not a final independent generalization estimate.",
        "- Candidate count is a proxy, not measured latency or energy.",
        "- Candidate-set regret assumes the strongest measured candidate can be",
        "  selected; this is not a free receiver-side operation.",
        "",
        "METHOD COMPARISON",
        comparison.to_string(index=False),
        "",
        "UTILITY-COST TRADEOFF",
        tradeoff.to_string(index=False),
    ]
    (OUTPUT_DIR / "experiment_notes.txt").write_text("\n".join(notes), encoding="utf-8")

    print(f"Matched samples: {len(data)} / {len(pred)}")
    print("Join keys: test_row + original_sample_id")
    print("\n=== METHOD COMPARISON ===")
    print(comparison.to_string(index=False))
    print("\n=== UTILITY-COST TRADEOFF ===")
    print(tradeoff.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
