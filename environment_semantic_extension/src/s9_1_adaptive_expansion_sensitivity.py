#!/usr/bin/env python3
"""
S9.1 — Adaptive expanded-budget sensitivity experiment.

Replays the S5.1 sequential adaptive policy on the existing S6 beam-gain
dataset, varying ONLY the expanded angular candidate count. The initial
acquisition, local candidate count, gain-drop threshold, residual threshold,
hysteresis, angular-distance calculation, and fallback rule remain unchanged.

This is CPU-only post-processing. It does not run Sionna RT.

Example (PowerShell):
  python .\src\s9_1_adaptive_expansion_sensitivity.py `
    --dataset-dir ".\reference\sionna-teammate\recovered_results\s6_physical_layer_dataset"

Default expanded budgets: 16, 20, 24, 28, 32.
Outputs:
  s9_1_adaptive_per_snapshot.csv
  s9_1_adaptive_summary.csv
  s9_1_cost_regret_frontier.csv
  s9_1_validation.txt

Important: Full-codebook gains are used only for offline oracle scoring and
the explicit full-sweep fallback, exactly as in S5.1. The adaptive selection
decision uses gains only for the candidates in the current search set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

N_BEAMS = 52
LOCAL_K = 9
GAIN_DROP_DB = 8.0
RESIDUAL_DROP_DB = 20.0
HYSTERESIS_DB = 3.0
DEFAULT_EXPANDED_BUDGETS = [16, 20, 24, 28, 32]
EPS = np.finfo(np.float64).tiny


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir", type=Path, default=Path.cwd())
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--expanded-budgets", type=int, nargs="+",
                   default=DEFAULT_EXPANDED_BUDGETS)
    return p.parse_args()


def to_db(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), EPS))


def load_data(dataset_dir: Path):
    metadata = pd.read_csv(dataset_dir / "snapshot_metadata.csv")
    measurements = pd.read_csv(dataset_dir / "beam_measurements.csv")
    codebook = pd.read_csv(dataset_dir / "beam_codebook.csv")

    req_meta = {"condition", "snapshot"}
    req_gain = {"condition", "snapshot", "beam_id",
                "beamformed_channel_gain_linear"}
    req_code = {"beam_id", "u_x_local", "u_y_local", "u_z_local"}
    for label, frame, required in [
        ("snapshot_metadata.csv", metadata, req_meta),
        ("beam_measurements.csv", measurements, req_gain),
        ("beam_codebook.csv", codebook, req_code),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")

    if metadata.duplicated(["condition", "snapshot"]).any():
        raise ValueError("Duplicate metadata condition/snapshot rows.")
    if measurements.duplicated(["condition", "snapshot", "beam_id"]).any():
        raise ValueError("Duplicate beam measurements.")

    codebook = codebook.sort_values("beam_id").reset_index(drop=True)
    if codebook["beam_id"].astype(int).tolist() != list(range(N_BEAMS)):
        raise ValueError("Expected codebook beam IDs 0 through 51.")
    directions = codebook[
        ["u_x_local", "u_y_local", "u_z_local"]
    ].to_numpy(float)
    if directions.shape != (N_BEAMS, 3) or not np.isfinite(directions).all():
        raise ValueError("Invalid codebook direction matrix.")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-6):
        raise ValueError("Codebook directions must be unit-normalized.")

    conditions = sorted(metadata["condition"].astype(str).unique())
    snapshots = sorted(metadata["snapshot"].astype(int).unique())
    expected = {(c, s) for c in conditions for s in snapshots}
    observed = set(zip(metadata["condition"].astype(str),
                       metadata["snapshot"].astype(int)))
    if observed != expected:
        raise ValueError("Expected paired conditions with identical snapshots.")

    pivot = measurements.pivot(
        index=["condition", "snapshot"],
        columns="beam_id",
        values="beamformed_channel_gain_linear",
    ).reindex(
        index=pd.MultiIndex.from_product(
            [conditions, snapshots], names=["condition", "snapshot"]),
        columns=range(N_BEAMS),
    )
    if pivot.isna().any().any():
        raise ValueError("Incomplete gain matrix; expected 52 beams per snapshot.")
    gains = pivot.to_numpy(float).reshape(
        len(conditions), len(snapshots), N_BEAMS)
    if not np.isfinite(gains).all() or np.any(gains < 0):
        raise ValueError("Gain matrix contains invalid values.")

    return {
        "metadata": metadata,
        "conditions": conditions,
        "snapshots": snapshots,
        "directions": directions,
        "gains": gains,
    }


def nearest_angular(directions, reference_beam, count):
    cosine = np.clip(directions @ directions[int(reference_beam)], -1.0, 1.0)
    angle = np.arccos(cosine)
    result = np.argsort(angle, kind="stable")[:int(count)].astype(int)
    if int(reference_beam) not in result:
        raise RuntimeError("Candidate set omitted its reference beam.")
    return result


def run_adaptive(data, expanded_k):
    """Exact S5.1 decision sequence, with only expanded candidate count varied."""
    output = []
    for ci, condition in enumerate(data["conditions"]):
        gains = data["gains"][ci]
        current_beam = None
        previous_selected_gain_db = None

        for si, snapshot in enumerate(data["snapshots"]):
            previous_beam = current_beam
            gain_drop_triggered = False
            expand = False
            fallback = False
            drop_db = np.nan
            residual_db = np.nan
            improvement_db = np.nan
            trigger_reason = ""

            if si == 0:
                candidates = np.arange(N_BEAMS, dtype=int)
                current_beam = int(np.argmax(gains[si]))
                stage = "initial_full_sweep"
                full_sweep = True
            else:
                current_gain_db = float(to_db(gains[si, current_beam]))
                drop_db = float(previous_selected_gain_db - current_gain_db)
                gain_drop_triggered = bool(drop_db > GAIN_DROP_DB)

                local_candidates = nearest_angular(
                    data["directions"], current_beam, LOCAL_K)
                local_best = int(
                    local_candidates[np.argmax(gains[si, local_candidates])])
                local_best_db = float(to_db(gains[si, local_best]))
                local_improvement_db = float(local_best_db - current_gain_db)
                residual_db = float(previous_selected_gain_db - local_best_db)

                expand = bool(
                    gain_drop_triggered and residual_db > RESIDUAL_DROP_DB)
                if gain_drop_triggered:
                    trigger_reason = f"same_beam_gain_drop>{GAIN_DROP_DB:.1f}dB"

                if expand:
                    trigger_reason += (
                        f";local_residual_drop>{RESIDUAL_DROP_DB:.1f}dB")
                    candidates = nearest_angular(
                        data["directions"], current_beam, expanded_k)
                    expanded_best = int(
                        candidates[np.argmax(gains[si, candidates])])
                    expanded_best_db = float(to_db(gains[si, expanded_best]))
                    improvement_db = float(expanded_best_db - current_gain_db)

                    if improvement_db >= HYSTERESIS_DB:
                        current_beam = expanded_best
                        stage = f"expanded_{expanded_k}"
                        full_sweep = False
                    else:
                        candidates = np.arange(N_BEAMS, dtype=int)
                        current_beam = int(np.argmax(gains[si]))
                        stage = "fallback_full_sweep"
                        full_sweep = True
                        fallback = True
                else:
                    candidates = local_candidates
                    improvement_db = local_improvement_db
                    if (local_best != current_beam and
                            improvement_db >= HYSTERESIS_DB):
                        current_beam = local_best
                    stage = ("local_9_after_gain_drop" if gain_drop_triggered
                             else "local_9")
                    full_sweep = False
                    if gain_drop_triggered:
                        trigger_reason += ";local_search_sufficient"

            selected_gain = float(gains[si, current_beam])
            selected_db = float(to_db(selected_gain))
            oracle_beam = int(np.argmax(gains[si]))
            oracle_db = float(to_db(gains[si, oracle_beam]))
            regret = max(0.0, oracle_db - selected_db)
            if current_beam not in candidates:
                raise RuntimeError("Selected beam was not in probed candidates.")

            output.append({
                "condition": condition,
                "snapshot": int(snapshot),
                "snapshot_index": int(si),
                "policy": f"adaptive_expanded_{expanded_k}",
                "expanded_budget": int(expanded_k),
                "selected_beam_id": int(current_beam),
                "oracle_beam_id": oracle_beam,
                "prior_beam_id": previous_beam if previous_beam is not None else np.nan,
                "selected_gain_db": selected_db,
                "oracle_gain_db": oracle_db,
                "regret_db": regret,
                "beams_probed": int(len(candidates)),
                "candidate_beam_ids": ";".join(map(str, candidates.tolist())),
                "search_stage": stage,
                "same_beam_gain_drop_db": drop_db,
                "local_residual_drop_db": residual_db,
                "gain_drop_triggered": gain_drop_triggered,
                "expanded_search": bool(expand and not fallback),
                "fallback_full_sweep": fallback,
                "trigger_reason": trigger_reason,
                "candidate_improvement_over_current_db": improvement_db,
                "oracle_in_candidates": bool(oracle_beam in candidates),
            })
            previous_selected_gain_db = selected_db

    return pd.DataFrame(output)


def summarize(per_snapshot):
    rows = []
    for (condition, policy, k), group in per_snapshot.groupby(
            ["condition", "policy", "expanded_budget"], sort=True):
        r = group["regret_db"].to_numpy(float)
        p = group["beams_probed"].to_numpy(float)
        rows.append({
            "condition": condition,
            "policy": policy,
            "expanded_budget": int(k),
            "num_snapshots": len(group),
            "total_beam_probes": int(p.sum()),
            "mean_beams_probed": float(p.mean()),
            "mean_probe_overhead_pct": float(100 * p.mean() / N_BEAMS),
            "mean_regret_db": float(r.mean()),
            "median_regret_db": float(np.median(r)),
            "p95_regret_db": float(np.percentile(r, 95)),
            "max_regret_db": float(r.max()),
            "within_1db_pct": float(100 * np.mean(r <= 1.0 + 1e-9)),
            "within_3db_pct": float(100 * np.mean(r <= 3.0 + 1e-9)),
            "expanded_search_count": int(group["expanded_search"].sum()),
            "fallback_full_sweep_count": int(group["fallback_full_sweep"].sum()),
            "oracle_in_candidate_set_pct": float(
                100 * group["oracle_in_candidates"].mean()),
        })
    return pd.DataFrame(rows)


def import_s51_baseline(dataset_dir):
    path = dataset_dir / "tracking_records.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    required = {"condition", "snapshot", "method", "regret_db", "beams_probed"}
    if not required.issubset(frame.columns):
        return None
    frame = frame[frame["method"].astype(str) == "adaptive_tracking"].copy()
    if frame.empty:
        return None
    rows = []
    for condition, group in frame.groupby("condition"):
        r = group["regret_db"].to_numpy(float)
        p = group["beams_probed"].to_numpy(float)
        rows.append({
            "condition": condition,
            "policy": "S5.1_original_adaptive_32",
            "expanded_budget": 32,
            "num_snapshots": len(group),
            "total_beam_probes": int(p.sum()),
            "mean_beams_probed": float(p.mean()),
            "mean_probe_overhead_pct": float(100 * p.mean() / N_BEAMS),
            "mean_regret_db": float(r.mean()),
            "median_regret_db": float(np.median(r)),
            "p95_regret_db": float(np.percentile(r, 95)),
            "max_regret_db": float(r.max()),
            "within_1db_pct": float(100 * np.mean(r <= 1.0 + 1e-9)),
            "within_3db_pct": float(100 * np.mean(r <= 3.0 + 1e-9)),
            "expanded_search_count": int(
                group.get("expanded_search", pd.Series(False, index=group.index)).astype(bool).sum()),
            "fallback_full_sweep_count": int(
                group.get("fallback_full_sweep", pd.Series(False, index=group.index)).astype(bool).sum()),
            "oracle_in_candidate_set_pct": np.nan,
        })
    return pd.DataFrame(rows)


def pareto_frontier(summary):
    rows = []
    for condition, group in summary.groupby("condition"):
        for idx, row in group.iterrows():
            dominated = (
                (group["mean_beams_probed"] <= row["mean_beams_probed"]) &
                (group["mean_regret_db"] <= row["mean_regret_db"]) &
                ((group["mean_beams_probed"] < row["mean_beams_probed"]) |
                 (group["mean_regret_db"] < row["mean_regret_db"]))
            ).any()
            rows.append({
                "condition": condition,
                "policy": row["policy"],
                "expanded_budget": row["expanded_budget"],
                "mean_beams_probed": row["mean_beams_probed"],
                "mean_regret_db": row["mean_regret_db"],
                "is_pareto_efficient": not bool(dominated),
            })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    base_dir = args.base_dir.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve()
                  if args.output_dir else base_dir / "s9_1_adaptive_analysis")
    budgets = sorted(set(args.expanded_budgets))
    if not budgets or any(k < LOCAL_K or k > N_BEAMS for k in budgets):
        raise ValueError("Expanded budgets must be between 9 and 52.")

    data = load_data(dataset_dir)
    per_snapshot = pd.concat(
        [run_adaptive(data, k) for k in budgets], ignore_index=True)
    summary = summarize(per_snapshot)

    baseline = import_s51_baseline(dataset_dir)
    if baseline is not None:
        summary = pd.concat([summary, baseline], ignore_index=True)

    frontier = pareto_frontier(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_snapshot.to_csv(output_dir / "s9_1_adaptive_per_snapshot.csv", index=False)
    summary.to_csv(output_dir / "s9_1_adaptive_summary.csv", index=False)
    frontier.to_csv(output_dir / "s9_1_cost_regret_frontier.csv", index=False)

    # Core structural and sequential checks.
    checks = {
        "two_paired_conditions": len(data["conditions"]) == 2,
        "11_snapshots_per_condition": len(data["snapshots"]) == 11,
        "52_codebook_directions": data["directions"].shape == (52, 3),
        "finite_complete_gain_tensor": bool(
            data["gains"].shape == (2, 11, 52) and
            np.isfinite(data["gains"]).all()),
        "one_initial_full_acquisition_per_condition_and_budget": bool(
            (per_snapshot[per_snapshot["search_stage"] == "initial_full_sweep"]
             .groupby(["condition", "policy"]).size() == 1).all()),
        "selected_beam_in_probed_set": bool(per_snapshot.apply(
            lambda r: int(r.selected_beam_id) in
            [int(x) for x in str(r.candidate_beam_ids).split(";")], axis=1).all()),
        "nonnegative_regret": bool((per_snapshot.regret_db >= -1e-9).all()),
        "expanded_search_requires_both_thresholds": bool(
            per_snapshot.loc[per_snapshot.expanded_search,
                "same_beam_gain_drop_db"].gt(GAIN_DROP_DB).all() and
            per_snapshot.loc[per_snapshot.expanded_search,
                "local_residual_drop_db"].gt(RESIDUAL_DROP_DB).all()),
        "fallback_only_after_expansion_trigger": bool(
            per_snapshot.loc[per_snapshot.fallback_full_sweep,
                "gain_drop_triggered"].all()),
        "candidate_count_matches_budget_or_full_sweep": bool(
            per_snapshot.apply(
                lambda r: int(r.beams_probed) ==
                (N_BEAMS if r.search_stage in
                 ("initial_full_sweep", "fallback_full_sweep")
                 else (int(r.expanded_budget) if r.expanded_search else LOCAL_K)),
                axis=1).all()),
    }

    # Exact parity test: expanded budget 32 should reproduce stored S5.1
    # selected beams, probe counts, stages, and regrets for every sample.
    original_path = dataset_dir / "tracking_records.csv"
    if 32 in budgets and original_path.is_file():
        original = pd.read_csv(original_path)
        original = original[original.method.astype(str) == "adaptive_tracking"]
        replay = per_snapshot[per_snapshot.expanded_budget == 32]
        merged = replay.merge(
            original[["condition", "snapshot", "selected_beam_id",
                      "beams_probed", "regret_db"]],
            on=["condition", "snapshot"], suffixes=("_replay", "_original"),
            validate="one_to_one")
        checks["budget_32_reproduces_s5_1_selected_beams"] = bool(
            len(merged) == len(replay) and
            (merged.selected_beam_id_replay == merged.selected_beam_id_original).all())
        checks["budget_32_reproduces_s5_1_probe_counts"] = bool(
            len(merged) == len(replay) and
            (merged.beams_probed_replay == merged.beams_probed_original).all())
        checks["budget_32_reproduces_s5_1_regret"] = bool(
            len(merged) == len(replay) and
            np.allclose(merged.regret_db_replay, merged.regret_db_original,
                        atol=1e-6, rtol=0))
    else:
        checks["budget_32_parity_test_available"] = False

    lines = [
        "S9.1 ADAPTIVE EXPANSION-BUDGET SENSITIVITY VALIDATION",
        "=" * 68,
        f"Dataset: {dataset_dir}",
        f"Conditions: {', '.join(data['conditions'])}",
        f"Snapshots per condition: {len(data['snapshots'])}",
        f"Expanded budgets: {budgets}",
        "Policy: S5.1 sequential logic; only expanded candidate count varies.",
        "",
    ]
    lines += [f"[{'PASS' if ok else 'FAIL'}] {name}"
              for name, ok in checks.items()]
    lines += [
        "",
        f"OVERALL: {'PASS' if all(checks.values()) else 'FAIL'}",
        "",
        "Interpretation limits:",
        "- CPU-only replay of existing S6 Sionna beam-gain measurements.",
        "- One scene, one quasi-static trajectory, 11 positions, two conditions.",
        "- Candidate counts are not measured latency or energy.",
        "- Full-codebook gains are used for offline oracle scoring and S5.1's",
        "  specified full-sweep fallback, not ordinary candidate selection.",
        "- This sensitivity experiment does not establish generalization.",
    ]
    (output_dir / "s9_1_validation.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    print("S9.1 completed.")
    print("Dataset:", dataset_dir)
    print("Output:", output_dir)
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nValidation:")
    print("\n".join(lines[-len(checks)-2:]))
    if not all(checks.values()):
        raise RuntimeError("S9.1 completed with failed validation checks.")


if __name__ == "__main__":
    main()
