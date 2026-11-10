#!/usr/bin/env python3
"""
S9.0 — CPU-only fixed-budget angular beam-tracking benchmark.

This is a new analysis stage. It does not modify or rerun S1–S8 and does not
import Sionna, Mitsuba, CUDA, or PyTorch.

For each condition, the first snapshot uses exhaustive acquisition. At every
later snapshot, the tracker uses ONLY its previously selected beam and the
known codebook directions to construct the K nearest angular candidates. It
then selects the strongest beam among those K candidates using the current
snapshot's gains for those candidates only.

Full-codebook gains are used offline to calculate oracle regret, never to
construct a candidate set or choose the selected beam.

Outputs:
  s9_fixed_budget_per_snapshot.csv
  s9_fixed_budget_summary.csv
  s9_cost_regret_frontier.csv
  s9_validation.txt

Example:
  python s9_fixed_budget_beam_tracking.py --base-dir "C:/path/to/sionna_project"
  python s9_fixed_budget_beam_tracking.py --dataset-dir "C:/path/to/s6_physical_layer_dataset"

Dependencies: numpy, pandas.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


EXPECTED_BEAMS = 52
DEFAULT_BUDGETS = [1, 3, 5, 9, 16, 24, 32, 40, 52]
EPS = np.finfo(np.float64).tiny


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-dir", type=Path, default=Path.cwd(),
        help="Project root. Used to discover the S6 dataset automatically.",
    )
    p.add_argument(
        "--dataset-dir", type=Path, default=None,
        help="Explicit S6 dataset directory; overrides --base-dir discovery.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory. Defaults to <base-dir>/s9_fixed_budget_analysis.",
    )
    p.add_argument(
        "--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS,
        help="Candidate counts to evaluate; each must be between 1 and 52.",
    )
    return p.parse_args()


def discover_dataset(base_dir: Path) -> Path:
    candidates = [
        base_dir / "s6_physical_layer_dataset",
        base_dir / "results" / "S6" / "s6_physical_layer_dataset",
        base_dir / "results" / "S6",
        base_dir / "results" / "s6_physical_layer_dataset",
    ]
    required = ["snapshot_metadata.csv", "beam_measurements.csv", "beam_codebook.csv"]
    for folder in candidates:
        if folder.is_dir() and all((folder / name).is_file() for name in required):
            return folder
    searched = "\n".join(f"  - {x}" for x in candidates)
    raise FileNotFoundError(
        "Could not locate the S6 dataset. Searched:\n"
        f"{searched}\nPass --dataset-dir with the folder containing "
        "snapshot_metadata.csv, beam_measurements.csv and beam_codebook.csv."
    )


def db10(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), EPS))


def load_dataset(dataset_dir: Path):
    metadata_path = dataset_dir / "snapshot_metadata.csv"
    gains_path = dataset_dir / "beam_measurements.csv"
    codebook_path = dataset_dir / "beam_codebook.csv"

    metadata = pd.read_csv(metadata_path)
    measurements = pd.read_csv(gains_path)
    codebook = pd.read_csv(codebook_path)

    req_meta = {"condition", "snapshot"}
    req_gain = {
        "condition", "snapshot", "beam_id", "beamformed_channel_gain_linear"
    }
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
        raise ValueError("Duplicate condition/snapshot keys in metadata.")
    if measurements.duplicated(["condition", "snapshot", "beam_id"]).any():
        raise ValueError("Duplicate condition/snapshot/beam_id measurements.")

    codebook = codebook.sort_values("beam_id").reset_index(drop=True)
    ids = codebook["beam_id"].astype(int).tolist()
    if ids != list(range(EXPECTED_BEAMS)):
        raise ValueError("Expected exactly codebook beam IDs 0 through 51.")

    directions = codebook[
        ["u_x_local", "u_y_local", "u_z_local"]
    ].to_numpy(dtype=float)
    if directions.shape != (EXPECTED_BEAMS, 3) or not np.isfinite(directions).all():
        raise ValueError("Invalid codebook direction matrix.")
    norms = np.linalg.norm(directions, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6):
        raise ValueError("Codebook direction vectors must be unit-normalized.")

    conditions = sorted(metadata["condition"].astype(str).unique().tolist())
    snapshots = sorted(metadata["snapshot"].astype(int).unique().tolist())
    expected_keys = {(c, s) for c in conditions for s in snapshots}
    observed_keys = set(
        zip(metadata["condition"].astype(str), metadata["snapshot"].astype(int))
    )
    if observed_keys != expected_keys:
        raise ValueError(
            "Conditions do not share the same complete snapshot index set; "
            "this script currently expects paired conditions."
        )

    gain_pivot = measurements.pivot(
        index=["condition", "snapshot"],
        columns="beam_id",
        values="beamformed_channel_gain_linear",
    )
    gain_pivot = gain_pivot.reindex(
        index=pd.MultiIndex.from_product(
            [conditions, snapshots], names=["condition", "snapshot"]
        ),
        columns=range(EXPECTED_BEAMS),
    )
    if gain_pivot.isna().any().any():
        raise ValueError("Incomplete beam-gain matrix; expected 52 gains per sample.")

    gains = gain_pivot.to_numpy(dtype=float).reshape(
        len(conditions), len(snapshots), EXPECTED_BEAMS
    )
    if not np.isfinite(gains).all() or np.any(gains < 0):
        raise ValueError("Beam gains must be finite and nonnegative.")

    return {
        "metadata": metadata,
        "conditions": conditions,
        "snapshots": snapshots,
        "directions": directions,
        "gains": gains,
    }


def nearest_angular_candidates(directions: np.ndarray, prior_beam: int, k: int):
    """Return K codebook entries nearest to the prior beam on the unit sphere."""
    cosines = np.clip(directions @ directions[int(prior_beam)], -1.0, 1.0)
    angles = np.arccos(cosines)
    candidates = np.argsort(angles, kind="stable")[:k].astype(int)
    if int(prior_beam) not in candidates:
        raise RuntimeError("Angular candidate set unexpectedly omitted its prior beam.")
    return candidates


def run_policy(data, k):
    rows = []
    for ci, condition in enumerate(data["conditions"]):
        gains = data["gains"][ci]
        prior_beam = None

        for si, snapshot in enumerate(data["snapshots"]):
            oracle_beam = int(np.argmax(gains[si]))
            oracle_gain_db = float(db10(gains[si, oracle_beam]))

            if si == 0:
                candidates = np.arange(EXPECTED_BEAMS, dtype=int)
                selected_beam = oracle_beam
                stage = "initial_exhaustive_acquisition"
            else:
                candidates = nearest_angular_candidates(
                    data["directions"], prior_beam, k
                )
                selected_beam = int(
                    candidates[np.argmax(gains[si, candidates])]
                )
                stage = f"prior_angular_top_{k}"

            selected_gain_db = float(db10(gains[si, selected_beam]))
            regret_db = max(0.0, oracle_gain_db - selected_gain_db)

            rows.append({
                "condition": condition,
                "snapshot": int(snapshot),
                "policy": f"fixed_angular_{k}",
                "candidate_budget": int(k),
                "search_stage": stage,
                "prior_beam_id": prior_beam if si > 0 else np.nan,
                "selected_beam_id": selected_beam,
                "oracle_beam_id": oracle_beam,
                "beams_probed": int(len(candidates)),
                "candidate_beam_ids": ";".join(map(str, candidates.tolist())),
                "selected_gain_db": selected_gain_db,
                "oracle_gain_db": oracle_gain_db,
                "regret_db": regret_db,
                "oracle_in_candidates": bool(oracle_beam in candidates),
                "within_1db": bool(regret_db <= 1.0 + 1e-9),
                "within_3db": bool(regret_db <= 3.0 + 1e-9),
            })
            # The next snapshot's prior is the beam selected by this policy.
            prior_beam = selected_beam
    return pd.DataFrame(rows)


def summarize(per_snapshot: pd.DataFrame):
    rows = []
    for (condition, policy, budget), group in per_snapshot.groupby(
        ["condition", "policy", "candidate_budget"], sort=True
    ):
        r = group["regret_db"].to_numpy(float)
        p = group["beams_probed"].to_numpy(float)
        rows.append({
            "condition": condition,
            "policy": policy,
            "candidate_budget": int(budget),
            "num_snapshots": int(len(group)),
            "total_beam_probes": int(p.sum()),
            "mean_beams_probed": float(p.mean()),
            "mean_probe_overhead_pct": float(100.0 * p.mean() / EXPECTED_BEAMS),
            "mean_regret_db": float(r.mean()),
            "median_regret_db": float(np.median(r)),
            "p95_regret_db": float(np.percentile(r, 95)),
            "max_regret_db": float(r.max()),
            "within_1db_pct": float(100.0 * np.mean(r <= 1.0 + 1e-9)),
            "within_3db_pct": float(100.0 * np.mean(r <= 3.0 + 1e-9)),
            "oracle_in_candidate_set_pct": float(
                100.0 * group["oracle_in_candidates"].mean()
            ),
        })
    return pd.DataFrame(rows)


def load_existing_baselines(dataset_dir: Path):
    path = dataset_dir / "tracking_records.csv"
    if not path.is_file():
        return None, "tracking_records.csv not present; S5.1 comparison omitted."
    frame = pd.read_csv(path)
    required = {
        "condition", "snapshot", "method", "regret_db", "beams_probed",
        "selected_beam_id", "oracle_beam_id",
    }
    if not required.issubset(frame.columns):
        return None, (
            "tracking_records.csv lacks required S5.1 comparison fields; "
            "fixed-budget results retained without imported baseline."
        )
    selected = frame[frame["method"].astype(str).isin(
        ["exhaustive_oracle", "hold_last_beam", "local_tracking", "adaptive_tracking"]
    )].copy()
    if selected.empty:
        return None, "No recognized S5.1 baseline methods found."
    out = selected[[
        "condition", "snapshot", "method", "regret_db", "beams_probed",
        "selected_beam_id", "oracle_beam_id",
    ]].copy()
    out["condition"] = out["condition"].astype(str)
    out["snapshot"] = out["snapshot"].astype(int)
    out["regret_db"] = out["regret_db"].astype(float)
    out["beams_probed"] = out["beams_probed"].astype(int)
    out["policy"] = out["method"]
    out["candidate_budget"] = out["beams_probed"]
    return out, "Imported existing S5.1 methods from tracking_records.csv."


def main():
    args = parse_args()
    base_dir = args.base_dir.expanduser().resolve()
    dataset_dir = (
        args.dataset_dir.expanduser().resolve()
        if args.dataset_dir is not None
        else discover_dataset(base_dir)
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else base_dir / "s9_fixed_budget_analysis"
    )

    budgets = sorted(set(args.budgets))
    if not budgets or any(k < 1 or k > EXPECTED_BEAMS for k in budgets):
        raise ValueError("Every candidate budget must be an integer from 1 to 52.")

    data = load_dataset(dataset_dir)
    frames = [run_policy(data, k) for k in budgets]
    per_snapshot = pd.concat(frames, ignore_index=True)
    summary = summarize(per_snapshot)

    baseline, baseline_note = load_existing_baselines(dataset_dir)
    if baseline is not None:
        baseline_rows = []
        for (condition, policy), group in baseline.groupby(
            ["condition", "policy"], sort=True
        ):
            r = group["regret_db"].to_numpy(float)
            p = group["beams_probed"].to_numpy(float)
            baseline_rows.append({
                "condition": condition,
                "policy": policy,
                "candidate_budget": np.nan,
                "num_snapshots": int(len(group)),
                "total_beam_probes": int(p.sum()),
                "mean_beams_probed": float(p.mean()),
                "mean_probe_overhead_pct": float(100.0 * p.mean() / EXPECTED_BEAMS),
                "mean_regret_db": float(r.mean()),
                "median_regret_db": float(np.median(r)),
                "p95_regret_db": float(np.percentile(r, 95)),
                "max_regret_db": float(r.max()),
                "within_1db_pct": float(100.0 * np.mean(r <= 1.0 + 1e-9)),
                "within_3db_pct": float(100.0 * np.mean(r <= 3.0 + 1e-9)),
                "oracle_in_candidate_set_pct": np.nan,
            })
        summary = pd.concat([summary, pd.DataFrame(baseline_rows)], ignore_index=True)

    # Pareto frontier: lower mean probes AND lower mean regret is better.
    frontier_rows = []
    for condition, group in summary.groupby("condition", sort=True):
        numeric = group.dropna(subset=["mean_beams_probed", "mean_regret_db"])
        for idx, row in numeric.iterrows():
            dominated = bool(
                (
                    (numeric["mean_beams_probed"] <= row["mean_beams_probed"])
                    & (numeric["mean_regret_db"] <= row["mean_regret_db"])
                    & (
                        (numeric["mean_beams_probed"] < row["mean_beams_probed"])
                        | (numeric["mean_regret_db"] < row["mean_regret_db"])
                    )
                ).any()
            )
            frontier_rows.append({
                "condition": condition,
                "policy": row["policy"],
                "candidate_budget": row["candidate_budget"],
                "mean_beams_probed": row["mean_beams_probed"],
                "mean_regret_db": row["mean_regret_db"],
                "is_pareto_efficient": not dominated,
            })
    frontier = pd.DataFrame(frontier_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_snapshot.to_csv(output_dir / "s9_fixed_budget_per_snapshot.csv", index=False)
    summary.to_csv(output_dir / "s9_fixed_budget_summary.csv", index=False)
    frontier.to_csv(output_dir / "s9_cost_regret_frontier.csv", index=False)

    checks = {
        "dataset_contains_two_conditions": len(data["conditions"]) == 2,
        "dataset_contains_11_snapshots_per_condition": len(data["snapshots"]) == 11,
        "all_52_codebook_beams_present": data["directions"].shape == (52, 3),
        "gain_matrix_complete_and_finite": bool(
            data["gains"].shape == (2, 11, 52)
            and np.isfinite(data["gains"]).all()
        ),
        "one_initial_full_acquisition_per_condition_and_policy": bool(
            (per_snapshot[per_snapshot["search_stage"] == "initial_exhaustive_acquisition"]
             .groupby(["condition", "policy"]).size() == 1).all()
        ),
        "selected_beam_always_belongs_to_probed_set": bool(
            per_snapshot.apply(
                lambda r: int(r["selected_beam_id"]) in
                [int(x) for x in str(r["candidate_beam_ids"]).split(";")],
                axis=1,
            ).all()
        ),
        "regret_nonnegative": bool((per_snapshot["regret_db"] >= -1e-9).all()),
        "full_codebook_policy_has_zero_regret": bool(
            all(
                np.allclose(
                    per_snapshot[
                        (per_snapshot["policy"] == "fixed_angular_52")
                        & (per_snapshot["condition"] == c)
                    ]["regret_db"],
                    0.0,
                    atol=1e-8,
                )
                for c in data["conditions"]
            )
        ),
    }

    lines = [
        "S9.0 FIXED-BUDGET ANGULAR BEAM-TRACKING VALIDATION",
        "=" * 68,
        f"Dataset: {dataset_dir}",
        f"Samples: {len(data['conditions']) * len(data['snapshots'])}",
        f"Conditions: {', '.join(data['conditions'])}",
        f"Snapshots per condition: {len(data['snapshots'])}",
        f"Codebook size: {len(data['directions'])}",
        f"Candidate budgets: {budgets}",
        baseline_note,
        "",
    ]
    lines.extend(
        f"[{'PASS' if ok else 'FAIL'}] {name}" for name, ok in checks.items()
    )
    lines += [
        "",
        f"OVERALL: {'PASS' if all(checks.values()) else 'FAIL'}",
        "",
        "Interpretation limits:",
        "- This is a CPU-only analysis of already generated Sionna outputs.",
        "- Current data cover one scene, one spatial quasi-static trajectory,",
        "  11 positions, and two paired obstruction conditions.",
        "- Probe counts are candidate-measurement counts, not measured latency or energy.",
        "- Regret uses the full codebook only as an offline evaluation oracle.",
        "- Fixed-budget results are a benchmark/sensitivity study, not evidence",
        "  of generalization to unseen scenes or trajectories.",
        "- No camera/semantic observations are consumed in this stage.",
    ]
    (output_dir / "s9_validation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nS9.0 completed.")
    print("Dataset:", dataset_dir)
    print("Output:", output_dir)
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nValidation:")
    print("\n".join(lines[-len(checks)-2:]))
    if not all(checks.values()):
        raise RuntimeError("S9.0 completed with failed validation checks.")


if __name__ == "__main__":
    main()
