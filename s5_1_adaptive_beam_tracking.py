#!/usr/bin/env python3
"""
S5.1: Adaptive beam reacquisition with automatic research-plot generation.

The script reuses the validated S4.1 beam-gain table and the S5 baseline
tracking output. It does not rerun Sionna RT.

Adaptive policy
---------------
- Initial snapshot: exhaustive 52-beam acquisition.
- Normal operation: probe the 9 codebook beams nearest in angle to the
  currently selected beam.
- Degradation detector: compare the current selected beam's gain at the
  current spatial snapshot with its gain at the previous snapshot.
- If that same-beam gain falls by more than 8 dB, inspect the best local
  candidate. Expand to 32 angularly nearest beams only if the local
  candidate remains more than 20 dB below the previous selected-beam gain.
- If the expanded search cannot improve on the current beam by at least
  3 dB, fall back to a full 52-beam reacquisition.
- The policy uses only the gains of beams in its candidate set to choose
  a beam. Full-codebook gains are used offline for evaluation only.

Important experimental scope
----------------------------
The inherited S2/S4 trajectory consists of spatial quasi-static snapshots.
This is not a continuous-time, Doppler, or measured-latency simulation.
Beam-probe counts are counts of candidate-beam measurements, not wall-clock
time or energy measurements.

Inputs
------
<base-dir>/s4_blockage_geometry/s4_beam_metrics.csv
<base-dir>/s4_blockage_geometry/s4_beam_codebook.csv
<base-dir>/s4_blockage_geometry/s4_snapshot_summary.csv
<base-dir>/s5_beam_tracking/s5_tracking_per_snapshot.csv

Outputs
-------
<base-dir>/s5_1_adaptive_tracking/
    s5_1_tracking_per_snapshot.csv
    s5_1_tracking_summary.csv
    s5_1_beam_switches.csv
    s5_1_search_events.csv
    s5_1_config.json
    s5_1_validation.txt
    plots/
        01_regret_vs_position.{png,pdf}
        02_selected_beam_vs_position.{png,pdf}
        03_beam_probes_per_snapshot.{png,pdf}
        04_cumulative_beam_probes.{png,pdf}
        05_training_overhead_vs_regret.{png,pdf}
        06_adaptive_trigger_diagnostics.{png,pdf}
        07_regret_heatmap.{png,pdf}
        s5_1_plot_report.pdf

Run
---
cd ~/sionna_project
source .venv/bin/activate
python s5_1_adaptive_beam_tracking.py

Optional:
python s5_1_adaptive_beam_tracking.py --base-dir /path/to/sionna_project

Dependencies: numpy, pandas, matplotlib.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ======================================================================
# 1. CONFIGURATION
# ======================================================================

NUM_EXPECTED_BEAMS = 52
LOCAL_CANDIDATE_COUNT = 9
EXPANDED_CANDIDATE_COUNT = 32
GAIN_DROP_TRIGGER_DB = 8.0
RESIDUAL_DROP_TRIGGER_DB = 20.0
HYSTERESIS_DB = 3.0

METHOD_ORACLE = "exhaustive_oracle"
METHOD_HOLD = "hold_last_beam"
METHOD_FIXED = "local_tracking"
METHOD_ADAPTIVE = "adaptive_tracking"

METHOD_ORDER = [
    METHOD_ORACLE,
    METHOD_HOLD,
    METHOD_FIXED,
    METHOD_ADAPTIVE,
]

METHOD_LABELS = {
    METHOD_ORACLE: "Exhaustive oracle",
    METHOD_HOLD: "Hold-last beam",
    METHOD_FIXED: "S5 fixed local tracker",
    METHOD_ADAPTIVE: "S5.1 adaptive tracker",
}

GAIN_FLOOR = np.finfo(np.float64).tiny
PLOT_DPI = 220


def parse_args():
    parser = argparse.ArgumentParser(
        description="S5.1 adaptive beam tracking and saved plot generation."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.home() / "sionna_project",
        help=(
            "Project directory containing s4_blockage_geometry/ and "
            "s5_beam_tracking/. Defaults to ~/sionna_project."
        ),
    )
    return parser.parse_args()


def gain_to_db(gain):
    return 10.0 * np.log10(np.maximum(np.asarray(gain, dtype=float), GAIN_FLOOR))


def nearest_angular_beams(directions, reference_beam_id, count):
    """Return the `count` nearest unit-direction codebook entries."""
    if not 0 <= int(reference_beam_id) < len(directions):
        raise ValueError(f"Invalid reference beam ID: {reference_beam_id}")
    if not 1 <= int(count) <= len(directions):
        raise ValueError(f"Invalid candidate count: {count}")

    reference = directions[int(reference_beam_id)]
    cosine = np.clip(directions @ reference, -1.0, 1.0)
    angular_distance = np.arccos(cosine)
    order = np.argsort(angular_distance, kind="stable")
    candidates = order[:int(count)].astype(int)

    if int(reference_beam_id) not in candidates:
        raise RuntimeError("Nearest-beam candidate set omitted its reference beam.")
    return candidates


# ======================================================================
# 2. LOAD AND VALIDATE S4.1 / S5 INPUTS
# ======================================================================

def load_inputs(base_dir):
    input_s4 = base_dir / "s4_blockage_geometry"
    input_s5 = base_dir / "s5_beam_tracking"

    paths = {
        "beam_metrics": input_s4 / "s4_beam_metrics.csv",
        "codebook": input_s4 / "s4_beam_codebook.csv",
        "snapshot_summary": input_s4 / "s4_snapshot_summary.csv",
        "s5_tracking": input_s5 / "s5_tracking_per_snapshot.csv",
    }

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required S4.1/S5 input files:\n - "
            + "\n - ".join(missing)
            + "\nConfirm that the original S4.1 and S5 output folders are present."
        )

    beam_metrics = pd.read_csv(paths["beam_metrics"])
    codebook = pd.read_csv(paths["codebook"])
    snapshot_summary = pd.read_csv(paths["snapshot_summary"])
    s5_tracking = pd.read_csv(paths["s5_tracking"])

    required_metric_columns = {
        "condition",
        "snapshot",
        "rx_x_m",
        "rx_y_m",
        "rx_z_m",
        "beam_id",
        "beamformed_channel_gain_linear",
    }
    missing_columns = required_metric_columns - set(beam_metrics.columns)
    if missing_columns:
        raise ValueError(
            f"s4_beam_metrics.csv is missing columns: {sorted(missing_columns)}"
        )

    required_codebook_columns = {
        "beam_id", "u_x_local", "u_y_local", "u_z_local"
    }
    missing_columns = required_codebook_columns - set(codebook.columns)
    if missing_columns:
        raise ValueError(
            f"s4_beam_codebook.csv is missing columns: {sorted(missing_columns)}"
        )

    required_s5_columns = {
        "condition",
        "snapshot",
        "snapshot_index",
        "rx_x_m",
        "rx_y_m",
        "rx_z_m",
        "method",
        "selected_beam_id",
        "selected_gain_linear",
        "selected_gain_db",
        "oracle_beam_id",
        "oracle_gain_db",
        "regret_db",
        "beam_switched",
        "is_full_sweep",
        "beams_probed",
        "candidate_beam_ids",
    }
    missing_columns = required_s5_columns - set(s5_tracking.columns)
    if missing_columns:
        raise ValueError(
            "s5_tracking_per_snapshot.csv is missing columns: "
            f"{sorted(missing_columns)}"
        )

    codebook = codebook.sort_values("beam_id").reset_index(drop=True)
    if codebook["beam_id"].astype(int).tolist() != list(range(NUM_EXPECTED_BEAMS)):
        raise ValueError(
            "S4.1 codebook must contain beam IDs 0 through 51 exactly once."
        )

    directions = codebook[
        ["u_x_local", "u_y_local", "u_z_local"]
    ].to_numpy(dtype=float)

    if directions.shape != (NUM_EXPECTED_BEAMS, 3):
        raise ValueError(f"Unexpected beam-direction shape: {directions.shape}")
    if not np.all(np.isfinite(directions)):
        raise ValueError("Non-finite codebook directions found.")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-7):
        raise ValueError("At least one codebook direction is not unit norm.")

    if beam_metrics.duplicated(["condition", "snapshot", "beam_id"]).any():
        raise ValueError("Duplicate condition/snapshot/beam rows in beam metrics.")

    conditions = ["unblocked", "blocked"]
    if set(beam_metrics["condition"].astype(str).unique()) != set(conditions):
        raise ValueError("Expected exactly unblocked and blocked S4.1 conditions.")

    snapshots = sorted(beam_metrics["snapshot"].astype(int).unique().tolist())
    if len(snapshots) < 2:
        raise ValueError("At least two spatial snapshots are required.")

    pivot = beam_metrics.pivot(
        index=["condition", "snapshot"],
        columns="beam_id",
        values="beamformed_channel_gain_linear",
    )
    pivot = pivot.reindex(
        index=pd.MultiIndex.from_product(
            [conditions, snapshots], names=["condition", "snapshot"]
        ),
        columns=range(NUM_EXPECTED_BEAMS),
    )

    if pivot.isna().any().any():
        raise ValueError(
            "Beam metrics do not contain a complete 2 × snapshots × 52 matrix."
        )

    gain_tensor = pivot.to_numpy(dtype=float).reshape(
        len(conditions), len(snapshots), NUM_EXPECTED_BEAMS
    )

    if not np.all(np.isfinite(gain_tensor)) or np.any(gain_tensor < 0.0):
        raise ValueError("Beam-gain matrix contains invalid values.")

    position_columns = ["condition", "snapshot", "rx_x_m", "rx_y_m", "rx_z_m"]
    if not set(position_columns).issubset(snapshot_summary.columns):
        raise ValueError(
            "s4_snapshot_summary.csv must contain condition, snapshot and RX XYZ."
        )

    position_table = snapshot_summary[position_columns].drop_duplicates(
        ["condition", "snapshot"]
    )
    positions = {}
    for condition in conditions:
        part = position_table[
            position_table["condition"].astype(str) == condition
        ].set_index("snapshot").reindex(snapshots)
        if part[["rx_x_m", "rx_y_m", "rx_z_m"]].isna().any().any():
            raise ValueError(f"Missing RX coordinates for condition {condition}.")
        positions[condition] = part[
            ["rx_x_m", "rx_y_m", "rx_z_m"]
        ].to_numpy(dtype=float)

    baseline_methods = set(s5_tracking["method"].astype(str).unique())
    expected_baseline_methods = {METHOD_ORACLE, METHOD_HOLD, METHOD_FIXED}
    if baseline_methods != expected_baseline_methods:
        raise ValueError(
            "S5 baseline must contain exactly exhaustive_oracle, "
            "hold_last_beam and local_tracking."
        )

    expected_rows = len(conditions) * len(snapshots) * len(expected_baseline_methods)
    if len(s5_tracking) != expected_rows:
        raise ValueError(
            f"S5 baseline row count mismatch: expected {expected_rows}, "
            f"received {len(s5_tracking)}."
        )

    # Confirm that the S5-selected gains and oracle gains agree with the
    # original S4.1 per-beam measurement table.
    condition_index = {name: index for index, name in enumerate(conditions)}
    snapshot_index = {int(value): index for index, value in enumerate(snapshots)}

    for row in s5_tracking.itertuples(index=False):
        ci = condition_index[str(row.condition)]
        si = snapshot_index[int(row.snapshot)]
        selected_id = int(row.selected_beam_id)
        oracle_id = int(row.oracle_beam_id)

        if not (0 <= selected_id < NUM_EXPECTED_BEAMS):
            raise ValueError("S5 baseline contains an invalid selected beam ID.")
        if not (0 <= oracle_id < NUM_EXPECTED_BEAMS):
            raise ValueError("S5 baseline contains an invalid oracle beam ID.")

        expected_selected = gain_tensor[ci, si, selected_id]
        expected_oracle = np.max(gain_tensor[ci, si])

        if not np.isclose(
            float(row.selected_gain_linear), expected_selected,
            rtol=1e-7, atol=1e-18
        ):
            raise RuntimeError(
                "S5 selected gain does not match S4.1 beam metrics at "
                f"{row.condition}, snapshot {row.snapshot}."
            )
        if oracle_id != int(np.argmax(gain_tensor[ci, si])):
            raise RuntimeError(
                "S5 oracle beam does not match S4.1 exhaustive optimum at "
                f"{row.condition}, snapshot {row.snapshot}."
            )
        if not np.isclose(
            float(row.oracle_gain_db),
            float(gain_to_db(expected_oracle)),
            rtol=0.0, atol=1e-6
        ):
            raise RuntimeError(
                "S5 oracle gain does not match S4.1 beam metrics."
            )

    return {
        "conditions": conditions,
        "snapshots": snapshots,
        "gain_tensor": gain_tensor,
        "directions": directions,
        "positions": positions,
        "s5_tracking": s5_tracking,
        "beam_metrics": beam_metrics,
        "snapshot_summary": snapshot_summary,
        "input_paths": paths,
    }


# ======================================================================
# 3. COPY S5 BASELINES WITHOUT ALTERING THEIR DECISIONS
# ======================================================================

def build_baseline_rows(data):
    rows = data["s5_tracking"].copy()

    def baseline_stage(row):
        if row["method"] == METHOD_ORACLE:
            return "full_sweep"
        if row["method"] == METHOD_HOLD:
            return "initial_full_sweep" if bool(row["is_full_sweep"]) else "hold"
        return "scheduled_full_sweep" if bool(row["is_full_sweep"]) else "local_9"

    rows["search_stage"] = rows.apply(baseline_stage, axis=1)
    rows["gain_drop_triggered"] = False
    rows["same_beam_gain_drop_db"] = np.nan
    rows["expanded_search"] = False
    rows["fallback_full_sweep"] = False
    rows["trigger_reason"] = ""
    rows["expansion_candidate_count"] = 0

    return rows


# ======================================================================
# 4. RUN THE S5.1 ADAPTIVE TRACKER
# ======================================================================

def run_adaptive_tracker(condition_index, condition, data):
    gains = data["gain_tensor"][condition_index]
    directions = data["directions"]
    snapshots = data["snapshots"]
    positions = data["positions"][condition]

    rows = []
    current_beam = None
    previous_selected_gain_db = None

    for snapshot_index, snapshot in enumerate(snapshots):
        previous_beam = current_beam
        gain_drop_detected = False
        expand_search = False
        fallback_full_sweep = False
        drop_db = np.nan
        local_residual_drop_db = np.nan
        trigger_reason = ""

        if snapshot_index == 0:
            candidates = np.arange(NUM_EXPECTED_BEAMS, dtype=int)
            current_beam = int(np.argmax(gains[snapshot_index]))
            search_stage = "initial_full_sweep"
            is_full_sweep = True
            improvement_db = np.nan

        else:
            # The current beam is included in every local candidate set.
            # Its current gain is therefore a legitimate tracker observation.
            current_beam_gain_db = float(
                gain_to_db(gains[snapshot_index, current_beam])
            )

            drop_db = float(
                previous_selected_gain_db - current_beam_gain_db
            )
            gain_drop_detected = bool(drop_db > GAIN_DROP_TRIGGER_DB)

            local_candidates = nearest_angular_beams(
                directions,
                current_beam,
                LOCAL_CANDIDATE_COUNT,
            )
            local_best = int(
                local_candidates[
                    np.argmax(gains[snapshot_index, local_candidates])
                ]
            )
            local_best_gain_db = float(
                gain_to_db(gains[snapshot_index, local_best])
            )
            local_improvement_db = float(
                local_best_gain_db - current_beam_gain_db
            )
            local_residual_drop_db = float(
                previous_selected_gain_db - local_best_gain_db
            )

            # First test the local neighbourhood. Expand only when the
            # same-beam drop is large AND the best local candidate still
            # leaves a severe residual degradation.
            expand_search = bool(
                gain_drop_detected
                and local_residual_drop_db > RESIDUAL_DROP_TRIGGER_DB
            )

            trigger_reason = ""
            if gain_drop_detected:
                trigger_reason = (
                    f"same_beam_gain_drop>{GAIN_DROP_TRIGGER_DB:.1f}dB"
                )

            if expand_search:
                trigger_reason += (
                    f";local_residual_drop>{RESIDUAL_DROP_TRIGGER_DB:.1f}dB"
                )
                candidates = nearest_angular_beams(
                    directions,
                    current_beam,
                    EXPANDED_CANDIDATE_COUNT,
                )
                candidate_best = int(
                    candidates[
                        np.argmax(gains[snapshot_index, candidates])
                    ]
                )
                candidate_best_gain_db = float(
                    gain_to_db(gains[snapshot_index, candidate_best])
                )
                improvement_db = float(
                    candidate_best_gain_db - current_beam_gain_db
                )

                if improvement_db >= HYSTERESIS_DB:
                    current_beam = candidate_best
                    search_stage = "expanded_32"
                    is_full_sweep = False
                else:
                    candidates = np.arange(NUM_EXPECTED_BEAMS, dtype=int)
                    current_beam = int(np.argmax(gains[snapshot_index]))
                    search_stage = "fallback_full_sweep"
                    is_full_sweep = True
                    fallback_full_sweep = True
            else:
                candidates = local_candidates
                improvement_db = local_improvement_db

                if (
                    local_best != current_beam
                    and improvement_db >= HYSTERESIS_DB
                ):
                    current_beam = local_best

                search_stage = (
                    "local_9_after_gain_drop"
                    if gain_drop_detected
                    else "local_9"
                )
                is_full_sweep = False
                if gain_drop_detected:
                    trigger_reason += ";local_search_sufficient"

        selected_gain = float(gains[snapshot_index, current_beam])
        selected_gain_db = float(gain_to_db(selected_gain))
        oracle_beam = int(np.argmax(gains[snapshot_index]))
        oracle_gain = float(gains[snapshot_index, oracle_beam])
        oracle_gain_db = float(gain_to_db(oracle_gain))
        regret_db = max(0.0, oracle_gain_db - selected_gain_db)

        switched = bool(
            previous_beam is not None and current_beam != previous_beam
        )

        # The current selected beam was necessarily measured: it belongs
        # to the initial/full set or to the local/expanded candidate set.
        if current_beam not in candidates:
            raise RuntimeError(
                "Adaptive tracker selected a beam that was not probed."
            )

        rows.append({
            "condition": condition,
            "snapshot": int(snapshot),
            "snapshot_index": int(snapshot_index),
            "rx_x_m": float(positions[snapshot_index, 0]),
            "rx_y_m": float(positions[snapshot_index, 1]),
            "rx_z_m": float(positions[snapshot_index, 2]),
            "method": METHOD_ADAPTIVE,
            "selected_beam_id": int(current_beam),
            "selected_gain_linear": selected_gain,
            "selected_gain_db": selected_gain_db,
            "oracle_beam_id": oracle_beam,
            "oracle_gain_db": oracle_gain_db,
            "regret_db": regret_db,
            "beam_switched": switched,
            "is_full_sweep": bool(is_full_sweep),
            "beams_probed": int(len(candidates)),
            "candidate_beam_ids": json.dumps(candidates.astype(int).tolist()),
            "search_stage": search_stage,
            "gain_drop_triggered": (
                gain_drop_detected if snapshot_index > 0 else False
            ),
            "same_beam_gain_drop_db": drop_db,
            "local_residual_drop_db": (
                local_residual_drop_db if snapshot_index > 0 else np.nan
            ),
            "reacquisition_triggered": (
                expand_search if snapshot_index > 0 else False
            ),
            "expanded_search": bool(
                expand_search
                and search_stage in {"expanded_32", "fallback_full_sweep"}
            ),
            "fallback_full_sweep": fallback_full_sweep,
            "trigger_reason": trigger_reason,
            "expansion_candidate_count": (
                EXPANDED_CANDIDATE_COUNT if expand_search else 0
            ),
            "candidate_improvement_over_current_db": improvement_db,
        })

        previous_selected_gain_db = selected_gain_db
 
    return rows


# ======================================================================
# 5. BUILD COMPLETE TRACKING RESULTS AND SUMMARIES
# ======================================================================

def build_all_results(data):
    baseline = build_baseline_rows(data)
    adaptive_rows = []

    for condition_index, condition in enumerate(data["conditions"]):
        adaptive_rows.extend(
            run_adaptive_tracker(condition_index, condition, data)
        )

    adaptive = pd.DataFrame(adaptive_rows)

    for column in [
        "candidate_improvement_over_current_db",
    ]:
        if column not in baseline.columns:
            baseline[column] = np.nan

    tracking = pd.concat(
        [baseline, adaptive],
        ignore_index=True,
        sort=False,
    )

    method_order = {method: index for index, method in enumerate(METHOD_ORDER)}
    tracking["_method_order"] = tracking["method"].map(method_order)
    tracking = tracking.sort_values(
        ["condition", "_method_order", "snapshot_index"]
    ).drop(columns="_method_order").reset_index(drop=True)

    summary_rows = []
    for (condition, method), group in tracking.groupby(
        ["condition", "method"], sort=False
    ):
        regret = group["regret_db"].to_numpy(dtype=float)
        probes = group["beams_probed"].to_numpy(dtype=float)

        summary_rows.append({
            "condition": condition,
            "method": method,
            "num_snapshots": int(len(group)),
            "num_beams_in_codebook": NUM_EXPECTED_BEAMS,
            "total_beam_probes": int(probes.sum()),
            "mean_beams_probed_per_snapshot": float(probes.mean()),
            "mean_beam_training_overhead_pct": float(
                100.0 * probes.mean() / NUM_EXPECTED_BEAMS
            ),
            "total_beam_switches": int(group["beam_switched"].astype(bool).sum()),
            "beam_switch_rate_per_transition_pct": float(
                100.0 * group["beam_switched"].astype(bool).sum()
                / max(1, len(group) - 1)
            ),
            "mean_selected_gain_db": float(group["selected_gain_db"].mean()),
            "mean_oracle_regret_db": float(regret.mean()),
            "median_oracle_regret_db": float(np.median(regret)),
            "p95_oracle_regret_db": float(np.percentile(regret, 95)),
            "maximum_oracle_regret_db": float(regret.max()),
            "snapshots_within_1db_of_oracle_pct": float(
                100.0 * np.mean(regret <= 1.0 + 1e-9)
            ),
            "snapshots_within_3db_of_oracle_pct": float(
                100.0 * np.mean(regret <= 3.0 + 1e-9)
            ),
            "full_sweep_count": int(group["is_full_sweep"].astype(bool).sum()),
            "expanded_search_count": int(
                group["expanded_search"].astype(bool).sum()
            ),
            "fallback_full_sweep_count": int(
                group["fallback_full_sweep"].astype(bool).sum()
            ),
            "gain_drop_trigger_count": int(
                group["gain_drop_triggered"].astype(bool).sum()
            ),
        })

    summary = pd.DataFrame(summary_rows)
    summary["_method_order"] = summary["method"].map(method_order)
    summary = summary.sort_values(
        ["condition", "_method_order"]
    ).drop(columns="_method_order").reset_index(drop=True)

    switch_rows = []
    for (condition, method), group in tracking.groupby(
        ["condition", "method"], sort=False
    ):
        group = group.sort_values("snapshot_index").reset_index(drop=True)
        condition_index = data["conditions"].index(condition)
        for index in range(1, len(group)):
            previous = group.iloc[index - 1]
            current = group.iloc[index]
            if bool(current["beam_switched"]):
                previous_beam = int(previous["selected_beam_id"])
                current_snapshot_index = int(current["snapshot_index"])
                gain_matrix = data["gain_tensor"][condition_index]
                switch_rows.append({
                    "condition": condition,
                    "method": method,
                    "from_snapshot": int(previous["snapshot"]),
                    "to_snapshot": int(current["snapshot"]),
                    "from_rx_x_m": float(previous["rx_x_m"]),
                    "to_rx_x_m": float(current["rx_x_m"]),
                    "previous_beam_id": previous_beam,
                    "new_beam_id": int(current["selected_beam_id"]),
                    "previous_beam_gain_at_new_snapshot_db": float(
                        gain_to_db(
                            gain_matrix[current_snapshot_index, previous_beam]
                        )
                    ),
                    "new_beam_gain_db": float(current["selected_gain_db"]),
                    "oracle_beam_id": int(current["oracle_beam_id"]),
                    "oracle_gain_db": float(current["oracle_gain_db"]),
                    "regret_after_switch_db": float(current["regret_db"]),
                    "search_stage": str(current["search_stage"]),
                })

    switches = pd.DataFrame(switch_rows)
    search_events = tracking[
        (tracking["method"] == METHOD_ADAPTIVE)
        & (
            tracking["gain_drop_triggered"].astype(bool)
            | tracking["is_full_sweep"].astype(bool)
        )
    ].copy()

    return tracking, summary, switches, search_events


# ======================================================================
# 6. PLOT GENERATION — ALL FIGURES SAVED AS PNG AND PDF
# ======================================================================

def save_figure(fig, plot_dir, stem, pdf_report):
    png_path = plot_dir / f"{stem}.png"
    pdf_path = plot_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    pdf_report.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    if not png_path.exists() or png_path.stat().st_size == 0:
        raise RuntimeError(f"Plot PNG was not saved correctly: {png_path}")
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise RuntimeError(f"Plot PDF was not saved correctly: {pdf_path}")


def generate_plots(tracking, summary, data, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    report_path = plot_dir / "s5_1_plot_report.pdf"
    with PdfPages(report_path) as pdf_report:

        # --------------------------------------------------------------
        # Figure 1: regret versus receiver position
        # --------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        methods = [METHOD_HOLD, METHOD_FIXED, METHOD_ADAPTIVE]

        for ax, condition in zip(axes, data["conditions"]):
            part = tracking[tracking["condition"] == condition]
            for method in methods:
                series = part[part["method"] == method].sort_values("snapshot_index")
                ax.plot(
                    series["rx_x_m"],
                    series["regret_db"],
                    marker="o",
                    linewidth=1.8,
                    label=METHOD_LABELS[method],
                )
            ax.axhline(0.0, linestyle="--", linewidth=1.0)
            ax.set_title(f"{condition.capitalize()} condition")
            ax.set_xlabel("Receiver X position (m)")
            ax.set_ylabel("Oracle regret (dB)")
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.suptitle("S5.1 — Beam-selection regret across the trajectory")
        fig.tight_layout()
        save_figure(fig, plot_dir, "01_regret_vs_position", pdf_report)
        generated.append("01_regret_vs_position")

        # --------------------------------------------------------------
        # Figure 2: selected beam ID versus receiver position
        # --------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, condition in zip(axes, data["conditions"]):
            part = tracking[tracking["condition"] == condition]
            for method in METHOD_ORDER:
                series = part[part["method"] == method].sort_values("snapshot_index")
                ax.step(
                    series["rx_x_m"],
                    series["selected_beam_id"],
                    where="mid",
                    marker="o",
                    linewidth=1.7,
                    label=METHOD_LABELS[method],
                )
            ax.set_title(f"{condition.capitalize()} condition")
            ax.set_xlabel("Receiver X position (m)")
            ax.set_ylabel("Selected beam ID")
            ax.set_yticks([0, 10, 20, 30, 40, 50])
            ax.set_ylim(-1, NUM_EXPECTED_BEAMS)
            ax.grid(True, alpha=0.25)
            ax.legend()
        fig.suptitle("S5.1 — Selected beam along the receiver trajectory")
        fig.tight_layout()
        save_figure(fig, plot_dir, "02_selected_beam_vs_position", pdf_report)
        generated.append("02_selected_beam_vs_position")

        # --------------------------------------------------------------
        # Figure 3: probes per snapshot
        # --------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        methods = [METHOD_HOLD, METHOD_FIXED, METHOD_ADAPTIVE]
        width = 0.24
        offsets = [-width, 0.0, width]

        for ax, condition in zip(axes, data["conditions"]):
            part = tracking[tracking["condition"] == condition]
            x_values = np.arange(len(data["snapshots"]))
            for method, offset in zip(methods, offsets):
                series = part[part["method"] == method].sort_values("snapshot_index")
                ax.bar(
                    x_values + offset,
                    series["beams_probed"].to_numpy(dtype=float),
                    width=width,
                    label=METHOD_LABELS[method],
                )
            ax.axhline(
                NUM_EXPECTED_BEAMS,
                linestyle="--",
                linewidth=1.0,
                label="Full codebook (52)" if condition == data["conditions"][0] else None,
            )
            ax.set_xticks(x_values)
            ax.set_xticklabels(
                [
                    f"{float(data['positions'][condition][i, 0]):.0f}"
                    for i in range(len(x_values))
                ]
            )
            ax.set_title(f"{condition.capitalize()} condition")
            ax.set_xlabel("Receiver X position (m)")
            ax.set_ylabel("Beam probes at snapshot")
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
        fig.suptitle("S5.1 — Beam-training measurements per snapshot")
        fig.tight_layout()
        save_figure(fig, plot_dir, "03_beam_probes_per_snapshot", pdf_report)
        generated.append("03_beam_probes_per_snapshot")

        # --------------------------------------------------------------
        # Figure 4: cumulative beam probes
        # --------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, condition in zip(axes, data["conditions"]):
            part = tracking[tracking["condition"] == condition]
            for method in METHOD_ORDER:
                series = part[part["method"] == method].sort_values("snapshot_index")
                cumulative = series["beams_probed"].cumsum().to_numpy(dtype=int)
                ax.plot(
                    series["rx_x_m"],
                    cumulative,
                    marker="o",
                    linewidth=1.8,
                    label=METHOD_LABELS[method],
                )
            ax.set_title(f"{condition.capitalize()} condition")
            ax.set_xlabel("Receiver X position (m)")
            ax.set_ylabel("Cumulative beam probes")
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.suptitle("S5.1 — Cumulative beam-training overhead")
        fig.tight_layout()
        save_figure(fig, plot_dir, "04_cumulative_beam_probes", pdf_report)
        generated.append("04_cumulative_beam_probes")

        # --------------------------------------------------------------
        # Figure 5: overhead versus mean regret
        # --------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 6))
        for condition in data["conditions"]:
            part = summary[summary["condition"] == condition]
            for method in METHOD_ORDER:
                row = part[part["method"] == method].iloc[0]
                ax.scatter(
                    row["mean_beam_training_overhead_pct"],
                    row["mean_oracle_regret_db"],
                    s=65,
                    label=f"{condition}: {METHOD_LABELS[method]}",
                )
        ax.set_xlabel("Mean beam-training overhead (% of 52-beam codebook)")
        ax.set_ylabel("Mean oracle regret (dB)")
        ax.set_title("S5.1 — Beam-training overhead versus tracking regret")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_figure(fig, plot_dir, "05_training_overhead_vs_regret", pdf_report)
        generated.append("05_training_overhead_vs_regret")

        # --------------------------------------------------------------
        # Figure 6: adaptive trigger diagnostics
        # --------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, condition in zip(axes, data["conditions"]):
            part = tracking[
                (tracking["condition"] == condition)
                & (tracking["method"] == METHOD_ADAPTIVE)
            ].sort_values("snapshot_index")

            ax.plot(
                part["rx_x_m"],
                part["same_beam_gain_drop_db"],
                marker="o",
                linewidth=1.8,
                label="Same-beam gain drop",
            )
            ax.plot(
                part["rx_x_m"],
                part["local_residual_drop_db"],
                marker="s",
                linewidth=1.6,
                label="Residual drop after local search",
            )
            ax.axhline(
                GAIN_DROP_TRIGGER_DB,
                linestyle="--",
                linewidth=1.2,
                label=f"Gain-drop detector ({GAIN_DROP_TRIGGER_DB:.0f} dB)",
            )
            ax.axhline(
                RESIDUAL_DROP_TRIGGER_DB,
                linestyle=":",
                linewidth=1.4,
                label=f"Expansion residual threshold ({RESIDUAL_DROP_TRIGGER_DB:.0f} dB)",
            )

            expanded = part[part["expanded_search"].astype(bool)]
            fallback = part[part["fallback_full_sweep"].astype(bool)]

            if not expanded.empty:
                ax.scatter(
                    expanded["rx_x_m"],
                    expanded["same_beam_gain_drop_db"],
                    marker="s",
                    s=75,
                    label="Expanded search",
                )
            if not fallback.empty:
                ax.scatter(
                    fallback["rx_x_m"],
                    fallback["same_beam_gain_drop_db"],
                    marker="X",
                    s=90,
                    label="Full-sweep fallback",
                )

            ax.set_title(f"{condition.capitalize()} condition")
            ax.set_xlabel("Receiver X position (m)")
            ax.set_ylabel("Same-beam gain drop (dB)")
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.suptitle("S5.1 — Adaptive reacquisition trigger diagnostics")
        fig.tight_layout()
        save_figure(fig, plot_dir, "06_adaptive_trigger_diagnostics", pdf_report)
        generated.append("06_adaptive_trigger_diagnostics")

        # --------------------------------------------------------------
        # Figure 7: regret heatmap
        # --------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        heatmap_max = max(
            1.0,
            float(tracking["regret_db"].max()),
        )
        for ax, condition in zip(axes, data["conditions"]):
            part = tracking[tracking["condition"] == condition]
            matrix = []
            for method in METHOD_ORDER:
                series = part[part["method"] == method].sort_values("snapshot_index")
                matrix.append(series["regret_db"].to_numpy(dtype=float))

            image = ax.imshow(
                np.asarray(matrix),
                aspect="auto",
                interpolation="nearest",
                vmin=0.0,
                vmax=heatmap_max,
            )
            ax.set_yticks(range(len(METHOD_ORDER)))
            ax.set_yticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
            ax.set_xticks(range(len(data["snapshots"])))
            ax.set_xticklabels(
                [
                    f"{float(data['positions'][condition][i, 0]):.0f}"
                    for i in range(len(data["snapshots"]))
                ]
            )
            ax.set_xlabel("Receiver X position (m)")
            ax.set_title(f"{condition.capitalize()} condition")
            fig.colorbar(image, ax=ax, label="Oracle regret (dB)")
        fig.suptitle("S5.1 — Per-snapshot regret comparison")
        fig.tight_layout()
        save_figure(fig, plot_dir, "07_regret_heatmap", pdf_report)
        generated.append("07_regret_heatmap")

    if not report_path.exists() or report_path.stat().st_size == 0:
        raise RuntimeError("Combined S5.1 plot report PDF was not created.")

    return generated, report_path


# ======================================================================
# 7. VALIDATION
# ======================================================================

def validate_results(tracking, summary, data, generated_plots, plot_dir):
    checks = {}

    expected_methods = set(METHOD_ORDER)
    checks["all_four_methods_present"] = (
        set(tracking["method"].astype(str).unique()) == expected_methods
    )

    expected_rows = (
        len(data["conditions"]) * len(data["snapshots"]) * len(METHOD_ORDER)
    )
    checks["each_method_covers_every_snapshot_once"] = (
        len(tracking) == expected_rows
        and not tracking.duplicated(
            ["condition", "method", "snapshot"]
        ).any()
    )

    checks["all_selected_beam_ids_are_valid"] = bool(
        tracking["selected_beam_id"].between(
            0, NUM_EXPECTED_BEAMS - 1
        ).all()
    )

    checks["all_probe_counts_are_within_codebook_size"] = bool(
        tracking["beams_probed"].between(0, NUM_EXPECTED_BEAMS).all()
    )

    selected_is_probed = True
    probe_count_matches_candidates = True
    regret_consistent = True
    condition_index = {
        condition: index for index, condition in enumerate(data["conditions"])
    }
    snapshot_index = {
        int(snapshot): index for index, snapshot in enumerate(data["snapshots"])
    }

    for row in tracking.itertuples(index=False):
        candidates = json.loads(str(row.candidate_beam_ids))
        if len(candidates) != int(row.beams_probed):
            probe_count_matches_candidates = False
        # Hold-last deliberately performs zero probes after initial acquisition;
        # its selected beam is retained from the preceding snapshot.
        if int(row.beams_probed) > 0 and int(row.selected_beam_id) not in candidates:
            selected_is_probed = False
        if int(row.beams_probed) == 0 and len(candidates) != 0:
            probe_count_matches_candidates = False

        ci = condition_index[str(row.condition)]
        si = snapshot_index[int(row.snapshot)]
        expected_gain = data["gain_tensor"][
            ci, si, int(row.selected_beam_id)
        ]
        expected_oracle = np.max(data["gain_tensor"][ci, si])
        expected_regret = max(
            0.0,
            float(gain_to_db(expected_oracle) - gain_to_db(expected_gain)),
        )

        if not np.isclose(
            float(row.selected_gain_linear),
            expected_gain,
            rtol=1e-7,
            atol=1e-18,
        ):
            regret_consistent = False
        if not np.isclose(
            float(row.regret_db),
            expected_regret,
            rtol=0.0,
            atol=1e-6,
        ):
            regret_consistent = False

    checks["every_selected_beam_was_in_its_probed_candidate_set"] = selected_is_probed
    checks["probe_counts_match_unique_candidate_lists"] = probe_count_matches_candidates
    checks["selected_gain_and_regret_match_s4_measurements"] = regret_consistent

    adaptive = tracking[tracking["method"] == METHOD_ADAPTIVE]
    checks["adaptive_tracker_starts_with_full_acquisition"] = bool(
        (
            adaptive[adaptive["snapshot_index"] == 0]["search_stage"]
            == "initial_full_sweep"
        ).all()
    )

    trigger_rows = adaptive[adaptive["gain_drop_triggered"].astype(bool)]
    checks["all_gain_drop_triggers_exceed_threshold"] = bool(
        (
            trigger_rows["same_beam_gain_drop_db"]
            > GAIN_DROP_TRIGGER_DB
        ).all()
    )

    expanded_rows = adaptive[adaptive["expanded_search"].astype(bool)]
    checks["expanded_search_uses_32_candidates_or_full_fallback"] = bool(
        (
            expanded_rows["expansion_candidate_count"]
            == EXPANDED_CANDIDATE_COUNT
        ).all()
    )

    checks["expanded_search_requires_both_drop_thresholds"] = bool(
        (
            (
                expanded_rows["same_beam_gain_drop_db"]
                > GAIN_DROP_TRIGGER_DB
            )
            & (
                expanded_rows["local_residual_drop_db"]
                > RESIDUAL_DROP_TRIGGER_DB
            )
        ).all()
    )

    checks["fallback_only_occurs_after_gain_drop_trigger"] = bool(
        (
            adaptive.loc[
                adaptive["fallback_full_sweep"].astype(bool),
                "gain_drop_triggered",
            ].astype(bool)
        ).all()
    )

    checks["adaptive_expansion_candidate_count_is_valid"] = bool(
        (
            adaptive.loc[
                adaptive["search_stage"] == "expanded_32",
                "beams_probed",
            ] == EXPANDED_CANDIDATE_COUNT
        ).all()
    )

    checks["all_seven_plot_pngs_and_pdfs_exist"] = bool(
        len(generated_plots) == 7
        and all(
            (plot_dir / f"{stem}.{extension}").exists()
            and (plot_dir / f"{stem}.{extension}").stat().st_size > 0
            for stem in generated_plots
            for extension in ("png", "pdf")
        )
        and (plot_dir / "s5_1_plot_report.pdf").exists()
    )

    return checks


# ======================================================================
# 8. MAIN
# ======================================================================

def main():
    args = parse_args()
    base_dir = args.base_dir.expanduser().resolve()
    output_dir = base_dir / "s5_1_adaptive_tracking"
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("SIONNA RT - S5.1 ADAPTIVE BEAM REACQUISITION")
    print("=" * 76)
    print("Project directory:", base_dir)

    data = load_inputs(base_dir)
    print("Conditions:", data["conditions"])
    print("Snapshots per condition:", len(data["snapshots"]))
    print("Codebook beams:", NUM_EXPECTED_BEAMS)
    print("Local candidate count:", LOCAL_CANDIDATE_COUNT)
    print("Expanded candidate count:", EXPANDED_CANDIDATE_COUNT)
    print("Same-beam gain-drop detector (dB):", GAIN_DROP_TRIGGER_DB)
    print("Residual-drop expansion threshold (dB):", RESIDUAL_DROP_TRIGGER_DB)
    print("Switch hysteresis (dB):", HYSTERESIS_DB)

    tracking, summary, switches, search_events = build_all_results(data)

    output_per_snapshot = output_dir / "s5_1_tracking_per_snapshot.csv"
    output_summary = output_dir / "s5_1_tracking_summary.csv"
    output_switches = output_dir / "s5_1_beam_switches.csv"
    output_events = output_dir / "s5_1_search_events.csv"
    output_config = output_dir / "s5_1_config.json"
    output_validation = output_dir / "s5_1_validation.txt"

    tracking.to_csv(output_per_snapshot, index=False)
    summary.to_csv(output_summary, index=False)
    switches.to_csv(output_switches, index=False)
    search_events.to_csv(output_events, index=False)

    generated_plots, plot_report = generate_plots(
        tracking, summary, data, plot_dir
    )

    checks = validate_results(
        tracking, summary, data, generated_plots, plot_dir
    )
    validation_passed = all(checks.values())

    config = {
        "experiment": "S5.1_Adaptive_Beam_Reacquisition",
        "input_stage": "Validated S4.1 channel-derived beam metrics and S5 baseline",
        "conditions": data["conditions"],
        "snapshot_ids": data["snapshots"],
        "codebook_size": NUM_EXPECTED_BEAMS,
        "adaptive_policy": {
            "initial_acquisition": "exhaustive full-codebook scan",
            "normal_search": {
                "candidate_count": LOCAL_CANDIDATE_COUNT,
                "selection": "angularly nearest codebook beams to current beam",
            },
            "degradation_detector": {
                "measurement": (
                    "previous snapshot selected-beam gain minus current "
                    "snapshot gain of the same beam"
                ),
                "same_beam_gain_drop_threshold_db": GAIN_DROP_TRIGGER_DB,
                "local_residual_drop_threshold_db": RESIDUAL_DROP_TRIGGER_DB,
                "trigger_condition": (
                    f"same_beam_gain_drop_db > {GAIN_DROP_TRIGGER_DB} AND "
                    f"local_residual_drop_db > {RESIDUAL_DROP_TRIGGER_DB}"
                ),
            },
            "expanded_search": {
                "candidate_count": EXPANDED_CANDIDATE_COUNT,
                "selection": (
                    "angularly nearest codebook beams to the current beam; "
                    "includes the local neighbourhood"
                ),
            },
            "hysteresis_threshold_db": HYSTERESIS_DB,
            "fallback": (
                "If expanded-search best candidate does not improve current "
                "beam by at least hysteresis threshold, perform full scan."
            ),
        },
        "evaluation": {
            "oracle": (
                "Per-snapshot exhaustive best beam from S4.1 full beam-gain "
                "table; used for offline scoring, not adaptive decisions."
            ),
            "probe_count": (
                "Number of unique candidate beams evaluated by the policy "
                "at each snapshot; not wall-clock latency or energy."
            ),
            "channel_gain": "dimensionless |h^T w|^2; dB = 10*log10(gain)",
        },
        "limitations": [
            "Spatial quasi-static snapshots; no continuous-time channel evolution.",
            "No receiver velocity, Doppler, or channel aging is modelled.",
            "Beam-training overhead is a candidate-measurement count proxy.",
            "Thresholds and candidate counts are fixed experimental settings, not optimized.",
            "Results are specific to the current scene, trajectory, and codebook.",
        ],
        "plot_files": [
            f"{stem}.{extension}"
            for stem in generated_plots
            for extension in ("png", "pdf")
        ] + ["s5_1_plot_report.pdf"],
        "validation": checks,
        "validation_passed": bool(validation_passed),
    }

    output_config.write_text(json.dumps(config, indent=4), encoding="utf-8")

    validation_lines = [
        "S5.1 ADAPTIVE BEAM TRACKING VALIDATION",
        "=" * 76,
        f"Conditions: {', '.join(data['conditions'])}",
        f"Snapshots per condition: {len(data['snapshots'])}",
        f"Codebook beams: {NUM_EXPECTED_BEAMS}",
        f"Local candidates: {LOCAL_CANDIDATE_COUNT}",
        f"Expanded candidates: {EXPANDED_CANDIDATE_COUNT}",
        f"Gain-drop trigger (dB): {GAIN_DROP_TRIGGER_DB}",
        f"Hysteresis (dB): {HYSTERESIS_DB}",
        "",
        "Checks:",
    ]
    for name, passed in checks.items():
        validation_lines.append(
            f"[{'PASS' if passed else 'FAIL'}] {name}"
        )
    validation_lines.extend([
        "",
        f"OVERALL: {'PASS' if validation_passed else 'FAIL'}",
        "",
        "Scope:",
        "- S5.1 reuses S4.1 beam metrics and the existing S5 baseline.",
        "- The exhaustive oracle is used only for offline evaluation.",
        "- Probe counts are candidate-beam measurement counts, not measured latency.",
        "- The trajectory contains spatial quasi-static snapshots, not continuous-time Doppler evolution.",
    ])
    output_validation.write_text(
        "\n".join(validation_lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("S5.1 SUMMARY")
    print("=" * 76)
    print(summary.to_string(index=False))
    print("\nAdaptive search events:")
    if search_events.empty:
        print("No adaptive reacquisition events.")
    else:
        print(
            search_events[
                [
                    "condition",
                    "snapshot",
                    "rx_x_m",
                    "same_beam_gain_drop_db",
                    "local_residual_drop_db",
                    "search_stage",
                    "beams_probed",
                    "selected_beam_id",
                    "oracle_beam_id",
                    "regret_db",
                ]
            ].to_string(index=False)
        )

    print("\nValidation:")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print("\nOutputs:")
    for path in [
        output_per_snapshot,
        output_summary,
        output_switches,
        output_events,
        output_config,
        output_validation,
        plot_report,
    ]:
        print(" -", path)
    print(" -", plot_dir, f"({len(generated_plots)} plots, PNG + PDF)")

    if not validation_passed:
        raise RuntimeError(
            "S5.1 completed but one or more validation checks failed. "
            "Inspect s5_1_validation.txt before using the results."
        )

    print("\nS5.1 adaptive tracking, plots and validation successful.")


if __name__ == "__main__":
    main()
