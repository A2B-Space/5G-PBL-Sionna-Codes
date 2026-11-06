#!/usr/bin/env python3
"""
S5: Spatially evolving beam tracking over the validated S4.1 channel dataset.

This stage intentionally reuses the saved S4.1 channel snapshots. It does
NOT rerun ray tracing and does NOT claim Doppler/time-continuous simulation.
The receiver follows the existing spatial trajectory, and a beam-management
policy updates its selected codebook beam as the channel changes.

Compared policies
-----------------
1. exhaustive_oracle:
   Full 52-beam scan at every snapshot. Upper-bound reference, not a
   low-overhead tracking algorithm.

2. hold_last_beam:
   Full initial scan, then retain that beam without further beam training.

3. local_tracking:
   Full initial scan; between periodic full sweeps, probe the K codebook
   directions closest in angle to the currently selected beam. Switch only
   when the best local candidate improves the currently selected beam by at
   least the configured hysteresis threshold. A full 52-beam reacquisition
   scan is performed every N snapshots.

The tracker is evaluated offline against all saved beam gains. Its reported
probe count is the number of beams the policy would measure, while the
all-beam channel data is used only to score the chosen beam and calculate
oracle regret.

Inputs (created by S4.1)
------------------------
~/sionna_project/s4_blockage_geometry/
    s4_channel_data.npz
    s4_beam_codebook.csv
    s4_beam_metrics.csv
    s4_snapshot_summary.csv
    s4_config.json

Outputs
-------
~/sionna_project/s5_beam_tracking/
    s5_tracking_per_snapshot.csv
    s5_tracking_summary.csv
    s5_tracking_beam_switches.csv
    s5_tracking_config.json
    s5_validation.txt

Run
---
cd ~/sionna_project
source .venv/bin/activate
python s5_beam_tracking.py

Dependencies: numpy, pandas (already used by S4.1).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# 1. CONFIGURATION
# ======================================================================

BASE_DIR = Path.home() / "sionna_project"
INPUT_DIR = BASE_DIR / "s4_blockage_geometry"
OUTPUT_DIR = BASE_DIR / "s5_beam_tracking"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CHANNELS = INPUT_DIR / "s4_channel_data.npz"
INPUT_CODEBOOK = INPUT_DIR / "s4_beam_codebook.csv"
INPUT_BEAM_METRICS = INPUT_DIR / "s4_beam_metrics.csv"
INPUT_SNAPSHOT_SUMMARY = INPUT_DIR / "s4_snapshot_summary.csv"
INPUT_CONFIG = INPUT_DIR / "s4_config.json"

OUTPUT_PER_SNAPSHOT = OUTPUT_DIR / "s5_tracking_per_snapshot.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "s5_tracking_summary.csv"
OUTPUT_SWITCHES = OUTPUT_DIR / "s5_tracking_beam_switches.csv"
OUTPUT_CONFIG = OUTPUT_DIR / "s5_tracking_config.json"
OUTPUT_VALIDATION = OUTPUT_DIR / "s5_validation.txt"

# Proposed local tracker settings. These are explicit experimental
# hyperparameters, not claimed to be globally optimal.
LOCAL_CANDIDATE_COUNT = 9
HYSTERESIS_DB = 3.0
FULL_SWEEP_INTERVAL = 4  # Snapshot indices 0, 4, 8, ... are full sweeps.

METHOD_ORACLE = "exhaustive_oracle"
METHOD_HOLD = "hold_last_beam"
METHOD_TRACKER = "local_tracking"
METHODS = [METHOD_ORACLE, METHOD_HOLD, METHOD_TRACKER]

GAIN_FLOOR = np.finfo(np.float64).tiny
RECOMPUTE_RTOL = 1e-7
RECOMPUTE_ATOL = 1e-18


# ======================================================================
# 2. INPUT VALIDATION AND DATA LOADING
# ======================================================================

print("=" * 76)
print("SIONNA RT - S5 SPATIALLY EVOLVING BEAM TRACKING")
print("=" * 76)

required_inputs = [
    INPUT_CHANNELS,
    INPUT_CODEBOOK,
    INPUT_BEAM_METRICS,
    INPUT_SNAPSHOT_SUMMARY,
    INPUT_CONFIG,
]
missing_inputs = [str(path) for path in required_inputs if not path.exists()]
if missing_inputs:
    raise FileNotFoundError(
        "Required S4.1 outputs are missing:\n - "
        + "\n - ".join(missing_inputs)
        + "\nRun s4_1_path_geometry_export.py successfully first."
    )

with np.load(INPUT_CHANNELS, allow_pickle=False) as data:
    required_npz_keys = {
        "condition_names",
        "snapshot_ids",
        "rx_positions_m",
        "tx_position_m",
        "h_complex",
        "beamformed_channel_gain_linear",
        "beam_directions_local",
        "beam_weights_complex",
    }
    missing_keys = required_npz_keys - set(data.files)
    if missing_keys:
        raise KeyError(
            f"S4.1 NPZ is missing required arrays: {sorted(missing_keys)}"
        )

    condition_names = data["condition_names"].astype(str)
    snapshot_ids = data["snapshot_ids"].astype(int)
    rx_positions_m = np.asarray(data["rx_positions_m"], dtype=float)
    tx_position_m = np.asarray(data["tx_position_m"], dtype=float)
    h_complex = np.asarray(data["h_complex"], dtype=np.complex128)
    saved_gain = np.asarray(
        data["beamformed_channel_gain_linear"], dtype=float
    )
    beam_directions = np.asarray(
        data["beam_directions_local"], dtype=float
    )
    saved_weights = np.asarray(
        data["beam_weights_complex"], dtype=np.complex128
    )

s4_config = json.loads(INPUT_CONFIG.read_text(encoding="utf-8"))
beam_codebook_df = pd.read_csv(INPUT_CODEBOOK)
beam_metrics_df = pd.read_csv(INPUT_BEAM_METRICS)
snapshot_summary_df = pd.read_csv(INPUT_SNAPSHOT_SUMMARY)

if condition_names.tolist() != ["unblocked", "blocked"]:
    raise ValueError(
        "Unexpected condition order in S4.1 NPZ. Expected "
        "['unblocked', 'blocked']; received "
        f"{condition_names.tolist()}."
    )

num_conditions, num_snapshots, num_antennas = h_complex.shape
if num_conditions != 2:
    raise ValueError(f"Expected two S4.1 conditions, got {num_conditions}.")
if rx_positions_m.shape != (num_snapshots, 3):
    raise ValueError(
        f"Unexpected RX position shape: {rx_positions_m.shape}"
    )
if snapshot_ids.shape != (num_snapshots,):
    raise ValueError(f"Unexpected snapshot ID shape: {snapshot_ids.shape}")

num_beams = len(beam_directions)
if beam_directions.shape != (num_beams, 3):
    raise ValueError(
        f"Unexpected beam direction shape: {beam_directions.shape}"
    )
if saved_weights.shape != (num_beams, num_antennas):
    raise ValueError(
        f"Unexpected beam weight shape: {saved_weights.shape}"
    )
if saved_gain.shape != (num_conditions, num_snapshots, num_beams):
    raise ValueError(
        f"Unexpected saved gain shape: {saved_gain.shape}"
    )
if num_beams != 52:
    raise ValueError(f"Expected the S3/S4 codebook of 52 beams, got {num_beams}.")
if num_antennas != 64:
    raise ValueError(f"Expected the S3/S4 64-element TX array, got {num_antennas}.")

if not np.all(np.isfinite(h_complex)):
    raise ValueError("Non-finite channel coefficients found in S4.1 NPZ.")
if not np.all(np.isfinite(saved_gain)) or np.any(saved_gain < 0.0):
    raise ValueError("Invalid beam-gain values found in S4.1 NPZ.")
if not np.all(np.isfinite(beam_directions)):
    raise ValueError("Non-finite beam directions found.")
if not np.all(np.isfinite(saved_weights)):
    raise ValueError("Non-finite beamforming weights found.")

if LOCAL_CANDIDATE_COUNT < 1 or LOCAL_CANDIDATE_COUNT > num_beams:
    raise ValueError("LOCAL_CANDIDATE_COUNT must be in [1, number of beams].")
if FULL_SWEEP_INTERVAL < 2:
    raise ValueError("FULL_SWEEP_INTERVAL must be at least 2.")
if HYSTERESIS_DB < 0.0:
    raise ValueError("HYSTERESIS_DB cannot be negative.")

print("Conditions:", condition_names.tolist())
print("Snapshots per condition:", num_snapshots)
print("TX antennas:", num_antennas)
print("Candidate beams:", num_beams)
print("RX trajectory:", rx_positions_m[0].tolist(), "->", rx_positions_m[-1].tolist())


# ======================================================================
# 3. VERIFY CODEBOOK AND RECOMPUTE THE SAVED BEAM GAINS
# ======================================================================

expected_weight_columns = [
    column
    for antenna_id in range(num_antennas)
    for column in (
        f"w{antenna_id:02d}_real",
        f"w{antenna_id:02d}_imag",
    )
]
missing_weight_columns = set(expected_weight_columns) - set(beam_codebook_df.columns)
if missing_weight_columns:
    raise ValueError(
        "S4.1 codebook is missing weight columns: "
        f"{sorted(missing_weight_columns)}"
    )

if "beam_id" not in beam_codebook_df.columns:
    raise ValueError("S4.1 codebook is missing beam_id.")

beam_codebook_df = beam_codebook_df.sort_values("beam_id").reset_index(drop=True)
if beam_codebook_df["beam_id"].tolist() != list(range(num_beams)):
    raise ValueError("Beam IDs in the codebook are not contiguous from 0.")

weights_from_csv = np.empty((num_beams, num_antennas), dtype=np.complex128)
for antenna_id in range(num_antennas):
    weights_from_csv[:, antenna_id] = (
        beam_codebook_df[f"w{antenna_id:02d}_real"].to_numpy(dtype=float)
        + 1j
        * beam_codebook_df[f"w{antenna_id:02d}_imag"].to_numpy(dtype=float)
    )

if not np.allclose(weights_from_csv, saved_weights, rtol=1e-10, atol=1e-12):
    raise RuntimeError(
        "Beam weights reconstructed from s4_beam_codebook.csv do not "
        "match beam_weights_complex in s4_channel_data.npz."
    )

weight_norms = np.linalg.norm(saved_weights, axis=1)
if not np.allclose(weight_norms, 1.0, rtol=1e-8, atol=1e-8):
    raise RuntimeError("At least one saved beamforming vector is not unit norm.")

# S4/S3 convention: |h^T w|^2 (transpose, no conjugation).
# This is consistent with the Sionna RT synthetic-array precoding convention.
recomputed_gain = np.empty_like(saved_gain)
for condition_index in range(num_conditions):
    for snapshot_index in range(num_snapshots):
        h = h_complex[condition_index, snapshot_index, :]
        effective = np.sum(
            h[np.newaxis, :] * saved_weights,
            axis=1,
        )
        recomputed_gain[condition_index, snapshot_index, :] = np.abs(effective) ** 2

if not np.allclose(
    recomputed_gain,
    saved_gain,
    rtol=RECOMPUTE_RTOL,
    atol=RECOMPUTE_ATOL,
):
    max_abs_error = float(np.max(np.abs(recomputed_gain - saved_gain)))
    raise RuntimeError(
        "Recomputed beam gains do not match S4.1 saved beam gains. "
        f"Maximum absolute error={max_abs_error:.6e}. "
        "Check channel/weight ordering and beamforming convention."
    )

# Use recomputed values after passing the integrity comparison.
gain_tensor = recomputed_gain

if len(beam_metrics_df) != num_conditions * num_snapshots * num_beams:
    raise ValueError(
        "s4_beam_metrics.csv row count does not match "
        "conditions × snapshots × beams."
    )

required_beam_metric_columns = {
    "condition",
    "snapshot",
    "beam_id",
    "beamformed_channel_gain_linear",
}
missing_metric_columns = required_beam_metric_columns - set(beam_metrics_df.columns)
if missing_metric_columns:
    raise ValueError(
        "s4_beam_metrics.csv is missing columns: "
        f"{sorted(missing_metric_columns)}"
    )

metric_pivot = beam_metrics_df.pivot(
    index=["condition", "snapshot"],
    columns="beam_id",
    values="beamformed_channel_gain_linear",
)
expected_index = pd.MultiIndex.from_product(
    [condition_names.tolist(), snapshot_ids.tolist()],
    names=["condition", "snapshot"],
)
metric_pivot = metric_pivot.reindex(index=expected_index, columns=range(num_beams))
if metric_pivot.isna().any().any():
    raise ValueError(
        "s4_beam_metrics.csv has missing/duplicate condition-snapshot-beam entries."
    )
csv_gain_tensor = metric_pivot.to_numpy(dtype=float).reshape(
    num_conditions, num_snapshots, num_beams
)
if not np.allclose(
    csv_gain_tensor,
    saved_gain,
    rtol=RECOMPUTE_RTOL,
    atol=RECOMPUTE_ATOL,
):
    raise RuntimeError(
        "s4_beam_metrics.csv values do not match the beam-gain tensor "
        "saved in s4_channel_data.npz."
    )

required_snapshot_columns = {"condition", "snapshot"}
if not required_snapshot_columns.issubset(snapshot_summary_df.columns):
    raise ValueError(
        "s4_snapshot_summary.csv must contain condition and snapshot columns."
    )
summary_pairs = set(
    zip(
        snapshot_summary_df["condition"].astype(str),
        snapshot_summary_df["snapshot"].astype(int),
    )
)
expected_pairs = {
    (str(condition), int(snapshot))
    for condition in condition_names
    for snapshot in snapshot_ids
}
if summary_pairs != expected_pairs:
    raise ValueError(
        "Snapshot IDs in s4_snapshot_summary.csv do not match the NPZ."
    )

print("Codebook integrity: PASS")
print("Complex-channel beam-gain recomputation: PASS")
print("S4.1 CSV-to-NPZ beam-gain consistency: PASS")
print("S4.1 snapshot index consistency: PASS")


# ======================================================================
# 4. TRACKING HELPERS
# ======================================================================

def gain_db(gain_linear):
    """Convert a nonnegative dimensionless gain to dB safely."""
    return 10.0 * np.log10(np.maximum(gain_linear, GAIN_FLOOR))


def nearest_angular_beams(direction_matrix, reference_beam_id, count):
    """
    Return `count` beam IDs closest in angular separation to a reference
    codebook direction. The reference beam is always included.
    """
    reference = direction_matrix[reference_beam_id]
    reference_norm = np.linalg.norm(reference)
    if not np.isclose(reference_norm, 1.0, atol=1e-8):
        raise ValueError("Reference beam direction is not unit norm.")

    direction_norms = np.linalg.norm(direction_matrix, axis=1)
    if not np.allclose(direction_norms, 1.0, atol=1e-8):
        raise ValueError("At least one codebook direction is not unit norm.")

    cosine = np.clip(direction_matrix @ reference, -1.0, 1.0)
    angular_distance = np.arccos(cosine)
    ordered = np.argsort(angular_distance, kind="stable")
    candidates = ordered[:count].astype(int)

    if reference_beam_id not in candidates:
        candidates[-1] = reference_beam_id
        candidates = np.unique(candidates)
        if len(candidates) < count:
            for beam_id in ordered:
                if int(beam_id) not in candidates:
                    candidates = np.append(candidates, int(beam_id))
                if len(candidates) == count:
                    break

    return np.sort(candidates.astype(int))


def run_tracking(condition_index, condition_name, gains):
    """
    Run exhaustive, hold-last, and local/hysteretic tracking on one
    condition. `gains` has shape [snapshot, beam].
    """
    oracle_rows = []
    hold_rows = []
    tracker_rows = []

    oracle_selected = []
    hold_selected = []
    tracker_selected = []

    # Full-scan oracle: upper bound with a full codebook sweep each snapshot.
    for snapshot_index in range(num_snapshots):
        best_beam = int(np.argmax(gains[snapshot_index]))
        oracle_selected.append(best_beam)
        oracle_rows.append({
            "condition": condition_name,
            "snapshot": int(snapshot_ids[snapshot_index]),
            "snapshot_index": snapshot_index,
            "rx_x_m": float(rx_positions_m[snapshot_index, 0]),
            "rx_y_m": float(rx_positions_m[snapshot_index, 1]),
            "rx_z_m": float(rx_positions_m[snapshot_index, 2]),
            "method": METHOD_ORACLE,
            "selected_beam_id": best_beam,
            "selected_gain_linear": float(gains[snapshot_index, best_beam]),
            "selected_gain_db": float(gain_db(gains[snapshot_index, best_beam])),
            "oracle_beam_id": best_beam,
            "oracle_gain_db": float(gain_db(np.max(gains[snapshot_index]))),
            "regret_db": 0.0,
            "beam_switched": bool(
                snapshot_index > 0
                and best_beam != oracle_selected[snapshot_index - 1]
            ),
            "is_full_sweep": True,
            "beams_probed": num_beams,
            "candidate_beam_ids": json.dumps(list(range(num_beams))),
        })

    # Hold-last baseline: one initial full scan, then no further training.
    current_hold = int(np.argmax(gains[0]))
    for snapshot_index in range(num_snapshots):
        if snapshot_index == 0:
            current_hold = int(np.argmax(gains[snapshot_index]))
            probes = num_beams
            full_sweep = True
        else:
            probes = 0
            full_sweep = False

        selected_gain = float(gains[snapshot_index, current_hold])
        oracle_gain = float(np.max(gains[snapshot_index]))
        regret = max(0.0, float(gain_db(oracle_gain) - gain_db(selected_gain)))
        previous = hold_selected[-1] if hold_selected else None
        hold_selected.append(current_hold)

        hold_rows.append({
            "condition": condition_name,
            "snapshot": int(snapshot_ids[snapshot_index]),
            "snapshot_index": snapshot_index,
            "rx_x_m": float(rx_positions_m[snapshot_index, 0]),
            "rx_y_m": float(rx_positions_m[snapshot_index, 1]),
            "rx_z_m": float(rx_positions_m[snapshot_index, 2]),
            "method": METHOD_HOLD,
            "selected_beam_id": current_hold,
            "selected_gain_linear": selected_gain,
            "selected_gain_db": float(gain_db(selected_gain)),
            "oracle_beam_id": int(np.argmax(gains[snapshot_index])),
            "oracle_gain_db": float(gain_db(oracle_gain)),
            "regret_db": regret,
            "beam_switched": bool(previous is not None and current_hold != previous),
            "is_full_sweep": full_sweep,
            "beams_probed": probes,
            "candidate_beam_ids": (
                json.dumps(list(range(num_beams))) if full_sweep else "[]"
            ),
        })

    # Proposed tracker: local angular search, hysteresis, periodic reacquisition.
    current_tracker = int(np.argmax(gains[0]))
    for snapshot_index in range(num_snapshots):
        previous = tracker_selected[-1] if tracker_selected else None

        if snapshot_index == 0 or snapshot_index % FULL_SWEEP_INTERVAL == 0:
            candidates = np.arange(num_beams, dtype=int)
            current_tracker = int(np.argmax(gains[snapshot_index]))
            full_sweep = True
        else:
            candidates = nearest_angular_beams(
                beam_directions,
                current_tracker,
                LOCAL_CANDIDATE_COUNT,
            )
            current_gain = float(gains[snapshot_index, current_tracker])
            local_best = int(
                candidates[np.argmax(gains[snapshot_index, candidates])]
            )
            local_best_gain = float(gains[snapshot_index, local_best])

            improvement_db = float(
                gain_db(local_best_gain) - gain_db(current_gain)
            )
            if local_best != current_tracker and improvement_db >= HYSTERESIS_DB:
                current_tracker = local_best

            full_sweep = False

        tracker_selected.append(current_tracker)

        selected_gain = float(gains[snapshot_index, current_tracker])
        oracle_gain = float(np.max(gains[snapshot_index]))
        regret = max(0.0, float(gain_db(oracle_gain) - gain_db(selected_gain)))

        tracker_rows.append({
            "condition": condition_name,
            "snapshot": int(snapshot_ids[snapshot_index]),
            "snapshot_index": snapshot_index,
            "rx_x_m": float(rx_positions_m[snapshot_index, 0]),
            "rx_y_m": float(rx_positions_m[snapshot_index, 1]),
            "rx_z_m": float(rx_positions_m[snapshot_index, 2]),
            "method": METHOD_TRACKER,
            "selected_beam_id": current_tracker,
            "selected_gain_linear": selected_gain,
            "selected_gain_db": float(gain_db(selected_gain)),
            "oracle_beam_id": int(np.argmax(gains[snapshot_index])),
            "oracle_gain_db": float(gain_db(oracle_gain)),
            "regret_db": regret,
            "beam_switched": bool(
                previous is not None and current_tracker != previous
            ),
            "is_full_sweep": full_sweep,
            "beams_probed": int(len(candidates)),
            "candidate_beam_ids": json.dumps(candidates.tolist()),
        })

    return oracle_rows + hold_rows + tracker_rows


# ======================================================================
# 5. RUN BOTH ENVIRONMENTAL CONDITIONS
# ======================================================================

all_tracking_rows = []

for condition_index, condition_name in enumerate(condition_names):
    print("\n" + "-" * 76)
    print("Tracking condition:", condition_name)
    condition_gains = gain_tensor[condition_index]
    condition_rows = run_tracking(
        condition_index,
        str(condition_name),
        condition_gains,
    )
    all_tracking_rows.extend(condition_rows)

tracking_df = pd.DataFrame(all_tracking_rows)
tracking_df = tracking_df.sort_values(
    ["condition", "method", "snapshot_index"]
).reset_index(drop=True)

# Reorder methods in the CSV for easier reading.
method_order = {
    METHOD_ORACLE: 0,
    METHOD_HOLD: 1,
    METHOD_TRACKER: 2,
}
tracking_df["_method_order"] = tracking_df["method"].map(method_order)
tracking_df = tracking_df.sort_values(
    ["condition", "_method_order", "snapshot_index"]
).drop(columns="_method_order").reset_index(drop=True)

tracking_df.to_csv(OUTPUT_PER_SNAPSHOT, index=False)


# ======================================================================
# 6. SUMMARIZE TRACKING PERFORMANCE AND TRAINING OVERHEAD
# ======================================================================

summary_rows = []

for (condition_name, method), group in tracking_df.groupby(
    ["condition", "method"],
    sort=False,
):
    regrets = group["regret_db"].to_numpy(dtype=float)
    probes = group["beams_probed"].to_numpy(dtype=float)
    switches = int(group["beam_switched"].sum())

    summary_rows.append({
        "condition": condition_name,
        "method": method,
        "num_snapshots": int(len(group)),
        "num_beams_in_codebook": int(num_beams),
        "total_beam_probes": int(probes.sum()),
        "mean_beams_probed_per_snapshot": float(probes.mean()),
        "mean_beam_training_overhead_pct": float(
            100.0 * probes.mean() / num_beams
        ),
        "total_beam_switches": switches,
        "beam_switch_rate_per_transition_pct": float(
            100.0 * switches / max(1, len(group) - 1)
        ),
        "mean_selected_gain_db": float(group["selected_gain_db"].mean()),
        "mean_oracle_regret_db": float(regrets.mean()),
        "median_oracle_regret_db": float(np.median(regrets)),
        "p95_oracle_regret_db": float(np.percentile(regrets, 95)),
        "maximum_oracle_regret_db": float(regrets.max()),
        "snapshots_within_1db_of_oracle_pct": float(
            100.0 * np.mean(regrets <= 1.0 + 1e-9)
        ),
        "snapshots_within_3db_of_oracle_pct": float(
            100.0 * np.mean(regrets <= 3.0 + 1e-9)
        ),
        "full_sweep_count": int(group["is_full_sweep"].sum()),
    })

tracking_summary_df = pd.DataFrame(summary_rows)
tracking_summary_df.to_csv(OUTPUT_SUMMARY, index=False)


# ======================================================================
# 7. EXPORT BEAM-SWITCH EVENTS
# ======================================================================

switch_rows = []
for (condition_name, method), group in tracking_df.groupby(
    ["condition", "method"],
    sort=False,
):
    group = group.sort_values("snapshot_index").reset_index(drop=True)
    for index in range(1, len(group)):
        previous_row = group.iloc[index - 1]
        current_row = group.iloc[index]
        if bool(current_row["beam_switched"]):
            switch_rows.append({
                "condition": condition_name,
                "method": method,
                "from_snapshot": int(previous_row["snapshot"]),
                "to_snapshot": int(current_row["snapshot"]),
                "from_rx_x_m": float(previous_row["rx_x_m"]),
                "to_rx_x_m": float(current_row["rx_x_m"]),
                "previous_beam_id": int(previous_row["selected_beam_id"]),
                "new_beam_id": int(current_row["selected_beam_id"]),
                "previous_gain_db_at_new_snapshot": float(
                    gain_db(
                        gain_tensor[
                            list(condition_names).index(condition_name),
                            int(current_row["snapshot_index"]),
                            int(previous_row["selected_beam_id"]),
                        ]
                    )
                ),
                "new_beam_gain_db": float(current_row["selected_gain_db"]),
                "oracle_beam_id": int(current_row["oracle_beam_id"]),
                "oracle_gain_db": float(current_row["oracle_gain_db"]),
                "regret_after_switch_db": float(current_row["regret_db"]),
                "is_full_sweep": bool(current_row["is_full_sweep"]),
            })

switch_df = pd.DataFrame(switch_rows)
switch_df.to_csv(OUTPUT_SWITCHES, index=False)


# ======================================================================
# 8. VALIDATION
# ======================================================================

checks = {}

checks["input_channel_tensor_has_expected_dimensions"] = (
    h_complex.shape == (2, num_snapshots, 64)
)
checks["codebook_weights_match_saved_npz_weights"] = bool(
    np.allclose(weights_from_csv, saved_weights, rtol=1e-10, atol=1e-12)
)
checks["recomputed_beam_gains_match_s4_saved_gains"] = bool(
    np.allclose(
        recomputed_gain,
        saved_gain,
        rtol=RECOMPUTE_RTOL,
        atol=RECOMPUTE_ATOL,
    )
)
checks["all_tracking_methods_cover_each_snapshot_once"] = bool(
    all(
        len(
            tracking_df[
                (tracking_df["condition"] == condition)
                & (tracking_df["method"] == method)
            ]
        ) == num_snapshots
        for condition in condition_names
        for method in METHODS
    )
)
checks["oracle_regret_is_zero"] = bool(
    np.allclose(
        tracking_df.loc[
            tracking_df["method"] == METHOD_ORACLE, "regret_db"
        ].to_numpy(dtype=float),
        0.0,
        atol=1e-9,
    )
)
checks["tracking_regret_is_nonnegative"] = bool(
    (tracking_df["regret_db"].to_numpy(dtype=float) >= -1e-9).all()
)
checks["selected_beam_ids_are_valid"] = bool(
    tracking_df["selected_beam_id"].between(0, num_beams - 1).all()
)
checks["probe_counts_do_not_exceed_codebook_size"] = bool(
    tracking_df["beams_probed"].between(0, num_beams).all()
)
checks["tracker_periodic_full_sweeps_follow_configuration"] = bool(
    all(
        bool(
            tracking_df.loc[
                (tracking_df["condition"] == condition)
                & (tracking_df["method"] == METHOD_TRACKER)
                & (tracking_df["snapshot_index"] == index),
                "is_full_sweep",
            ].iloc[0]
        )
        == (index == 0 or index % FULL_SWEEP_INTERVAL == 0)
        for condition in condition_names
        for index in range(num_snapshots)
    )
)
checks["local_tracker_probe_count_matches_candidate_count"] = bool(
    all(
        int(row["beams_probed"])
        == (
            num_beams
            if bool(row["is_full_sweep"])
            else LOCAL_CANDIDATE_COUNT
        )
        for _, row in tracking_df[
            tracking_df["method"] == METHOD_TRACKER
        ].iterrows()
    )
)

validation_passed = all(checks.values())

validation_lines = [
    "S5 BEAM TRACKING VALIDATION",
    "=" * 72,
    f"Conditions: {', '.join(condition_names.tolist())}",
    f"Snapshots per condition: {num_snapshots}",
    f"TX antennas: {num_antennas}",
    f"Codebook beams: {num_beams}",
    f"Local candidate count: {LOCAL_CANDIDATE_COUNT}",
    f"Hysteresis threshold (dB): {HYSTERESIS_DB}",
    f"Full-sweep interval (snapshots): {FULL_SWEEP_INTERVAL}",
    "",
    "Checks:",
]
for check_name, passed in checks.items():
    validation_lines.append(
        f"[{'PASS' if passed else 'FAIL'}] {check_name}"
    )

validation_lines.extend([
    "",
    f"OVERALL: {'PASS' if validation_passed else 'FAIL'}",
    "",
    "Interpretation and limitations:",
    "- The trajectory is a sequence of spatial quasi-static channel snapshots.",
    "- No receiver velocity, elapsed-time model, Doppler evolution, or channel aging is introduced here.",
    "- exhaustive_oracle is a full-codebook upper-bound reference.",
    "- local_tracking uses a local angular candidate set, hysteresis, and periodic full sweeps.",
    "- The all-beam saved channel data is used offline to score the selected beam; it is not counted as online probing.",
    "- Probe counts model beam-training overhead in units of candidate beam measurements, not wall-clock latency.",
    "- Gain is the dimensionless |h^T w|^2 metric, not calibrated received power.",
    "- The tracker is a defined baseline policy; its hyperparameters have not been optimized.",
])

OUTPUT_VALIDATION.write_text(
    "\n".join(validation_lines) + "\n",
    encoding="utf-8",
)


# ======================================================================
# 9. SAVE EXPERIMENT CONFIGURATION
# ======================================================================

config = {
    "experiment": "S5_Spatially_Evolving_Beam_Tracking",
    "input_stage": "S4.1",
    "input_directory": str(INPUT_DIR),
    "output_directory": str(OUTPUT_DIR),
    "conditions": condition_names.tolist(),
    "trajectory": {
        "snapshot_ids": snapshot_ids.tolist(),
        "receiver_positions_m": rx_positions_m.tolist(),
        "interpretation": (
            "Spatial quasi-static snapshots inherited from S2/S4.1; "
            "not a time-continuous mobility or Doppler simulation."
        ),
    },
    "radio": {
        "carrier_frequency_hz": s4_config.get("carrier_frequency_hz"),
        "tx_position_m": tx_position_m.tolist(),
        "num_tx_antennas": int(num_antennas),
        "num_candidate_beams": int(num_beams),
        "channel_tensor_shape": list(h_complex.shape),
        "beam_gain_metric": "abs(sum(h * w))^2 = |h^T w|^2",
        "gain_units": "dimensionless linear channel gain; dB is 10*log10(gain)",
    },
    "tracking_methods": {
        METHOD_ORACLE: {
            "description": "Full 52-beam exhaustive scan at every snapshot.",
            "role": "Upper-bound reference; not low-overhead tracking.",
            "beams_probed_per_snapshot": int(num_beams),
        },
        METHOD_HOLD: {
            "description": "Initial full scan, then hold the selected beam.",
            "beams_probed": (
                f"{num_beams} at the first snapshot, zero afterwards"
            ),
        },
        METHOD_TRACKER: {
            "description": (
                "Initial full scan; local angular search between periodic "
                "full sweeps; switch only when the best local beam improves "
                "the current beam by at least the hysteresis threshold."
            ),
            "local_candidate_count": int(LOCAL_CANDIDATE_COUNT),
            "hysteresis_threshold_db": float(HYSTERESIS_DB),
            "full_sweep_interval_snapshots": int(FULL_SWEEP_INTERVAL),
            "full_sweep_indices": [
                index for index in range(num_snapshots)
                if index == 0 or index % FULL_SWEEP_INTERVAL == 0
            ],
        },
    },
    "validation": checks,
    "validation_passed": bool(validation_passed),
    "important_limitations": [
        "No explicit velocity or timestamp exists in the inherited trajectory.",
        "No Doppler or time-continuous channel evolution is simulated.",
        "Beam probe counts are measurement-count proxies, not measured latency.",
        "The tracker hyperparameters are fixed experimental settings, not optimized.",
        "Offline all-beam gains are used for evaluation, not as tracker observations.",
    ],
}

OUTPUT_CONFIG.write_text(
    json.dumps(config, indent=4),
    encoding="utf-8",
)


# ======================================================================
# 10. REPORT
# ======================================================================

print("\n" + "=" * 76)
print("S5 TRACKING SUMMARY")
print("=" * 76)
print(tracking_summary_df.to_string(index=False))

print("\nBeam switch events:", len(switch_df))
print("Validation:")
for check_name, passed in checks.items():
    print(f"[{'PASS' if passed else 'FAIL'}] {check_name}")

print("\nOutput files:")
for path in [
    OUTPUT_PER_SNAPSHOT,
    OUTPUT_SUMMARY,
    OUTPUT_SWITCHES,
    OUTPUT_CONFIG,
    OUTPUT_VALIDATION,
]:
    print(" -", path)

if not validation_passed:
    raise RuntimeError(
        "S5 tracking completed but validation failed. "
        "Inspect s5_validation.txt before using the results."
    )

print("\nS5 beam tracking and validation successful.")
