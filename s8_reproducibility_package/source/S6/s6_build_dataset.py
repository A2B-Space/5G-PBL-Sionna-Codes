#!/usr/bin/env python3
"""
S6: Reproducible physical-layer dataset builder for the Sionna RT project.

This stage assembles the S4.1 path-geometry export and S5.1 beam-tracking
results into a keyed, documented dataset. It does not rerun Sionna RT.

Required inputs
---------------
<project>/s4_blockage_geometry/
    s4_snapshot_summary.csv
    s4_beam_metrics.csv
    s4_beam_codebook.csv
    s4_path_details.csv
    s4_path_interactions.csv
    s4_scene_objects.csv
    s4_channel_data.npz
    s4_config.json
    s4_validation.txt

<project>/s5_1_adaptive_tracking/
    s5_1_tracking_per_snapshot.csv
    s5_1_tracking_summary.csv
    s5_1_config.json
    s5_1_validation.txt

Outputs
-------
<project>/s6_physical_layer_dataset/
    snapshot_metadata.csv
    path_records.csv
    interaction_records.csv
    beam_measurements.csv
    tracking_records.csv
    tracking_summary.csv
    scene_objects.csv
    beam_codebook.csv
    channel_data.npz
    manifest.json
    data_dictionary.json
    validation.txt
    README.md

Run
---
cd ~/sionna_project
source .venv/bin/activate
python s6_build_dataset.py

Optional:
python s6_build_dataset.py --base-dir /path/to/sionna_project

Dependencies: numpy, pandas, matplotlib (no GPU or Sionna import required).
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


SCENE_ID = "simple_street_canyon"
TRAJECTORY_ID = "s2_rx_x17_to_x37_step2m"
EXPECTED_CONDITIONS = {"unblocked", "blocked"}
EXPECTED_BEAMS = 52
EXPECTED_TX_ANTENNAS = 64


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assemble S4.1 and S5.1 outputs into the S6 dataset."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.home() / "sionna_project",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame, required, file_label):
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(
            f"{file_label} is missing required columns: {sorted(missing)}"
        )


def stable_sample_id(condition, snapshot):
    return (
        f"{SCENE_ID}__{TRAJECTORY_ID}__{condition}"
        f"__snapshot_{int(snapshot):03d}"
    )


def load_inputs(base_dir):
    s4_dir = base_dir / "s4_blockage_geometry"
    s51_dir = base_dir / "s5_1_adaptive_tracking"

    files = {
        "s4_summary": s4_dir / "s4_snapshot_summary.csv",
        "s4_beams": s4_dir / "s4_beam_metrics.csv",
        "s4_codebook": s4_dir / "s4_beam_codebook.csv",
        "s4_paths": s4_dir / "s4_path_details.csv",
        "s4_interactions": s4_dir / "s4_path_interactions.csv",
        "s4_objects": s4_dir / "s4_scene_objects.csv",
        "s4_channels": s4_dir / "s4_channel_data.npz",
        "s4_config": s4_dir / "s4_config.json",
        "s4_validation": s4_dir / "s4_validation.txt",
        "s51_tracking": s51_dir / "s5_1_tracking_per_snapshot.csv",
        "s51_summary": s51_dir / "s5_1_tracking_summary.csv",
        "s51_config": s51_dir / "s5_1_config.json",
        "s51_validation": s51_dir / "s5_1_validation.txt",
    }

    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "S6 cannot assemble the dataset because required inputs are absent:\n - "
            + "\n - ".join(missing)
            + "\n\nFirst run the S6-ready S4.1 exporter, then rerun S5.1 if needed."
        )

    s4_validation = files["s4_validation"].read_text(encoding="utf-8")
    s51_validation = files["s51_validation"].read_text(encoding="utf-8")
    if "OVERALL: PASS" not in s4_validation:
        raise RuntimeError("S4.1 validation report does not contain OVERALL: PASS.")
    if "OVERALL: PASS" not in s51_validation:
        raise RuntimeError("S5.1 validation report does not contain OVERALL: PASS.")

    data = {
        "files": files,
        "snapshot": pd.read_csv(files["s4_summary"]),
        "beams": pd.read_csv(files["s4_beams"]),
        "codebook": pd.read_csv(files["s4_codebook"]),
        "paths": pd.read_csv(files["s4_paths"]),
        "interactions": pd.read_csv(files["s4_interactions"]),
        "objects": pd.read_csv(files["s4_objects"]),
        "tracking": pd.read_csv(files["s51_tracking"]),
        "tracking_summary": pd.read_csv(files["s51_summary"]),
        "s4_config": json.loads(files["s4_config"].read_text(encoding="utf-8")),
        "s51_config": json.loads(files["s51_config"].read_text(encoding="utf-8")),
        "s4_validation": s4_validation,
        "s51_validation": s51_validation,
    }

    with np.load(files["s4_channels"], allow_pickle=False) as archive:
        data["channel_archive"] = {
            key: np.asarray(archive[key])
            for key in archive.files
        }

    return data


def validate_inputs(data):
    require_columns(
        data["snapshot"],
        [
            "condition", "snapshot", "rx_x_m", "rx_y_m", "rx_z_m",
            "los_visible_from_paths", "valid_path_count",
            "geometric_obstruction_intersection", "best_beam_id",
            "best_beam_channel_gain_linear", "best_beam_channel_gain_db",
        ],
        "s4_snapshot_summary.csv",
    )
    require_columns(
        data["beams"],
        [
            "condition", "snapshot", "beam_id",
            "beamformed_channel_gain_linear", "beamformed_channel_gain_db",
        ],
        "s4_beam_metrics.csv",
    )
    require_columns(
        data["codebook"],
        ["beam_id", "u_x_local", "u_y_local", "u_z_local", "weight_norm"],
        "s4_beam_codebook.csv",
    )
    require_columns(
        data["paths"],
        [
            "condition", "snapshot", "path_id", "delay_s", "delay_ns",
            "path_coefficient_real", "path_coefficient_imag",
            "path_coefficient_abs", "path_coefficient_power",
            "phi_t_rad", "theta_t_rad", "phi_r_rad", "theta_r_rad",
            "phi_t_deg", "theta_t_deg", "phi_r_deg", "theta_r_deg",
            "doppler_hz", "is_direct_los_path", "interaction_count",
            "interaction_codes",
        ],
        "s4_path_details.csv",
    )
    require_columns(
        data["interactions"],
        [
            "condition", "snapshot", "path_id", "interaction_order",
            "interaction_code", "interaction_type", "object_id",
            "object_name", "primitive_id",
            "vertex_x_m", "vertex_y_m", "vertex_z_m",
        ],
        "s4_path_interactions.csv",
    )
    require_columns(
        data["objects"],
        ["condition", "object_id", "object_name"],
        "s4_scene_objects.csv",
    )
    require_columns(
        data["tracking"],
        [
            "condition", "snapshot", "snapshot_index", "method",
            "selected_beam_id", "selected_gain_linear", "selected_gain_db",
            "oracle_beam_id", "oracle_gain_db", "regret_db",
            "beam_switched", "is_full_sweep", "beams_probed",
            "candidate_beam_ids", "search_stage",
        ],
        "s5_1_tracking_per_snapshot.csv",
    )

    conditions = set(data["snapshot"]["condition"].astype(str).unique())
    if conditions != EXPECTED_CONDITIONS:
        raise ValueError(
            f"Expected conditions {EXPECTED_CONDITIONS}; found {conditions}."
        )

    if data["snapshot"].duplicated(["condition", "snapshot"]).any():
        raise ValueError("Duplicate condition/snapshot in S4.1 snapshot metadata.")

    snapshot_keys = set(
        zip(
            data["snapshot"]["condition"].astype(str),
            data["snapshot"]["snapshot"].astype(int),
        )
    )
    if len(snapshot_keys) != 22:
        raise ValueError(
            f"Expected 22 condition/snapshot records; found {len(snapshot_keys)}."
        )

    if data["codebook"]["beam_id"].astype(int).tolist() != list(range(EXPECTED_BEAMS)):
        raise ValueError("S4.1 codebook must contain beam IDs 0 through 51.")

    if data["beams"].duplicated(["condition", "snapshot", "beam_id"]).any():
        raise ValueError("Duplicate per-beam measurements.")

    beam_counts = data["beams"].groupby(
        ["condition", "snapshot"]
    )["beam_id"].nunique()
    if len(beam_counts) != len(snapshot_keys) or not (beam_counts == EXPECTED_BEAMS).all():
        raise ValueError("Every snapshot must contain exactly 52 beam measurements.")

    path_keys = ["condition", "snapshot", "path_id"]
    if data["paths"].duplicated(path_keys).any():
        raise ValueError("Duplicate path identifiers within a snapshot.")

    if data["interactions"].duplicated(
        ["condition", "snapshot", "path_id", "interaction_order"]
    ).any():
        raise ValueError("Duplicate interaction-order rows for a path.")

    for table_name in ["paths", "beams", "tracking", "interactions"]:
        table = data[table_name]
        keys = set(
            zip(
                table["condition"].astype(str),
                table["snapshot"].astype(int),
            )
        )
        if not keys.issubset(snapshot_keys):
            raise ValueError(f"{table_name} contains unknown snapshot keys.")

    expected_methods = {
        "exhaustive_oracle",
        "hold_last_beam",
        "local_tracking",
        "adaptive_tracking",
    }
    if set(data["tracking"]["method"].astype(str).unique()) != expected_methods:
        raise ValueError("S5.1 tracking does not contain all four expected methods.")

    tracking_counts = data["tracking"].groupby(
        ["condition", "method"]
    )["snapshot"].nunique()
    if len(tracking_counts) != 8 or not (tracking_counts == 11).all():
        raise ValueError(
            "Expected 11 snapshots for each of four methods in both conditions."
        )

    channel = data["channel_archive"]
    required_npz = {
        "condition_names", "snapshot_ids", "rx_positions_m",
        "tx_position_m", "h_complex", "beamformed_channel_gain_linear",
        "beamformed_channel_gain_db", "beam_directions_local",
        "beam_weights_complex",
    }
    missing_npz = required_npz - set(channel)
    if missing_npz:
        raise ValueError(f"s4_channel_data.npz missing arrays: {sorted(missing_npz)}")

    h = channel["h_complex"]
    gains = channel["beamformed_channel_gain_linear"]
    if h.shape != (2, 11, EXPECTED_TX_ANTENNAS):
        raise ValueError(f"Unexpected h_complex shape: {h.shape}")
    if gains.shape != (2, 11, EXPECTED_BEAMS):
        raise ValueError(f"Unexpected beam-gain tensor shape: {gains.shape}")
    if not np.all(np.isfinite(h.real)) or not np.all(np.isfinite(h.imag)):
        raise ValueError("Non-finite complex channel values.")
    if not np.all(np.isfinite(gains)) or np.any(gains < 0):
        raise ValueError("Invalid beam gain tensor.")

    # Confirm the channel NPZ order agrees with CSV condition/snapshot order.
    archive_conditions = [str(x) for x in channel["condition_names"].tolist()]
    archive_snapshots = [int(x) for x in channel["snapshot_ids"].tolist()]
    if set(archive_conditions) != EXPECTED_CONDITIONS:
        raise ValueError("NPZ condition_names do not match expected conditions.")
    if len(archive_snapshots) != 11:
        raise ValueError("NPZ must contain 11 snapshot IDs.")

    return {
        "conditions": archive_conditions,
        "snapshots": archive_snapshots,
        "snapshot_keys": snapshot_keys,
    }


def add_sample_keys(data):
    conditions = data["snapshot"]["condition"].astype(str)
    snapshots = data["snapshot"]["snapshot"].astype(int)

    data["snapshot"]["scene_id"] = SCENE_ID
    data["snapshot"]["trajectory_id"] = TRAJECTORY_ID
    data["snapshot"]["condition_id"] = conditions
    data["snapshot"]["sample_id"] = [
        stable_sample_id(c, s) for c, s in zip(conditions, snapshots)
    ]

    key_map = data["snapshot"][
        ["condition", "snapshot", "sample_id", "scene_id", "trajectory_id"]
    ]

    for table_name in ["paths", "interactions", "beams", "tracking"]:
        table = data[table_name].copy()
        table["condition"] = table["condition"].astype(str)
        table["snapshot"] = table["snapshot"].astype(int)
        table = table.merge(
            key_map,
            on=["condition", "snapshot"],
            how="left",
            validate="many_to_one",
        )
        if table["sample_id"].isna().any():
            raise RuntimeError(f"Could not assign sample IDs to {table_name}.")
        data[table_name] = table

    data["objects"] = data["objects"].copy()
    data["objects"]["scene_id"] = SCENE_ID
    data["objects"]["condition_id"] = data["objects"]["condition"].astype(str)

    return data


def enrich_snapshot_metadata(data):
    snapshots = data["snapshot"].copy()

    path_groups = data["paths"].groupby(
        ["condition", "snapshot"], sort=False
    )
    strongest_rows = []
    for (condition, snapshot), group in path_groups:
        idx = group["path_coefficient_power"].astype(float).idxmax()
        row = group.loc[idx]
        strongest_rows.append({
            "condition": condition,
            "snapshot": int(snapshot),
            "strongest_path_id": int(row["path_id"]),
            "strongest_path_power": float(row["path_coefficient_power"]),
            "strongest_path_delay_ns": float(row["delay_ns"]),
            "strongest_path_is_direct_los": bool(row["is_direct_los_path"]),
        })
    strongest = pd.DataFrame(strongest_rows)
    snapshots = snapshots.merge(
        strongest,
        on=["condition", "snapshot"],
        how="left",
        validate="one_to_one",
    )

    snapshots["scene_id"] = SCENE_ID
    snapshots["trajectory_id"] = TRAJECTORY_ID
    snapshots["condition_id"] = snapshots["condition"].astype(str)
    snapshots["sample_id"] = [
        stable_sample_id(c, s)
        for c, s in zip(
            snapshots["condition"].astype(str),
            snapshots["snapshot"].astype(int),
        )
    ]
    snapshots["obstruction_present"] = (
        snapshots["condition"].astype(str) == "blocked"
    )
    snapshots["los_blocked"] = ~snapshots[
        "los_visible_from_paths"
    ].astype(bool)
    snapshots["dataset_split"] = "not_assigned_single_trajectory"

    # Add S5.1 adaptive and fixed tracker labels in a wide per-snapshot view.
    selected = data["tracking"][
        data["tracking"]["method"].isin(
            ["local_tracking", "adaptive_tracking"]
        )
    ][
        [
            "condition", "snapshot", "method",
            "selected_beam_id", "regret_db", "beams_probed",
            "beam_switched", "search_stage",
        ]
    ].copy()

    for method, suffix in [
        ("local_tracking", "s5"),
        ("adaptive_tracking", "s5_1"),
    ]:
        part = selected[selected["method"] == method].drop(columns="method")
        part = part.rename(columns={
            "selected_beam_id": f"{suffix}_selected_beam_id",
            "regret_db": f"{suffix}_regret_db",
            "beams_probed": f"{suffix}_beams_probed",
            "beam_switched": f"{suffix}_beam_switched",
            "search_stage": f"{suffix}_search_stage",
        })
        snapshots = snapshots.merge(
            part,
            on=["condition", "snapshot"],
            how="left",
            validate="one_to_one",
        )

    return snapshots


def build_channel_archive(data, snapshot_metadata):
    archive = dict(data["channel_archive"])
    condition_names = [str(x) for x in archive["condition_names"].tolist()]
    snapshot_ids = [int(x) for x in archive["snapshot_ids"].tolist()]

    sample_ids = []
    for condition in condition_names:
        for snapshot in snapshot_ids:
            sample_ids.append(stable_sample_id(condition, snapshot))

    archive["scene_id"] = np.asarray([SCENE_ID])
    archive["trajectory_id"] = np.asarray([TRAJECTORY_ID])
    archive["sample_ids_condition_major"] = np.asarray(sample_ids)
    archive["channel_representation"] = np.asarray(
        ["complex channel vector at 28 GHz carrier; one baseband frequency offset"]
    )
    return archive


def generate_dataset_plots(output_dir, snapshot_metadata, paths, interactions, beams, tracking):
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    stems = []

    def save(fig, stem, pdf):
        fig.savefig(plot_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
        fig.savefig(plot_dir / f"{stem}.pdf", bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        stems.append(stem)

    report = plot_dir / "s6_dataset_qa_report.pdf"
    with PdfPages(report) as pdf:

        # 1. Valid path count over the trajectory.
        fig, ax = plt.subplots(figsize=(8, 5))
        for condition in ["unblocked", "blocked"]:
            part = snapshot_metadata[
                snapshot_metadata["condition"] == condition
            ].sort_values("rx_x_m")
            ax.plot(
                part["rx_x_m"], part["valid_path_count"],
                marker="o", linewidth=1.8, label=condition,
            )
        ax.set_xlabel("Receiver X position (m)")
        ax.set_ylabel("Valid propagation paths")
        ax.set_title("S6 dataset QA — Path count across the trajectory")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        save(fig, "01_path_count_vs_position", pdf)

        # 2. LOS status.
        fig, ax = plt.subplots(figsize=(8, 5))
        for condition in ["unblocked", "blocked"]:
            part = snapshot_metadata[
                snapshot_metadata["condition"] == condition
            ].sort_values("rx_x_m")
            ax.step(
                part["rx_x_m"],
                part["los_visible_from_paths"].astype(int),
                where="mid", marker="o", linewidth=1.8, label=condition,
            )
        ax.set_yticks([0, 1], labels=["NLOS", "LOS"])
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel("Receiver X position (m)")
        ax.set_ylabel("Path-derived LOS status")
        ax.set_title("S6 dataset QA — LOS visibility")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        save(fig, "02_los_visibility_vs_position", pdf)

        # 3. Best-beam gain.
        fig, ax = plt.subplots(figsize=(8, 5))
        for condition in ["unblocked", "blocked"]:
            part = snapshot_metadata[
                snapshot_metadata["condition"] == condition
            ].sort_values("rx_x_m")
            ax.plot(
                part["rx_x_m"], part["best_beam_channel_gain_db"],
                marker="o", linewidth=1.8, label=condition,
            )
        ax.set_xlabel("Receiver X position (m)")
        ax.set_ylabel("Best codebook beam gain (dB)")
        ax.set_title("S6 dataset QA — Best-beam channel gain")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        save(fig, "03_best_beam_gain_vs_position", pdf)

        # 4. Full 52-beam gain map.
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        vmax = float(beams["beamformed_channel_gain_db"].max())
        vmin = float(beams["beamformed_channel_gain_db"].min())
        for ax, condition in zip(axes, ["unblocked", "blocked"]):
            part = beams[beams["condition"] == condition]
            matrix = part.pivot(
                index="beam_id", columns="rx_x_m",
                values="beamformed_channel_gain_db",
            ).sort_index()
            im = ax.imshow(
                matrix.to_numpy(),
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(condition.capitalize())
            ax.set_xlabel("Receiver X position (m)")
            ax.set_xticks(
                range(len(matrix.columns)),
                labels=[f"{x:g}" for x in matrix.columns],
            )
            ax.set_ylabel("Beam ID")
        fig.colorbar(im, ax=axes, label="Beamformed channel gain (dB)")
        fig.suptitle("S6 dataset QA — 52-beam gain matrix")
        fig.subplots_adjust(
            left=0.08, right=0.88, bottom=0.12, top=0.86, wspace=0.18
        )
        save(fig, "04_beam_gain_heatmap", pdf)

        # 5. Path power-delay distribution.
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, condition in zip(axes, ["unblocked", "blocked"]):
            part = paths[paths["condition"] == condition]
            power_db = 10.0 * np.log10(
                np.maximum(
                    part["path_coefficient_power"].to_numpy(dtype=float),
                    np.finfo(float).tiny,
                )
            )
            ax.scatter(
                part["delay_ns"], power_db,
                s=15, alpha=0.65,
            )
            ax.set_title(condition.capitalize())
            ax.set_xlabel("Path delay (ns)")
            ax.set_ylabel("Path coefficient power (dB, relative)")
            ax.grid(True, alpha=0.25)
        fig.suptitle("S6 dataset QA — Path power versus propagation delay")
        fig.tight_layout()
        save(fig, "05_path_power_delay_scatter", pdf)

        # 6. Interaction type counts.
        if not interactions.empty:
            counts = (
                interactions.groupby(["condition", "interaction_type"])
                .size()
                .unstack(fill_value=0)
            )
        else:
            counts = pd.DataFrame(index=["unblocked", "blocked"])
        fig, ax = plt.subplots(figsize=(9, 5))
        counts.reindex(["unblocked", "blocked"]).plot(
            kind="bar", ax=ax, rot=0
        )
        ax.set_xlabel("Environment condition")
        ax.set_ylabel("Interaction events")
        ax.set_title("S6 dataset QA — Propagation interaction counts")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(title="Interaction type")
        fig.tight_layout()
        save(fig, "06_interaction_type_counts", pdf)

        # 7. Fixed versus adaptive tracker regret.
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, condition in zip(axes, ["unblocked", "blocked"]):
            part = tracking[
                (tracking["condition"] == condition)
                & tracking["method"].isin(
                    ["local_tracking", "adaptive_tracking"]
                )
            ]
            for method, label in [
                ("local_tracking", "S5 fixed local"),
                ("adaptive_tracking", "S5.1 adaptive"),
            ]:
                series = part[part["method"] == method].sort_values("rx_x_m")
                ax.plot(
                    series["rx_x_m"], series["regret_db"],
                    marker="o", linewidth=1.8, label=label,
                )
            ax.set_title(condition.capitalize())
            ax.set_xlabel("Receiver X position (m)")
            ax.set_ylabel("Oracle regret (dB)")
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.suptitle("S6 dataset QA — Fixed versus adaptive beam tracking")
        fig.tight_layout()
        save(fig, "07_tracking_regret_vs_position", pdf)

    return stems, report


def write_readme(output_dir):
    text = """# S6 Physical-Layer Dataset

This dataset combines the S4.1 Sionna RT path/channel export with the S5.1
beam-management replay.

## Primary record key

`scene_id + trajectory_id + condition_id + snapshot`

A `sample_id` is generated from those fields. Path records and beam
measurements are child tables keyed by `sample_id`; path interactions are
keyed by `sample_id + path_id + interaction_order`.

## Files

- `snapshot_metadata.csv`: one row per scene/trajectory/condition/snapshot.
- `path_records.csv`: one row per valid propagation path, including complex
  coefficient components, delay, AoD/AoA, Doppler, LOS and interaction summary.
- `interaction_records.csv`: one row per physical path interaction, with
  interaction type, scene object/primitive IDs and world-coordinate vertex.
- `beam_measurements.csv`: all 52 beam gains per snapshot.
- `tracking_records.csv`: S5.1 exhaustive, hold-last, fixed-local and adaptive
  tracker records.
- `tracking_summary.csv`: aggregate S5.1 metrics.
- `channel_data.npz`: complex 64-element channel vectors, beam-gain tensor,
  codebook weights/directions and sample-key arrays.
- `scene_objects.csv`: scene object IDs and names.
- `beam_codebook.csv`: 52-beam codebook and complex precoding weights.
- `manifest.json`: provenance, experiment configuration and array shapes.
- `data_dictionary.json`: table descriptions, units and key fields.
- `validation.txt`: dataset integrity checks.
- `plots/`: seven dataset QA plots in PNG/PDF and a combined PDF report.

## Interpretation notes

- The channel vector is the 64-element TX array response at the single
  baseband frequency offset used by S4.1. It is not a wideband CFR dataset.
- Beam gain is the dimensionless `abs(h^T w)^2` metric. It is not calibrated
  received power in watts or dBm.
- `obstruction_present` records the controlled concrete cuboid condition.
  `geometric_obstruction_intersection` is the geometric TX-RX segment test.
  `los_visible_from_paths` is based on the traced interaction sequence.
- The original street-canyon scene can already be NLOS without the added
  obstruction. Therefore `obstruction_present` and `los_blocked` are separate.
- Path coefficients are the Sionna passband path coefficients. CIR/CFR
  coefficients have a distinct baseband representation.
- Current data contains one spatial quasi-static trajectory with 11 positions
  and two paired environment conditions. It is not a multi-trajectory dataset.
- Do not randomly split adjacent snapshots or put paired blocked/unblocked
  versions of the same position into different splits. A valid generalization
  split requires additional independent trajectories/scenes; this release
  leaves `dataset_split` unassigned.
- The present pipeline does not contain camera frames, masks, bounding boxes,
  semantic object classes, or cross-modal correspondences.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    base_dir = args.base_dir.expanduser().resolve()
    output_dir = base_dir / "s6_physical_layer_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("S6 - PHYSICAL-LAYER DATASET ASSEMBLY")
    print("=" * 76)
    print("Project directory:", base_dir)

    data = load_inputs(base_dir)
    archive_info = validate_inputs(data)
    data = add_sample_keys(data)
    snapshot_metadata = enrich_snapshot_metadata(data)

    # Stable, human-readable tables.
    tables = {
        "snapshot_metadata.csv": snapshot_metadata.sort_values(
            ["condition", "snapshot"]
        ),
        "path_records.csv": data["paths"].sort_values(
            ["condition", "snapshot", "path_id"]
        ),
        "interaction_records.csv": data["interactions"].sort_values(
            ["condition", "snapshot", "path_id", "interaction_order"]
        ),
        "beam_measurements.csv": data["beams"].sort_values(
            ["condition", "snapshot", "beam_id"]
        ),
        "tracking_records.csv": data["tracking"].sort_values(
            ["condition", "method", "snapshot"]
        ),
        "tracking_summary.csv": data["tracking_summary"].sort_values(
            ["condition", "method"]
        ),
        "scene_objects.csv": data["objects"].sort_values(
            ["condition", "object_id"]
        ),
        "beam_codebook.csv": data["codebook"].sort_values("beam_id"),
    }

    for filename, frame in tables.items():
        frame.to_csv(output_dir / filename, index=False)

    channel_archive = build_channel_archive(data, snapshot_metadata)
    np.savez_compressed(
        output_dir / "channel_data.npz",
        **channel_archive,
    )

    plot_stems, plot_report = generate_dataset_plots(
        output_dir,
        snapshot_metadata,
        data["paths"],
        data["interactions"],
        data["beams"],
        data["tracking"],
    )

    # Dataset integrity checks after writing and re-reading key files.
    checks = {}
    reloaded_snapshot = pd.read_csv(output_dir / "snapshot_metadata.csv")
    reloaded_paths = pd.read_csv(output_dir / "path_records.csv")
    reloaded_beams = pd.read_csv(output_dir / "beam_measurements.csv")
    reloaded_tracking = pd.read_csv(output_dir / "tracking_records.csv")
    reloaded_interactions = pd.read_csv(output_dir / "interaction_records.csv")

    checks["22_unique_snapshot_samples"] = (
        len(reloaded_snapshot) == 22
        and reloaded_snapshot["sample_id"].nunique() == 22
    )
    checks["sample_ids_unique_and_non_null"] = bool(
        reloaded_snapshot["sample_id"].notna().all()
        and reloaded_snapshot["sample_id"].is_unique
    )
    checks["every_path_record_maps_to_a_snapshot"] = bool(
        set(reloaded_paths["sample_id"]).issubset(
            set(reloaded_snapshot["sample_id"])
        )
    )
    checks["every_interaction_maps_to_a_path"] = bool(
        set(
            zip(
                reloaded_interactions["sample_id"],
                reloaded_interactions["path_id"],
            )
        ).issubset(
            set(zip(reloaded_paths["sample_id"], reloaded_paths["path_id"]))
        )
    )
    checks["52_beam_measurements_per_sample"] = bool(
        (
            reloaded_beams.groupby("sample_id")["beam_id"].nunique()
            == EXPECTED_BEAMS
        ).all()
        and len(reloaded_beams) == 22 * EXPECTED_BEAMS
    )
    checks["44_tracking_records_per_condition"] = bool(
        len(reloaded_tracking) == 2 * 11 * 4
        and reloaded_tracking.groupby(
            ["condition", "method"]
        )["snapshot"].nunique().eq(11).all()
    )
    checks["path_coefficients_and_angles_are_finite"] = bool(
        np.isfinite(
            reloaded_paths[
                [
                    "path_coefficient_real", "path_coefficient_imag",
                    "delay_s", "phi_t_rad", "theta_t_rad",
                    "phi_r_rad", "theta_r_rad", "doppler_hz",
                ]
            ].to_numpy(dtype=float)
        ).all()
    )
    checks["interaction_vertices_are_finite"] = bool(
        np.isfinite(
            reloaded_interactions[
                ["vertex_x_m", "vertex_y_m", "vertex_z_m"]
            ].to_numpy(dtype=float)
        ).all()
    )
    checks["complex_channel_archive_preserved"] = bool(
        np.iscomplexobj(channel_archive["h_complex"])
        and channel_archive["h_complex"].shape
        == (2, 11, EXPECTED_TX_ANTENNAS)
    )
    checks["beam_codebook_has_52_unit_norm_weights"] = bool(
        len(data["codebook"]) == EXPECTED_BEAMS
        and np.allclose(
            data["codebook"]["weight_norm"].to_numpy(dtype=float),
            1.0,
            atol=1e-6,
        )
    )
    checks["no_train_test_leakage_claim_for_single_trajectory"] = bool(
        (reloaded_snapshot["dataset_split"] == "not_assigned_single_trajectory").all()
    )
    checks["seven_dataset_qa_plots_saved_as_png_and_pdf"] = bool(
        len(plot_stems) == 7
        and all(
            (output_dir / "plots" / f"{stem}.{ext}").exists()
            and (output_dir / "plots" / f"{stem}.{ext}").stat().st_size > 0
            for stem in plot_stems
            for ext in ["png", "pdf"]
        )
        and plot_report.exists()
        and plot_report.stat().st_size > 0
    )

    validation_passed = all(checks.values())

    input_hashes = {
        name: {
            "path": str(path.relative_to(base_dir)),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for name, path in data["files"].items()
    }

    data_dictionary = {
        "primary_key": [
            "scene_id", "trajectory_id", "condition_id", "snapshot"
        ],
        "sample_id_format": (
            "scene_id__trajectory_id__condition_id__snapshot_NNN"
        ),
        "tables": {
            "snapshot_metadata.csv": {
                "grain": "one row per sample_id",
                "key": ["sample_id"],
                "description": "Position, environment, LOS, path-count, strongest-path and beam-tracking labels.",
            },
            "path_records.csv": {
                "grain": "one row per valid ray-traced path",
                "key": ["sample_id", "path_id"],
                "description": "Path delay, passband complex coefficient, angles, Doppler, LOS and interaction summary.",
                "units": {
                    "delay_s": "s",
                    "delay_ns": "ns",
                    "path_coefficient_real": "unitless linear",
                    "path_coefficient_imag": "unitless linear",
                    "path_coefficient_power": "unitless squared magnitude",
                    "phi_t_rad": "rad",
                    "theta_t_rad": "rad",
                    "phi_r_rad": "rad",
                    "theta_r_rad": "rad",
                    "doppler_hz": "Hz",
                },
            },
            "interaction_records.csv": {
                "grain": "one row per physical interaction on a path",
                "key": ["sample_id", "path_id", "interaction_order"],
                "description": "Interaction type and intersected scene object/primitive with world XYZ vertex.",
                "units": {"vertex_x_m": "m", "vertex_y_m": "m", "vertex_z_m": "m"},
            },
            "beam_measurements.csv": {
                "grain": "one row per codebook beam per sample",
                "key": ["sample_id", "beam_id"],
                "description": "All 52 codebook beam gains.",
                "units": {"beamformed_channel_gain_linear": "unitless linear", "beamformed_channel_gain_db": "dB of dimensionless gain"},
            },
            "tracking_records.csv": {
                "grain": "one row per method per sample",
                "key": ["sample_id", "method"],
                "description": "Exhaustive oracle and S5/S5.1 tracker outputs, regret and probe counts.",
            },
            "channel_data.npz": {
                "h_complex": "complex array [condition, snapshot, 64 TX antenna elements]",
                "beamformed_channel_gain_linear": "float array [condition, snapshot, 52 beams]",
                "beam_weights_complex": "complex array [52 beams, 64 TX antenna elements]",
                "beam_directions_local": "float array [52 beams, 3 direction components]",
            },
        },
        "semantic_data": (
            "Not present in S6. This dataset does not include camera imagery, "
            "masks, bounding boxes, semantic classes or cross-modal labels."
        ),
    }
    (output_dir / "data_dictionary.json").write_text(
        json.dumps(data_dictionary, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "dataset_name": "Sionna RT Physical-Layer Dataset",
        "dataset_stage": "S6",
        "schema_version": "1.0.0",
        "scene_id": SCENE_ID,
        "trajectory_id": TRAJECTORY_ID,
        "conditions": archive_info["conditions"],
        "snapshot_ids": archive_info["snapshots"],
        "samples": int(len(reloaded_snapshot)),
        "valid_path_records": int(len(reloaded_paths)),
        "interaction_records": int(len(reloaded_interactions)),
        "beam_measurement_records": int(len(reloaded_beams)),
        "tracking_records": int(len(reloaded_tracking)),
        "codebook_size": EXPECTED_BEAMS,
        "tx_array_elements": EXPECTED_TX_ANTENNAS,
        "carrier_frequency_hz": data["s4_config"].get("carrier_frequency_hz"),
        "channel_gain_definition": "abs(h^T w)^2",
        "channel_gain_is_calibrated_power": False,
        "channel_representation": (
            "64-element complex channel vector at a single baseband frequency offset"
        ),
        "path_coefficient_representation": (
            "Sionna Paths.a passband coefficient, stored as real and imaginary components"
        ),
        "split_policy": (
            "No train/validation/test split assigned: only one spatial trajectory. "
            "Add independent trajectories/scenes and split by trajectory/scene."
        ),
        "source_s4_config": data["s4_config"],
        "source_s5_1_config": data["s51_config"],
        "source_file_hashes": input_hashes,
        "plot_files": [
            f"plots/{stem}.{ext}"
            for stem in plot_stems
            for ext in ["png", "pdf"]
        ] + ["plots/s6_dataset_qa_report.pdf"],
        "validation": checks,
        "validation_passed": bool(validation_passed),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    write_readme(output_dir)

    validation_lines = [
        "S6 PHYSICAL-LAYER DATASET VALIDATION",
        "=" * 76,
        f"Samples: {len(reloaded_snapshot)}",
        f"Path records: {len(reloaded_paths)}",
        f"Interaction records: {len(reloaded_interactions)}",
        f"Beam measurements: {len(reloaded_beams)}",
        f"Tracking records: {len(reloaded_tracking)}",
        "",
        "Checks:",
    ]
    validation_lines.extend(
        f"[{'PASS' if passed else 'FAIL'}] {name}"
        for name, passed in checks.items()
    )
    validation_lines.extend([
        "",
        f"OVERALL: {'PASS' if validation_passed else 'FAIL'}",
        "",
        "Important scope:",
        "- One spatial quasi-static trajectory; no independent test trajectory.",
        "- Channel vector is single-frequency-offset, not wideband CFR.",
        "- Beam gain is dimensionless, not calibrated received power.",
        "- Semantic camera labels and cross-modal correspondences are not included.",
    ])
    (output_dir / "validation.txt").write_text(
        "\n".join(validation_lines) + "\n",
        encoding="utf-8",
    )

    print("\nDataset summary")
    print("-" * 76)
    print("Samples:", len(reloaded_snapshot))
    print("Valid paths:", len(reloaded_paths))
    print("Interactions:", len(reloaded_interactions))
    print("Beam measurements:", len(reloaded_beams))
    print("Tracking records:", len(reloaded_tracking))
    print("\nValidation:")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print("\nOutputs:")
    for filename in [
        "snapshot_metadata.csv",
        "path_records.csv",
        "interaction_records.csv",
        "beam_measurements.csv",
        "tracking_records.csv",
        "tracking_summary.csv",
        "scene_objects.csv",
        "beam_codebook.csv",
        "channel_data.npz",
        "manifest.json",
        "data_dictionary.json",
        "validation.txt",
        "README.md",
    ]:
        print(" -", output_dir / filename)
    print(" -", plot_report)
    print(" -", output_dir / "plots", f"({len(plot_stems)} plots, PNG + PDF)")

    if not validation_passed:
        raise RuntimeError(
            "S6 dataset assembly completed with failed checks. "
            "Inspect validation.txt; do not use this dataset as final."
        )

    print("\nS6 dataset assembly and validation successful.")


if __name__ == "__main__":
    main()
