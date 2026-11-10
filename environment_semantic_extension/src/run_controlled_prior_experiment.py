import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "reference/sionna-teammate/recovered_results/"
    "s6_physical_layer_dataset"
)

OUT = Path("results/analysis/controlled_prior_experiment")
OUT.mkdir(parents=True, exist_ok=True)

NUM_BEAMS = 52
LOCAL_K = 9
EXPANDED_K = 32

GAIN_DROP_TRIGGER_DB = 8.0
RESIDUAL_DROP_TRIGGER_DB = 20.0
HYSTERESIS_DB = 3.0

# Controlled azimuth offsets applied to the previous adaptive
# track's beam direction. These are prior perturbations, not
# measured semantic prediction errors.
AZIMUTH_OFFSETS_DEG = [0, 15, 30, 45]


def load_inputs():
    archive = np.load(ROOT / "channel_data.npz")
    codebook = pd.read_csv(ROOT / "beam_codebook.csv")

    gains = archive["beamformed_channel_gain_db"]

    if gains.shape != (2, 11, NUM_BEAMS):
        raise ValueError(f"Unexpected gain shape: {gains.shape}")

    if len(codebook) != NUM_BEAMS:
        raise ValueError("Expected 52 codebook entries.")

    required = [
        "beam_id",
        "azimuth_local_deg",
        "elevation_local_deg",
        "u_x_local",
        "u_y_local",
        "u_z_local",
    ]

    missing = [c for c in required if c not in codebook.columns]
    if missing:
        raise ValueError(f"Missing codebook columns: {missing}")

    return gains, codebook


def build_geometry(codebook):
    directions = codebook[
        ["u_x_local", "u_y_local", "u_z_local"]
    ].to_numpy(dtype=float)

    norms = np.linalg.norm(directions, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-6):
        raise ValueError("Codebook directions are not unit-normalized.")

    cosine = np.clip(directions @ directions.T, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    np.fill_diagonal(angles, np.inf)

    return directions, angles


def local_candidates(angles, center, count=LOCAL_K):
    neighbors = np.argsort(angles[center])[:count - 1]
    return np.sort(np.append(neighbors, center)).astype(int)


def azimuth_perturbed_prior(
    center_beam,
    offset_deg,
    codebook,
    directions,
):
    """
    Construct a prior by shifting the previous beam's azimuth,
    retaining its elevation, then mapping to the nearest valid
    codebook direction.

    This uses only the previous selected beam and fixed offset.
    It never reads current-snapshot beam gains.
    """

    base_az = float(
        codebook.iloc[center_beam]["azimuth_local_deg"]
    )

    base_el = float(
        codebook.iloc[center_beam]["elevation_local_deg"]
    )

    target_az = base_az + offset_deg

    # Keep the target inside the codebook's azimuth coverage.
    min_az = float(codebook["azimuth_local_deg"].min())
    max_az = float(codebook["azimuth_local_deg"].max())

    target_az = float(np.clip(target_az, min_az, max_az))

    az = np.radians(target_az)
    el = np.radians(base_el)

    target_direction = np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])

    target_direction /= np.linalg.norm(target_direction)

    cosine = np.clip(directions @ target_direction, -1.0, 1.0)
    angular_distances = np.degrees(np.arccos(cosine))

    predicted_beam = int(np.argmin(angular_distances))

    actual_perturbation = float(
        np.degrees(
            np.arccos(
                np.clip(
                    np.dot(
                        directions[center_beam],
                        directions[predicted_beam],
                    ),
                    -1.0,
                    1.0,
                )
            )
        )
    )

    return predicted_beam, actual_perturbation


def angular_error(angles, predicted_beam, oracle_beam):
    return float(angles[predicted_beam, oracle_beam]) if (
        predicted_beam != oracle_beam
    ) else 0.0


def regret_db(current_gains, selected_beam):
    return float(
        np.max(current_gains) - current_gains[selected_beam]
    )


def add_record(
    records,
    condition,
    snapshot,
    method,
    selected,
    oracle,
    gains,
    candidates,
    stage,
    predicted_prior=None,
    perturbation_deg=0.0,
    angles=None,
):
    if predicted_prior is None:
        prior_error = np.nan
    else:
        prior_error = angular_error(
            angles, predicted_prior, oracle
        )

    records.append({
        "condition": condition,
        "snapshot": snapshot,
        "method": method,
        "selected_beam": int(selected),
        "oracle_beam": int(oracle),
        "regret_db": regret_db(gains, selected),
        "probes": int(len(candidates)),
        "candidate_count": int(len(candidates)),
        "oracle_in_candidates": bool(oracle in candidates),
        "stage": stage,
        "predicted_prior_beam": (
            int(predicted_prior)
            if predicted_prior is not None else np.nan
        ),
        "prior_angular_error_to_oracle_deg": prior_error,
        "prior_perturbation_from_base_deg": perturbation_deg,
    })


def run():
    gains, codebook = load_inputs()
    directions, angles = build_geometry(codebook)

    records = []

    conditions = ["unblocked", "blocked"]

    for condition_index, condition in enumerate(conditions):

        previous_adaptive = None
        previous_adaptive_gain = None
        previous_local = None
        previous_hold = None

        for snapshot in range(11):

            current = gains[condition_index, snapshot]
            oracle = int(np.argmax(current))

            # Preserve the previous adaptive beam as the prior
            # source for this snapshot before updating the tracker.
            prior_base = previous_adaptive

            # --------------------------------------------------
            # 1. Exhaustive reference
            # --------------------------------------------------
            full = np.arange(NUM_BEAMS)

            add_record(
                records, condition, snapshot,
                "exhaustive", oracle, oracle,
                current, full, "full_sweep",
            )

            # --------------------------------------------------
            # 2. Hold-last baseline
            # --------------------------------------------------
            if snapshot == 0:
                hold_selected = oracle
                hold_candidates = full
                hold_stage = "initial_full_sweep"
            else:
                hold_selected = previous_hold
                hold_candidates = np.array([], dtype=int)
                hold_stage = "hold"

            add_record(
                records, condition, snapshot,
                "hold_last", hold_selected, oracle,
                current, hold_candidates, hold_stage,
            )

            previous_hold = hold_selected

            # --------------------------------------------------
            # 3. Fixed local-9 baseline
            # --------------------------------------------------
            if snapshot == 0:
                local_selected = oracle
                local_set = full
                local_stage = "initial_full_sweep"
            else:
                local_set = local_candidates(
                    angles, previous_local
                )

                local_selected = int(
                    local_set[np.argmax(current[local_set])]
                )
                local_stage = "local_9"

            add_record(
                records, condition, snapshot,
                "local_9", local_selected, oracle,
                current, local_set, local_stage,
            )

            previous_local = local_selected

            # --------------------------------------------------
            # 4. Adaptive 9/32/52 search
            # --------------------------------------------------
            if snapshot == 0:
                adaptive_set = full
                adaptive_selected = oracle
                adaptive_stage = "initial_full_sweep"

            else:
                current_beam = previous_adaptive

                current_beam_gain = float(
                    current[current_beam]
                )

                drop_db = (
                    previous_adaptive_gain - current_beam_gain
                )

                local_set_adaptive = local_candidates(
                    angles, current_beam
                )

                local_best = int(
                    local_set_adaptive[
                        np.argmax(current[local_set_adaptive])
                    ]
                )

                local_residual = (
                    previous_adaptive_gain -
                    float(current[local_best])
                )

                if (
                    drop_db > GAIN_DROP_TRIGGER_DB
                    and local_residual > RESIDUAL_DROP_TRIGGER_DB
                ):
                    expanded_set = local_candidates(
                        angles, current_beam, EXPANDED_K
                    )

                    expanded_best = int(
                        expanded_set[
                            np.argmax(current[expanded_set])
                        ]
                    )

                    improvement = (
                        float(current[expanded_best])
                        - current_beam_gain
                    )

                    if improvement >= HYSTERESIS_DB:
                        adaptive_set = expanded_set
                        adaptive_selected = expanded_best
                        adaptive_stage = "expanded_32"
                    else:
                        adaptive_set = full
                        adaptive_selected = oracle
                        adaptive_stage = "fallback_full_52"

                else:
                    adaptive_set = local_set_adaptive

                    improvement = (
                        float(current[local_best])
                        - current_beam_gain
                    )

                    if improvement >= HYSTERESIS_DB:
                        adaptive_selected = local_best
                    else:
                        adaptive_selected = current_beam

                    adaptive_stage = "local_9"

            add_record(
                records, condition, snapshot,
                "adaptive", adaptive_selected, oracle,
                current, adaptive_set, adaptive_stage,
            )

            previous_adaptive = adaptive_selected
            previous_adaptive_gain = float(
                current[adaptive_selected]
            )

            # --------------------------------------------------
            # 5. Controlled prior perturbation
            # --------------------------------------------------
            if snapshot == 0:
                for offset in AZIMUTH_OFFSETS_DEG:
                    add_record(
                        records,
                        condition,
                        snapshot,
                        f"prior_az_{offset}deg",
                        oracle,
                        oracle,
                        current,
                        full,
                        "initial_full_sweep",
                        predicted_prior=oracle,
                        perturbation_deg=0.0,
                        angles=angles,
                    )

            else:
                # The base prior is from the previous snapshot's
                # adaptive selection, not the current oracle.
                for offset in AZIMUTH_OFFSETS_DEG:

                    prior_beam, actual_perturbation = (
                        azimuth_perturbed_prior(
                            prior_base,
                            offset,
                            codebook,
                            directions,
                        )
                    )

                    candidates = local_candidates(
                        angles, prior_beam, LOCAL_K
                    )

                    selected = int(
                        candidates[
                            np.argmax(current[candidates])
                        ]
                    )

                    add_record(
                        records,
                        condition,
                        snapshot,
                        f"prior_az_{offset}deg",
                        selected,
                        oracle,
                        current,
                        candidates,
                        "prior_guided_top9",
                        predicted_prior=prior_beam,
                        perturbation_deg=actual_perturbation,
                        angles=angles,
                    )

    results = pd.DataFrame(records)

    results.to_csv(
        OUT / "per_snapshot_results.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    summary = (
        results.groupby(["condition", "method"])
        .agg(
            samples=("snapshot", "count"),
            mean_regret_db=("regret_db", "mean"),
            median_regret_db=("regret_db", "median"),
            max_regret_db=("regret_db", "max"),
            mean_probes=("probes", "mean"),
            total_probes=("probes", "sum"),
            oracle_inclusion_rate=("oracle_in_candidates", "mean"),
            mean_prior_error_deg=(
                "prior_angular_error_to_oracle_deg", "mean"
            ),
            mean_perturbation_deg=(
                "prior_perturbation_from_base_deg", "mean"
            ),
        )
        .reset_index()
    )

    summary["probe_reduction_pct"] = (
        1 - summary["mean_probes"] / NUM_BEAMS
    ) * 100

    summary.to_csv(
        OUT / "method_summary.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Prior sensitivity summary
    # ----------------------------------------------------------
    prior_results = results[
        results["method"].str.startswith("prior_az_")
    ].copy()

    prior_summary = (
        prior_results.groupby(["condition", "method"])
        .agg(
            mean_regret_db=("regret_db", "mean"),
            max_regret_db=("regret_db", "max"),
            mean_probes=("probes", "mean"),
            oracle_inclusion_rate=("oracle_in_candidates", "mean"),
            mean_prior_error_deg=(
                "prior_angular_error_to_oracle_deg", "mean"
            ),
            mean_perturbation_deg=(
                "prior_perturbation_from_base_deg", "mean"
            ),
        )
        .reset_index()
    )

    prior_summary.to_csv(
        OUT / "prior_sensitivity_summary.csv",
        index=False,
    )

    stage_summary = (
        results.groupby(["condition", "method", "stage"])
        .size()
        .reset_index(name="snapshots")
    )

    stage_summary.to_csv(
        OUT / "search_stage_summary.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------
    checks = {
        "expected_total_rows": len(results) == 2 * 11 * 8,
        "no_missing_selected_beams": results[
            "selected_beam"
        ].notna().all(),
        "selected_beams_in_range": results[
            "selected_beam"
        ].between(0, NUM_BEAMS - 1).all(),
        "oracle_beams_in_range": results[
            "oracle_beam"
        ].between(0, NUM_BEAMS - 1).all(),
        "nonnegative_regret": (
            results["regret_db"] >= -1e-8
        ).all(),
        "probe_counts_valid": results[
            "probes"
        ].between(0, NUM_BEAMS).all(),
        "all_prior_searches_have_nine_candidates": (
            results[
                results["method"].str.startswith("prior_az_")
                & (results["snapshot"] > 0)
            ]["probes"].eq(LOCAL_K).all()
        ),
    }

    with open(OUT / "validation_report.txt", "w") as f:
        f.write("CONTROLLED PRIOR EXPERIMENT VALIDATION\n\n")

        for name, passed in checks.items():
            f.write(
                f"[{'PASS' if passed else 'FAIL'}] {name}\n"
            )

        f.write(
            "\nOVERALL: "
            + ("PASS" if all(checks.values()) else "FAIL")
            + "\n"
        )

    config = {
        "dataset": "S6 physical-layer dataset",
        "scene": "simple_street_canyon",
        "frequency_ghz": 28,
        "codebook_beams": NUM_BEAMS,
        "transmit_antennas": 64,
        "prior_source": (
            "Previous snapshot's adaptive selected beam"
        ),
        "azimuth_offsets_deg": AZIMUTH_OFFSETS_DEG,
        "prior_candidate_count": LOCAL_K,
        "prior_mapping": (
            "Shift previous beam azimuth, preserve elevation, "
            "map to nearest valid codebook direction"
        ),
        "current_snapshot_oracle_used_to_construct_prior": False,
        "initial_acquisition": "Exhaustive 52-beam scan",
        "conditions": conditions,
        "snapshots_per_condition": 11,
        "scope": (
            "One trajectory, two paired obstruction conditions. "
            "Controlled prior perturbation experiment; "
            "not actual semantic prediction evaluation."
        ),
    }

    with open(OUT / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nMETHOD SUMMARY")
    print(summary.round(4).to_string(index=False))

    print("\nPRIOR SENSITIVITY")
    print(prior_summary.round(4).to_string(index=False))

    print("\nSEARCH STAGES")
    print(stage_summary.to_string(index=False))

    print("\nVALIDATION")
    print((OUT / "validation_report.txt").read_text())

    print("\nSaved outputs:", OUT.resolve())


if __name__ == "__main__":
    run()
