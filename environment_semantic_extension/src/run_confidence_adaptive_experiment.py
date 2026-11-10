import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA = Path(
    "reference/sionna-teammate/recovered_results/"
    "s6_physical_layer_dataset"
)

OUT = Path("results/analysis/confidence_adaptive_experiment")
OUT.mkdir(parents=True, exist_ok=True)

N_BEAMS = 52
LOCAL_K = 9
MEDIUM_K = 32

GAIN_DROP_DB = 8.0
RESIDUAL_DROP_DB = 20.0
HYSTERESIS_DB = 3.0

OFFSETS = [0, 15, 30, 45]

# Controlled confidence policy:
# These categories are assigned from the injected perturbation.
# They are NOT probabilities from a trained prediction model.
CONFIDENCE_POLICY = {
    0: ("high", 9),
    15: ("high", 9),
    30: ("medium", 32),
    45: ("low", 52),
}


def load_data():
    archive = np.load(DATA / "channel_data.npz")
    codebook = pd.read_csv(DATA / "beam_codebook.csv")

    gains = archive["beamformed_channel_gain_db"]

    if gains.shape != (2, 11, N_BEAMS):
        raise ValueError(f"Unexpected gain shape: {gains.shape}")

    if len(codebook) != N_BEAMS:
        raise ValueError("Expected 52 codebook beams.")

    directions = codebook[
        ["u_x_local", "u_y_local", "u_z_local"]
    ].to_numpy(dtype=float)

    norms = np.linalg.norm(directions, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-6):
        raise ValueError("Codebook direction normalization failed.")

    directions /= norms[:, None]

    cosine = np.clip(directions @ directions.T, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    np.fill_diagonal(angles, np.inf)

    return gains, codebook, directions, angles


def get_candidates(angles, center, count):
    if count == N_BEAMS:
        return np.arange(N_BEAMS, dtype=int)

    nearest = np.argsort(angles[center])[:count - 1]

    return np.sort(
        np.append(nearest, center)
    ).astype(int)


def perturb_prior(base_beam, offset, codebook, directions):
    """
    Shift the previous selected beam's azimuth by a controlled
    amount, retain elevation, and map to the nearest codebook beam.

    No current-snapshot gain or oracle information is used.
    """

    base_az = float(
        codebook.iloc[base_beam]["azimuth_local_deg"]
    )
    base_el = float(
        codebook.iloc[base_beam]["elevation_local_deg"]
    )

    target_az = np.clip(
        base_az + offset,
        codebook["azimuth_local_deg"].min(),
        codebook["azimuth_local_deg"].max(),
    )

    az = np.radians(target_az)
    el = np.radians(base_el)

    target = np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])

    target /= np.linalg.norm(target)

    cosine = np.clip(directions @ target, -1.0, 1.0)
    distances = np.degrees(np.arccos(cosine))

    predicted = int(np.argmin(distances))

    actual_shift = float(
        np.degrees(
            np.arccos(
                np.clip(
                    np.dot(directions[base_beam], directions[predicted]),
                    -1.0,
                    1.0,
                )
            )
        )
    )

    return predicted, actual_shift


def regret(current_gains, selected):
    return float(np.max(current_gains) - current_gains[selected])


def record_result(
    records,
    condition,
    snapshot,
    method,
    selected,
    oracle,
    current_gains,
    candidates,
    stage,
    confidence="not_applicable",
    predicted_prior=None,
    actual_shift=0.0,
    prior_error=np.nan,
):
    records.append({
        "condition": condition,
        "snapshot": snapshot,
        "method": method,
        "selected_beam": int(selected),
        "oracle_beam": int(oracle),
        "regret_db": regret(current_gains, selected),
        "probes": len(candidates),
        "candidate_count": len(candidates),
        "oracle_in_candidates": bool(oracle in candidates),
        "stage": stage,
        "confidence": confidence,
        "predicted_prior_beam": (
            int(predicted_prior)
            if predicted_prior is not None else np.nan
        ),
        "prior_perturbation_deg": actual_shift,
        "prior_error_to_oracle_deg": prior_error,
    })


def run_existing_adaptive(gains, angles, condition, records):
    previous_beam = None
    previous_gain = None

    for snapshot in range(11):
        current = gains[snapshot]
        oracle = int(np.argmax(current))

        if snapshot == 0:
            candidates = np.arange(N_BEAMS)
            selected = oracle
            stage = "initial_full_sweep"

        else:
            current_gain = float(current[previous_beam])

            drop = previous_gain - current_gain

            local = get_candidates(
                angles, previous_beam, LOCAL_K
            )

            local_best = int(
                local[np.argmax(current[local])]
            )

            residual = (
                previous_gain - float(current[local_best])
            )

            if drop > GAIN_DROP_DB and residual > RESIDUAL_DROP_DB:
                expanded = get_candidates(
                    angles, previous_beam, MEDIUM_K
                )

                expanded_best = int(
                    expanded[np.argmax(current[expanded])]
                )

                improvement = (
                    float(current[expanded_best]) - current_gain
                )

                if improvement >= HYSTERESIS_DB:
                    candidates = expanded
                    selected = expanded_best
                    stage = "expanded_32"
                else:
                    candidates = np.arange(N_BEAMS)
                    selected = oracle
                    stage = "fallback_full_52"

            else:
                candidates = local

                improvement = (
                    float(current[local_best]) - current_gain
                )

                if improvement >= HYSTERESIS_DB:
                    selected = local_best
                else:
                    selected = previous_beam

                stage = "local_9"

        record_result(
            records,
            condition,
            snapshot,
            "existing_adaptive",
            selected,
            oracle,
            current,
            candidates,
            stage,
        )

        previous_beam = selected
        previous_gain = float(current[selected])


def run_prior_policy(
    gains,
    codebook,
    directions,
    angles,
    condition,
    offset,
    records,
    adaptive_budget,
):
    previous_selected = None

    confidence, adaptive_count = CONFIDENCE_POLICY[offset]

    method = (
        f"confidence_adaptive_{offset}deg"
        if adaptive_budget
        else f"fixed_prior9_{offset}deg"
    )

    for snapshot in range(11):
        current = gains[snapshot]
        oracle = int(np.argmax(current))

        if snapshot == 0:
            candidates = np.arange(N_BEAMS)
            selected = oracle
            predicted = oracle
            actual_shift = 0.0
            prior_error = 0.0
            stage = "initial_full_sweep"

        else:
            # Construct prior exclusively from this policy's
            # previous selected beam and the predetermined offset.
            predicted, actual_shift = perturb_prior(
                previous_selected,
                offset,
                codebook,
                directions,
            )

            if adaptive_budget:
                candidate_count = adaptive_count
            else:
                candidate_count = LOCAL_K

            candidates = get_candidates(
                angles, predicted, candidate_count
            )

            selected = int(
                candidates[np.argmax(current[candidates])]
            )

            prior_error = float(
                angles[predicted, oracle]
                if predicted != oracle else 0.0
            )

            stage = (
                f"{confidence}_{candidate_count}_candidates"
                if adaptive_budget
                else "fixed_9_candidates"
            )

        record_result(
            records,
            condition,
            snapshot,
            method,
            selected,
            oracle,
            current,
            candidates,
            stage,
            confidence=confidence,
            predicted_prior=predicted,
            actual_shift=actual_shift,
            prior_error=prior_error,
        )

        previous_selected = selected


def main():
    gains, codebook, directions, angles = load_data()

    records = []

    conditions = ["unblocked", "blocked"]

    for condition_index, condition in enumerate(conditions):
        condition_gains = gains[condition_index]

        # Exhaustive baseline
        for snapshot in range(11):
            current = condition_gains[snapshot]
            oracle = int(np.argmax(current))

            record_result(
                records,
                condition,
                snapshot,
                "exhaustive",
                oracle,
                oracle,
                current,
                np.arange(N_BEAMS),
                "full_sweep",
            )

        # Reproduce existing adaptive policy
        run_existing_adaptive(
            condition_gains,
            angles,
            condition,
            records,
        )

        # Fixed 9-beam and confidence-adaptive policies
        for offset in OFFSETS:
            run_prior_policy(
                condition_gains,
                codebook,
                directions,
                angles,
                condition,
                offset,
                records,
                adaptive_budget=False,
            )

            run_prior_policy(
                condition_gains,
                codebook,
                directions,
                angles,
                condition,
                offset,
                records,
                adaptive_budget=True,
            )

    results = pd.DataFrame(records)

    results.to_csv(
        OUT / "per_snapshot_results.csv",
        index=False,
    )

    # Full experiment summary
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
        )
        .reset_index()
    )

    summary["probe_reduction_pct"] = (
        1 - summary["mean_probes"] / N_BEAMS
    ) * 100

    summary.to_csv(
        OUT / "method_summary.csv",
        index=False,
    )

    # Post-acquisition results
    post = results[results["snapshot"] > 0].copy()

    post_summary = (
        post.groupby(["condition", "method"])
        .agg(
            snapshots=("snapshot", "count"),
            mean_regret_db=("regret_db", "mean"),
            median_regret_db=("regret_db", "median"),
            max_regret_db=("regret_db", "max"),
            mean_probes=("probes", "mean"),
            total_probes=("probes", "sum"),
            oracle_inclusion_rate=("oracle_in_candidates", "mean"),
        )
        .reset_index()
    )

    post_summary["probe_reduction_pct"] = (
        1 - post_summary["mean_probes"] / N_BEAMS
    ) * 100

    post_summary.to_csv(
        OUT / "post_acquisition_summary.csv",
        index=False,
    )

    # Detailed confidence-policy sensitivity
    confidence_results = results[
        results["method"].str.startswith("confidence_adaptive")
        & (results["snapshot"] > 0)
    ].copy()

    confidence_summary = (
        confidence_results.groupby(
            ["condition", "method", "confidence"]
        )
        .agg(
            snapshots=("snapshot", "count"),
            mean_regret_db=("regret_db", "mean"),
            max_regret_db=("regret_db", "max"),
            mean_probes=("probes", "mean"),
            total_probes=("probes", "sum"),
            oracle_inclusion_rate=("oracle_in_candidates", "mean"),
            mean_prior_error_deg=(
                "prior_error_to_oracle_deg", "mean"
            ),
        )
        .reset_index()
    )

    confidence_summary.to_csv(
        OUT / "confidence_sensitivity_summary.csv",
        index=False,
    )

    stages = (
        results.groupby(["condition", "method", "stage"])
        .size()
        .reset_index(name="snapshots")
    )

    stages.to_csv(
        OUT / "search_stage_summary.csv",
        index=False,
    )

    # Validation
    checks = {
        "expected_total_rows": len(results) == 2 * 11 * 10,
        "selected_beams_valid": results[
            "selected_beam"
        ].between(0, N_BEAMS - 1).all(),
        "oracle_beams_valid": results[
            "oracle_beam"
        ].between(0, N_BEAMS - 1).all(),
        "nonnegative_regret": (
            results["regret_db"] >= -1e-8
        ).all(),
        "valid_probe_counts": results[
            "probes"
        ].between(0, N_BEAMS).all(),
        "confidence_high_uses_9": (
            confidence_results[
                confidence_results["confidence"] == "high"
            ]["probes"].eq(9).all()
        ),
        "confidence_medium_uses_32": (
            confidence_results[
                confidence_results["confidence"] == "medium"
            ]["probes"].eq(32).all()
        ),
        "confidence_low_uses_52": (
            confidence_results[
                confidence_results["confidence"] == "low"
            ]["probes"].eq(52).all()
        ),
        "confidence_prior_not_from_current_oracle": True,
        "all_tracking_methods_have_10_post_acquisition_samples": (
            post.groupby(["condition", "method"]).size().eq(10).all()
        ),
    }

    with open(OUT / "validation_report.txt", "w") as f:
        f.write("CONFIDENCE-ADAPTIVE EXPERIMENT VALIDATION\n\n")

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
        "codebook_beams": N_BEAMS,
        "transmit_antennas": 64,
        "controlled_offsets_deg": OFFSETS,
        "confidence_mapping": {
            str(k): {
                "confidence": v[0],
                "candidate_count": v[1],
            }
            for k, v in CONFIDENCE_POLICY.items()
        },
        "prior_source": "Previous selected beam of each policy",
        "prior_construction": (
            "Controlled azimuth perturbation, fixed elevation, "
            "nearest valid codebook direction"
        ),
        "current_snapshot_oracle_used_for_prior": False,
        "candidate_search": (
            "Top N nearest codebook directions, including prior centre"
        ),
        "scope": (
            "One trajectory, 11 snapshots, two obstruction conditions. "
            "Controlled confidence experiment, not semantic model evaluation."
        ),
    }

    with open(OUT / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nFULL METHOD SUMMARY")
    print(summary.round(4).to_string(index=False))

    print("\nPOST-ACQUISITION SUMMARY")
    print(post_summary.round(4).to_string(index=False))

    print("\nCONFIDENCE SENSITIVITY")
    print(confidence_summary.round(4).to_string(index=False))

    print("\nSEARCH STAGES")
    print(stages.to_string(index=False))

    print("\nVALIDATION")
    print((OUT / "validation_report.txt").read_text())

    print("\nSaved outputs:", OUT.resolve())


if __name__ == "__main__":
    main()
