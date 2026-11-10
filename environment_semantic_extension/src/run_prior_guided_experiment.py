import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "reference/sionna-teammate/recovered_results/"
    "s6_physical_layer_dataset"
)

OUT = Path("results/analysis/prior_guided_experiment")
OUT.mkdir(parents=True, exist_ok=True)

K_LOCAL = 9
K_PRIOR = 9
K_EXPANDED = 32
HYSTERESIS_DB = 3.0
GAIN_DROP_TRIGGER_DB = 8.0
RESIDUAL_DROP_TRIGGER_DB = 20.0


def load_data():
    metadata = pd.read_csv(ROOT / "snapshot_metadata.csv")
    measurements = pd.read_csv(ROOT / "beam_measurements.csv")
    tracking = pd.read_csv(ROOT / "tracking_records.csv")
    codebook = pd.read_csv(ROOT / "beam_codebook.csv")

    archive = np.load(ROOT / "channel_data.npz")

    gains = archive["beamformed_channel_gain_db"]

    sample_ids = archive["sample_ids_condition_major"].astype(str)

    return metadata, measurements, tracking, codebook, gains, sample_ids


def build_geometry(codebook):
    directions = codebook[
        ["u_x_local", "u_y_local", "u_z_local"]
    ].to_numpy(dtype=float)

    directions /= np.linalg.norm(
        directions, axis=1, keepdims=True
    )

    cosine = np.clip(directions @ directions.T, -1.0, 1.0)

    angles = np.degrees(np.arccos(cosine))

    np.fill_diagonal(angles, np.inf)

    return angles


def nearest_candidates(angles, beam, count):
    return np.argsort(angles[beam])[:count].astype(int)


def regret_db(gains, selected):
    oracle_gain = np.max(gains)
    selected_gain = gains[selected]

    return float(oracle_gain - selected_gain)


def run_experiment():
    metadata, measurements, tracking, codebook, gains, sample_ids = load_data()

    angles = build_geometry(codebook)

    beam_ids = codebook["beam_id"].astype(int).to_numpy()

    if len(beam_ids) != 52:
        raise ValueError("Expected exactly 52 codebook beams.")

    if gains.shape != (2, 11, 52):
        raise ValueError(
            f"Unexpected gain tensor shape: {gains.shape}"
        )

    records = []

    for condition_index, condition in enumerate(["unblocked", "blocked"]):

        previous_selected = None
        previous_selected_gain = None

        for snapshot in range(11):

            current_gains = gains[condition_index, snapshot]

            oracle_beam = int(np.argmax(current_gains))
            oracle_gain = float(current_gains[oracle_beam])

            # --------------------------------------------------
            # 1. Exhaustive baseline
            # --------------------------------------------------
            exhaustive_candidates = np.arange(52)
            exhaustive_selected = oracle_beam

            records.append({
                "condition": condition,
                "snapshot": snapshot,
                "method": "exhaustive",
                "selected_beam": exhaustive_selected,
                "oracle_beam": oracle_beam,
                "regret_db": 0.0,
                "probes": 52,
                "candidate_count": 52,
                "oracle_in_candidates": True,
                "stage": "full_sweep",
            })

            # --------------------------------------------------
            # 2. Hold-last baseline
            # --------------------------------------------------
            if snapshot == 0:
                hold_selected = oracle_beam
                hold_probes = 52
                hold_candidates = np.arange(52)
            else:
                hold_selected = previous_hold
                hold_probes = 0
                hold_candidates = np.array([], dtype=int)

            records.append({
                "condition": condition,
                "snapshot": snapshot,
                "method": "hold_last",
                "selected_beam": hold_selected,
                "oracle_beam": oracle_beam,
                "regret_db": regret_db(current_gains, hold_selected),
                "probes": hold_probes,
                "candidate_count": len(hold_candidates),
                "oracle_in_candidates": (
                    oracle_beam in hold_candidates
                ),
                "stage": "initial_full_sweep" if snapshot == 0 else "hold",
            })

            previous_hold = hold_selected

            # --------------------------------------------------
            # 3. Fixed local-9 tracking
            # --------------------------------------------------
            if snapshot == 0:
                local_selected = oracle_beam
                local_candidates = np.arange(52)
                local_probes = 52
                local_stage = "initial_full_sweep"
            else:
                local_candidates = nearest_candidates(
                    angles, previous_local, K_LOCAL - 1
                )
                local_candidates = np.unique(
                    np.append(local_candidates, previous_local)
                )

                local_selected = int(
                    local_candidates[
                        np.argmax(current_gains[local_candidates])
                    ]
                )

                local_probes = len(local_candidates)
                local_stage = "local_search"

            records.append({
                "condition": condition,
                "snapshot": snapshot,
                "method": "local_9",
                "selected_beam": local_selected,
                "oracle_beam": oracle_beam,
                "regret_db": regret_db(current_gains, local_selected),
                "probes": local_probes,
                "candidate_count": len(local_candidates),
                "oracle_in_candidates": (
                    oracle_beam in local_candidates
                ),
                "stage": local_stage,
            })

            previous_local = local_selected

            # --------------------------------------------------
            # 4. Existing adaptive policy
            # --------------------------------------------------
            if snapshot == 0:
                adaptive_candidates = np.arange(52)
                adaptive_selected = oracle_beam
                adaptive_stage = "initial_full_sweep"

            else:
                current_beam = previous_selected

                current_beam_gain = float(
                    current_gains[current_beam]
                )

                drop = (
                    previous_selected_gain - current_beam_gain
                )

                local = nearest_candidates(
                    angles, current_beam, K_LOCAL - 1
                )
                local = np.unique(np.append(local, current_beam))

                local_best = int(
                    local[np.argmax(current_gains[local])]
                )

                local_residual = (
                    previous_selected_gain -
                    float(current_gains[local_best])
                )

                if drop > GAIN_DROP_TRIGGER_DB and (
                    local_residual > RESIDUAL_DROP_TRIGGER_DB
                ):
                    expanded = nearest_candidates(
                        angles, current_beam, K_EXPANDED - 1
                    )
                    expanded = np.unique(
                        np.append(expanded, current_beam)
                    )

                    expanded_best = int(
                        expanded[
                            np.argmax(current_gains[expanded])
                        ]
                    )

                    improvement = (
                        float(current_gains[expanded_best]) -
                        current_beam_gain
                    )

                    if improvement >= HYSTERESIS_DB:
                        adaptive_candidates = expanded
                        adaptive_selected = expanded_best
                        adaptive_stage = "expanded_search"
                    else:
                        adaptive_candidates = np.arange(52)
                        adaptive_selected = oracle_beam
                        adaptive_stage = "fallback_full_sweep"

                else:
                    adaptive_candidates = local

                    improvement = (
                        float(current_gains[local_best]) -
                        current_beam_gain
                    )

                    if improvement >= HYSTERESIS_DB:
                        adaptive_selected = local_best
                    else:
                        adaptive_selected = current_beam

                    adaptive_stage = "local_search"

            records.append({
                "condition": condition,
                "snapshot": snapshot,
                "method": "adaptive",
                "selected_beam": adaptive_selected,
                "oracle_beam": oracle_beam,
                "regret_db": regret_db(current_gains, adaptive_selected),
                "probes": len(adaptive_candidates),
                "candidate_count": len(adaptive_candidates),
                "oracle_in_candidates": (
                    oracle_beam in adaptive_candidates
                ),
                "stage": adaptive_stage,
            })

            previous_selected = adaptive_selected
            previous_selected_gain = float(
                current_gains[adaptive_selected]
            )

            # --------------------------------------------------
            # 5. Temporal-prior-guided search
            # --------------------------------------------------
            if snapshot == 0:
                prior_selected = oracle_beam
                prior_candidates = np.arange(52)
                prior_stage = "initial_full_sweep"

            else:
                # The prior is the previous snapshot's selected
                # beam. It uses no current-snapshot oracle data.
                prior_center = prior_selected

                prior_candidates = nearest_candidates(
                    angles, prior_center, K_PRIOR - 1
                )

                prior_candidates = np.unique(
                    np.append(prior_candidates, prior_center)
                )

                prior_best = int(
                    prior_candidates[
                        np.argmax(current_gains[prior_candidates])
                    ]
                )

                prior_selected = prior_best
                prior_stage = "temporal_prior_top9"

            records.append({
                "condition": condition,
                "snapshot": snapshot,
                "method": "temporal_prior_top9",
                "selected_beam": prior_selected,
                "oracle_beam": oracle_beam,
                "regret_db": regret_db(current_gains, prior_selected),
                "probes": len(prior_candidates),
                "candidate_count": len(prior_candidates),
                "oracle_in_candidates": (
                    oracle_beam in prior_candidates
                ),
                "stage": prior_stage,
            })

    results = pd.DataFrame(records)

    results.to_csv(
        OUT / "per_snapshot_results.csv",
        index=False
    )

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
        1 - summary["mean_probes"] / 52
    ) * 100

    summary.to_csv(
        OUT / "method_summary.csv",
        index=False
    )

    stage_summary = (
        results.groupby(["condition", "method", "stage"])
        .size()
        .reset_index(name="snapshots")
    )

    stage_summary.to_csv(
        OUT / "search_stage_summary.csv",
        index=False
    )

    config = {
        "dataset": "S6 physical-layer dataset",
        "scene": "simple_street_canyon",
        "frequency_ghz": 28,
        "codebook_beams": 52,
        "transmit_antennas": 64,
        "local_candidates": K_LOCAL,
        "expanded_candidates": K_EXPANDED,
        "gain_drop_trigger_db": GAIN_DROP_TRIGGER_DB,
        "residual_drop_trigger_db": RESIDUAL_DROP_TRIGGER_DB,
        "hysteresis_db": HYSTERESIS_DB,
        "prior_method": (
            "Previous snapshot's selected beam; "
            "not a semantic prediction"
        ),
        "current_snapshot_oracle_used_for_prior": False,
        "scope": (
            "One trajectory, 11 snapshots, two paired "
            "obstruction conditions. Descriptive simulation only."
        ),
    }

    with open(OUT / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nMETHOD SUMMARY")
    print(summary.to_string(index=False))

    print("\nSEARCH STAGES")
    print(stage_summary.to_string(index=False))

    print("\nSaved outputs to:", OUT.resolve())


if __name__ == "__main__":
    run_experiment()
