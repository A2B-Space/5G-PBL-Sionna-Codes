
"""
S3: 8x8 UPA and directional beam selection using Sionna RT 2.2.0

Inputs:
    s2_mobility_summary.csv

Outputs:
    s3_beam_selection_summary.csv
    s3_beam_selection_all_beams.csv
    s3_beam_codebook.csv
    s3_config.json

Metric:
    |h^T w|^2

This is a dimensionless beamformed channel-gain metric,
NOT received power in watts or dBm.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import mitsuba as mi
import sionna

from sionna.rt import (
    load_scene,
    PlanarArray,
    Transmitter,
    Receiver,
    PathSolver,
)


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

BASE_DIR = Path.home() / "sionna_project"

INPUT_CSV = BASE_DIR / "s2_mobility_summary.csv"

OUTPUT_SUMMARY = BASE_DIR / "s3_beam_selection_summary.csv"
OUTPUT_ALL_BEAMS = BASE_DIR / "s3_beam_selection_all_beams.csv"
OUTPUT_CODEBOOK = BASE_DIR / "s3_beam_codebook.csv"
OUTPUT_CONFIG = BASE_DIR / "s3_config.json"

CARRIER_FREQUENCY = 28e9
WAVELENGTH = 299792458.0 / CARRIER_FREQUENCY

TX_POSITION = np.array([-33.0, 11.0, 32.0], dtype=float)

NUM_ROWS = 8
NUM_COLS = 8
NUM_TX_ANTENNAS = NUM_ROWS * NUM_COLS

VERTICAL_SPACING = 0.5
HORIZONTAL_SPACING = 0.5

MAX_DEPTH = 5
SEED = 41

# Sionna uses baseband frequency offsets for Paths.cfr().
# Zero offset evaluates the channel at the carrier centre.
CFR_FREQUENCIES = np.array([0.0], dtype=np.float32)


# ============================================================
# 2. INPUT VALIDATION
# ============================================================

print("=" * 72)
print("SIONNA RT - S3 ANTENNA ARRAY AND BEAM SELECTION")
print("=" * 72)

print("Mitsuba:", mi.__version__)
print("Mitsuba variant:", mi.variant())

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"S2 input file not found: {INPUT_CSV}\n"
        "Run S2 successfully before running S3."
    )

trajectory = pd.read_csv(INPUT_CSV)

required_columns = {
    "snapshot",
    "rx_x_m",
    "rx_y_m",
    "rx_z_m",
}

missing_columns = required_columns - set(trajectory.columns)

if missing_columns:
    raise ValueError(
        f"S2 CSV is missing required columns: {missing_columns}"
    )

trajectory = trajectory.sort_values("snapshot").reset_index(drop=True)

if trajectory.empty:
    raise ValueError("S2 trajectory contains no receiver positions.")

if trajectory["snapshot"].duplicated().any():
    raise ValueError("Duplicate snapshot identifiers found in S2 CSV.")

position_values = trajectory[
    ["rx_x_m", "rx_y_m", "rx_z_m"]
].to_numpy(dtype=float)

if not np.all(np.isfinite(position_values)):
    raise ValueError("Non-finite receiver coordinates found.")

print("\nS2 trajectory loaded successfully.")
print("Number of snapshots:", len(trajectory))


# ============================================================
# 3. LOAD SCENE AND CONFIGURE ANTENNA ARRAYS
# ============================================================

print("\nLoading street canyon scene...")

scene = load_scene(
    sionna.rt.scene.simple_street_canyon,
    merge_shapes=True
)

scene.frequency = CARRIER_FREQUENCY

# Fixed directional 8x8 transmit array.
scene.tx_array = PlanarArray(
    num_rows=NUM_ROWS,
    num_cols=NUM_COLS,
    vertical_spacing=VERTICAL_SPACING,
    horizontal_spacing=HORIZONTAL_SPACING,
    pattern="tr38901",
    polarization="V",
)

# Single isotropic receive antenna.
scene.rx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    pattern="iso",
    polarization="V",
)

# Keep the transmitter orientation fixed.
# At zero orientation, the array normal points along global +x.
tx = Transmitter(
    name="tx",
    position=TX_POSITION.tolist(),
    orientation=[0.0, 0.0, 0.0],
)

first_position = position_values[0]

rx = Receiver(
    name="rx",
    position=first_position.tolist(),
    orientation=[0.0, 0.0, 0.0],
)

scene.add(tx)
scene.add(rx)

print("Scene loaded.")
print("Carrier frequency:", CARRIER_FREQUENCY / 1e9, "GHz")
print("Wavelength:", WAVELENGTH, "m")
print("Transmit antennas:", scene.tx_array.array_size)
print("Receive antennas:", scene.rx_array.array_size)
print("Transmitter position:", TX_POSITION.tolist())
print("Transmitter orientation: [0, 0, 0] rad")


# ============================================================
# 4. GENERATE DIRECTIONAL DFT-STYLE BEAM CODEBOOK
# ============================================================

print("\nGenerating beam codebook...")

# Extract the actual normalized antenna element coordinates.
# PlanarArray stores positions in wavelength units.
antenna_positions = scene.tx_array.normalized_positions

antenna_y = np.asarray(
    antenna_positions.y.numpy(),
    dtype=float
)

antenna_z = np.asarray(
    antenna_positions.z.numpy(),
    dtype=float
)

if len(antenna_y) != NUM_TX_ANTENNAS:
    raise RuntimeError("Unexpected transmit antenna y-coordinate count.")

if len(antenna_z) != NUM_TX_ANTENNAS:
    raise RuntimeError("Unexpected transmit antenna z-coordinate count.")

# Standard 8-point DFT spatial-frequency grid.
grid_y = (
    np.arange(NUM_COLS) - (NUM_COLS - 1) / 2
) * (2.0 / NUM_COLS)

grid_z = (
    np.arange(NUM_ROWS) - (NUM_ROWS - 1) / 2
) * (2.0 / NUM_ROWS)

beam_directions = []
beam_weights = []

for u_z in grid_z:
    for u_y in grid_y:

        radial_squared = u_y**2 + u_z**2

        # Only retain physically valid forward-hemisphere directions.
        if radial_squared > 1.0 + 1e-12:
            continue

        u_x = np.sqrt(max(0.0, 1.0 - radial_squared))

        # Direction vector in the fixed transmitter's local frame.
        direction = np.array(
            [u_x, u_y, u_z],
            dtype=float
        )

        # Array steering weights.
        #
        # Sionna's synthetic-array phase convention produces
        # h_k proportional to exp(+j*2*pi*r_k.u).
        #
        # Precoding weights therefore use the opposite phase:
        # w_k = exp(-j*2*pi*r_k.u)/sqrt(N).
        #
        # The final channel combination is h^T w, not h^H w.

        weights = np.exp(
            -1j * 2.0 * np.pi *
            (antenna_y * u_y + antenna_z * u_z)
        ) / np.sqrt(NUM_TX_ANTENNAS)

        if not np.isclose(np.linalg.norm(weights), 1.0, atol=1e-10):
            raise RuntimeError("Beam weights are not unit norm.")

        beam_directions.append(direction)
        beam_weights.append(weights)

beam_directions = np.asarray(beam_directions)
beam_weights = np.asarray(beam_weights)

NUM_BEAMS = len(beam_weights)

if NUM_BEAMS != 52:
    raise RuntimeError(
        f"Expected 52 forward-hemisphere beams, found {NUM_BEAMS}."
    )

print("Number of candidate beams:", NUM_BEAMS)
print("Number of weights per beam:", NUM_TX_ANTENNAS)
print("All precoding vectors have unit norm.")


# ============================================================
# 5. SAVE BEAM CODEBOOK
# ============================================================

codebook_rows = []

for beam_id in range(NUM_BEAMS):

    ux, uy, uz = beam_directions[beam_id]
    weights = beam_weights[beam_id]

    codebook_row = {
        "beam_id": beam_id,
        "u_x_local": ux,
        "u_y_local": uy,
        "u_z_local": uz,
        "azimuth_local_deg": np.degrees(np.arctan2(uy, ux)),
        "elevation_local_deg": np.degrees(np.arcsin(uz)),
        "weight_norm": np.linalg.norm(weights),
    }

    for antenna_id, weight in enumerate(weights):
        codebook_row[f"w{antenna_id:02d}_real"] = weight.real
        codebook_row[f"w{antenna_id:02d}_imag"] = weight.imag

    codebook_rows.append(codebook_row)

pd.DataFrame(codebook_rows).to_csv(
    OUTPUT_CODEBOOK,
    index=False
)

print("Beam codebook saved:", OUTPUT_CODEBOOK.name)


# ============================================================
# 6. INITIALIZE PATH SOLVER
# ============================================================

path_solver = PathSolver(deterministic=True)


# ============================================================
# 7. RUN BEAM SELECTION AT EACH RECEIVER POSITION
# ============================================================

summary_rows = []
all_beam_rows = []

previous_best_beam = None

print("\nStarting spatial beam-selection simulation...")

for snapshot_index, row in trajectory.iterrows():

    snapshot = int(row["snapshot"])

    rx_position = np.array(
        [
            float(row["rx_x_m"]),
            float(row["rx_y_m"]),
            float(row["rx_z_m"]),
        ],
        dtype=float
    )

    rx.position = rx_position.tolist()

    # --------------------------------------------------------
    # Compute propagation paths
    # --------------------------------------------------------

    print("\n" + "-" * 64)
    print(
        f"Snapshot {snapshot_index + 1}/{len(trajectory)}"
    )
    print("Receiver position:", rx_position.tolist())

    paths = path_solver(
        scene=scene,
        max_depth=MAX_DEPTH,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=True,
        synthetic_array=True,
        seed=SEED,
    )

    # --------------------------------------------------------
    # Compute channel frequency response at carrier centre
    # --------------------------------------------------------

    h_frequency = paths.cfr(
        frequencies=CFR_FREQUENCIES,
        normalize=False,
        normalize_delays=False,
        out_type="numpy",
    )

    # Expected shape:
    # [num_rx, num_rx_ant, num_tx, num_tx_ant,
    #  num_time_steps, num_frequencies]
    expected_shape = (
        1,
        1,
        1,
        NUM_TX_ANTENNAS,
        1,
        1,
    )

    if h_frequency.shape != expected_shape:
        raise RuntimeError(
            "Unexpected channel frequency response shape.\n"
            f"Expected: {expected_shape}\n"
            f"Received: {h_frequency.shape}"
        )

    # Complex channel vector across the 64 transmit antennas.
    h = np.asarray(
        h_frequency[0, 0, 0, :, 0, 0],
        dtype=np.complex128
    )

    if not np.all(np.isfinite(h)):
        raise RuntimeError("Non-finite channel coefficients detected.")

    if np.allclose(h, 0.0):
        raise RuntimeError(
            f"Zero channel response at snapshot {snapshot}."
        )

    # Number of valid propagation paths.
    valid_mask = np.asarray(
        paths.valid.numpy(),
        dtype=bool
    )

    valid_path_count = int(
        np.count_nonzero(valid_mask[0, 0, :])
    )

    # --------------------------------------------------------
    # Calculate geometric line-of-sight direction
    # --------------------------------------------------------

    tx_to_rx = rx_position - TX_POSITION

    distance = float(np.linalg.norm(tx_to_rx))

    if distance <= 0:
        raise RuntimeError("TX and RX positions coincide.")

    los_direction = tx_to_rx / distance

    # Transmitter orientation is zero, so local and global
    # coordinate directions are identical.
    los_u_x, los_u_y, los_u_z = los_direction

    # --------------------------------------------------------
    # Evaluate every beam
    # --------------------------------------------------------

    beam_metrics = []

    for beam_id in range(NUM_BEAMS):

        weights = beam_weights[beam_id]
        beam_direction = beam_directions[beam_id]

        # Sionna precoding convention:
        # effective channel = h^T * w (no conjugation).
        effective_channel = np.sum(h * weights)

        channel_gain = float(
            np.abs(effective_channel) ** 2
        )

        channel_gain_db = float(
            10.0 * np.log10(
                max(channel_gain, np.finfo(float).tiny)
            )
        )

        cosine_angle = np.clip(
            np.dot(beam_direction, los_direction),
            -1.0,
            1.0
        )

        pointing_error_deg = float(
            np.degrees(np.arccos(cosine_angle))
        )

        beam_metrics.append(channel_gain)

        all_beam_rows.append(
            {
                "snapshot": snapshot,
                "rx_x_m": rx_position[0],
                "rx_y_m": rx_position[1],
                "rx_z_m": rx_position[2],
                "beam_id": beam_id,
                "beamformed_channel_gain_linear": channel_gain,
                "beamformed_channel_gain_db": channel_gain_db,
                "los_pointing_error_deg": pointing_error_deg,
            }
        )

    beam_metrics = np.asarray(beam_metrics)

    if not np.all(np.isfinite(beam_metrics)):
        raise RuntimeError("Invalid beam metric detected.")

    # --------------------------------------------------------
    # Identify best beam
    # --------------------------------------------------------

    best_beam_index = int(np.argmax(beam_metrics))

    best_gain = float(beam_metrics[best_beam_index])

    best_gain_db = float(
        10.0 * np.log10(
            max(best_gain, np.finfo(float).tiny)
        )
    )

    best_direction = beam_directions[best_beam_index]

    best_pointing_error = float(
        np.degrees(
            np.arccos(
                np.clip(
                    np.dot(best_direction, los_direction),
                    -1.0,
                    1.0
                )
            )
        )
    )

    sorted_metrics = np.sort(beam_metrics)

    second_best_gain = float(sorted_metrics[-2])

    beam_margin_db = float(
        10.0 * np.log10(
            max(best_gain, np.finfo(float).tiny) /
            max(second_best_gain, np.finfo(float).tiny)
        )
    )

    # --------------------------------------------------------
    # Track beam transitions
    # --------------------------------------------------------

    if previous_best_beam is None:
        beam_switched = False
    else:
        beam_switched = (
            best_beam_index != previous_best_beam
        )

    previous_best_beam = best_beam_index

    # --------------------------------------------------------
    # Save snapshot-level result
    # --------------------------------------------------------

    summary_rows.append(
        {
            "snapshot": snapshot,
            "rx_x_m": rx_position[0],
            "rx_y_m": rx_position[1],
            "rx_z_m": rx_position[2],
            "tx_rx_distance_m": distance,
            "valid_path_count": valid_path_count,
            "channel_vector_norm_squared": float(
                np.sum(np.abs(h) ** 2)
            ),
            "los_u_x": los_u_x,
            "los_u_y": los_u_y,
            "los_u_z": los_u_z,
            "best_beam_id": best_beam_index,
            "best_beam_u_x": best_direction[0],
            "best_beam_u_y": best_direction[1],
            "best_beam_u_z": best_direction[2],
            "best_beam_azimuth_deg": np.degrees(
                np.arctan2(
                    best_direction[1],
                    best_direction[0]
                )
            ),
            "best_beam_elevation_deg": np.degrees(
                np.arcsin(best_direction[2])
            ),
            "best_beam_los_pointing_error_deg": best_pointing_error,
            "best_beam_channel_gain_linear": best_gain,
            "best_beam_channel_gain_db": best_gain_db,
            "second_best_beam_channel_gain_linear": second_best_gain,
            "best_to_second_best_margin_db": beam_margin_db,
            "beam_switched_from_previous": beam_switched,
        }
    )

    print("Valid propagation paths:", valid_path_count)
    print("Channel response shape:", h_frequency.shape)
    print("Best beam ID:", best_beam_index)
    print("Best beam channel gain:", best_gain)
    print("Best beam channel gain (dB):", best_gain_db)
    print("LOS pointing error (degrees):", best_pointing_error)
    print("Beam margin (dB):", beam_margin_db)
    print("Beam switched:", beam_switched)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

summary_df = pd.DataFrame(summary_rows)
all_beams_df = pd.DataFrame(all_beam_rows)

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False
)

all_beams_df.to_csv(
    OUTPUT_ALL_BEAMS,
    index=False
)


# ============================================================
# 9. SAVE EXPERIMENT CONFIGURATION
# ============================================================

config = {
    "experiment": "S3_8x8_UPA_Beam_Selection",
    "carrier_frequency_hz": CARRIER_FREQUENCY,
    "wavelength_m": WAVELENGTH,
    "transmitter_position_m": TX_POSITION.tolist(),
    "transmitter_orientation_rad": [0.0, 0.0, 0.0],
    "transmitter_array": {
        "rows": NUM_ROWS,
        "columns": NUM_COLS,
        "elements": NUM_TX_ANTENNAS,
        "pattern": "tr38901",
        "polarization": "V",
        "vertical_spacing_lambda": VERTICAL_SPACING,
        "horizontal_spacing_lambda": HORIZONTAL_SPACING,
    },
    "receiver_array": {
        "rows": 1,
        "columns": 1,
        "pattern": "iso",
        "polarization": "V",
    },
    "num_candidate_beams": NUM_BEAMS,
    "max_depth": MAX_DEPTH,
    "seed": SEED,
    "synthetic_array": True,
    "cfr_baseband_frequency_offset_hz": 0.0,
    "beam_metric": "abs(h_transpose_times_w)^2",
    "precoding_weights_unit_norm": True,
    "channel_normalization": False,
    "delay_normalization": False,
}

with open(OUTPUT_CONFIG, "w", encoding="utf-8") as file:
    json.dump(config, file, indent=4)


# ============================================================
# 10. FINAL REPORT
# ============================================================

number_of_switches = int(
    summary_df["beam_switched_from_previous"].sum()
)

print("\n" + "=" * 72)
print("S3 BEAM-SELECTION EXPERIMENT COMPLETED")
print("=" * 72)

print("Receiver snapshots:", len(summary_df))
print("Transmit antenna elements:", NUM_TX_ANTENNAS)
print("Candidate beams:", NUM_BEAMS)
print("Total beam evaluations:", len(all_beams_df))
print("Beam switches:", number_of_switches)

print("\nBest-beam sequence:")
print(
    summary_df[
        ["snapshot", "rx_x_m", "best_beam_id",
         "best_beam_channel_gain_db"]
    ].to_string(index=False)
)

print("\nOutput files:")
print(OUTPUT_SUMMARY.name)
print(OUTPUT_ALL_BEAMS.name)
print(OUTPUT_CODEBOOK.name)
print(OUTPUT_CONFIG.name)

print("\nS3 execution successful.")
