
import csv
import importlib.metadata as metadata

import numpy as np
import mitsuba as mi
import sionna.rt

from sionna.rt import (
    load_scene,
    PlanarArray,
    Transmitter,
    Receiver,
    PathSolver,
)

# ============================================================
# S2: VEHICLE MOBILITY AND SPATIAL CHANNEL VARIATION
# ============================================================

# ------------------------------------------------------------
# 1. Experiment configuration
# ------------------------------------------------------------

CARRIER_FREQUENCY = 28e9
MAX_DEPTH = 5
SEED = 41

TX_POSITION = [-33.0, 11.0, 32.0]

RX_HEIGHT = 1.5
RX_Y = -13.0

TRAJECTORY_X = np.arange(17.0, 37.0 + 0.1, 2.0)

SUMMARY_FILE = "s2_mobility_summary.csv"
PATHS_FILE = "s2_mobility_paths.csv"

C = 299792458.0

# ------------------------------------------------------------
# 2. Verify environment
# ------------------------------------------------------------

print("=" * 70)
print("SIONNA RT - S2 VEHICLE MOBILITY EXPERIMENT")
print("=" * 70)

print("Sionna RT:", metadata.version("sionna-rt"))
print("Mitsuba:", metadata.version("mitsuba"))
print("Dr.Jit:", metadata.version("drjit"))
print("Mitsuba variant:", mi.variant())

if not mi.variant() or not mi.variant().startswith("cuda"):
    raise RuntimeError("CUDA Mitsuba variant is not active.")

# ------------------------------------------------------------
# 3. Load scene
# ------------------------------------------------------------

print("\nLoading street canyon scene...")

scene = load_scene(
    sionna.rt.scene.simple_street_canyon,
    merge_shapes=True
)

scene.frequency = CARRIER_FREQUENCY

print("Scene loaded.")
print("Scene geometry bounds:", scene.mi_scene.bbox())

# ------------------------------------------------------------
# 4. Configure antenna arrays
# ------------------------------------------------------------

scene.tx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    pattern="iso",
    polarization="V"
)

scene.rx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    pattern="iso",
    polarization="V"
)

# ------------------------------------------------------------
# 5. Configure fixed transmitter
# ------------------------------------------------------------

tx = Transmitter(
    name="base_station",
    position=TX_POSITION
)

scene.add(tx)

# ------------------------------------------------------------
# 6. Configure moving receiver
# ------------------------------------------------------------

initial_position = [
    float(TRAJECTORY_X[0]),
    RX_Y,
    RX_HEIGHT
]

rx = Receiver(
    name="vehicle",
    position=initial_position
)

scene.add(rx)

tx.look_at(rx)

print("\nBase station:", TX_POSITION)
print("Initial receiver position:", initial_position)
print("Trajectory snapshots:", len(TRAJECTORY_X))

# ------------------------------------------------------------
# 7. Initialize deterministic path solver
# ------------------------------------------------------------

path_solver = PathSolver(deterministic=True)

summary_rows = []
path_rows = []

# ------------------------------------------------------------
# 8. Execute ray tracing at each receiver position
# ------------------------------------------------------------

print("\nStarting vehicle trajectory simulation...")

for snapshot_idx, x in enumerate(TRAJECTORY_X):

    rx_position = [
        float(x),
        RX_Y,
        RX_HEIGHT
    ]

    # Update receiver location
    rx.position = rx_position

    # Keep transmitter pointing towards receiver
    tx.look_at(rx)

    print("\n" + "-" * 60)
    print(f"Snapshot {snapshot_idx + 1}/{len(TRAJECTORY_X)}")
    print("Receiver position:", rx_position)

    # --------------------------------------------------------
    # Compute propagation paths
    # --------------------------------------------------------

    paths = path_solver(
        scene=scene,
        max_depth=MAX_DEPTH,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=True,
        synthetic_array=True,
        seed=SEED
    )

    # --------------------------------------------------------
    # Generate CIR
    # --------------------------------------------------------

    a, tau = paths.cir(
        normalize_delays=False,
        out_type="numpy"
    )

    a = np.asarray(a)
    tau = np.asarray(tau)

    valid_mask = np.asarray(
        paths.valid.numpy(),
        dtype=bool
    )

    if tau.ndim != 3 or a.ndim != 6:
        raise RuntimeError(
            f"Unexpected tensor dimensions: "
            f"a={a.shape}, tau={tau.shape}"
        )

    if valid_mask.shape != tau.shape:
        raise RuntimeError(
            f"Validity mask mismatch: "
            f"{valid_mask.shape} versus {tau.shape}"
        )

    # Single TX and single RX
    path_delays = tau[0, 0, :]
    path_coefficients = a[0, 0, 0, 0, :, 0]

    valid_indices = np.flatnonzero(
        valid_mask[0, 0, :]
    )

    if len(valid_indices) == 0:
        raise RuntimeError(
            f"No valid paths at snapshot {snapshot_idx}"
        )

    valid_delays = path_delays[valid_indices]
    valid_coefficients = path_coefficients[valid_indices]

    finite_mask = (
        np.isfinite(valid_delays)
        & np.isfinite(valid_coefficients.real)
        & np.isfinite(valid_coefficients.imag)
    )

    valid_indices = valid_indices[finite_mask]
    valid_delays = valid_delays[finite_mask]
    valid_coefficients = valid_coefficients[finite_mask]

    if len(valid_indices) == 0:
        raise RuntimeError(
            f"No finite valid paths at snapshot {snapshot_idx}"
        )

    # --------------------------------------------------------
    # Calculate channel metrics
    # --------------------------------------------------------

    path_powers = np.abs(valid_coefficients) ** 2

    total_coefficient_power = float(
        np.sum(path_powers)
    )

    strongest_local_idx = int(
        np.argmax(path_powers)
    )

    strongest_path_idx = int(
        valid_indices[strongest_local_idx]
    )

    strongest_path_power = float(
        path_powers[strongest_local_idx]
    )

    strongest_path_delay = float(
        valid_delays[strongest_local_idx]
    )

    min_path_delay = float(
        np.min(valid_delays)
    )

    # Geometric direct-path reference
    distance = float(
        np.linalg.norm(
            np.asarray(rx_position, dtype=float)
            - np.asarray(TX_POSITION, dtype=float)
        )
    )

    geometric_los_delay = distance / C

    total_power_db = (
        10.0 * np.log10(total_coefficient_power)
        if total_coefficient_power > 0
        else -np.inf
    )

    strongest_power_db = (
        10.0 * np.log10(strongest_path_power)
        if strongest_path_power > 0
        else -np.inf
    )

    # --------------------------------------------------------
    # Store snapshot-level metrics
    # --------------------------------------------------------

    summary_rows.append([
        snapshot_idx,
        rx_position[0],
        rx_position[1],
        rx_position[2],
        distance,
        len(valid_indices),
        min_path_delay,
        geometric_los_delay,
        strongest_path_idx,
        strongest_path_delay,
        total_coefficient_power,
        total_power_db,
        strongest_path_power,
        strongest_power_db
    ])

    # --------------------------------------------------------
    # Store individual propagation paths
    # --------------------------------------------------------

    for local_idx, original_idx in enumerate(valid_indices):

        coefficient = valid_coefficients[local_idx]
        delay = float(valid_delays[local_idx])
        power = float(abs(coefficient) ** 2)

        path_rows.append([
            snapshot_idx,
            rx_position[0],
            rx_position[1],
            rx_position[2],
            int(original_idx),
            delay,
            delay * 1e9,
            float(coefficient.real),
            float(coefficient.imag),
            power
        ])

    print("Valid paths:", len(valid_indices))
    print("Minimum path delay (ns):", min_path_delay * 1e9)
    print("Geometric LOS delay (ns):", geometric_los_delay * 1e9)
    print("Strongest path delay (ns):", strongest_path_delay * 1e9)
    print("Total coefficient power:", total_coefficient_power)

# ------------------------------------------------------------
# 9. Save snapshot summary
# ------------------------------------------------------------

with open(SUMMARY_FILE, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "snapshot",
        "rx_x_m",
        "rx_y_m",
        "rx_z_m",
        "tx_rx_distance_m",
        "valid_path_count",
        "minimum_path_delay_s",
        "geometric_los_delay_s",
        "strongest_path_index",
        "strongest_path_delay_s",
        "total_coefficient_power",
        "total_coefficient_power_db",
        "strongest_path_power",
        "strongest_path_power_db"
    ])

    writer.writerows(summary_rows)

# ------------------------------------------------------------
# 10. Save path-level data
# ------------------------------------------------------------

with open(PATHS_FILE, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "snapshot",
        "rx_x_m",
        "rx_y_m",
        "rx_z_m",
        "path_index",
        "delay_seconds",
        "delay_ns",
        "coefficient_real",
        "coefficient_imag",
        "coefficient_power"
    ])

    writer.writerows(path_rows)

# ------------------------------------------------------------
# 11. Final experiment summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("S2 VEHICLE MOBILITY EXPERIMENT COMPLETED")
print("=" * 70)

print("Number of snapshots:", len(summary_rows))
print("Total recorded paths:", len(path_rows))

print("Summary file:", SUMMARY_FILE)
print("Path-level file:", PATHS_FILE)

print("\nS2 execution successful.")
