
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

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

CARRIER_FREQUENCY = 28e9
MAX_DEPTH = 5
SEED = 41

TX_POSITION = [-33.0, 11.0, 32.0]
RX_POSITION = [27.0, -13.0, 1.5]

OUTPUT_FILE = "s1_ray_tracing_results.csv"

# --------------------------------------------------
# 2. Verify installation and backend
# --------------------------------------------------

print("=" * 65)
print("SIONNA RT - FIRST RAY TRACING EXPERIMENT")
print("=" * 65)

print("Sionna RT:", metadata.version("sionna-rt"))
print("Mitsuba:", metadata.version("mitsuba"))
print("Dr.Jit:", metadata.version("drjit"))
print("Mitsuba variant:", mi.variant())

if not mi.variant() or not mi.variant().startswith("cuda"):
    raise RuntimeError(
        "CUDA Mitsuba variant is not active. "
        "Check the GPU backend before continuing."
    )

# --------------------------------------------------
# 3. Load the built-in street canyon scene
# --------------------------------------------------

print("\nLoading scene...")

scene = load_scene(
    sionna.rt.scene.simple_street_canyon,
    merge_shapes=True
)

scene.frequency = CARRIER_FREQUENCY

# --------------------------------------------------
# 4. Configure antenna arrays
# --------------------------------------------------

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

# --------------------------------------------------
# 5. Add base station and vehicle receiver
# --------------------------------------------------

tx = Transmitter(
    name="base_station",
    position=TX_POSITION
)

rx = Receiver(
    name="vehicle",
    position=RX_POSITION
)

scene.add(tx)
scene.add(rx)

tx.look_at(rx)

print("Transmitter position:", TX_POSITION)
print("Receiver position:", RX_POSITION)
print("Carrier frequency:", CARRIER_FREQUENCY / 1e9, "GHz")

# --------------------------------------------------
# 6. Calculate propagation paths
# --------------------------------------------------

print("\nComputing propagation paths...")

path_solver = PathSolver(deterministic=True)

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

print("PathSolver completed successfully.")

# --------------------------------------------------
# 7. Extract channel impulse response
# --------------------------------------------------

a, tau = paths.cir(
    normalize_delays=False,
    out_type="numpy"
)

a = np.asarray(a)
tau = np.asarray(tau)

# For synthetic_array=True:
# tau: [num_rx, num_tx, num_paths]
#
# a: [num_rx, num_rx_ant, num_tx,
#     num_tx_ant, num_paths, num_time_steps]

print("\nTensor diagnostics")
print("CIR coefficient shape:", a.shape)
print("Delay tensor shape:", tau.shape)

if tau.ndim != 3:
    raise RuntimeError(
        f"Unexpected delay tensor dimensions: {tau.shape}"
    )

if a.ndim != 6:
    raise RuntimeError(
        f"Unexpected CIR coefficient dimensions: {a.shape}"
    )

# --------------------------------------------------
# 8. Extract valid propagation paths
# --------------------------------------------------

valid_mask = np.asarray(paths.valid.numpy(), dtype=bool)

if valid_mask.shape != tau.shape:
    raise RuntimeError(
        f"Validity mask shape {valid_mask.shape} "
        f"does not match delay shape {tau.shape}"
    )

# Single receiver and single transmitter
path_delays = tau[0, 0, :]
path_coefficients = a[0, 0, 0, 0, :, 0]

# Only retain physically valid paths
valid_indices = np.flatnonzero(valid_mask[0, 0, :])

num_path_slots = tau.shape[-1]
num_valid_paths = len(valid_indices)

if num_valid_paths == 0:
    raise RuntimeError(
        "No valid propagation paths found. "
        "Check scene geometry and TX/RX positions."
    )

valid_delays = path_delays[valid_indices]
valid_coefficients = path_coefficients[valid_indices]

path_powers = np.abs(valid_coefficients) ** 2

total_path_power = float(np.sum(path_powers))

strongest_local_idx = int(np.argmax(path_powers))
strongest_path_idx = int(valid_indices[strongest_local_idx])

strongest_path_power = float(
    path_powers[strongest_local_idx]
)

strongest_path_delay = float(
    path_delays[strongest_path_idx]
)

# --------------------------------------------------
# 9. Display results
# --------------------------------------------------

print("\n========== RESULTS ==========")

print("CIR coefficient shape:", a.shape)
print("Delay tensor shape:", tau.shape)

print("Allocated path slots:", num_path_slots)
print("Valid propagation paths:", num_valid_paths)

print("Total path coefficient power:", total_path_power)
print("Strongest path coefficient power:", strongest_path_power)

print(
    "Strongest path delay (ns):",
    strongest_path_delay * 1e9
)

print("Strongest path index:", strongest_path_idx)

print("================================")

# --------------------------------------------------
# 10. Save experiment results
# --------------------------------------------------

with open(OUTPUT_FILE, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "frequency_hz",
        "tx_x", "tx_y", "tx_z",
        "rx_x", "rx_y", "rx_z",
        "max_depth",
        "allocated_path_slots",
        "valid_num_paths",
        "total_path_coefficient_power",
        "strongest_path_coefficient_power",
        "strongest_path_delay_seconds"
    ])

    writer.writerow([
        CARRIER_FREQUENCY,
        *TX_POSITION,
        *RX_POSITION,
        MAX_DEPTH,
        num_path_slots,
        num_valid_paths,
        total_path_power,
        strongest_path_power,
        strongest_path_delay
    ])

print("\nResults saved to:", OUTPUT_FILE)
print("S1 RAY-TRACING EXPERIMENT COMPLETED.")
