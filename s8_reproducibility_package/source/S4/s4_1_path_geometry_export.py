"""
S4.1: S4 blockage experiment with path interaction geometry export

Purpose
-------
Compare an unmodified street-canyon scene against the same scene with a
controlled concrete obstruction. The obstruction is a vertical, ground-
connected slab. Its geometry is chosen to intersect the geometric TX-RX
segment for a subset of the existing S2 trajectory.

This is a controlled building-like obstruction, NOT a vehicle model.

Inputs
------
~/sionna_project/s2_mobility_summary.csv

Outputs
-------
~/sionna_project/s4_blockage_geometry/
    s4_blockage_comparison.csv
    s4_snapshot_summary.csv
    s4_beam_metrics.csv
    s4_path_details.csv
    s4_channel_data.npz
    s4_path_coefficients.npz
    s4_beam_codebook.csv
    s4_blocker.obj
    s4_config.json
    s4_validation.txt

Beam metric
-----------
|h^T w|^2

This is a dimensionless beamformed channel-gain metric, not received power
in watts or dBm. Channel coefficients are preserved as complex values.

Run
---
cd ~/sionna_project
source .venv/bin/activate
python s4_blockage.py
"""

import json
from pathlib import Path

import mitsuba as mi
import numpy as np
import pandas as pd
import sionna

from sionna.rt import (
    load_scene,
    PlanarArray,
    Transmitter,
    Receiver,
    PathSolver,
    SceneObject,
    ITURadioMaterial,
)


# ======================================================================
# 1. CONFIGURATION
# ======================================================================

BASE_DIR = Path.home() / "sionna_project"
INPUT_CSV = BASE_DIR / "s2_mobility_summary.csv"

OUTPUT_DIR = BASE_DIR / "s4_blockage_geometry"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_COMPARISON = OUTPUT_DIR / "s4_blockage_comparison.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "s4_snapshot_summary.csv"
OUTPUT_BEAMS = OUTPUT_DIR / "s4_beam_metrics.csv"
OUTPUT_PATHS = OUTPUT_DIR / "s4_path_details.csv"
OUTPUT_INTERACTIONS = OUTPUT_DIR / "s4_path_interactions.csv"
OUTPUT_OBJECTS = OUTPUT_DIR / "s4_scene_objects.csv"
OUTPUT_CHANNELS = OUTPUT_DIR / "s4_channel_data.npz"
OUTPUT_PATH_COEFFICIENTS = OUTPUT_DIR / "s4_path_coefficients.npz"
OUTPUT_CODEBOOK = OUTPUT_DIR / "s4_beam_codebook.csv"
OUTPUT_BLOCKER_OBJ = OUTPUT_DIR / "s4_blocker.obj"
OUTPUT_CONFIG = OUTPUT_DIR / "s4_config.json"
OUTPUT_VALIDATION = OUTPUT_DIR / "s4_validation.txt"

CARRIER_FREQUENCY = 28e9
WAVELENGTH = 299792458.0 / CARRIER_FREQUENCY
SPEED_OF_LIGHT = 299792458.0

TX_POSITION = np.array([-33.0, 11.0, 32.0], dtype=float)
TX_ORIENTATION = [0.0, 0.0, 0.0]

NUM_ROWS = 8
NUM_COLS = 8
NUM_TX_ANTENNAS = NUM_ROWS * NUM_COLS

VERTICAL_SPACING = 0.5
HORIZONTAL_SPACING = 0.5

MAX_DEPTH = 5
SEED = 41
CFR_FREQUENCIES = np.array([0.0], dtype=np.float32)

# ----------------------------------------------------------------------
# Controlled obstruction geometry (world coordinates, metres)
#
# The slab's local X thickness is 1 m, local Y width is 3.0 m, and
# local Z height is 22 m. It is ground-connected.
#
# Its global axis-aligned bounds are:
#   x: [-3.5, -2.5]
#   y: [-2.0, 1.0]
#   z: [0.0, 22.0]
#
# This is a simplified building-like obstruction, not a detailed
# architectural model.
# ----------------------------------------------------------------------

BLOCKER_NAME = "s4_concrete_obstruction"
BLOCKER_CENTER = np.array([-3.0, -0.5, 11.0], dtype=float)
BLOCKER_SIZE = np.array([1.0, 3.0, 22.0], dtype=float)
BLOCKER_MATERIAL_TYPE = "concrete"
BLOCKER_MATERIAL_THICKNESS_M = 0.30

BLOCKER_HALF_SIZE = BLOCKER_SIZE / 2.0
BLOCKER_MIN = BLOCKER_CENTER - BLOCKER_HALF_SIZE
BLOCKER_MAX = BLOCKER_CENTER + BLOCKER_HALF_SIZE

CONDITION_UNBLOCKED = "unblocked"
CONDITION_BLOCKED = "blocked"
CONDITIONS = [CONDITION_UNBLOCKED, CONDITION_BLOCKED]


# ======================================================================
# 2. VALIDATE INPUT TRAJECTORY
# ======================================================================

print("=" * 76)
print("SIONNA RT - S4.1 BLOCKAGE + PATH GEOMETRY EXPORT")
print("=" * 76)
print("Mitsuba:", mi.__version__)
print("Mitsuba variant:", mi.variant())

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Required S2 trajectory not found: {INPUT_CSV}\n"
        "Run S2 successfully before running S4."
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
        f"S2 CSV is missing required columns: {sorted(missing_columns)}"
    )

trajectory = trajectory.sort_values("snapshot").reset_index(drop=True)

if trajectory.empty:
    raise ValueError("S2 trajectory contains no receiver positions.")

if trajectory["snapshot"].duplicated().any():
    raise ValueError("Duplicate snapshot identifiers in S2 trajectory.")

position_values = trajectory[
    ["rx_x_m", "rx_y_m", "rx_z_m"]
].to_numpy(dtype=float)

if not np.all(np.isfinite(position_values)):
    raise ValueError("Non-finite receiver coordinates in S2 trajectory.")

snapshots = trajectory["snapshot"].to_numpy()

print("\nS2 trajectory loaded.")
print("Snapshots:", len(trajectory))
print("First receiver position:", position_values[0].tolist())
print("Last receiver position:", position_values[-1].tolist())


# ======================================================================
# 3. GEOMETRIC LINE-SEGMENT / AABB INTERSECTION
# ======================================================================

def segment_intersects_aabb(p0, p1, box_min, box_max, eps=1e-10):
    """
    Slab-method intersection test for a finite 3D line segment and AABB.

    Returns True only if the segment intersects the box. The segment
    parameter is restricted to [0, 1], so intersections beyond TX/RX
    are not counted.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    box_min = np.asarray(box_min, dtype=float)
    box_max = np.asarray(box_max, dtype=float)

    direction = p1 - p0
    t_enter = 0.0
    t_exit = 1.0

    for axis in range(3):
        if abs(direction[axis]) < eps:
            if (
                p0[axis] < box_min[axis] - eps
                or p0[axis] > box_max[axis] + eps
            ):
                return False
            continue

        t1 = (box_min[axis] - p0[axis]) / direction[axis]
        t2 = (box_max[axis] - p0[axis]) / direction[axis]

        t_near = min(t1, t2)
        t_far = max(t1, t2)

        t_enter = max(t_enter, t_near)
        t_exit = min(t_exit, t_far)

        if t_enter > t_exit + eps:
            return False

    return (t_exit >= 0.0 - eps) and (t_enter <= 1.0 + eps)


geometric_blockage = []

for rx_position in position_values:
    geometric_blockage.append(
        segment_intersects_aabb(
            TX_POSITION,
            rx_position,
            BLOCKER_MIN,
            BLOCKER_MAX,
        )
    )

geometric_blockage = np.asarray(geometric_blockage, dtype=bool)

print("\nGeometric obstruction intersection prediction:")
for snapshot, rx_position, intersects in zip(
    snapshots, position_values, geometric_blockage
):
    print(
        f"snapshot={snapshot}, "
        f"rx_x={rx_position[0]:.2f} m, "
        f"segment_intersects_obstruction={bool(intersects)}"
    )

if not np.any(geometric_blockage):
    raise RuntimeError(
        "The configured obstruction does not intersect any TX-RX segment. "
        "Do not run the ray-tracing experiment with this geometry."
    )

if np.all(geometric_blockage):
    raise RuntimeError(
        "The configured obstruction intersects every TX-RX segment. "
        "This experiment is intended to contain both blocked and "
        "unblocked snapshots."
    )


# ======================================================================
# 4. GENERATE THE 8x8 ARRAY AND THE SAME 52-BEAM CODEBOOK AS S3
# ======================================================================

print("\nGenerating the S3-compatible 8x8 UPA and beam codebook...")

beam_directions = []
beam_weights = []

# This is the same scene and antenna-array configuration as S3.
scene = load_scene(
    sionna.rt.scene.simple_street_canyon,
    merge_shapes=True,
)

scene.frequency = CARRIER_FREQUENCY

scene.tx_array = PlanarArray(
    num_rows=NUM_ROWS,
    num_cols=NUM_COLS,
    vertical_spacing=VERTICAL_SPACING,
    horizontal_spacing=HORIZONTAL_SPACING,
    pattern="tr38901",
    polarization="V",
)

scene.rx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    pattern="iso",
    polarization="V",
)

antenna_positions = scene.tx_array.normalized_positions

antenna_y = np.asarray(antenna_positions.y.numpy(), dtype=float)
antenna_z = np.asarray(antenna_positions.z.numpy(), dtype=float)

if len(antenna_y) != NUM_TX_ANTENNAS:
    raise RuntimeError("Unexpected TX antenna y-coordinate count.")

if len(antenna_z) != NUM_TX_ANTENNAS:
    raise RuntimeError("Unexpected TX antenna z-coordinate count.")

grid_y = (
    np.arange(NUM_COLS) - (NUM_COLS - 1) / 2
) * (2.0 / NUM_COLS)

grid_z = (
    np.arange(NUM_ROWS) - (NUM_ROWS - 1) / 2
) * (2.0 / NUM_ROWS)

for u_z in grid_z:
    for u_y in grid_y:
        radial_squared = u_y**2 + u_z**2

        if radial_squared > 1.0 + 1e-12:
            continue

        u_x = np.sqrt(max(0.0, 1.0 - radial_squared))

        direction = np.array([u_x, u_y, u_z], dtype=float)

        weights = np.exp(
            -1j
            * 2.0
            * np.pi
            * (antenna_y * u_y + antenna_z * u_z)
        ) / np.sqrt(NUM_TX_ANTENNAS)

        if not np.isclose(np.linalg.norm(weights), 1.0, atol=1e-10):
            raise RuntimeError("A beamforming vector is not unit norm.")

        beam_directions.append(direction)
        beam_weights.append(weights)

beam_directions = np.asarray(beam_directions, dtype=float)
beam_weights = np.asarray(beam_weights, dtype=np.complex128)
NUM_BEAMS = len(beam_weights)

if NUM_BEAMS != 52:
    raise RuntimeError(
        f"S3 compatibility check failed: expected 52 beams, got {NUM_BEAMS}."
    )

codebook_rows = []

for beam_id, (direction, weights) in enumerate(
    zip(beam_directions, beam_weights)
):
    ux, uy, uz = direction

    record = {
        "beam_id": beam_id,
        "u_x_local": ux,
        "u_y_local": uy,
        "u_z_local": uz,
        "azimuth_local_deg": np.degrees(np.arctan2(uy, ux)),
        "elevation_local_deg": np.degrees(np.arcsin(uz)),
        "weight_norm": np.linalg.norm(weights),
    }

    for antenna_id, weight in enumerate(weights):
        record[f"w{antenna_id:02d}_real"] = weight.real
        record[f"w{antenna_id:02d}_imag"] = weight.imag

    codebook_rows.append(record)

pd.DataFrame(codebook_rows).to_csv(OUTPUT_CODEBOOK, index=False)

print("TX antennas:", NUM_TX_ANTENNAS)
print("Candidate beams:", NUM_BEAMS)
print("Codebook saved:", OUTPUT_CODEBOOK)


# ======================================================================
# 5. CREATE A SIMPLE CONCRETE OBSTRUCTION MESH
# ======================================================================

def write_box_obj(path, size_xyz):
    """
    Write a closed cuboid OBJ mesh centred at the local origin.

    The SceneObject is subsequently positioned using its world-space
    centre. The six quad faces form a closed rectangular solid.
    """
    sx, sy, sz = np.asarray(size_xyz, dtype=float) / 2.0

    vertices = [
        (-sx, -sy, -sz),
        ( sx, -sy, -sz),
        ( sx,  sy, -sz),
        (-sx,  sy, -sz),
        (-sx, -sy,  sz),
        ( sx, -sy,  sz),
        ( sx,  sy,  sz),
        (-sx,  sy,  sz),
    ]

    faces = [
        (1, 4, 3, 2),  # bottom
        (5, 6, 7, 8),  # top
        (1, 2, 6, 5),  # -Y
        (2, 3, 7, 6),  # +X
        (3, 4, 8, 7),  # +Y
        (4, 1, 5, 8),  # -X
    ]

    with open(path, "w", encoding="utf-8") as obj_file:
        obj_file.write("# S4 controlled concrete obstruction cuboid\n")
        for vertex in vertices:
            obj_file.write(
                f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n"
            )
        for face in faces:
            obj_file.write("f " + " ".join(map(str, face)) + "\n")


write_box_obj(OUTPUT_BLOCKER_OBJ, BLOCKER_SIZE)

concrete_material = ITURadioMaterial(
    name="s4_concrete_material",
    itu_type=BLOCKER_MATERIAL_TYPE,
    thickness=BLOCKER_MATERIAL_THICKNESS_M,
    color=(0.55, 0.55, 0.55),
)

blocker = SceneObject(
    fname=str(OUTPUT_BLOCKER_OBJ),
    name=BLOCKER_NAME,
    radio_material=concrete_material,
    position=mi.Point3f(
        float(BLOCKER_CENTER[0]),
        float(BLOCKER_CENTER[1]),
        float(BLOCKER_CENTER[2]),
    ),
)

print("\nObstruction mesh created:", OUTPUT_BLOCKER_OBJ)
print("Obstruction centre:", BLOCKER_CENTER.tolist())
print("Obstruction size:", BLOCKER_SIZE.tolist())
print("Obstruction bounds min:", BLOCKER_MIN.tolist())
print("Obstruction bounds max:", BLOCKER_MAX.tolist())
print("Radio material:", BLOCKER_MATERIAL_TYPE)


# ======================================================================
# 6. ADD TX/RX DEVICES AND INITIALIZE DETERMINISTIC PATH SOLVER
# ======================================================================

tx = Transmitter(
    name="tx",
    position=TX_POSITION.tolist(),
    orientation=TX_ORIENTATION,
)

rx = Receiver(
    name="rx",
    position=position_values[0].tolist(),
    orientation=[0.0, 0.0, 0.0],
)

scene.add([tx, rx])

path_solver = PathSolver(deterministic=True)


# ======================================================================
# 7. PATH ARRAY HELPERS
# ======================================================================

def extract_valid_and_delay_arrays(paths):
    """
    Extract path-validity and delay arrays for the single TX/RX pair.

    The S3 configuration uses synthetic_array=True, for which Sionna
    represents the path arrays without explicit per-element dimensions.
    """
    valid = np.asarray(paths.valid.numpy(), dtype=bool)
    delays = np.asarray(paths.tau.numpy(), dtype=float)

    if valid.ndim != 3 or valid.shape[0:2] != (1, 1):
        raise RuntimeError(
            f"Unexpected Paths.valid shape: {valid.shape}"
        )

    if delays.ndim != 3 or delays.shape[0:2] != (1, 1):
        raise RuntimeError(
            f"Unexpected Paths.tau shape: {delays.shape}"
        )

    if valid.shape[-1] != delays.shape[-1]:
        raise RuntimeError(
            "Paths.valid and Paths.tau disagree on path count."
        )

    return valid[0, 0, :], delays[0, 0, :]


def extract_path_vector(paths, property_name, number_of_paths, dtype=float):
    """
    Extract one scalar per path from a Sionna Paths property.

    With synthetic_array=True, the expected layouts are [1, 1, N]
    or [1, 1, 1, 1, N], depending on the property/API representation.
    """
    value = getattr(paths, property_name)
    if isinstance(value, tuple):
        raise TypeError(
            f"{property_name} is a tuple-valued property; use "
            "extract_complex_path_coefficients() for Paths.a."
        )
    raw = np.asarray(value.numpy(), dtype=dtype)

    if raw.ndim == 3 and raw.shape[:2] == (1, 1):
        result = raw[0, 0, :]
    elif raw.ndim == 5 and raw.shape[:4] == (1, 1, 1, 1):
        result = raw[0, 0, 0, 0, :]
    else:
        raise RuntimeError(
            f"Unexpected {property_name} shape for synthetic_array=True: "
            f"{raw.shape}"
        )

    if result.shape != (number_of_paths,):
        raise RuntimeError(
            f"{property_name} has {result.size} paths; expected {number_of_paths}."
        )
    return result


def extract_complex_path_coefficients(paths, number_of_paths):
    """
    Extract all TX-antenna path coefficients from Sionna Paths.a.

    With this experiment's single RX, single TX, 1-element RX array,
    64-element TX array, and synthetic_array=True, Sionna 2.2 exposes:

        Paths.a.real / Paths.a.imag:
        [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
        = [1, 1, 1, 64, num_paths]

    Returns a complex array shaped [num_tx_ant, num_paths].
    """
    coefficients = paths.a
    if not isinstance(coefficients, (tuple, list)) or len(coefficients) != 2:
        raise RuntimeError(
            "Unexpected Paths.a representation; expected (real_tensor, imag_tensor)."
        )

    real = np.asarray(coefficients[0].numpy(), dtype=np.float64)
    imag = np.asarray(coefficients[1].numpy(), dtype=np.float64)

    expected_prefix = (1, 1, 1, NUM_TX_ANTENNAS)
    if real.ndim != 5 or real.shape[:4] != expected_prefix:
        raise RuntimeError(
            "Unexpected Paths.a.real shape for this experiment: "
            f"{real.shape}; expected (1, 1, 1, {NUM_TX_ANTENNAS}, num_paths)."
        )
    if imag.shape != real.shape:
        raise RuntimeError(
            f"Paths.a real/imaginary shape mismatch: {real.shape} vs {imag.shape}."
        )
    if real.shape[-1] != number_of_paths:
        raise RuntimeError(
            f"Paths.a has {real.shape[-1]} paths; expected {number_of_paths}."
        )

    # Keep every TX antenna coefficient. Remove only the singleton
    # RX-device, RX-antenna, and TX-device axes.
    result = real[0, 0, 0, :, :] + 1j * imag[0, 0, 0, :, :]

    if not np.all(np.isfinite(result.real)) or not np.all(np.isfinite(result.imag)):
        raise RuntimeError("Non-finite complex path coefficients.")

    return result


def extract_interactions_by_path(paths, number_of_paths):
    """Return interaction codes shaped [max_depth, num_paths]."""
    interactions = np.asarray(paths.interactions.numpy())
    if interactions.ndim == 4:
        result = interactions[:, 0, 0, :]
    elif interactions.ndim == 6:
        result = interactions[:, 0, 0, 0, 0, :]
    else:
        raise RuntimeError(
            f"Unexpected Paths.interactions shape: {interactions.shape}"
        )
    if result.shape[-1] != number_of_paths:
        raise RuntimeError("Paths.interactions and Paths.valid path counts differ.")
    return result


def extract_interaction_geometry(paths, number_of_paths):
    """
    Extract object IDs, primitive IDs, and XYZ vertices for each interaction.
    With synthetic_array=True, tensors omit antenna-element dimensions.
    """
    objects_raw = np.asarray(paths.objects.numpy())
    primitives_raw = np.asarray(paths.primitives.numpy())
    vertices_raw = np.asarray(paths.vertices.numpy(), dtype=float)

    if objects_raw.ndim == 4:
        objects = objects_raw[:, 0, 0, :]
        primitives = primitives_raw[:, 0, 0, :]
        vertices = vertices_raw[:, 0, 0, :, :]
    elif objects_raw.ndim == 6:
        objects = objects_raw[:, 0, 0, 0, 0, :]
        primitives = primitives_raw[:, 0, 0, 0, 0, :]
        vertices = vertices_raw[:, 0, 0, 0, 0, :, :]
    else:
        raise RuntimeError(f"Unexpected Paths.objects shape: {objects_raw.shape}")

    if objects.shape != primitives.shape or objects.shape != vertices.shape[:-1]:
        raise RuntimeError(
            f"Interaction tensor shape mismatch: objects={objects.shape}, "
            f"primitives={primitives.shape}, vertices={vertices.shape}"
        )
    if objects.shape[-1] != number_of_paths:
        raise RuntimeError("Interaction geometry and path counts differ.")
    return objects, primitives, vertices


INTERACTION_NAMES = {
    0: "none",
    1: "specular_reflection",
    2: "diffuse_reflection",
    4: "refraction",
    8: "diffraction",
}
INVALID_OBJECT_ID = 4294967295


# ======================================================================
# 8. RUN ONE CONDITION OVER THE FULL TRAJECTORY
# ======================================================================

def run_condition(condition_name):
    summary_rows = []
    beam_rows = []
    path_rows = []
    interaction_rows = []
    object_catalog_rows = []

    h_snapshots = []
    beam_gain_snapshots = []
    path_coefficient_snapshots = []

    previous_best_beam = None

    print("\n" + "=" * 76)
    print("RUNNING CONDITION:", condition_name.upper())
    print("=" * 76)

    for row_index, row in trajectory.iterrows():
        snapshot = int(row["snapshot"])

        rx_position = np.array(
            [
                float(row["rx_x_m"]),
                float(row["rx_y_m"]),
                float(row["rx_z_m"]),
            ],
            dtype=float,
        )

        rx.position = rx_position.tolist()

        print(
            f"\n[{condition_name}] "
            f"snapshot {row_index + 1}/{len(trajectory)} "
            f"(ID={snapshot})"
        )

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

        h_frequency = paths.cfr(
            frequencies=CFR_FREQUENCIES,
            normalize=False,
            normalize_delays=False,
            out_type="numpy",
        )

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

        h = np.asarray(
            h_frequency[0, 0, 0, :, 0, 0],
            dtype=np.complex128,
        )

        if not np.all(np.isfinite(h)):
            raise RuntimeError(
                f"Non-finite channel coefficients at snapshot {snapshot}."
            )

        if np.allclose(h, 0.0):
            raise RuntimeError(
                f"Zero channel response at snapshot {snapshot}, "
                f"condition={condition_name}."
            )

        valid_paths, path_delays = extract_valid_and_delay_arrays(paths)
        number_of_paths = len(valid_paths)

        path_coefficients = extract_complex_path_coefficients(
            paths, number_of_paths
        )
        phi_t = extract_path_vector(paths, "phi_t", number_of_paths)
        theta_t = extract_path_vector(paths, "theta_t", number_of_paths)
        phi_r = extract_path_vector(paths, "phi_r", number_of_paths)
        theta_r = extract_path_vector(paths, "theta_r", number_of_paths)
        doppler_hz = extract_path_vector(paths, "doppler", number_of_paths)

        interactions = extract_interactions_by_path(
            paths,
            number_of_paths,
        )

        object_ids, primitive_ids, interaction_vertices = (
            extract_interaction_geometry(paths, len(valid_paths))
        )

        # Resolve numeric IDs against the current scene. Numeric IDs remain
        # the primary keys; names are descriptive metadata.
        object_id_to_name = {}
        for scene_object in scene.objects.values():
            object_id = int(scene_object.object_id)
            object_name = str(getattr(scene_object, "name", ""))
            object_id_to_name[object_id] = object_name
            object_catalog_rows.append({
                "condition": condition_name,
                "object_id": object_id,
                "object_name": object_name,
            })

        valid_indices = np.flatnonzero(valid_paths)
        valid_path_count = int(len(valid_indices))

        # A direct LOS path has no interactions at any depth.
        # InteractionType.NONE is represented by integer zero.
        no_interaction = np.all(interactions == 0, axis=0)
        direct_path_indices = np.flatnonzero(
            valid_paths & no_interaction
        )
        los_visible = bool(len(direct_path_indices) > 0)

        tx_to_rx = rx_position - TX_POSITION
        distance = float(np.linalg.norm(tx_to_rx))

        if distance <= 0.0:
            raise RuntimeError("TX and RX positions coincide.")

        geometric_los_delay_s = distance / SPEED_OF_LIGHT
        los_delay_s = np.nan
        los_delay_error_ns = np.nan

        if los_visible:
            # In a direct path, the delay should equal geometric distance/c.
            direct_delays = path_delays[direct_path_indices]
            closest_direct_index = int(
                np.argmin(np.abs(direct_delays - geometric_los_delay_s))
            )
            los_delay_s = float(direct_delays[closest_direct_index])
            los_delay_error_ns = float(
                (los_delay_s - geometric_los_delay_s) * 1e9
            )

        los_direction = tx_to_rx / distance

        # Evaluate the same 52 unit-norm beams used in S3.
        beam_metrics = np.empty(NUM_BEAMS, dtype=float)

        for beam_id, weights in enumerate(beam_weights):
            effective_channel = np.sum(h * weights)
            beam_metrics[beam_id] = float(
                np.abs(effective_channel) ** 2
            )

        if not np.all(np.isfinite(beam_metrics)):
            raise RuntimeError(
                f"Invalid beam metrics at snapshot {snapshot}."
            )

        best_beam_id = int(np.argmax(beam_metrics))
        best_gain = float(beam_metrics[best_beam_id])
        best_gain_db = float(
            10.0 * np.log10(max(best_gain, np.finfo(float).tiny))
        )

        sorted_metrics = np.sort(beam_metrics)
        second_best_gain = float(sorted_metrics[-2])
        beam_margin_db = float(
            10.0
            * np.log10(
                max(best_gain, np.finfo(float).tiny)
                / max(second_best_gain, np.finfo(float).tiny)
            )
        )

        best_direction = beam_directions[best_beam_id]
        pointing_error_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(best_direction, los_direction),
                        -1.0,
                        1.0,
                    )
                )
            )
        )

        beam_switched = (
            previous_best_beam is not None
            and best_beam_id != previous_best_beam
        )
        previous_best_beam = best_beam_id

        geometric_intersection = bool(
            segment_intersects_aabb(
                TX_POSITION,
                rx_position,
                BLOCKER_MIN,
                BLOCKER_MAX,
            )
        )

        summary_rows.append(
            {
                "condition": condition_name,
                "snapshot": snapshot,
                "rx_x_m": rx_position[0],
                "rx_y_m": rx_position[1],
                "rx_z_m": rx_position[2],
                "tx_rx_distance_m": distance,
                "geometric_obstruction_intersection": geometric_intersection,
                "los_visible_from_paths": los_visible,
                "los_path_count": int(len(direct_path_indices)),
                "geometric_los_delay_ns": geometric_los_delay_s * 1e9,
                "ray_traced_los_delay_ns": (
                    los_delay_s * 1e9 if np.isfinite(los_delay_s) else np.nan
                ),
                "los_delay_error_ns": los_delay_error_ns,
                "valid_path_count": valid_path_count,
                "channel_vector_norm_squared": float(
                    np.sum(np.abs(h) ** 2)
                ),
                "best_beam_id": best_beam_id,
                "best_beam_channel_gain_linear": best_gain,
                "best_beam_channel_gain_db": best_gain_db,
                "second_best_beam_channel_gain_linear": second_best_gain,
                "best_to_second_best_margin_db": beam_margin_db,
                "best_beam_los_pointing_error_deg": pointing_error_deg,
                "beam_switched_from_previous": bool(beam_switched),
            }
        )

        for beam_id, gain in enumerate(beam_metrics):
            direction = beam_directions[beam_id]
            pointing_error = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            np.dot(direction, los_direction),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )

            beam_rows.append(
                {
                    "condition": condition_name,
                    "snapshot": snapshot,
                    "rx_x_m": rx_position[0],
                    "rx_y_m": rx_position[1],
                    "rx_z_m": rx_position[2],
                    "beam_id": beam_id,
                    "beamformed_channel_gain_linear": float(gain),
                    "beamformed_channel_gain_db": float(
                        10.0
                        * np.log10(
                            max(gain, np.finfo(float).tiny)
                        )
                    ),
                    "los_pointing_error_deg": pointing_error,
                }
            )

        # Preserve each valid path and export one normalized row per
        # physical interaction. Zero/padding entries are not physical events.
        for path_id in valid_indices:
            interaction_codes = [
                int(value) for value in interactions[:, path_id]
                if int(value) != 0
            ]
            path_interaction_records = []

            for depth_index, interaction_value in enumerate(interactions[:, path_id]):
                interaction_code = int(interaction_value)
                if interaction_code == 0:
                    continue

                object_id = int(object_ids[depth_index, path_id])
                primitive_id = int(primitive_ids[depth_index, path_id])
                vertex_xyz = interaction_vertices[depth_index, path_id]

                if object_id == INVALID_OBJECT_ID:
                    raise RuntimeError(
                        "Non-zero interaction code has invalid object ID "
                        f"(snapshot={snapshot}, path={path_id}, depth={depth_index})."
                    )
                if not np.all(np.isfinite(vertex_xyz)):
                    raise RuntimeError(
                        "Non-finite interaction vertex "
                        f"(snapshot={snapshot}, path={path_id}, depth={depth_index})."
                    )

                interaction_record = {
                    "condition": condition_name,
                    "snapshot": snapshot,
                    "path_id": int(path_id),
                    "interaction_order": int(depth_index),
                    "interaction_code": interaction_code,
                    "interaction_type": INTERACTION_NAMES.get(
                        interaction_code, f"unknown_{interaction_code}"
                    ),
                    "object_id": object_id,
                    "object_name": object_id_to_name.get(object_id, ""),
                    "primitive_id": primitive_id,
                    "vertex_x_m": float(vertex_xyz[0]),
                    "vertex_y_m": float(vertex_xyz[1]),
                    "vertex_z_m": float(vertex_xyz[2]),
                }
                interaction_rows.append(interaction_record)
                path_interaction_records.append(interaction_record)

            path_rows.append({
                "condition": condition_name,
                "snapshot": snapshot,
                "path_id": int(path_id),
                "delay_s": float(path_delays[path_id]),
                "delay_ns": float(path_delays[path_id] * 1e9),
                "path_coefficient_real": float(path_coefficients[0, path_id].real),
                "path_coefficient_imag": float(path_coefficients[0, path_id].imag),
                "path_coefficient_abs": float(np.abs(path_coefficients[0, path_id])),
                "path_coefficient_power": float(np.abs(path_coefficients[0, path_id]) ** 2),
                "path_coefficient_reference_tx_antenna": 0,
                "phi_t_rad": float(phi_t[path_id]),
                "theta_t_rad": float(theta_t[path_id]),
                "phi_r_rad": float(phi_r[path_id]),
                "theta_r_rad": float(theta_r[path_id]),
                "phi_t_deg": float(np.degrees(phi_t[path_id])),
                "theta_t_deg": float(np.degrees(theta_t[path_id])),
                "phi_r_deg": float(np.degrees(phi_r[path_id])),
                "theta_r_deg": float(np.degrees(theta_r[path_id])),
                "doppler_hz": float(doppler_hz[path_id]),
                "is_direct_los_path": bool(np.all(interactions[:, path_id] == 0)),
                "interaction_count": len(interaction_codes),
                "interaction_codes": ",".join(map(str, interaction_codes)),
                "interaction_object_ids_json": json.dumps(
                    [r["object_id"] for r in path_interaction_records]
                ),
                "interaction_vertices_xyz_m_json": json.dumps([
                    [r["vertex_x_m"], r["vertex_y_m"], r["vertex_z_m"]]
                    for r in path_interaction_records
                ]),
            })


        h_snapshots.append(h)
        beam_gain_snapshots.append(beam_metrics)
        path_coefficient_snapshots.append(path_coefficients)

        print("Valid paths:", valid_path_count)
        print("Direct LOS visible:", los_visible)
        print("Geometric segment intersects obstruction:", geometric_intersection)
        print("Best beam:", best_beam_id)
        print("Best beam channel gain (dB):", best_gain_db)

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(beam_rows),
        pd.DataFrame(path_rows),
        pd.DataFrame(interaction_rows),
        pd.DataFrame(object_catalog_rows).drop_duplicates(),
        np.asarray(h_snapshots, dtype=np.complex128),
        np.asarray(beam_gain_snapshots, dtype=float),
        path_coefficient_snapshots,
    )


# ======================================================================
# 9. RUN UNBLOCKED REFERENCE, THEN ADD OBSTRUCTION AND RUN BLOCKED CASE
# ======================================================================

print("\nRunning unblocked reference on the original scene...")
unblocked_summary, unblocked_beams, unblocked_paths, unblocked_interactions, unblocked_objects, unblocked_h, unblocked_gains, unblocked_path_coefficients = (
    run_condition(CONDITION_UNBLOCKED)
)

print("\nAdding the concrete obstruction to the same scene...")
scene.edit(add=blocker)

print("Obstruction added. Running blocked condition...")
blocked_summary, blocked_beams, blocked_paths, blocked_interactions, blocked_objects, blocked_h, blocked_gains, blocked_path_coefficients = (
    run_condition(CONDITION_BLOCKED)
)


# ======================================================================
# 10. COMBINE AND SAVE TABULAR DATA
# ======================================================================

summary_df = pd.concat(
    [unblocked_summary, blocked_summary],
    ignore_index=True,
)

beam_df = pd.concat(
    [unblocked_beams, blocked_beams],
    ignore_index=True,
)

path_df = pd.concat(
    [unblocked_paths, blocked_paths],
    ignore_index=True,
)
interaction_df = pd.concat(
    [unblocked_interactions, blocked_interactions],
    ignore_index=True,
)
object_df = pd.concat(
    [unblocked_objects, blocked_objects],
    ignore_index=True,
).drop_duplicates()

summary_df.to_csv(OUTPUT_SUMMARY, index=False)
beam_df.to_csv(OUTPUT_BEAMS, index=False)
path_df.to_csv(OUTPUT_PATHS, index=False)
interaction_df.to_csv(OUTPUT_INTERACTIONS, index=False)
object_df.to_csv(OUTPUT_OBJECTS, index=False)

# Paired comparison: same receiver snapshot in both conditions.
comparison = unblocked_summary.merge(
    blocked_summary,
    on="snapshot",
    suffixes=("_unblocked", "_blocked"),
    validate="one_to_one",
)

comparison["best_beam_gain_delta_db_blocked_minus_unblocked"] = (
    comparison["best_beam_channel_gain_db_blocked"]
    - comparison["best_beam_channel_gain_db_unblocked"]
)

comparison["channel_norm_delta_db_blocked_minus_unblocked"] = (
    10.0
    * np.log10(
        np.maximum(
            comparison["channel_vector_norm_squared_blocked"],
            np.finfo(float).tiny,
        )
        / np.maximum(
            comparison["channel_vector_norm_squared_unblocked"],
            np.finfo(float).tiny,
        )
    )
)

comparison["valid_path_count_delta_blocked_minus_unblocked"] = (
    comparison["valid_path_count_blocked"]
    - comparison["valid_path_count_unblocked"]
)

comparison["los_visibility_changed"] = (
    comparison["los_visible_from_paths_unblocked"]
    != comparison["los_visible_from_paths_blocked"]
)

comparison["expected_geometric_blockage"] = (
    comparison["geometric_obstruction_intersection_blocked"]
)

comparison.to_csv(OUTPUT_COMPARISON, index=False)


# ======================================================================
# 11. SAVE COMPLEX CHANNELS AND BEAM METRICS
# ======================================================================

# Axis order:
#   condition: [unblocked, blocked]
#   snapshot:  trajectory row order
#   antenna:   64 TX antenna elements
#
# Beam gain array:
#   [condition, snapshot, beam_id]

channel_tensor = np.stack(
    [unblocked_h, blocked_h],
    axis=0,
)

beam_gain_tensor = np.stack(
    [unblocked_gains, blocked_gains],
    axis=0,
)

np.savez_compressed(
    OUTPUT_CHANNELS,
    condition_names=np.asarray(CONDITIONS),
    snapshot_ids=np.asarray(snapshots),
    rx_positions_m=position_values,
    tx_position_m=TX_POSITION,
    h_complex=channel_tensor,
    h_real=channel_tensor.real,
    h_imag=channel_tensor.imag,
    beamformed_channel_gain_linear=beam_gain_tensor,
    beamformed_channel_gain_db=10.0
    * np.log10(
        np.maximum(beam_gain_tensor, np.finfo(float).tiny)
    ),
    beam_directions_local=beam_directions,
    beam_weights_complex=beam_weights,
)

# Preserve every path coefficient for every TX antenna. Path counts vary
# across snapshots, so the final axis is padded with complex NaNs and
# path_counts records the number of valid path slots for each sample.
all_path_coefficients = [
    unblocked_path_coefficients,
    blocked_path_coefficients,
]
max_paths = max(
    matrix.shape[1]
    for condition_matrices in all_path_coefficients
    for matrix in condition_matrices
)
path_coefficient_tensor = np.full(
    (len(CONDITIONS), len(snapshots), NUM_TX_ANTENNAS, max_paths),
    np.nan + 1j * np.nan,
    dtype=np.complex128,
)
path_coefficient_counts = np.zeros(
    (len(CONDITIONS), len(snapshots)),
    dtype=np.int32,
)

for condition_index, condition_matrices in enumerate(all_path_coefficients):
    for snapshot_index, matrix in enumerate(condition_matrices):
        if matrix.ndim != 2 or matrix.shape[0] != NUM_TX_ANTENNAS:
            raise RuntimeError(
                "Unexpected antenna-resolved path coefficient matrix: "
                f"{matrix.shape}"
            )
        count = matrix.shape[1]
        path_coefficient_tensor[
            condition_index, snapshot_index, :, :count
        ] = matrix
        path_coefficient_counts[condition_index, snapshot_index] = count

np.savez_compressed(
    OUTPUT_PATH_COEFFICIENTS,
    condition_names=np.asarray(CONDITIONS),
    snapshot_ids=np.asarray(snapshots),
    tx_antenna_ids=np.arange(NUM_TX_ANTENNAS, dtype=np.int32),
    path_coefficients_complex=path_coefficient_tensor,
    path_coefficients_real=path_coefficient_tensor.real,
    path_coefficients_imag=path_coefficient_tensor.imag,
    path_counts=path_coefficient_counts,
    axis_order=np.asarray(
        ["condition", "snapshot", "tx_antenna", "path_index"]
    ),
    coefficient_reference_for_csv=np.asarray(
        "TX antenna index 0; full antenna-resolved values are in this archive"
    ),
)


# ======================================================================
# 12. VALIDATION
# ======================================================================

baseline_los = unblocked_summary[
    "los_visible_from_paths"
].to_numpy(dtype=bool)

blocked_los = blocked_summary[
    "los_visible_from_paths"
].to_numpy(dtype=bool)

blocked_expected_indices = np.flatnonzero(geometric_blockage)
unblocked_expected_indices = np.flatnonzero(~geometric_blockage)

# The original street-canyon scene may already be NLOS at some positions.
# Therefore, the valid control is that adding the obstruction removes LOS
# where it intersects the direct segment, while leaving LOS classification
# unchanged where it does not intersect the segment.
blocked_los_absent_where_expected = bool(
    (~blocked_los[blocked_expected_indices]).all()
)

los_status_unchanged_where_new_blocker_does_not_intersect = bool(
    np.array_equal(
        blocked_los[unblocked_expected_indices],
        baseline_los[unblocked_expected_indices],
    )
)

baseline_contains_los_and_nlos = bool(
    baseline_los.any() and (~baseline_los).any()
)

baseline_los_delay_consistent = bool(
    np.nanmax(
        np.abs(
            unblocked_summary.loc[
                unblocked_summary["los_visible_from_paths"],
                "los_delay_error_ns",
            ].to_numpy()
        )
    ) < 1.0
)

validation_checks = {
    "baseline_contains_both_los_and_nlos_snapshots": (
        baseline_contains_los_and_nlos
    ),
    "blocked_direct_los_absent_at_geometrically_intersected_snapshots": (
        blocked_los_absent_where_expected
    ),
    "los_classification_unchanged_where_new_blocker_does_not_intersect": (
        los_status_unchanged_where_new_blocker_does_not_intersect
    ),
    "baseline_direct_los_delay_matches_geometric_delay_within_1ns": (
        baseline_los_delay_consistent
    ),
}

validation_passed = all(validation_checks.values())

validation_lines = [
    "S4 VALIDATION REPORT",
    "=" * 72,
    f"Snapshots: {len(trajectory)}",
    f"Geometrically obstructed snapshots: {int(geometric_blockage.sum())}",
    f"Geometrically unobstructed snapshots: {int((~geometric_blockage).sum())}",
    "",
    "Checks:",
]

for check_name, passed in validation_checks.items():
    validation_lines.append(
        f"[{'PASS' if passed else 'FAIL'}] {check_name}"
    )

validation_lines.extend(
    [
        "",
        f"OVERALL: {'PASS' if validation_passed else 'FAIL'}",
        "",
        "Interpretation:",
        "- LOS is identified from valid paths whose interaction sequence contains only NONE (zero); refracted paths are not LOS.",
        "- Geometric blockage is independently determined by segment-AABB intersection.",
        "- The channel-gain metric is dimensionless and is not dBm or watts.",
        "- The concrete obstruction is a simplified controlled geometry.",
        "- No Doppler/time evolution is simulated in this spatial snapshot experiment.",
    ]
)

OUTPUT_VALIDATION.write_text(
    "\n".join(validation_lines) + "\n",
    encoding="utf-8",
)

print("\n" + "=" * 76)
print("S4 VALIDATION")
print("=" * 76)

for check_name, passed in validation_checks.items():
    print(f"[{'PASS' if passed else 'FAIL'}] {check_name}")

print("\nOverall validation:", "PASS" if validation_passed else "FAIL")


# ======================================================================
# 13. SAVE CONFIGURATION
# ======================================================================

config = {
    "experiment": "S4_Controlled_Environmental_Blockage",
    "sionna_rt_api_target": "2.2.0",
    "carrier_frequency_hz": CARRIER_FREQUENCY,
    "wavelength_m": WAVELENGTH,
    "transmitter_position_m": TX_POSITION.tolist(),
    "transmitter_orientation_rad": TX_ORIENTATION,
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
    "beam_codebook": {
        "num_candidate_beams": NUM_BEAMS,
        "construction": "S3-compatible forward-hemisphere DFT-style grid",
        "precoding_weights_unit_norm": True,
        "beam_metric": "abs(h_transpose_times_w)^2",
    },
    "solver": {
        "deterministic": True,
        "max_depth": MAX_DEPTH,
        "los": True,
        "specular_reflection": True,
        "diffuse_reflection": False,
        "refraction": True,
        "synthetic_array": True,
        "seed": SEED,
    },
    "path_geometry_export": {
        "enabled": True,
        "exports": [
            "interaction_type_per_depth",
            "passband_complex_path_coefficient_real_imag",
            "path_delay_s",
            "AoD_AoA_azimuth_zenith_rad_and_deg",
            "path_doppler_hz",
            "intersected_scene_object_id_per_interaction",
            "scene_object_name_lookup",
            "intersected_mesh_primitive_id_per_interaction",
            "world_coordinate_interaction_vertex_xyz_m",
        ],
        "invalid_object_id": INVALID_OBJECT_ID,
        "note": (
            "Only non-zero interaction codes are exported as physical "
            "interaction rows; padded entries are excluded."
        ),
    },
    "channel": {
        "cfr_baseband_frequency_offsets_hz": CFR_FREQUENCIES.tolist(),
        "channel_normalization": False,
        "delay_normalization": False,
        "exported_channel_shape": [
            len(CONDITIONS),
            len(trajectory),
            NUM_TX_ANTENNAS,
        ],
        "exported_channel_dtype": "complex128",
    },
    "obstruction": {
        "name": BLOCKER_NAME,
        "geometry": "axis-aligned closed cuboid",
        "interpretation": "simplified ground-connected building-like obstruction",
        "material_itu_type": BLOCKER_MATERIAL_TYPE,
        "material_thickness_m": BLOCKER_MATERIAL_THICKNESS_M,
        "center_m": BLOCKER_CENTER.tolist(),
        "size_xyz_m": BLOCKER_SIZE.tolist(),
        "bounds_min_m": BLOCKER_MIN.tolist(),
        "bounds_max_m": BLOCKER_MAX.tolist(),
    },
    "trajectory": {
        "input_file": str(INPUT_CSV),
        "num_snapshots": int(len(trajectory)),
        "snapshot_ids": [int(value) for value in snapshots],
    },
    "validation": validation_checks,
    "validation_passed": validation_passed,
    "metric_warning": (
        "Beamformed channel gain is dimensionless; it is not calibrated "
        "received power in watts or dBm."
    ),
}

with open(OUTPUT_CONFIG, "w", encoding="utf-8") as config_file:
    json.dump(config, config_file, indent=4)


# ======================================================================
# 14. FINAL REPORT
# ======================================================================

print("\n" + "=" * 76)
print("S4 EXPERIMENT FINISHED")
print("=" * 76)

print("Snapshots per condition:", len(trajectory))
print("TX antenna elements:", NUM_TX_ANTENNAS)
print("Candidate beams:", NUM_BEAMS)
print("Geometrically obstructed snapshots:", int(geometric_blockage.sum()))
print("Geometrically unobstructed snapshots:", int((~geometric_blockage).sum()))

print("\nPaired blockage comparison:")
display_columns = [
    "snapshot",
    "geometric_obstruction_intersection_blocked",
    "los_visible_from_paths_unblocked",
    "los_visible_from_paths_blocked",
    "best_beam_gain_delta_db_blocked_minus_unblocked",
    "valid_path_count_delta_blocked_minus_unblocked",
]
print(comparison[display_columns].to_string(index=False))

print("\nOutput files:")
for output_path in [
    OUTPUT_COMPARISON,
    OUTPUT_SUMMARY,
    OUTPUT_BEAMS,
    OUTPUT_PATHS,
    OUTPUT_INTERACTIONS,
    OUTPUT_OBJECTS,
    OUTPUT_CHANNELS,
    OUTPUT_CODEBOOK,
    OUTPUT_BLOCKER_OBJ,
    OUTPUT_CONFIG,
    OUTPUT_VALIDATION,
]:
    print(" -", output_path)

if not validation_passed:
    raise RuntimeError(
        "S4 completed but validation failed. Inspect s4_validation.txt "
        "and the per-snapshot CSV files before using this dataset."
    )

print("\nS4 execution and validation successful.")
