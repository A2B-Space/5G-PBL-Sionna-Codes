import numpy as np
import pandas as pd
from pathlib import Path

root = Path("reference/sionna-teammate/recovered_results/s6_physical_layer_dataset")
codebook = pd.read_csv(root / "beam_codebook.csv")

directions = codebook[
    ["u_x_local", "u_y_local", "u_z_local"]
].to_numpy(dtype=float)

# Angular distance using the angle between unit direction vectors.
norms = np.linalg.norm(directions, axis=1, keepdims=True)
unit = directions / norms

cosine = np.clip(unit @ unit.T, -1.0, 1.0)
angles = np.degrees(np.arccos(cosine))
np.fill_diagonal(angles, np.inf)

rows = []
for i, beam in enumerate(codebook["beam_id"].astype(int)):
    nearest = np.argsort(angles[i])[:10]
    for rank, j in enumerate(nearest, start=1):
        rows.append({
            "beam_id": beam,
            "azimuth_deg": codebook.iloc[i]["azimuth_local_deg"],
            "elevation_deg": codebook.iloc[i]["elevation_local_deg"],
            "neighbor_rank": rank,
            "neighbor_beam_id": int(codebook.iloc[j]["beam_id"]),
            "angular_distance_deg": angles[i, j],
        })

out = Path("results/analysis")
out.mkdir(parents=True, exist_ok=True)

neighbors = pd.DataFrame(rows)
neighbors.to_csv(out / "sionna_beam_neighbors.csv", index=False)

print("CODEBOOK DIRECTION NORMS:")
print(norms.min(), norms.max())

print("\nANGULAR NEIGHBORS OF BEAM 12:")
print(neighbors[neighbors.beam_id == 12].to_string(index=False))

print("\nANGULAR NEIGHBORS OF BEAM 23:")
print(neighbors[neighbors.beam_id == 23].to_string(index=False))

print("\nOVERALL NEAREST-NEIGHBOR ANGULAR DISTANCE:")
print(neighbors.groupby("beam_id")["angular_distance_deg"].first().describe())

print("\nSaved:", out / "sionna_beam_neighbors.csv")
