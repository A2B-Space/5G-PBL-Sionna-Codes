# Sionna RT S1-S8 reproducibility package

This package consolidates the available S1-S7 code, results, configurations,
validation reports and dataset documentation. It is a provenance and handoff
package; it does not claim bitwise reproducibility across different GPUs,
drivers, Mitsuba variants or software stacks.

## Current experiment scope

- Scene: `simple_street_canyon`
- Carrier: 28 GHz
- TX: 8x8 UPA (64 elements), RX: 1x1
- Codebook: 52 beams
- Receiver trajectory: 11 spatial snapshots, x=17..37 m in 2 m increments
- Conditions: original/unblocked and added-concrete-obstruction/blocked
- Solver: deterministic PathSolver, max depth 5, LOS/specular/refraction enabled,
  diffuse reflection disabled, synthetic arrays enabled, seed 41
- Channel archive: one baseband frequency offset (0 Hz), not wideband CFR
- Dataset: 22 paired samples, one scene and one quasi-static trajectory

## Important interpretation limits

1. The beam metric `abs(h^T w)^2` is dimensionless, not calibrated watts/dBm.
2. The `Paths.a` values are passband path coefficients; do not conflate them
   with the baseband-equivalent CIR coefficients.
3. The 64-element path-coefficient archive is antenna-resolved, but the scalar
   `path_records.csv` coefficient is the reference TX antenna (index 0).
4. Doppler is zero in the current static-position experiment.
5. Probe counts are candidate-beam measurement counts, not measured latency,
   energy or real hardware training duration.
6. Results describe one scene and one trajectory; no population-level or
   unseen-trajectory generalization claim is supported.
7. Camera imagery, masks, bounding boxes, semantic classes and cross-modal
   labels are not part of this physical-layer dataset.

## Reproduction

See `docs/REPRODUCIBILITY_RUNBOOK.md`. The environment snapshot and exact file
hashes are in `environment/` and `manifests/`.

## Stage inventory

`manifests/stage_registry.json` records which source scripts, outputs and
validation reports were found. Missing S1/S2 source scripts are explicitly
reported if they are not present in the source project directory. No missing
script is reconstructed or represented as original source.
