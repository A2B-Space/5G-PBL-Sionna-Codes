# Methodology, equations and assumptions

## Ray tracing and geometry

The physical-layer pipeline uses Sionna RT PathSolver on the
`simple_street_canyon` scene. The recorded configuration uses a 28 GHz carrier,
8x8 TX UPA, 1x1 RX, deterministic path generation, max interaction depth 5,
LOS/specular reflection/refraction enabled, diffuse reflection disabled,
synthetic arrays enabled and seed 41. The blocked condition adds a simplified
concrete cuboid. It is a controlled obstruction experiment, not a detailed
city-scale digital twin.

A fixed seed alone is not the full determinism condition: Sionna RT documents
that deterministic path generation requires constructing PathSolver with
`deterministic=True` as well as passing a seed. The recorded experiment uses
both settings.

## Path and channel quantities

For path p, the export records delay, passband complex coefficient, departure
and arrival angles, Doppler and interaction metadata. The antenna-resolved
archive preserves coefficients for 64 TX antenna elements and valid path slots.

The channel vector is exported at a single baseband frequency offset of 0 Hz.
It is not a sampled wideband CFR. Doppler values are zero for the current
static-position snapshots.

## Beamforming metric

The evaluated codebook metric is:

`G_b = |h^T w_b|^2`

where `h` is the 64-element channel vector and `w_b` is a unit-norm codebook
weight vector. This is a dimensionless channel-gain metric. Its dB form is
`10 log10(G_b)`; it is not calibrated received power or path loss in dBm.

## RMS delay spread

For path powers `P_p` and delays `tau_p`, normalized power weights are
`q_p = P_p / sum_p P_p`. The mean delay is
`tau_bar = sum_p q_p tau_p`, and RMS delay spread is
`sqrt(sum_p q_p (tau_p - tau_bar)^2)`.
The result depends on the paths retained by the ray tracer and the path-power
quantity used in the S7 implementation; it is not a measured channel sounding.

## Adaptive tracking

The S5.1 policy uses exhaustive initial acquisition, a 9-beam local candidate
set, 8 dB same-beam degradation detection, 20 dB residual degradation to
expand to 32 angularly nearest beams, 3 dB switching hysteresis and full
52-beam fallback. The exhaustive codebook is used as an offline oracle for
scoring. Candidate probe counts are not measured latency or energy.

## Statistical scope

There is one scene and one spatial quasi-static receiver trajectory, evaluated
under two paired obstruction conditions. Treat all reported comparisons as
descriptive paired simulation results. Adjacent snapshots are not independent
replicates. Do not report population-level confidence intervals or unseen
scene/trajectory generalization from this dataset alone.
