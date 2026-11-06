Sionna RT Physical-Layer Simulation and Dataset Pipeline
A reproducible research codebase for ray-tracing-based physical-layer simulation, multipath analysis, environmental blockage experiments, beam selection, adaptive beam tracking, and physical-layer dataset generation using NVIDIA Sionna RT.

This repository documents the Sionna workstream developed for the Antenna PBL project. It is designed to provide a physical-layer simulation and dataset foundation that can later be aligned with a separate camera/semantic-communication pipeline.
Research status: S1–S7 have been implemented and evaluated for the current scene and receiver trajectory. S8 consolidates the available code, results, configurations, and provenance. This is a controlled simulation study, not yet a multi-scene generalization study or a completed semantic-to-physical-layer integration.

Table of contents
- Project objectives
- S1–S8 at a glance
- Experimental configuration
- Repository layout
- Environment and dependencies
- Execution workflow
- Stage descriptions
- Dataset description
- Reported results
- Reproducibility and validation
- Scientific scope and limitations
- Git and large-file policy
- References
Project objectives
The Sionna RT workstream has five practical objectives:
1. Build a three-dimensional propagation scenario and generate physically meaningful ray-tracing paths.
2. Study how receiver movement and environmental obstruction change propagation paths and channel characteristics.
3. Construct a transmit antenna array and finite beam codebook, then evaluate beam selection and tracking.
4. Export structured physical-layer data—including paths, interactions, channel vectors, beam measurements, and tracking records—for downstream analysis.
5. Preserve the experiment configuration, outputs, validation evidence, and execution instructions so that the work can be audited and reproduced.
The intended future integration key is a shared scenario/snapshot identity. The semantic pipeline must eventually provide compatible scene, trajectory, timestamp/snapshot, coordinate-system, and object identity information. That integration is not included in the current S1–S8 physical-layer dataset.
S1–S8 at a glance
Stage	Name	Main work	Status
S1	Ray-tracing baseline	Load the street-canyon scene, configure a baseline link, solve paths, and inspect CIR/path delays	Completed
S2	Receiver mobility	Evaluate 11 quasi-static receiver positions and export path counts and delays	Completed
S3	Antenna array and beam selection	Configure the 8×8 TX UPA, construct a 52-beam codebook, and evaluate beam gains	Completed
S4	Environmental blockage	Add a concrete obstruction and compare blocked/unblocked propagation and beam metrics	Completed
S5	Beam tracking baseline	Evaluate fixed/local tracking across the receiver trajectory	Completed
S5.1	Adaptive beam tracking	Add local search, degradation detection, expanded search, hysteresis, and fallback	Completed
S6	Physical-layer dataset	Assemble snapshot, path, interaction, channel, beam, and tracking records	Completed; antenna-resolved path-coefficient extension added
S7	Performance evaluation	Analyze channel characteristics, blockage, multipath, beam selection, and adaptive tracking	Completed for the current experiment
S8	Consolidation and reproducibility	Consolidate available source, results, manifests, hashes, methodology, and handoff documentation	Package prepared; final local audit depends on the user's environment snapshot


S5.1 is a refinement of S5, not an additional top-level phase in the original S1–S8 roadmap.
Source availability note
The consolidated artifact inventory contains S1 and S2 result files, but the original S1 and S2 Python source scripts were not present in the files used to prepare the S8 package. If those scripts are available in the working WSL project, retain and add them to the repository. Do not represent reconstructed scripts as the original implementations.
Experimental configuration
Scenario
- Scene: simple_street_canyon
- Carrier frequency: 28 GHz
- TX position: [-33.0, 11.0, 32.0] metres
- RX trajectory: (x, -13.0, 1.5) metres
- RX x positions: 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, and 37 metres
- Trajectory type: spatial, quasi-static snapshots
- Conditions: original/unblocked scene and scene with an added concrete obstruction
- Samples: 11 positions × 2 conditions = 22 snapshot samples
Antenna and beam configuration
For S3 onward and the S4–S7 beam experiments:
- TX array: 8×8 uniform planar array (64 elements)
- TX antenna pattern: tr38901
- TX polarization: vertical
- TX element spacing: 0.5 wavelength
- RX array: 1×1 isotropic, vertically polarized
- Beam codebook: 52 candidate beams
- Codebook weights: complex, unit-norm vectors of length 64
- The codebook covers a finite forward-hemisphere angular grid; the selected beam is optimal only relative to this codebook.
S1 and S2 were initial propagation baselines using 1×1 isotropic TX/RX arrays. Do not assume that the S1/S2 channel exports are antenna-resolved 8×8-array data.
Ray-tracing configuration
The recorded S4/S6 configuration uses:
- PathSolver(deterministic=True)
- max_depth=5
- LOS enabled
- Specular reflection enabled
- Refraction enabled
- Diffuse reflection disabled
- synthetic_array=True
- Seed: 41
A fixed seed alone does not guarantee deterministic path generation in Sionna RT; the solver's deterministic=True option is also used.
Added obstruction
The blocked condition adds a simplified concrete cuboid:
- Centre: (-3.0, -0.5, 11.0) metres
- Dimensions: (1.0, 3.0, 22.0) metres
- AABB minimum: (-3.5, -2.0, 0.0) metres
- AABB maximum: (-2.5, 1.0, 22.0) metres
- Material: ITU concrete
- Thickness: 0.30 metres
This is a controlled building-like obstruction experiment. It is not a detailed city-scale digital twin or a calibrated material survey.
Repository layout
The exact layout may differ between a working directory and the S8 handoff package. A recommended Git repository structure is:
sionna-rt-physical-layer/
├── README.md
├── .gitignore
├── requirements.txt                 # Optional: project-level Python dependencies
├── src/
│   ├── S1/                           # Add original S1 script if available
│   ├── S2/                           # Add original S2 script if available
│   ├── S3/
│   │   └── s3_beam_selection.py
│   ├── S4/
│   │   ├── s4_blockage.py
│   │   └── s4_1_path_geometry_export_s6_fixed.py
│   ├── S5/
│   │   └── s5_beam_tracking.py
│   ├── S5_1/
│   │   └── s5_1_adaptive_beam_tracking.py
│   ├── S6/
│   │   └── s6_build_dataset.py
│   ├── S7/
│   │   └── s7_extend_and_evaluate.py
│   └── S8/
│       └── s8_consolidate.py
├── docs/
│   ├── METHODOLOGY_AND_ASSUMPTIONS.md
│   ├── REPRODUCIBILITY_RUNBOOK.md
│   ├── SEMANTIC_INTEGRATION_CONTRACT.md
│   └── SOURCE_GAPS.md
├── configs/                          # Frozen experiment configurations
├── manifests/                        # Stage registry and checksums
├── results/                           # Curated, versioned result artifacts
└── .venv/                             # Local environment; ignored by Git
The generated data directories used by the scripts in the working WSL project are typically:
~/sionna_project/
├── .venv/
├── s4_blockage_geometry/
├── s5_beam_tracking/
├── s5_1_adaptive_tracking/
├── s6_physical_layer_dataset/
├── s6_physical_layer_dataset_extended/
├── s7_performance_evaluation/
└── S8_reproducibility_package/
Keep generated outputs outside src/. Commit source code and documentation; version curated results deliberately rather than committing every intermediate file.
Environment and dependencies
Reference environment
The working environment used for the implementation was:
- Ubuntu under WSL2
- Python 3.12
- Sionna RT 2.2.0
- Mitsuba 3.9.1
- Dr.Jit 1.5.0
- NVIDIA RTX 3050 (4 GB) on the teammate's machine
The NVIDIA driver and WSL CUDA user-mode stack are host-specific. Record the actual output of nvidia-smi and the installed package list on the machine used for a reproduction run.

Do not install a Linux display driver inside WSL as a substitute for the Windows NVIDIA driver. The WSL GPU stack is supplied through the Windows host/WSL integration.
Activate the existing environment
cd ~/sionna_project
source .venv/bin/activate

python --version
python -m pip freeze
nvidia-smi
Do not recreate or reinstall the validated Sionna/Mitsuba/Dr.Jit environment merely to run the CPU-based dataset assembly, evaluation, or consolidation stages.
Python dependencies used by the analysis scripts
The scripts use the following Python packages where applicable:
- numpy
- pandas
- matplotlib
- scipy (only if imported by a particular analysis utility)
- Sionna RT, Mitsuba, and Dr.Jit for ray-tracing stages
Use the exact package versions recorded in the local environment snapshot or a project lock file. Do not treat an unpinned dependency list as a fully reproducible environment.
Execution workflow
Important
The commands below assume that the corresponding scripts are present in ~/sionna_project. S1/S2 source scripts must be supplied separately if you want to rerun those historical stages from source.
Downstream regeneration from the available S4.1 source
The current validated downstream sequence is:
cd ~/sionna_project
source .venv/bin/activate

# S4.1: regenerate enriched path geometry, antenna coefficients,
# channel exports, beam metrics, and related S4 outputs.
python s4_1_path_geometry_export_s6_fixed.py

# S5: regenerate the baseline tracking outputs required by S5.1.
python s5_beam_tracking.py

# S5.1: adaptive tracking.
python s5_1_adaptive_beam_tracking.py

# S6: assemble the original physical-layer dataset.
python s6_build_dataset.py

# S7: create the antenna-resolved extension and run S7.1-S7.5.
python s7_extend_and_evaluate.py

# S8: consolidate source, results, manifests, and local environment metadata.
python s8_consolidate.py --project-dir ~/sionna_project
If S4.1 is rerun, rerun all downstream stages. This prevents S5.1, S6, and S7 from referring to different generations of S4 outputs.

S4.1 is a Sionna RT/GPU stage. S5, S5.1, S6, S7, and S8 operate on generated data and are principally CPU/data-processing stages.
S1–S3
S1 and S2 are historical baseline stages. Their output CSVs are included in the available consolidation artifacts, but their original scripts were not supplied to the S8 package build. S3 source is available as s3_beam_selection.py. Retain the original S1/S2 scripts if they exist locally; do not infer their complete implementation from result CSVs.
Stage descriptions
S1 — Ray-tracing baseline
The initial experiment used sionna.rt.scene.simple_street_canyon, a 28 GHz carrier, and a single-antenna isotropic TX/RX baseline.

Recorded link positions:
- TX: [-33, 11, 32] metres
- RX: [27, -13, 1.5] metres
The deterministic path solution used maximum interaction depth 5, LOS/specular reflection/refraction enabled, diffuse reflection disabled, and seed 41.

The baseline produced 20 valid paths. The strongest path delay was approximately 238.358 ns, consistent with the geometric LOS propagation delay for the configured link. The exported coefficient power is a simulation channel-coefficient quantity, not calibrated received power in watts.

Available result: s1_ray_tracing_results.csv.
S2 — Receiver mobility
The receiver was moved through 11 spatial snapshots:
TRAJECTORY_X = np.arange(17.0, 37.0 + 0.1, 2.0)
The receiver positions were (x, -13, 1.5) metres. The experiment used quasi-static snapshots; receiver velocity was zero and continuous-time Doppler evolution was not simulated.

Available results:
- s2_mobility_summary.csv
- s2_mobility_paths.csv
The experiment recorded changes in valid path count and path delays across the trajectory. The minimum path delay agreed with the geometric LOS delay at each snapshot where the direct path existed.
S3 — Antenna array and beam selection
S3 introduced the 8×8 TX UPA (64 elements), a 1×1 isotropic RX, and a 52-beam codebook. The channel was evaluated at a zero baseband frequency offset. The beam metric was:
\[
G_b = \left|\mathbf{h}^{T}\mathbf{w}_b\right|^2
\]
where \(\mathbf{h}\) is the 64-element channel vector and \(\mathbf{w}_b\) is the complex weight vector for beam \(b\).

The transpose in this expression is intentional and follows the implemented Sionna precoding convention used in this experiment. The metric is dimensionless; its dB representation is \(10\log_{10}(G_b)\), not dBm.

Available results:
- s3_beam_codebook.csv
- s3_beam_selection_all_beams.csv
- s3_beam_selection_summary.csv
S4 — Environmental blockage
S4 added the concrete cuboid described in Experimental configuration and evaluated both the original and blocked scene at the same 11 receiver positions.

The added obstruction removed an existing direct LOS path at RX x positions 27, 29, and 31 metres. Other positions were already NLOS in the original scene. Therefore, do not label every blocked-condition NLOS sample as a newly caused LOS-to-NLOS transition.

The geometry-intersection flag in the original comparison experiment is a segment-versus-hypothetical-obstruction test. In the unblocked condition it indicates whether the TX–RX segment would intersect the obstruction's location if the obstruction were present; it does not mean the obstruction exists in the unblocked scene.

Available outputs include: beam metrics, codebook, snapshot summary, path details, blockage comparison, and validation report. The enriched S4.1 exporter additionally writes path interaction geometry, per-path attributes, and antenna-resolved coefficients.
S5 — Baseline beam tracking
S5 evaluated the fixed/local beam-tracking baseline using the 52-beam codebook and spatial snapshots. It provides the baseline against which S5.1 is compared.
S5.1 — Adaptive beam tracking
The adaptive policy uses:
- Initial exhaustive acquisition over all 52 beams.
- Normal local search over 9 angularly nearest codebook beams.
- Same-beam gain-drop trigger: greater than 8 dB.
- Expanded search over 32 angularly nearest beams when the local result still has more than 20 dB residual degradation relative to the previous selected-beam gain.
- Switching hysteresis: at least 3 dB improvement.
- Full 52-beam fallback if expanded search does not meet the hysteresis threshold.
The exhaustive codebook is used as an offline oracle for evaluation. The adaptive decision itself uses only the candidate set. Probe counts represent candidate-beam measurement counts, not measured wall-clock latency or energy consumption.
S6 — Physical-layer dataset generation
S6 assembles a structured dataset with 22 samples (11 positions × 2 conditions), including:
- Snapshot metadata
- Per-path records
- Path interaction records
- Per-beam measurements
- Tracking records and summaries
- Scene object information
- Beam codebook
- Complex channel archive
- Manifest, data dictionary, validation report, and QA plots
The S6 extension adds the antenna-resolved path coefficient export:
- path_antenna_coefficients.csv
- path_coefficients_antenna_resolved.npz
The extended coefficient archive uses condition, snapshot, TX antenna, and path-slot axes. The recorded archive shape is (2, 11, 64, 22). The final dimension is the allocated path-slot dimension; not every slot is a valid path. Invalid slots are represented separately and must not be counted as physical paths.

The scalar coefficient retained in path_records.csv refers to TX antenna index 0. The antenna-resolved archive stores the 64 TX-element coefficients per valid path.
S7 — Performance evaluation
S7.1–S7.5 produce quantitative CSVs, figures, a findings summary, a manifest, and a combined PDF report.
S7.1 — Channel characterization
Evaluates channel-vector norm, path counts/powers, strongest-path behaviour, and RMS delay spread.

RMS delay spread is computed from path delays and normalized path-power weights:
\[
q_p = \frac{P_p}{\sum_k P_k}, \qquad
\bar{\tau} = \sum_p q_p \tau_p
\]
\[
\tau_{\mathrm{RMS}} =
\sqrt{\sum_p q_p(\tau_p-\bar{\tau})^2}
\]
The result depends on the paths retained by the ray tracer and the path-power definition used in the analysis. It is a simulation-derived statistic, not a measured channel-sounding result.
S7.2 — Blockage evaluation
Performs paired comparisons between blocked and unblocked conditions at identical receiver positions. It reports path-count changes, LOS visibility, and best-codebook-beam gain differences.
S7.3 — Multipath evaluation
Summarizes reflection/refraction events, interaction depth, and path-delay characteristics. Interaction-event counts are not counts of unique objects or necessarily independent paths.
S7.4 — Beam-selection evaluation
Compares selected beams with the exhaustive best beam in the finite 52-beam codebook. Reports regret, beam agreement, and per-snapshot behaviour.
S7.5 — Adaptive tracking evaluation
Compares fixed/local and adaptive policies using probe counts, beam switches, and oracle regret. Results are specific to the current scene and trajectory.
S8 — Consolidation and reproducibility
S8 consolidates the available scripts, results, validation reports, configurations, environment metadata, and file hashes. It also records source gaps and documents the proposed semantic integration interface.

The S8 package is a provenance and handoff package. It does not guarantee bitwise-identical reruns across different GPUs, drivers, operating systems, Mitsuba variants, or software stacks.
Dataset description
Sample identity
The physical-layer dataset uses a sample identity based on:
scene_id + trajectory_id + condition_id + snapshot
The current experiment contains:
- scene_id: simple_street_canyon
- trajectory_id: s2_rx_x17_to_x37_step2m
- condition_id: unblocked or blocked
- snapshot: the receiver-position snapshot identifier
Principal files
File	Description
snapshot_metadata.csv	One row per scene/trajectory/condition/snapshot
path_records.csv	Valid propagation path attributes, including delay, scalar reference coefficient, angles, Doppler, and interaction summary
interaction_records.csv	Per-path interaction order/type, object/primitive identifiers, and interaction vertex coordinates
beam_measurements.csv	Gain for each of the 52 beams at each sample
tracking_records.csv	Tracking decisions and method-specific measurements
tracking_summary.csv	Aggregated tracking metrics
scene_objects.csv	Scene object metadata
beam_codebook.csv	Beam IDs, directions, and complex weights
channel_data.npz	Complex channel vectors and beam-gain arrays
path_antenna_coefficients.csv	Long-form antenna-resolved path coefficients
path_coefficients_antenna_resolved.npz	Dense antenna-resolved coefficient array with validity information
manifest.json	Dataset configuration and provenance
data_dictionary.json	Field definitions and units
validation.txt	Dataset validation results


Use the data_dictionary.json shipped with the dataset as the authoritative field-level schema for a particular dataset version.
Array conventions
The recorded S6/S7 artifacts include these key shapes:
- Channel vectors: (2, 11, 64)
- Beam gains: (2, 11, 52)
- Codebook weights: (52, 64)
- Antenna-resolved path coefficients: (2, 11, 64, 22)
The axes are condition, snapshot, TX antenna, and path slot as applicable. Always inspect the array keys and metadata in the exact .npz archive before relying on axis order in a new dataset version.
Reported results
The following are descriptive results from the current 22-sample simulation. They are not population-level estimates.
Blockage and channel statistics
- Mean RMS delay spread: approximately 8.54 ns in the unblocked condition and 53.85 ns in the blocked condition for the reported S7 calculation.
- The blocked-condition mean is approximately 6.31 times the unblocked mean.
- Mean paired best-codebook-beam gain change (blocked minus unblocked): approximately −18.21 dB.
- The added obstruction removed existing LOS at RX x positions 27, 29, and 31 metres.
- The original scene was already NLOS at the other eight receiver positions.
Interaction-event counts
Event type	Unblocked	Blocked
Specular reflection events	316	242
Refraction events	235	219
Total interaction events	551	461


These are interaction-event counts, not unique object counts.
Adaptive tracking
Metric	Fixed/local tracker	Adaptive tracker
Unblocked probes	228	142
Blocked probes	228	188
Total probes	456	330
Mean blocked-condition regret	2.51 dB	0 dB
Maximum blocked-condition regret	14.65 dB	0 dB


Across both conditions, adaptive tracking uses 330 candidate-beam probes compared with 1,144 probes for exhaustive search, a reduction of approximately 71.15%. This is a simulated measurement-count reduction, not a measured latency or energy saving.

The adaptive policy matched the exhaustive 52-beam codebook optimum at all 22 evaluated snapshots. This is a result for the current experiment only and does not establish performance on unseen trajectories or scenes.
Reproducibility and validation
Validation reports
Inspect the validation files produced by each stage before using its outputs:
- S4: s4_validation.txt
- S5: s5_validation.txt
- S5.1: s5_1_validation.txt
- S6: validation.txt
- S7: s7_validation.txt
- S8: audit report and stage registry in the S8 package
A PASS report establishes that the listed checks passed for those stored artifacts. It does not independently prove physical realism, calibration, or generalization.
Checksums
The S8 package includes a SHA-256 checksum manifest. From the package root, verify it with:
sha256sum -c manifests/file_checksums.sha256
A checksum match confirms file identity relative to the manifest; it does not prove that a new simulation run will reproduce bit-identical results.
Record the local environment
Before a new experiment, capture:
python --version
python -m pip freeze > environment_freeze.txt
nvidia-smi > nvidia_smi.txt
Do not use sionna.__version__ as the version-reporting mechanism in project scripts. Record installed distribution metadata or the environment freeze instead.
Freeze policy
- Treat validated S6/S7 result directories as immutable evidence.
- Write new experiments to new scenario/trajectory/dataset-version directories.
- If S4.1 is regenerated, rerun S5, S5.1, S6, and S7.
- Update the manifest, data dictionary, validation report, and checksums when the dataset schema or contents change.
- Never silently overwrite a published result set.
Scientific scope and limitations
The current experiment has important boundaries:
1. One scene: only simple_street_canyon is evaluated.
2. One receiver trajectory: 11 spatial snapshots are used; adjacent snapshots are not independent experimental replicates.
3. Two paired conditions: original/unblocked and one added concrete obstruction.
4. Quasi-static snapshots: receiver velocity and continuous-time channel evolution are not modeled.
5. Zero Doppler: the current exported Doppler values are zero for the static-position experiment.
6. Single frequency offset: the exported channel vector is evaluated at a zero baseband frequency offset; it is not a sampled wideband CFR.
7. Finite codebook: “best beam” means best among the 52 codebook entries, not a globally optimal continuous steering direction.
8. Uncalibrated gain: |h^T w|^2 is dimensionless and is not received power in watts/dBm.
9. No measured training latency or energy: probe counts are a proxy for candidate measurements.
10. No semantic observations in S6: camera frames, masks, bounding boxes, semantic classes, and cross-modal labels are not part of the current physical-layer dataset.
11. No generalization claim: further independent scenes, trajectories, obstacle geometries, and channel conditions are required for robust generalization claims.
These limitations should be retained in reports, presentations, and papers.
Git and large-file policy
The repository should normally track:
- Python source scripts
- README and other documentation
- Small configuration files
- Data dictionaries and manifests
- Validation reports
- Selected, curated result tables and plots
The repository should normally ignore:
- .venv/ and other local Python environments
- Python bytecode and cache directories
- IDE/OS metadata
- Local .env files and secrets
- Temporary logs and scratch files
- Regenerable simulation output directories
- Large binary arrays and archives unless deliberately versioned
Use the accompanying .gitignore file as a starting point. It intentionally does not ignore all CSV, JSON, PDF, or PNG files, because selected research results and documentation may be worth versioning.

For large datasets that must be shared, use a deliberate artifact strategy such as Git LFS or an external research-data store. Do not commit virtual environments, driver installers, or generated dependency folders.
References
- NVIDIA Sionna RT 2.2.0 — Introduction to Sionna RT
- NVIDIA Sionna RT 2.2.0 — Path Solvers API
- NVIDIA Sionna RT 2.2.0 — Understanding the Paths Object
- NVIDIA — CUDA on WSL User Guide
- Git project — gitignore documentation
Project status
S1–S8 physical-layer implementation: completed for the documented single-scene, single-trajectory experiment, with the S8 local environment audit and source inventory to be maintained alongside the actual working repository.

Next research direction: expand the physical-layer dataset with independent scenes and trajectories, then align the physical-layer sample identifiers and geometry with the separate semantic-perception/communication pipeline. That integration should be treated as a new experimental stage with its own validation, not as an already completed S1–S8 result.
