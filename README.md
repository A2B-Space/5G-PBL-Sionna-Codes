# Sionna RT – GitHub Repository Setup, Collaboration & Next Steps

## 1. Verify the GitHub Upload

After successfully pushing the Sionna project to the private GitHub repository:

**Repository:** [5G-PBL-Sionna-Codes](https://github.com/A2B-Space/5G-PBL-Sionna-Codes)

Run the following commands in the Antigravity terminal:

```powershell
git status
git remote -v
git log --oneline --graph --decorate -10
```

Verify that:
- The working tree is clean.
- The remote points to the correct private GitHub repository.
- The latest S1–S8 implementation commit is present on `main`.
- The uploaded README, Python scripts, documentation, and `.gitignore` are visible on GitHub.

## 2. Recommended Repository Structure

The following is a suggested organization for maintaining the project:

```text
5G-PBL-Sionna-Codes/
│
├── README.md
├── .gitignore
├── LICENSE
├── CITATION.cff
├── requirements.txt
│
├── S1_Ray_Tracing/
├── S2_Vehicle_Mobility/
├── S3_Beam_Selection/
├── S4_Blockage/
├── S5_Beam_Tracking/
├── S5_1_Adaptive_Tracking/
├── S6_Dataset_Generation/
├── S7_Performance_Evaluation/
├── S8_Reproducibility/
│
├── docs/
│
└── configs/
```

**Note:** This is a recommended structure, not a requirement to rename existing folders immediately. Confirm licensing rights before adding a project-wide LICENSE, particularly if the repository contains code adapted from upstream research repositories.

## 3. Add Teammates as GitHub Collaborators

To enable collaborative development:

1. Open the private GitHub repository.
2. Navigate to **Settings → Collaborators**.
3. Select **Add people**.
4. Enter your teammate's GitHub username.
5. Send the invitation.

Once the invitation is accepted, the teammate can access the repository and clone it onto their machine.

## 4. Establish a Git Branching Workflow

To avoid conflicts and protect the stable implementation, use the following branching strategy.

### Main Branch

- Contains stable and verified implementations.
- Should only receive tested and integrated changes.
- Serves as the reference version of the project.

### Development Branch

- Used for integrating new experiments and features.
- Acts as the intermediate branch before changes are merged into `main`.

Create the development branch using:

```powershell
git switch -c development
git push -u origin development
```

### Feature Branches

Individual development tasks should use separate branches.

Examples:

- `feature/multi-scene-generation`
- `feature/trajectory-generalization`
- `feature/semantic-integration`
- `feature/channel-analysis`

Merge completed and tested feature branches into `development`, and periodically merge stable development into `main`.

## 5. Recommended Research Development Roadmap

With the Sionna S1–S8 implementation uploaded, the next steps should focus on reproducibility and research expansion.

| Priority | Task | Objective |
|---|---|---|
| 1 | Verify GitHub contents | Ensure all source files and documentation are present |
| 2 | Environment specification | Record exact dependencies and software versions |
| 3 | Reproducibility test | Verify that another machine can reproduce the existing S1 baseline |
| 4 | Multiple scene generation | Move beyond the single street-canyon scenario |
| 5 | Multiple trajectory generation | Evaluate whether beam tracking generalizes across trajectories |
| 6 | Semantic integration | Connect Sionna physical-layer outputs with the existing semantic pipeline |

## 6. Immediate Next Milestone: Independent Reproducibility

Before introducing additional features, the first objective should be to verify the existing implementation on the teammate's laptop.

Recommended procedure:

1. Clone the private GitHub repository onto the teammate's laptop.
2. Configure the WSL2 and Python environment.
3. Install the documented Sionna RT dependencies.
4. Run the available baseline ray-tracing implementation.
5. Compare the generated path and channel outputs against the existing reference results.
6. Record any differences in the environment and validation reports.

This will establish an independent reproducibility check for the current implementation.

## 7. Long-Term Research Direction

The next major research milestone is:

**Environment-Semantic-Assisted Beam Prediction and Adaptive Beam Tracking Using Sionna RT-Generated Physical-Layer Data.**

The intended direction is to expand the physical-layer dataset across multiple scenes and receiver trajectories, then align its geometry and sample identifiers with the existing camera-based semantic-perception and communication pipeline.

This integration should be treated as a new experimental stage, with its own methodology, validation, and performance evaluation.

**Current status:** S1–S8 have been implemented and evaluated for the documented single-scene, single-trajectory experiment. Multi-scene generalization and semantic-to-physical-layer integration remain future work.

## Research Extension: Adaptive Semantic Beam Management

An additional research extension investigates beam-utility-aware adaptive semantic communication for distributed sensing-aided mmWave beam management.

It includes:
- Scenario 41 data-validation and synchronization analysis tools.
- DeepSense beam-codebook and Sionna beam-neighbour analysis.
- Confidence-adaptive, prior-guided, and utility-aware beam-selection experiments.
- Communication-cost, beam-reliability, and post-acquisition analysis.

**Implementation and experimental results:** [`environment_semantic_extension/`](./environment_semantic_extension/)

See the extension's [README](./environment_semantic_extension/README.md) for its structure, current scope, and usage notes.

