\# Environment Semantic Communication — Research Extension



\## Overview



This directory contains the research extension developed for adaptive semantic communication and beam management in distributed sensing-aided mmWave networks.



The work builds upon the existing Sionna-based 5G physical-layer project and investigates how semantic information, beam utility, confidence, and communication cost can support adaptive beam-selection decisions.



\## Research Direction



\*\*Beam-Utility-Aware Adaptive Semantic Communication for Distributed Sensing-Aided mmWave Beam Management\*\*



The central objective is to reduce semantic communication and beam-acquisition overhead while maintaining acceptable beam-selection reliability.



\## Repository Structure



```text

environment\_semantic\_extension/

├── src/

│   ├── data/

│   └── integration/

├── s9\_1\_adaptive\_analysis/

├── s9\_fixed\_budget\_analysis/

├── results/

│   ├── analysis/

│   ├── calibration\_cost\_utility\_audit/

│   ├── confidence\_adaptive\_experiment/

│   ├── controlled\_prior\_experiment/

│   ├── deepsense\_codebook\_inspection/

│   ├── post\_acquisition/

│   ├── prior\_guided\_experiment/

│   ├── risk\_aware\_adaptive\_selector/

│   └── utility\_aware\_adaptive\_selector/

└── README.md

```



\## Current Work



The extension includes:



\- Scenario 41 data and synchronization validation tools.

\- DeepSense beam-codebook inspection and beam-pattern analysis.

\- Sionna beam-neighbour analysis.

\- Confidence-adaptive and prior-guided experiments.

\- Controlled-prior and risk-aware adaptive selection experiments.

\- Utility-aware beam-selection and cost–reliability analysis.

\- Post-acquisition cost and utility evaluation.

\- Integration utilities for semantic predictions and adaptive beam selection.



\## Requirements



The scripts are developed in Python. Individual scripts may require additional packages depending on the experiment, including NumPy, Pandas, Matplotlib, SciPy, and Sionna.



Install the dependencies required by the particular script before running it.



\## Running the Experiments



Run commands from the repository root. For example:



```powershell

python .\\environment\_semantic\_extension\\src\\run\_confidence\_adaptive\_experiment.py

```



Other scripts are available under `src/`. Check each script's arguments and configuration before execution.



\## Results



The `results/` directory contains selected aggregate experiment summaries, validation reports, CSV metrics, and plots.



Individual sample-level outputs and raw dataset files are intentionally excluded from this initial public release.



\## Data and Attribution



The project builds on the existing Sionna-based repository and uses external research resources and datasets during development.



External datasets, research code, and reference materials remain subject to their respective licenses and redistribution conditions. Users should obtain those resources from their official sources and follow the applicable terms.



\## Status



This is an initial research release. Experimental results and implementation details are subject to further validation and refinement.



