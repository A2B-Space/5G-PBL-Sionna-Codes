# Reproducibility runbook

## 1. Environment

Activate the existing environment; do not reinstall Sionna/Mitsuba/Dr.Jit as
part of S8:

```bash
cd ~/sionna_project
source .venv/bin/activate
python --version
python -m pip freeze > environment_freeze_local.txt
nvidia-smi
```

Copy `s8_consolidate.py` from this package into `~/sionna_project/` before
running the consolidation command below.

S8 captures package metadata and `nvidia-smi` output where available. It does
not infer a package version from `sionna.__version__`.

## 2. Execution order

The completed dataset and S7 results should remain frozen. To regenerate the
current downstream outputs from the source pipeline, use the following order:

```bash
python s4_1_path_geometry_export_s6_fixed.py
python s5_1_adaptive_beam_tracking.py
python s6_build_dataset.py
python s7_extend_and_evaluate.py
python s8_consolidate.py --project-dir ~/sionna_project
```

S4.1 is a GPU/Sionna RT stage and regenerates the S4 geometry/path/channel
exports. S5.1, S6 and S7 operate on generated files. If S4.1 is rerun, rerun
all downstream stages so that the artifacts share consistent provenance.

S1-S3 are historical upstream stages. Their original source scripts should be
retained and executed only when intentionally reproducing those stages. The
S8 inventory reports whether those scripts are available; it does not invent
missing source code.

## 3. Validation and hashes

- Inspect `validation/s8_audit_report.txt`.
- Inspect `manifests/stage_registry.json` for missing sources and validation
  states.
- Verify files against `manifests/file_checksums.sha256` with:

```bash
sha256sum -c manifests/file_checksums.sha256
```

A checksum match verifies file identity relative to this package; it does not
prove the simulator will regenerate bit-identical output on different hardware.

## 4. Freeze policy

Treat `results/` as immutable evidence. New experiments should use a new
trajectory/scenario identifier and a new dataset version. Do not overwrite
S6/S7 outputs after analysis without incrementing the dataset/schema version
and regenerating downstream results.
