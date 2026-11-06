#!/usr/bin/env python3
"""S8: Consolidate S1-S7 artifacts and create a reproducibility package.

Standard-library only. Run from the Sionna project virtual environment:
    python s8_consolidate.py --project-dir ~/sionna_project

The script copies available artifacts; it does not rerun simulations or modify
S1-S7 source data. Missing source scripts are reported, never fabricated.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

SCRIPT_MAP = {
    "S1": ["s1*.py"],
    "S2": ["s2*.py"],
    "S3": ["s3_beam_selection.py"],
    "S4": ["s4_blockage.py", "s4_1_path_geometry_export.py",
           "s4_1_path_geometry_export_s6.py", "s4_1_path_geometry_export_s6_fixed.py"],
    "S5": ["s5_beam_tracking.py"],
    "S5_1": ["s5_1_adaptive_beam_tracking.py"],
    "S6": ["s6_build_dataset.py"],
    "S7": ["s7_extend_and_evaluate.py"],
    "S8": ["s8_consolidate.py"],
}

FILE_MAP = {
    "S1": ["s1_ray_tracing_results.csv"],
    "S2": ["s2_mobility_summary.csv", "s2_mobility_paths.csv"],
    "S3": ["s3_beam_codebook.csv", "s3_beam_selection_all_beams.csv",
           "s3_beam_selection_summary.csv"],
    "S4": ["s4_blockage_geometry", "s4_*.csv", "s4_*.npz", "s4_*.json", "s4_validation.txt"],
    "S5": ["s5_beam_tracking", "s5_tracking_*.csv", "s5_validation.txt"],
    "S5_1": ["s5_1_adaptive_tracking"],
    "S6": ["s6_physical_layer_dataset", "s6_physical_layer_dataset_extended"],
    "S7": ["s7_performance_evaluation"],
}

VALIDATION_CANDIDATES = {
    "S4": ["s4_blockage_geometry/s4_validation.txt", "s4_validation.txt"],
    "S5": ["s5_beam_tracking/s5_validation.txt", "s5_validation.txt"],
    "S5_1": ["s5_1_adaptive_tracking/s5_1_validation.txt"],
    "S6": ["s6_physical_layer_dataset_extended/validation.txt",
           "s6_physical_layer_dataset/validation.txt"],
    "S7": ["s7_performance_evaluation/s7_validation.txt"],
}

DOCS = {
"README.md": r'''# Sionna RT S1-S8 reproducibility package

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
''',
"docs/REPRODUCIBILITY_RUNBOOK.md": r'''# Reproducibility runbook

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
''',
"docs/METHODOLOGY_AND_ASSUMPTIONS.md": r'''# Methodology, equations and assumptions

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
''',
"docs/SEMANTIC_INTEGRATION_CONTRACT.md": r'''# Proposed semantic-to-physical-layer integration contract

This document defines a future interface. The current S6/S7 data does NOT
contain camera images, masks, bounding boxes, semantic classes or synchronized
cross-modal labels.

## Existing physical-layer keys

The physical-layer tables carry `scene_id`, `trajectory_id`, `condition_id`,
`snapshot`, and `sample_id`. `snapshot_metadata.csv` also contains receiver XYZ
coordinates and the LOS/blockage/channel/beam descriptors. Use `sample_id` as
the primary join key after verifying it is unique in each table at its declared
grain.

## Required semantic-side fields before joining

- `scene_id`
- `trajectory_id`
- `snapshot` or an explicit timestamp-to-snapshot mapping
- `camera_timestamp` with declared clock/time base
- camera pose and coordinate frame
- camera calibration and field of view
- object/instance IDs and semantic labels
- object geometry or 3D location if a physical object-to-ray interaction link
  is intended

## Join rules

1. Establish a common coordinate frame and document every transform.
2. Establish synchronization tolerance and timestamp provenance; do not join
   by row order alone.
3. Preserve one-to-many relationships: one snapshot can have multiple paths,
   multiple interactions, 52 beam measurements and multiple semantic objects.
4. Keep `sample_id` as the sample-level key and use `(sample_id, path_id)` or
   `(sample_id, beam_id)` for child tables.
5. Do not infer that a ray interaction object ID is a camera object ID. A
   geometry/object correspondence table must be explicitly constructed and
   validated.
6. Keep train/validation/test splits grouped by independent trajectory or
   scene, not by adjacent snapshots from the same trajectory.

## Suggested future correspondence table

`sample_id, semantic_object_id, scene_object_id, correspondence_method,
confidence, coordinate_transform_id, timestamp_delta_ms`

This table is a proposed integration artifact, not an existing S6 output.
''',
"docs/REFERENCES.md": r'''# Official technical references

- Sionna 2.2 installation and environment guidance: https://nvlabs.github.io/sionna/installation.html
- Sionna RT 2.2 PathSolver API, including deterministic mode and seed semantics: https://nvlabs.github.io/sionna/rt/api/paths_solvers.html
- Sionna RT 2.2 Understanding the Paths object, including synthetic-array behavior and path metadata: https://nvlabs.github.io/sionna/v2.2.0/rt/developer/dev_understanding_paths.html
- Sionna RT 2.2 overview: https://nvlabs.github.io/sionna/rt/index.html

These references support the simulator/API descriptions. They do not independently validate this project's numerical results; those are validated against the uploaded experiment artifacts and local validation reports.
''',
"docs/SOURCE_GAPS.md": r'''# Source gaps and provenance notes

The files available to this consolidation step include the S1 result CSV and
S2 result CSVs, but the original S1 and S2 Python source scripts were not
present in the supplied artifact set used to prepare this package. The S8
script searches the user's project directory for `s1*.py` and `s2*.py` and
will include them if present. If absent, they remain explicitly marked as
missing source, while their supplied result artifacts are retained.

The package does not claim that result CSVs can replace the original scripts.
Likewise, a successful S4-S7 validation report is evidence for those stored
outputs, not proof that every historical stage can be regenerated from source.
'''
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_item(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".venv", "__pycache__", "*.pyc"))
    else:
        shutil.copy2(src, dst)


def matches(root: Path, pattern: str) -> list[Path]:
    # Patterns are intentionally one level deep; directory names are explicit.
    if any(ch in pattern for ch in "*?["):
        return sorted(p for p in root.glob(pattern) if p.is_file())
    p = root / pattern
    return [p] if p.exists() else []


def capture_environment() -> dict[str, Any]:
    package_names = ["sionna-rt", "sionna", "mitsuba", "drjit", "numpy", "pandas", "matplotlib", "scipy", "torch"]
    packages = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    pip_freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=60)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=15) if shutil.which("nvidia-smi") else None
    proc_version = Path("/proc/version").read_text(errors="replace").strip() if Path("/proc/version").exists() else None
    return {
        "captured_unix_time": int(time.time()),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "kernel_release": platform.release(),
        "proc_version": proc_version,
        "package_versions": packages,
        "nvidia_smi_available": gpu is not None,
        "nvidia_smi_stdout": gpu.stdout.strip() if gpu else None,
        "nvidia_smi_stderr": gpu.stderr.strip() if gpu else None,
        "pip_freeze": pip_freeze.stdout.splitlines() if pip_freeze.returncode == 0 else [],
        "pip_freeze_returncode": pip_freeze.returncode,
        "note": "Package versions are read from installed distribution metadata; no sionna.__version__ access is used."
    }


def read_validation(path: Path) -> str:
    text = path.read_text(errors="replace")
    if "OVERALL: FAIL" in text or "OVERALL EXTENSION: FAIL" in text:
        return "FAIL"
    if "OVERALL: PASS" in text or "OVERALL EXTENSION: PASS" in text:
        return "PASS"
    return "UNPARSED"


def main() -> int:
    parser = argparse.ArgumentParser(description="Consolidate S1-S7 Sionna artifacts into an S8 reproducibility package.")
    parser.add_argument("--project-dir", type=Path, default=Path.home() / "sionna_project")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    project = args.project_dir.expanduser().resolve()
    if not project.is_dir():
        raise SystemExit(f"Project directory does not exist: {project}")
    out = (args.output_dir.expanduser().resolve() if args.output_dir else project / "s8_reproducibility_package")
    if out == project or project not in out.parents:
        raise SystemExit("Output directory must be inside the project directory and cannot be the project root.")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for rel, content in DOCS.items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content.strip() + "\n", encoding="utf-8")

    registry: dict[str, Any] = {"project_dir": str(project), "package_dir": str(out), "stages": {}, "notes": []}
    copied_relpaths: set[str] = set()

    # Source scripts
    for stage, patterns in SCRIPT_MAP.items():
        found = []
        for pattern in patterns:
            for src in matches(project, pattern):
                # Never ingest files from the package being generated.
                if out in src.parents:
                    continue
                rel = src.relative_to(project)
                dst = out / "source" / stage / rel.name
                copy_item(src, dst)
                found.append(str(rel))
                copied_relpaths.add(str(dst.relative_to(out)))
        registry["stages"][stage] = {"source_scripts_found": found, "source_script_count": len(found)}

    # Results and configurations
    for stage, patterns in FILE_MAP.items():
        found = []
        for pattern in patterns:
            for src in matches(project, pattern):
                if out in src.parents:
                    continue
                rel = src.relative_to(project)
                dst = out / "results" / stage / rel.name
                # Avoid duplicate copying when overlapping patterns match.
                if str(dst.relative_to(out)) not in copied_relpaths:
                    copy_item(src, dst)
                    copied_relpaths.add(str(dst.relative_to(out)))
                found.append(str(rel))
        registry["stages"].setdefault(stage, {})["result_artifacts_found"] = sorted(set(found))

    # Validation status from the live project tree
    validation_failures = []
    for stage, candidates in VALIDATION_CANDIDATES.items():
        selected = next((project / c for c in candidates if (project / c).is_file()), None)
        if selected is None:
            status = "NOT_FOUND"
            registry["stages"].setdefault(stage, {})["validation_report"] = None
            registry["stages"].setdefault(stage, {})["validation_status"] = status
        else:
            status = read_validation(selected)
            registry["stages"].setdefault(stage, {})["validation_report"] = str(selected.relative_to(project))
            registry["stages"].setdefault(stage, {})["validation_status"] = status
            if status == "FAIL":
                validation_failures.append(stage)

    for stage in ("S1", "S2"):
        if not registry["stages"].get(stage, {}).get("source_scripts_found"):
            registry["notes"].append(f"{stage} original Python source not found; available outputs, if any, are retained. Add original source if available.")

    # Add environment snapshot and registry
    env_dir = out / "environment"
    env_dir.mkdir(exist_ok=True)
    env = capture_environment()
    (env_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (env_dir / "requirements-freeze.txt").write_text("\n".join(env["pip_freeze"]) + "\n", encoding="utf-8")

    registry["validation_failures"] = validation_failures
    registry["status"] = "FAIL" if validation_failures else ("PASS_WITH_SOURCE_GAPS" if registry["notes"] else "PASS")
    (out / "manifests").mkdir(exist_ok=True)
    (out / "manifests/stage_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")

    # File inventory and SHA256 manifest. Exclude checksum file itself.
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "file_checksums.sha256")
    manifest_rows = []
    for p in files:
        manifest_rows.append({"path": str(p.relative_to(out)), "size_bytes": p.stat().st_size, "sha256": sha256(p)})
    package_manifest = {
        "package_name": "Sionna_RT_S1-S8_Reproducibility_Package",
        "package_schema_version": "1.0.0",
        "created_unix_time": int(time.time()),
        "source_project": str(project),
        "stage_registry_status": registry["status"],
        "file_count_before_manifest": len(manifest_rows),
        "files": manifest_rows,
        "reproducibility_statement": "Hashes establish artifact identity. Bitwise regeneration across hardware/software stacks is not asserted.",
    }
    (out / "manifests/reproducibility_manifest.json").write_text(json.dumps(package_manifest, indent=2), encoding="utf-8")

    # Refresh checksum list to include registry and reproducibility manifest.
    checksum_files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "file_checksums.sha256")
    checksum_text = "".join(f"{sha256(p)}  {p.relative_to(out)}\n" for p in checksum_files)
    (out / "manifests/file_checksums.sha256").write_text(checksum_text, encoding="utf-8")

    # Human-readable audit
    report_lines = [
        "S8 CONSOLIDATION AUDIT", "=" * 76,
        f"Project: {project}", f"Package: {out}", f"Registry status: {registry['status']}", "",
        "Stage inventory:",
    ]
    for stage in ["S1", "S2", "S3", "S4", "S5", "S5_1", "S6", "S7", "S8"]:
        item = registry["stages"].get(stage, {})
        report_lines.append(f"- {stage}: {item.get('source_script_count', 0)} source script(s); {len(item.get('result_artifacts_found', []))} result item(s); validation={item.get('validation_status', 'N/A')}")
    report_lines += ["", "Notes:"] + [f"- {n}" for n in registry["notes"]]
    report_lines += ["", "Validation failures:"] + ([f"- {s}" for s in validation_failures] or ["- None detected in reports found."])
    report_lines += ["", "Interpretation:",
        "- This audit checks artifact availability and parses existing validation summaries.",
        "- It does not rerun Sionna RT or independently reproduce the physical simulation.",
        "- A missing historical source script is a provenance gap, not a failed result-data check.",
        "- One scene/one trajectory does not support generalization claims.",
    ]
    validation_dir = out / "validation"
    validation_dir.mkdir(exist_ok=True)
    (validation_dir / "s8_audit_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Add audit report to checksum list (and update manifest file list too).
    checksum_files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "file_checksums.sha256")
    checksum_text = "".join(f"{sha256(p)}  {p.relative_to(out)}\n" for p in checksum_files)
    (out / "manifests/file_checksums.sha256").write_text(checksum_text, encoding="utf-8")

    zip_path = project / "S8_reproducibility_package.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(Path(out.name) / p.relative_to(out)))

    print("=" * 76)
    print("S8 CONSOLIDATION COMPLETE")
    print("=" * 76)
    print("Registry status:", registry["status"])
    print("Package directory:", out)
    print("ZIP archive:", zip_path)
    print("Files packaged:", sum(1 for p in out.rglob("*") if p.is_file()))
    print("\nAudit report:", validation_dir / "s8_audit_report.txt")
    print("Checksum verification command:")
    print("  cd", out, "&& sha256sum -c manifests/file_checksums.sha256")
    if validation_failures:
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
