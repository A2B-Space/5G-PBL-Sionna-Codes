from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("results/analysis/controlled_prior_experiment")
OUT = ROOT / "post_acquisition"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "per_snapshot_results.csv")

# Exclude snapshot 0: it uses exhaustive initial acquisition.
post = df[df["snapshot"] > 0].copy()

# Summarize every method over the 10 tracking snapshots.
summary = (
    post.groupby(["condition", "method"])
    .agg(
        snapshots=("snapshot", "count"),
        mean_regret_db=("regret_db", "mean"),
        median_regret_db=("regret_db", "median"),
        max_regret_db=("regret_db", "max"),
        mean_probes=("probes", "mean"),
        total_probes=("probes", "sum"),
        oracle_inclusion_rate=("oracle_in_candidates", "mean"),
        mean_prior_error_deg=(
            "prior_angular_error_to_oracle_deg", "mean"
        ),
        mean_perturbation_deg=(
            "prior_perturbation_from_base_deg", "mean"
        ),
    )
    .reset_index()
)

summary.to_csv(OUT / "post_acquisition_summary.csv", index=False)

prior = summary[
    summary["method"].str.startswith("prior_az_")
].copy()

prior.to_csv(OUT / "post_acquisition_prior_sensitivity.csv", index=False)

print("\nPOST-ACQUISITION METHOD SUMMARY")
print(summary.round(4).to_string(index=False))

print("\nPOST-ACQUISITION PRIOR SENSITIVITY")
print(prior.round(4).to_string(index=False))


# Plot 1: Mean regret against requested azimuth perturbation.
fig, ax = plt.subplots()

for condition in ["blocked", "unblocked"]:
    part = prior[prior["condition"] == condition].copy()
    part["offset_deg"] = (
        part["method"].str.extract(r"prior_az_(\d+)deg")[0].astype(int)
    )
    part = part.sort_values("offset_deg")

    ax.plot(
        part["offset_deg"],
        part["mean_regret_db"],
        marker="o",
        label=condition,
    )

ax.set_xlabel("Requested prior azimuth perturbation (degrees)")
ax.set_ylabel("Mean beam-selection regret (dB)")
ax.set_title("Post-acquisition regret versus prior perturbation")
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "post_acquisition_regret.png", dpi=200)
plt.close(fig)


# Plot 2: Oracle inclusion rate.
fig, ax = plt.subplots()

for condition in ["blocked", "unblocked"]:
    part = prior[prior["condition"] == condition].copy()
    part["offset_deg"] = (
        part["method"].str.extract(r"prior_az_(\d+)deg")[0].astype(int)
    )
    part = part.sort_values("offset_deg")

    ax.plot(
        part["offset_deg"],
        part["oracle_inclusion_rate"] * 100,
        marker="o",
        label=condition,
    )

ax.set_xlabel("Requested prior azimuth perturbation (degrees)")
ax.set_ylabel("Oracle beam inclusion (%)")
ax.set_title("Post-acquisition oracle candidate inclusion")
ax.set_ylim(0, 105)
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "post_acquisition_oracle_inclusion.png", dpi=200)
plt.close(fig)


# Plot 3: Actual prior error against oracle.
fig, ax = plt.subplots()

for condition in ["blocked", "unblocked"]:
    part = prior[prior["condition"] == condition].copy()
    part["offset_deg"] = (
        part["method"].str.extract(r"prior_az_(\d+)deg")[0].astype(int)
    )
    part = part.sort_values("offset_deg")

    ax.plot(
        part["offset_deg"],
        part["mean_prior_error_deg"],
        marker="o",
        label=condition,
    )

ax.set_xlabel("Requested prior azimuth perturbation (degrees)")
ax.set_ylabel("Mean prior-to-oracle angular error (degrees)")
ax.set_title("Actual prior error after initial acquisition")
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "post_acquisition_actual_prior_error.png", dpi=200)
plt.close(fig)


print("\nSaved corrected analysis to:", OUT.resolve())
print("Generated three PNG plots.")
