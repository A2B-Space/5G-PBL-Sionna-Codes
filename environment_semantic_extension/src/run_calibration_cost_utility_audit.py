from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("results/analysis")
CONF_DIR = ROOT / "confidence_adaptive_experiment"
PRIOR_DIR = ROOT / "controlled_prior_experiment"
OUT = ROOT / "calibration_cost_utility_audit"

OUT.mkdir(parents=True, exist_ok=True)


def pareto_frontier(df):
    """
    Lower regret and fewer probes are better.
    A method is dominated if another method is no worse
    in both metrics and strictly better in at least one.
    """
    rows = []

    for _, candidate in df.iterrows():
        dominated = False

        for _, other in df.iterrows():
            if other["method"] == candidate["method"]:
                continue

            no_worse = (
                other["mean_probes"] <= candidate["mean_probes"]
                and other["mean_regret_db"] <= candidate["mean_regret_db"]
            )

            strictly_better = (
                other["mean_probes"] < candidate["mean_probes"]
                or other["mean_regret_db"] < candidate["mean_regret_db"]
            )

            if no_worse and strictly_better:
                dominated = True
                break

        if not dominated:
            rows.append(candidate)

    if not rows:
        return pd.DataFrame(columns=df.columns)

    return pd.DataFrame(rows).sort_values("mean_probes")


def main():
    # ---------------------------------------------------------
    # 1. Read prior prediction and confidence experiment
    # ---------------------------------------------------------
    confidence = pd.read_csv(
        CONF_DIR / "per_snapshot_results.csv"
    )

    confidence = confidence[
        (confidence["snapshot"] > 0)
        & confidence["method"].str.startswith(
            "confidence_adaptive_"
        )
    ].copy()

    confidence["offset_deg"] = (
        confidence["method"]
        .str.extract(r"confidence_adaptive_(\d+)deg")[0]
        .astype(int)
    )

    reliability = (
        confidence.groupby(
            ["condition", "confidence", "offset_deg"]
        )
        .agg(
            snapshots=("snapshot", "count"),
            mean_actual_prior_error_deg=(
                "prior_error_to_oracle_deg", "mean"
            ),
            median_actual_prior_error_deg=(
                "prior_error_to_oracle_deg", "median"
            ),
            max_actual_prior_error_deg=(
                "prior_error_to_oracle_deg", "max"
            ),
            mean_regret_db=("regret_db", "mean"),
            mean_probes=("probes", "mean"),
            oracle_inclusion_rate=("oracle_in_candidates", "mean"),
        )
        .reset_index()
    )

    reliability.to_csv(
        OUT / "prior_reliability_by_confidence.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 2. Retrospective confidence-order audit
    # ---------------------------------------------------------
    order_rows = []

    for condition in ["blocked", "unblocked"]:
        part = reliability[
            reliability["condition"] == condition
        ].sort_values("offset_deg")

        errors = part["mean_actual_prior_error_deg"].to_numpy()

        if len(errors) >= 2 and np.std(errors) > 0:
            correlation = float(
                np.corrcoef(
                    part["offset_deg"].to_numpy(),
                    errors
                )[0, 1]
            )
        else:
            correlation = np.nan

        monotonic = bool(np.all(np.diff(errors) >= -1e-9))

        order_rows.append({
            "condition": condition,
            "requested_offsets": "0,15,30,45",
            "actual_mean_errors_deg": ", ".join(
                f"{x:.3f}" for x in errors
            ),
            "offset_error_correlation": correlation,
            "strictly_nondecreasing_error": monotonic,
        })

    order_audit = pd.DataFrame(order_rows)

    order_audit.to_csv(
        OUT / "confidence_order_audit.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 3. Read post-acquisition cost-utility data
    # ---------------------------------------------------------
    results = pd.read_csv(
        CONF_DIR / "per_snapshot_results.csv"
    )

    post = results[results["snapshot"] > 0].copy()

    cost_utility = (
        post.groupby(["condition", "method"])
        .agg(
            snapshots=("snapshot", "count"),
            mean_regret_db=("regret_db", "mean"),
            max_regret_db=("regret_db", "max"),
            mean_probes=("probes", "mean"),
            total_probes=("probes", "sum"),
            oracle_inclusion_rate=("oracle_in_candidates", "mean"),
        )
        .reset_index()
    )

    cost_utility["probe_reduction_pct"] = (
        1 - cost_utility["mean_probes"] / 52
    ) * 100

    cost_utility.to_csv(
        OUT / "post_acquisition_cost_utility.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 4. Pareto frontier by condition
    # ---------------------------------------------------------
    frontier_frames = []

    for condition in ["blocked", "unblocked"]:
        part = cost_utility[
            cost_utility["condition"] == condition
        ].copy()

        frontier = pareto_frontier(part)
        frontier_frames.append(frontier)

    frontier_all = pd.concat(
        frontier_frames,
        ignore_index=True,
    )

    frontier_all.to_csv(
        OUT / "pareto_frontier.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 5. Plot confidence versus actual prior error
    # ---------------------------------------------------------
    fig, ax = plt.subplots()

    for condition in ["blocked", "unblocked"]:
        part = reliability[
            reliability["condition"] == condition
        ].sort_values("offset_deg")

        ax.plot(
            part["offset_deg"],
            part["mean_actual_prior_error_deg"],
            marker="o",
            label=condition,
        )

    ax.set_xlabel("Injected azimuth perturbation (degrees)")
    ax.set_ylabel("Mean actual prior-to-oracle error (degrees)")
    ax.set_title("Retrospective prior reliability audit")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        OUT / "prior_reliability.png",
        dpi=200,
    )
    plt.close(fig)

    # ---------------------------------------------------------
    # 6. Plot regret-probe trade-off
    # ---------------------------------------------------------
    for condition in ["blocked", "unblocked"]:
        part = cost_utility[
            cost_utility["condition"] == condition
        ]

        frontier = frontier_all[
            frontier_all["condition"] == condition
        ]

        fig, ax = plt.subplots()

        ax.scatter(
            part["mean_probes"],
            part["mean_regret_db"],
        )

        for _, row in part.iterrows():
            ax.annotate(
                row["method"],
                (row["mean_probes"], row["mean_regret_db"]),
                fontsize=7,
            )

        if not frontier.empty:
            frontier = frontier.sort_values("mean_probes")
            ax.plot(
                frontier["mean_probes"],
                frontier["mean_regret_db"],
                marker="o",
            )

        ax.set_xlabel("Mean candidate measurements per snapshot")
        ax.set_ylabel("Mean beam-selection regret (dB)")
        ax.set_title(
            f"Post-acquisition cost-utility frontier: {condition}"
        )
        ax.grid(True)
        fig.tight_layout()

        fig.savefig(
            OUT / f"cost_utility_{condition}.png",
            dpi=200,
        )
        plt.close(fig)

    # ---------------------------------------------------------
    # 7. Write interpretation and limitations
    # ---------------------------------------------------------
    report = [
        "CONFIDENCE CALIBRATION AND COST-UTILITY AUDIT",
        "",
        "CONFIDENCE ORDER AUDIT",
        order_audit.to_string(index=False),
        "",
        "PARETO FRONTIER",
        frontier_all[
            [
                "condition",
                "method",
                "mean_probes",
                "mean_regret_db",
            ]
        ].to_string(index=False),
        "",
        "IMPORTANT LIMITATIONS",
        "- Actual prior-to-oracle error is retrospective.",
        "- Oracle information is used for evaluation only.",
        "- Injected offsets are not learned prediction uncertainty.",
        "- No semantic probability outputs are available.",
        "- Results cover one trajectory and 10 tracking snapshots",
        "  per obstruction condition.",
        "- Pareto dominance applies only to the methods tested.",
        "",
        "This analysis does not establish probabilistic calibration.",
        "A real calibration study requires held-out predictions",
        "with confidence scores and corresponding ground-truth outcomes.",
    ]

    (OUT / "audit_report.txt").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\nPRIOR RELIABILITY")
    print(reliability.round(4).to_string(index=False))

    print("\nCONFIDENCE ORDER AUDIT")
    print(order_audit.to_string(index=False))

    print("\nPOST-ACQUISITION COST-UTILITY")
    print(cost_utility.round(4).to_string(index=False))

    print("\nPARETO FRONTIER")
    print(
        frontier_all[
            [
                "condition",
                "method",
                "mean_probes",
                "mean_regret_db",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nSaved audit to:", OUT.resolve())


if __name__ == "__main__":
    main()
