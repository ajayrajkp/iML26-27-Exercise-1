"""Create the figures used to present the results."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def create_plots(results, output_folder):
    """Create the accuracy plot and index distribution plot."""
    output_folder.mkdir(parents=True, exist_ok=True)
    colors = {
        "Decision Tree": "#4778a8",
        "Random Forest": "#59a14f",
        "Gradient Boosting": "#e28e2c",
    }

    dataset_names = sorted(results["dataset"].unique())
    figure, axes = plt.subplots(1, len(dataset_names), figsize=(5 * len(dataset_names), 4.8), sharex=True, sharey=True)

    for axis, dataset in zip(axes, dataset_names):
        dataset_results = results[results["dataset"] == dataset]
        for family, family_results in dataset_results.groupby("model_family"):
            axis.scatter(
                family_results["simulatability_index"],
                family_results["accuracy"],
                s=24,
                alpha=0.55,
                label=family,
                color=colors[family],
            )
        axis.axvline(50.0, color="#b44", linestyle="--", linewidth=2)
        axis.set_title(dataset.replace("_", " ").title())
        axis.set_xlabel("Simulatability Index")
        axis.grid(alpha=0.2)

    axes[0].set_ylabel("Test accuracy")
    axes[-1].legend(fontsize=8)
    figure.suptitle("Accuracy and Simulatability Index under the moderate limits")
    figure.tight_layout()
    figure.savefig(output_folder / "accuracy_vs_simulatability.png", dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    failed = results[~results["passes_threshold"]]
    passed = results[results["passes_threshold"]]
    bins = np.arange(0, 102, 2)
    axis.hist(failed["simulatability_index"], bins=bins, alpha=0.58, label="Below 50")
    axis.hist(passed["simulatability_index"], bins=bins, alpha=0.58, label="50 or higher")
    axis.axvline(50.0, color="#b44", linestyle="--", linewidth=2, label="Threshold = 50")
    axis.set_xlabel("Simulatability Index")
    axis.set_ylabel("Model configurations")
    axis.set_title("Distribution of the Simulatability Index")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_folder / "index_distribution.png", dpi=170)
    plt.close(figure)

