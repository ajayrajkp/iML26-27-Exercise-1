"""Create the figures used to present the results."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_plots(results, output_folder):
    """Create the accuracy and Simulatability Index plot."""
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

