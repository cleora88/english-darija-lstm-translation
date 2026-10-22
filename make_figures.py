"""Scientific figures and qualitative cases from the measured experiment."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sacrebleu.metrics import CHRF

from data import ROOT, save_json

INK, TEAL, GOLD = "#18364B", "#007F7A", "#B26D16"


def make_data_figures():
    directory = ROOT / "results"
    training = pd.read_csv(directory / "train_pairs.csv")
    summary = json.loads((directory / "data_summary.json").read_text())
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].bar(
        ["Missing English", "Complete pairs"],
        [summary["missing_english"], summary["complete_pairs_before_deduplication"]],
        color=[GOLD, TEAL],
        width=0.6,
    )
    for index, value in enumerate(
        [summary["missing_english"], summary["complete_pairs_before_deduplication"]]
    ):
        axes[0].text(index, value + 1200, f"{value:,}", ha="center", fontweight="bold")
    axes[0].set(title="Audit of the 87,785-row file", ylabel="Rows", ylim=(0, 85000))
    axes[1].hist(
        [training.source.str.split().str.len(), training.target.str.split().str.len()],
        bins=np.arange(0, 40, 2),
        color=[INK, TEAL],
        label=["English", "Latin Darija"],
    )
    axes[1].set(
        title="Training sentence lengths",
        xlabel="Whitespace-separated words",
        ylabel="Pairs",
    )
    axes[1].legend(frameon=False)
    fig.tight_layout(pad=1.5)
    fig.savefig(directory / "data_exploration.png", dpi=190)
    plt.close(fig)


def main():
    make_data_figures()
    directory = ROOT / "results"
    runs = json.loads((directory / "experiments.json").read_text())
    results = json.loads((directory / "final_results.json").read_text())
    selection = json.loads((directory / "selection.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for axis, kind in zip(axes, ("vanilla", "cifg", "wmc")):
        for run, color, style in zip(
            [r for r in runs if r["model"] == kind], (INK, TEAL, GOLD), ("-", "--", ":")
        ):
            axis.plot(
                [h["epoch"] for h in run["history"]],
                [h["validation_chrf"] for h in run["history"]],
                color=color,
                linestyle=style,
                linewidth=2,
                label=f"LR {run['learning_rate']} / batch {run['batch_size']}",
            )
        axis.set(title=kind.upper(), xlabel="Epoch")
        axis.legend(frameon=False, fontsize=7, loc="lower right")
        axis.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("Validation chrF (0-100)")
    fig.tight_layout(pad=1.5)
    fig.savefig(directory / "learning_curves.png", dpi=190)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for axis, metric, label in zip(axes, ("bleu", "chrf"), ("Test BLEU", "Test chrF")):
        values = [r[metric] for r in results]
        axis.bar(
            [r["model"].upper() for r in results],
            values,
            color=[INK, TEAL, GOLD],
            width=0.55,
        )
        for i, value in enumerate(values):
            axis.text(
                i,
                value + max(values) * 0.03,
                f"{value:.2f}",
                ha="center",
                fontweight="bold",
            )
        axis.set(title=label, ylim=(0, max(values) * 1.2), ylabel="Score (0-100)")
    fig.tight_layout(pad=1.5)
    fig.savefig(directory / "test_comparison.png", dpi=190)
    plt.close(fig)
    best_kind = max(selection, key=lambda kind: selection[kind]["best_validation_chrf"])
    predictions = pd.read_csv(
        directory / f"{best_kind}_predictions.csv", keep_default_na=False
    )
    metric = CHRF()
    predictions["sentence_chrf"] = [
        metric.sentence_score(h, [r]).score
        for h, r in zip(predictions.prediction, predictions.reference)
    ]
    predictions["row"] = range(len(predictions))
    ordered = predictions.sort_values(["sentence_chrf", "row"])
    examples = []
    for label, position in (("weak", 0.1), ("typical", 0.5), ("strong", 0.9)):
        row = ordered.iloc[int((len(ordered) - 1) * position)]
        examples.append(
            {
                "case": label,
                "source": row.source,
                "reference": row.reference,
                "prediction": row.prediction,
                "sentence_chrf": float(row.sentence_chrf),
            }
        )
    save_json(
        directory / "examples.json",
        {
            "model": best_kind,
            "selection": "10th, 50th and 90th percentile sentence chrF, ordered deterministically",
            "examples": examples,
        },
    )
    print("Figures and actual qualitative examples saved")


if __name__ == "__main__":
    main()
