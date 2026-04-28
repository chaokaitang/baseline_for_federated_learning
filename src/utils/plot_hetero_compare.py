import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


DIRICHLET_RUN = "dirichlet_stp_beta0.5"
SHARD_RUN = "hetero_emnist_balanced_0_shard_continual_t3_spt5_niid.pkl"
OUTPUT_NAME = "results_hetero_compare.png"


def pick_summary(result_dir: Path) -> Path:
    files = sorted(result_dir.glob("*sequential_summary*.json"))
    if not files:
        raise FileNotFoundError(f"No sequential summary json found in: {result_dir}")
    return files[-1]


def valid(x) -> bool:
    return isinstance(x, (int, float)) and not math.isnan(x)


def avg_last(mat):
    if not mat:
        raise ValueError("Empty accuracy matrix.")
    vals = [x for x in mat[-1] if valid(x)]
    if not vals:
        raise ValueError("No valid values in the last row of accuracy matrix.")
    return sum(vals) / len(vals)


def load_metrics(result_dir: Path):
    summary = pick_summary(result_dir)
    data = json.loads(summary.read_text(encoding="utf-8"))

    required = [
        "eval_acc_matrix_global",
        "eval_acc_matrix_personalized",
        "eval_acc_matrix_collab_beta",
        "mean_forgetting_global",
        "mean_forgetting_personalized",
        "mean_forgetting_collab_beta",
    ]
    for key in required:
        if key not in data:
            raise KeyError(f"Missing key `{key}` in summary: {summary}")

    return {
        "summary": summary,
        "global_acc": avg_last(data["eval_acc_matrix_global"]),
        "personal_acc": avg_last(data["eval_acc_matrix_personalized"]),
        "collab_acc": avg_last(data["eval_acc_matrix_collab_beta"]),
        "global_forget": data["mean_forgetting_global"],
        "personal_forget": data["mean_forgetting_personalized"],
        "collab_forget": data["mean_forgetting_collab_beta"],
    }


def main():
    root = Path(__file__).resolve().parents[2]
    result_root = root / "result"
    fig_root = root / "processing_result"
    fig_root.mkdir(parents=True, exist_ok=True)

    d_dir = result_root / DIRICHLET_RUN
    s_dir = result_root / SHARD_RUN
    if not d_dir.exists():
        raise FileNotFoundError(f"Dirichlet result directory not found: {d_dir}")
    if not s_dir.exists():
        raise FileNotFoundError(f"Shard result directory not found: {s_dir}")

    d_metrics = load_metrics(d_dir)
    s_metrics = load_metrics(s_dir)

    names = ["Global", "Personal", "Collaborative"]
    d_acc = [d_metrics["global_acc"], d_metrics["personal_acc"], d_metrics["collab_acc"]]
    s_acc = [s_metrics["global_acc"], s_metrics["personal_acc"], s_metrics["collab_acc"]]
    d_forget = [d_metrics["global_forget"], d_metrics["personal_forget"], d_metrics["collab_forget"]]
    s_forget = [s_metrics["global_forget"], s_metrics["personal_forget"], s_metrics["collab_forget"]]

    x = range(len(names))
    width = 0.36

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9))

    axes[0].bar([i - width / 2 for i in x], d_acc, width=width, label="Dirichlet", color="#4e79a7")
    axes[0].bar([i + width / 2 for i in x], s_acc, width=width, label="Shard", color="#f28e2b")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("AvgAcc")
    axes[0].set_title("Heterogeneity comparison: AvgAcc")
    axes[0].grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)

    axes[1].bar([i - width / 2 for i in x], d_forget, width=width, label="Dirichlet", color="#4e79a7")
    axes[1].bar([i + width / 2 for i in x], s_forget, width=width, label="Shard", color="#f28e2b")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("AvgForget")
    axes[1].set_title("Heterogeneity comparison: AvgForget")
    axes[1].grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)

    # Ensure we only use a shared figure legend.
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.948),
        bbox_transform=fig.transFigure,
    )

    fig.suptitle("Dirichlet vs Shard under STP-FedCL", fontsize=14, y=0.988)
    fig.tight_layout(rect=[0, 0, 1, 0.86])

    out_file = fig_root / OUTPUT_NAME
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

    print(f"Dirichlet summary: {d_metrics['summary']}")
    print(f"Shard summary: {s_metrics['summary']}")
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
