import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT_DIR / "result"
OUTPUT_DIR = ROOT_DIR / "processing_result"
TASK_NAME = "task3"

BASELINE_SPECS = [
    {
        "label": "FedAvg (global)",
        "short_label": "FedAvg\n(global)",
        "run_dir": "fedavg_baseline_dirichlet_2nn_r40_c50_lr0.01_sd0",
        "curve_key": "acc_on_eval_data",
        "client_file": "client_acc.json",
        "client_key": "client_acc",
        "color": "#1f77b4",
    },
    {
        "label": "FedProx (global)",
        "short_label": "FedProx\n(global)",
        "run_dir": "fedprox_baseline_dirichlet_2nn_r40_c50_lr0.01_sd0",
        "curve_key": "acc_on_eval_data",
        "client_file": "client_acc.json",
        "client_key": "client_acc",
        "color": "#ff7f0e",
    },
    {
        "label": "Ditto (personalized)",
        "short_label": "Ditto\n(personalized)",
        "run_dir": "ditto_baseline_dirichlet_2nn_r40_c50_lr0.01_sd0",
        "curve_key": "personalized_mean_acc",
        "client_file": "client_acc_personal.json",
        "client_key": "client_acc_personal",
        "color": "#2ca02c",
    },
    {
        "label": "FedAvg+EWC (global)",
        "short_label": "FedAvg+EWC\n(global)",
        "run_dir": "fedavg_ewc_baseline_dirichlet_2nn_r40_c50_lr0.01_sd0",
        "curve_key": "acc_on_eval_data",
        "client_file": "client_acc.json",
        "client_key": "client_acc",
        "color": "#d62728",
    },
    {
        "label": "STP-FedCL (collaborative)",
        "short_label": "STP-FedCL\n(collaborative)",
        "run_dir": "dirichlet_stp_beta0.5",
        "curve_key": "collab_mean_acc",
        "client_file": "client_acc_collab.json",
        "client_key": "client_acc_collab",
        "color": "#9467bd",
    },
]

LEARNING_CURVE_NAME = "task3_baseline_learning_curves.png"
DISTRIBUTION_NAME = "task3_client_accuracy_distribution.png"
BETA_NAME = "dirichlet_beta_sensitivity.png"
SUMMARY_PATTERN = "*_summary_*.json"
BETA_DIR_PATTERN = re.compile(r"^dirichlet_stp_beta(?P<beta>\d+(?:\.\d+)?)$")


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _task_dir(run_dir: str) -> Path:
    path = RESULT_DIR / run_dir / TASK_NAME
    if not path.exists():
        raise FileNotFoundError(f"Required task directory not found: {path}")
    return path


def _extract_curve(metrics_path: Path, curve_key: str) -> np.ndarray:
    metrics = _load_json(metrics_path)
    if curve_key not in metrics:
        raise KeyError(f"Curve key `{curve_key}` not found in {metrics_path}")
    curve = np.asarray(metrics[curve_key], dtype=float)
    if curve.ndim != 1 or curve.size == 0:
        raise ValueError(f"Curve `{curve_key}` in {metrics_path} is empty or malformed.")
    return curve


def _repair_terminal_zero(curve: np.ndarray, label: str) -> np.ndarray:
    fixed = curve.copy()
    if fixed.size >= 2 and fixed[-1] == 0.0 and fixed[-2] > 0.0:
        warnings.warn(
            f"{label} has a terminal zero in metrics; reusing round {fixed.size - 2} for the final plotting point.",
            stacklevel=2,
        )
        fixed[-1] = fixed[-2]
    return fixed


def _extract_client_acc(client_path: Path, client_key: str) -> np.ndarray:
    payload = _load_json(client_path)
    if client_key not in payload:
        raise KeyError(f"Client key `{client_key}` not found in {client_path}")
    client_acc = payload[client_key]
    if isinstance(client_acc, dict):
        ordered = [client_acc[k] for k in sorted(client_acc, key=lambda x: int(x))]
    else:
        ordered = client_acc
    values = np.asarray(ordered, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"Client accuracy `{client_key}` in {client_path} is empty or malformed.")
    return values


def _style_axes(ax):
    ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_alpha(0.3)
    ax.spines["right"].set_alpha(0.3)


def plot_learning_curves():
    fig, ax = plt.subplots(figsize=(11.0, 6.8))

    for spec in BASELINE_SPECS:
        metrics_path = _task_dir(spec["run_dir"]) / "metrics.json"
        curve = _extract_curve(metrics_path, spec["curve_key"])
        if spec["run_dir"] == "ditto_baseline_dirichlet_2nn_r40_c50_lr0.01_sd0":
            curve = _repair_terminal_zero(curve, spec["label"])
        rounds = np.arange(curve.size)
        ax.plot(
            rounds,
            curve,
            label=spec["label"],
            color=spec["color"],
            linewidth=2.4,
        )
        if curve.size != 41:
            warnings.warn(
                f"{spec['label']} has {curve.size} points instead of the expected 41.",
                stacklevel=2,
            )

    ax.set_title("Task-3 Accuracy Curves Under Representative Evaluation Modes", fontsize=18, pad=12)
    ax.set_xlabel("Communication rounds", fontsize=13)
    ax.set_ylabel("Test accuracy", fontsize=13)
    ax.set_xlim(0, 40)
    ax.tick_params(labelsize=11)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        fontsize=11,
        handlelength=3.0,
        columnspacing=1.4,
    )
    _style_axes(ax)
    fig.subplots_adjust(bottom=0.24, top=0.90)
    output_path = OUTPUT_DIR / LEARNING_CURVE_NAME
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_client_distribution():
    labels = []
    color_map = []
    client_values = []

    for spec in BASELINE_SPECS:
        client_path = _task_dir(spec["run_dir"]) / spec["client_file"]
        values = _extract_client_acc(client_path, spec["client_key"])
        labels.append(spec["label"])
        color_map.append(spec["color"])
        client_values.append(values)

    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    positions = np.arange(1, len(labels) + 1)

    violin = ax.violinplot(
        client_values,
        positions=positions,
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(violin["bodies"], color_map):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.6)

    box = ax.boxplot(
        client_values,
        positions=positions,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
    )
    for patch in box["boxes"]:
        patch.set(facecolor="white", edgecolor="#666666", linewidth=1.0, alpha=0.75)
    for key in ("whiskers", "caps", "medians"):
        for artist in box[key]:
            artist.set(color="#666666", linewidth=1.0)

    rng = np.random.default_rng(0)
    for idx, values in enumerate(client_values, start=1):
        x = rng.normal(loc=idx, scale=0.045, size=values.size)
        ax.scatter(x, values, s=12, color="#444444", alpha=0.35, marker=".")
        mean = float(np.mean(values)) * 100.0
        std = float(np.std(values)) * 100.0
        ax.text(
            idx,
            0.992,
            f"Mean={mean:.2f}\nStd={std:.2f}",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            linespacing=1.4,
            bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 2.5},
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([spec["short_label"] for spec in BASELINE_SPECS], rotation=0, ha="center")
    ax.set_ylabel("Client accuracy", fontsize=13)
    ax.set_title("Task-3 Client Accuracy Distribution Across Baselines", fontsize=18, pad=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_ylim(0.43, 1.01)
    _style_axes(ax)
    fig.subplots_adjust(bottom=0.20, top=0.90)
    output_path = OUTPUT_DIR / DISTRIBUTION_NAME
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _summary_file(run_root: Path) -> Path:
    candidates = sorted(run_root.glob(SUMMARY_PATTERN))
    if not candidates:
        raise FileNotFoundError(f"No summary json found under {run_root}")
    return candidates[-1]


def _compute_avg_acc_and_forget(acc_matrix: np.ndarray) -> tuple[float, float]:
    if acc_matrix.ndim != 2 or acc_matrix.shape[0] != acc_matrix.shape[1]:
        raise ValueError("Expected a square task accuracy matrix.")

    final_stage = acc_matrix.shape[0] - 1
    seen_acc = acc_matrix[final_stage, : final_stage + 1]
    if np.isnan(seen_acc).any():
        raise ValueError("Final-stage task accuracies contain NaN values.")
    avg_acc = float(np.mean(seen_acc))

    forgetting = []
    for task_idx in range(final_stage):
        history = acc_matrix[task_idx:final_stage, task_idx]
        final_acc = acc_matrix[final_stage, task_idx]
        if np.isnan(history).any() or np.isnan(final_acc):
            raise ValueError(f"Task {task_idx + 1} forgetting cannot be computed due to NaN values.")
        forgetting.append(float(np.max(history) - final_acc))

    avg_forget = float(np.mean(forgetting)) if forgetting else 0.0
    return avg_acc, avg_forget


def _collect_beta_points():
    points = []
    for run_root in sorted(RESULT_DIR.iterdir()):
        if not run_root.is_dir():
            continue
        match = BETA_DIR_PATTERN.match(run_root.name)
        if not match:
            continue

        beta = float(match.group("beta"))
        metrics_path = run_root / TASK_NAME / "metrics.json"
        try:
            summary_path = _summary_file(run_root)
            metrics = _load_json(metrics_path)
            summary = _load_json(summary_path)
            curve = np.asarray(metrics["collab_mean_acc"], dtype=float)
            if curve.size == 0:
                raise ValueError(f"Empty `collab_mean_acc` in {metrics_path}")
            if "eval_acc_matrix_collab_beta" not in summary:
                raise KeyError(f"`eval_acc_matrix_collab_beta` missing in {summary_path}")
            acc_matrix = np.asarray(summary["eval_acc_matrix_collab_beta"], dtype=float)
            avg_acc, avg_forget = _compute_avg_acc_and_forget(acc_matrix)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            warnings.warn(f"Skipping incomplete beta run `{run_root.name}`: {exc}", stacklevel=2)
            continue

        points.append(
            {
                "beta": beta,
                "avg_acc": avg_acc,
                "avg_forget": avg_forget,
            }
        )

    if not points:
        raise FileNotFoundError("No dirichlet STP beta runs were found under result/.")

    points.sort(key=lambda item: item["beta"])
    missing = [beta for beta in np.round(np.arange(0.0, 1.01, 0.1), 1) if beta not in {round(p["beta"], 1) for p in points}]
    if missing:
        warnings.warn(
            "Missing beta runs for: " + ", ".join(f"{beta:.1f}" for beta in missing) + ". Plotting only available points.",
            stacklevel=2,
        )
    return points


def plot_beta_sensitivity():
    points = _collect_beta_points()
    betas = np.asarray([p["beta"] for p in points], dtype=float)
    avg_acc = np.asarray([p["avg_acc"] for p in points], dtype=float)
    avg_forget = np.asarray([p["avg_forget"] for p in points], dtype=float)

    fig, ax1 = plt.subplots(figsize=(9.8, 6.0))
    ax2 = ax1.twinx()

    acc_line = ax1.plot(
        betas,
        avg_acc,
        color="#1f77b4",
        marker="o",
        linewidth=2.0,
        markersize=9,
        markeredgewidth=1.1,
        label="Collaborative AvgAcc",
        clip_on=False,
    )
    forget_line = ax2.plot(
        betas,
        avg_forget,
        color="#d62728",
        marker="s",
        linewidth=2.0,
        markersize=9,
        markeredgewidth=1.1,
        label="Collaborative AvgForget",
        clip_on=False,
    )

    ax1.set_title("Beta Sensitivity of Collaborative Inference", fontsize=18, pad=12)
    ax1.set_xlabel("Beta", fontsize=13)
    ax1.set_ylabel("AvgAcc", color="#1f77b4", fontsize=13)
    ax2.set_ylabel("AvgForget", color="#d62728", fontsize=13)
    ax1.tick_params(axis="y", colors="#1f77b4")
    ax2.tick_params(axis="y", colors="#d62728")
    ax1.tick_params(axis="x", labelsize=11)
    ax1.set_xlim(-0.03, 0.8)
    ax1.set_xticks(np.linspace(0, 0.8, 9))
    _style_axes(ax1)

    lines = acc_line + forget_line
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=False, loc="lower right", fontsize=11)

    fig.tight_layout()
    output_path = OUTPUT_DIR / BETA_NAME
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    sns.set_theme(style="whitegrid", context="talk")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = [
        plot_learning_curves(),
        plot_client_distribution(),
        plot_beta_sensitivity(),
    ]

    print("Generated figures:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
