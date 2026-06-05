import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional


TASKS = (
    "cyclops",
    "bbbc048",
    "bloodmnist",
    "chestmnist",
    "pathmnist",
    "dermamnist",
    "octmnist",
    "pneumoniamnist",
    "breastmnist",
    "retinamnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
    "tissuemnist",
)

TASK_DISPLAY_NAMES = {
    "cyclops": "CyclOPS",
    "bbbc048": "BBBC048",
    "bloodmnist": "BloodMNIST",
    "chestmnist": "ChestMNIST",
    "pathmnist": "PathMNIST",
    "dermamnist": "DermaMNIST",
    "octmnist": "OCTMNIST",
    "pneumoniamnist": "PneumoniaMNIST",
    "breastmnist": "BreastMNIST",
    "retinamnist": "RetinaMNIST",
    "organamnist": "OrganAMNIST",
    "organcmnist": "OrganCMNIST",
    "organsmnist": "OrganSMNIST",
    "tissuemnist": "TissueMNIST",
}

DINOV2_LABEL = "dinov2crop_bs4096"
RATIO20_LABEL = "ratio20_dinov2crop_bs4096"
WEAKAUG_LABEL = "weakaug"
OFFICIAL_LABEL = "official_ep399"

MODEL_ORDER = (OFFICIAL_LABEL, WEAKAUG_LABEL, DINOV2_LABEL, RATIO20_LABEL)
CURVE_MODEL_ORDER = (WEAKAUG_LABEL, DINOV2_LABEL, RATIO20_LABEL)

MODEL_STYLES = {
    OFFICIAL_LABEL: {"label": "official ep399", "color": "#6f6f6f", "marker": "o", "radar_marker": "D"},
    WEAKAUG_LABEL: {"label": "weakaug best", "color": "#cc6677", "marker": "s", "radar_marker": "D"},
    DINOV2_LABEL: {
        "label": "dinov2crop bs4096 best",
        "color": "#0072b2",
        "marker": "^",
        "radar_marker": "D",
    },
    RATIO20_LABEL: {
        "label": "ratio20 dinov2crop best",
        "color": "#009e73",
        "marker": "P",
        "radar_marker": "D",
    },
}

COLOR_VARIANTS = {
    "gray_red_blue": {
        OFFICIAL_LABEL: {"label": "official ep399", "color": "#6f6f6f", "marker": "o", "radar_marker": "D"},
        WEAKAUG_LABEL: {"label": "weakaug best", "color": "#cc6677", "marker": "s", "radar_marker": "D"},
        DINOV2_LABEL: {
            "label": "dinov2crop bs4096 best",
            "color": "#0072b2",
            "marker": "^",
            "radar_marker": "D",
        },
        RATIO20_LABEL: {
            "label": "ratio20 dinov2crop best",
            "color": "#009e73",
            "marker": "P",
            "radar_marker": "D",
        },
    },
    "gray_orange_blue": {
        OFFICIAL_LABEL: {"label": "official ep399", "color": "#6f6f6f", "marker": "o", "radar_marker": "D"},
        WEAKAUG_LABEL: {"label": "weakaug best", "color": "#d55e00", "marker": "s", "radar_marker": "D"},
        DINOV2_LABEL: {
            "label": "dinov2crop bs4096 best",
            "color": "#0072b2",
            "marker": "^",
            "radar_marker": "D",
        },
        RATIO20_LABEL: {
            "label": "ratio20 dinov2crop best",
            "color": "#009e73",
            "marker": "P",
            "radar_marker": "D",
        },
    },
    "gray_magenta_green": {
        OFFICIAL_LABEL: {"label": "official ep399", "color": "#6f6f6f", "marker": "o", "radar_marker": "D"},
        WEAKAUG_LABEL: {"label": "weakaug best", "color": "#cc79a7", "marker": "s", "radar_marker": "D"},
        DINOV2_LABEL: {
            "label": "dinov2crop bs4096 best",
            "color": "#009e73",
            "marker": "^",
            "radar_marker": "D",
        },
        RATIO20_LABEL: {
            "label": "ratio20 dinov2crop best",
            "color": "#0072b2",
            "marker": "P",
            "radar_marker": "D",
        },
    },
    "gray_vermillion_teal": {
        OFFICIAL_LABEL: {"label": "official ep399", "color": "#6f6f6f", "marker": "o", "radar_marker": "D"},
        WEAKAUG_LABEL: {"label": "weakaug best", "color": "#b2182b", "marker": "s", "radar_marker": "D"},
        DINOV2_LABEL: {
            "label": "dinov2crop bs4096 best",
            "color": "#1b9e77",
            "marker": "^",
            "radar_marker": "D",
        },
        RATIO20_LABEL: {
            "label": "ratio20 dinov2crop best",
            "color": "#7570b3",
            "marker": "P",
            "radar_marker": "D",
        },
    },
}


def display_task(task: object) -> str:
    return TASK_DISPLAY_NAMES.get(str(task), str(task))


def read_metric_csv(path: Path) -> Dict[str, object]:
    with path.open("r", newline="") as f:
        row = next(csv.DictReader(f))
    if "mean_auroc" in row:
        return {
            "primary_metric": "mean_auroc",
            "secondary_metric": "mean_ap",
            "primary": float(row["mean_auroc"]),
            "secondary": float(row["mean_ap"]),
        }
    return {
        "primary_metric": "acc@1",
        "secondary_metric": "acc@5",
        "primary": float(row["acc@1"]),
        "secondary": float(row["acc@5"]),
    }


def parse_epoch(ckpt_id: str) -> Optional[int]:
    match = re.search(r"ep_(\d+)", ckpt_id)
    return int(match.group(1)) if match else None


def parse_task_ckpt(path: Path) -> Optional[Dict[str, str]]:
    name = path.name
    suffix = "_knn_offline_eval.csv"
    if not name.endswith(suffix):
        return None
    for task in TASKS:
        prefix = f"knn_{task}_"
        if name.startswith(prefix):
            return {"task": task, "ckpt_id": name[len(prefix) : -len(suffix)]}
    return None


def model_from_ckpt(ckpt_id: str, source_path: Optional[Path] = None) -> Optional[str]:
    if ckpt_id == OFFICIAL_LABEL:
        return OFFICIAL_LABEL
    path_text = "" if source_path is None else str(source_path).lower()
    if RATIO20_LABEL in path_text or "ratio20" in path_text:
        return RATIO20_LABEL
    if DINOV2_LABEL in path_text or "webds_shuffle_dinov2crop" in path_text:
        return DINOV2_LABEL
    if WEAKAUG_LABEL in path_text:
        return WEAKAUG_LABEL
    if "dinov2" in ckpt_id.lower():
        return DINOV2_LABEL
    if re.fullmatch(r"ep_\d+_run\d+", ckpt_id):
        return DINOV2_LABEL
    if re.fullmatch(r"ep_\d+", ckpt_id):
        return WEAKAUG_LABEL
    return None


def collect_rows(repo_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    paths = list(repo_root.glob("knn_*_knn_offline_eval.csv"))
    paths.extend((repo_root / "csv_archive" / "knn_offline_eval").glob("**/knn_*_knn_offline_eval.csv"))
    for path in sorted(paths):
        if {"duplicates", "legacy_or_other"} & set(path.parts):
            continue
        parsed = parse_task_ckpt(path)
        if not parsed:
            continue
        model = model_from_ckpt(parsed["ckpt_id"], path)
        if not model:
            continue
        metrics = read_metric_csv(path)
        rows.append(
            {
                "model": model,
                "task": parsed["task"],
                "ckpt_id": parsed["ckpt_id"],
                "epoch": parse_epoch(parsed["ckpt_id"]),
                "source_csv": str(path),
                **metrics,
            }
        )
    rows.sort(key=lambda r: (str(r["task"]), str(r["model"]), int(r["epoch"] or 10**9)))
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def best_for(rows: List[Dict[str, object]], task: str, model: str) -> Optional[Dict[str, object]]:
    candidates = [r for r in rows if r["task"] == task and r["model"] == model]
    return max(candidates, key=lambda r: float(r["primary"])) if candidates else None


def build_best_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for task in TASKS:
        dinov2 = best_for(rows, task, DINOV2_LABEL)
        ratio20 = best_for(rows, task, RATIO20_LABEL)
        weakaug = best_for(rows, task, WEAKAUG_LABEL)
        official = best_for(rows, task, OFFICIAL_LABEL)
        if not any((dinov2, ratio20, weakaug, official)):
            continue
        metric = next(str(r["primary_metric"]) for r in (dinov2, ratio20, weakaug, official) if r)
        secondary_metric = next(str(r["secondary_metric"]) for r in (dinov2, ratio20, weakaug, official) if r)

        def field(row: Optional[Dict[str, object]], key: str) -> object:
            return "" if row is None else row[key]

        def fnum(row: Optional[Dict[str, object]], key: str) -> str:
            return "" if row is None else f"{float(row[key]):.8f}"

        d = float(dinov2["primary"]) if dinov2 else None
        r20 = float(ratio20["primary"]) if ratio20 else None
        w = float(weakaug["primary"]) if weakaug else None
        o = float(official["primary"]) if official else None
        out.append(
            {
                "task": task,
                "primary_metric": metric,
                "secondary_metric": secondary_metric,
                "dinov2_best_ckpt": field(dinov2, "ckpt_id"),
                "dinov2_best_epoch": field(dinov2, "epoch"),
                "dinov2_best_primary": fnum(dinov2, "primary"),
                "dinov2_best_secondary": fnum(dinov2, "secondary"),
                "ratio20_best_ckpt": field(ratio20, "ckpt_id"),
                "ratio20_best_epoch": field(ratio20, "epoch"),
                "ratio20_best_primary": fnum(ratio20, "primary"),
                "ratio20_best_secondary": fnum(ratio20, "secondary"),
                "weakaug_best_ckpt": field(weakaug, "ckpt_id"),
                "weakaug_best_epoch": field(weakaug, "epoch"),
                "weakaug_best_primary": fnum(weakaug, "primary"),
                "weakaug_best_secondary": fnum(weakaug, "secondary"),
                "official_primary": fnum(official, "primary"),
                "official_secondary": fnum(official, "secondary"),
                "dinov2_minus_official": "" if d is None or o is None else f"{d - o:.8f}",
                "ratio20_minus_official": "" if r20 is None or o is None else f"{r20 - o:.8f}",
                "ratio20_minus_dinov2": "" if r20 is None or d is None else f"{r20 - d:.8f}",
                "weakaug_minus_official": "" if w is None or o is None else f"{w - o:.8f}",
                "dinov2_minus_weakaug": "" if d is None or w is None else f"{d - w:.8f}",
            }
        )
    return out


def plot_best_summary(summary: List[Dict[str, object]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    tasks = [display_task(r["task"]) for r in summary]
    x = list(range(len(tasks)))
    width = 0.20

    def vals(key: str) -> List[float]:
        return [float("nan") if r[key] == "" else float(r[key]) for r in summary]

    plt.figure(figsize=(18, 7))
    plt.bar(
        [i - 1.5 * width for i in x],
        vals("official_primary"),
        width=width,
        label=MODEL_STYLES[OFFICIAL_LABEL]["label"],
        color=MODEL_STYLES[OFFICIAL_LABEL]["color"],
    )
    plt.bar(
        [i - 0.5 * width for i in x],
        vals("weakaug_best_primary"),
        width=width,
        label=MODEL_STYLES[WEAKAUG_LABEL]["label"],
        color=MODEL_STYLES[WEAKAUG_LABEL]["color"],
    )
    plt.bar(
        [i + 0.5 * width for i in x],
        vals("dinov2_best_primary"),
        width=width,
        label=MODEL_STYLES[DINOV2_LABEL]["label"],
        color=MODEL_STYLES[DINOV2_LABEL]["color"],
    )
    plt.bar(
        [i + 1.5 * width for i in x],
        vals("ratio20_best_primary"),
        width=width,
        label=MODEL_STYLES[RATIO20_LABEL]["label"],
        color=MODEL_STYLES[RATIO20_LABEL]["color"],
    )
    plt.xticks(x, tasks, rotation=35, ha="right")
    plt.ylabel("Primary metric (%)")
    plt.title("KNN Best Checkpoint Comparison")
    plt.grid(axis="y", color="#e3e3e3", linewidth=0.8)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_radar_data(summary: List[Dict[str, object]], out_path: Path) -> None:
    rows = []
    for row in summary:
        rows.append(
            {
                "task": row["task"],
                "task_label": display_task(row["task"]),
                "primary_metric": row["primary_metric"],
                "official_ep399": row["official_primary"],
                "weakaug_best": row["weakaug_best_primary"],
                "dinov2crop_bs4096_best": row["dinov2_best_primary"],
                "ratio20_dinov2crop_bs4096_best": row["ratio20_best_primary"],
            }
        )
    write_csv(
        out_path,
        rows,
        [
            "task",
            "task_label",
            "primary_metric",
            "official_ep399",
            "weakaug_best",
            "dinov2crop_bs4096_best",
            "ratio20_dinov2crop_bs4096_best",
        ],
    )


def plot_radar_summary(
    summary: List[Dict[str, object]],
    out_path: Path,
    styles: Dict[str, Dict[str, str]] = MODEL_STYLES,
) -> None:
    import math
    import matplotlib.pyplot as plt

    labels = [display_task(r["task"]) for r in summary]
    n = len(labels)
    angles = [2 * math.pi * i / n for i in range(n)]
    angles += angles[:1]

    series = (
        (
            styles[OFFICIAL_LABEL]["label"],
            "official_primary",
            styles[OFFICIAL_LABEL]["color"],
            styles[OFFICIAL_LABEL]["radar_marker"],
        ),
        (
            styles[WEAKAUG_LABEL]["label"],
            "weakaug_best_primary",
            styles[WEAKAUG_LABEL]["color"],
            styles[WEAKAUG_LABEL]["radar_marker"],
        ),
        (
            styles[DINOV2_LABEL]["label"],
            "dinov2_best_primary",
            styles[DINOV2_LABEL]["color"],
            styles[DINOV2_LABEL]["radar_marker"],
        ),
        (
            styles[RATIO20_LABEL]["label"],
            "ratio20_best_primary",
            styles[RATIO20_LABEL]["color"],
            styles[RATIO20_LABEL]["radar_marker"],
        ),
    )

    fig = plt.figure(figsize=(11.5, 11.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(40, 100)
    ax.set_yticks([40, 50, 60, 70, 80, 90, 100])
    ax.set_yticklabels(["40", "50", "60", "70", "80", "90", "100"], fontsize=8, color="#777777")
    ax.grid(color="#d9d9d9", linewidth=0.8, alpha=0.75)
    ax.spines["polar"].set_visible(False)
    ax.tick_params(axis="x", pad=9)
    ax.tick_params(axis="y", colors="#777777")

    for label, key, color, marker in series:
        values = [float("nan") if r[key] == "" else float(r[key]) for r in summary]
        values += values[:1]
        ax.plot(
            angles,
            values,
            linewidth=2.0,
            color=color,
            marker=marker,
            markersize=4.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=label,
        )

    ax.set_title("KNN Best Checkpoint Radar Comparison", pad=30, fontsize=14)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.06, right=0.94, top=0.90, bottom=0.13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_dotline_data(summary: List[Dict[str, object]], out_path: Path) -> None:
    rows = []
    for row in summary:
        rows.append(
            {
                "task": row["task"],
                "task_label": display_task(row["task"]),
                "primary_metric": row["primary_metric"],
                "official_ep399": row["official_primary"],
                "weakaug_best": row["weakaug_best_primary"],
                "dinov2crop_bs4096_best": row["dinov2_best_primary"],
                "ratio20_dinov2crop_bs4096_best": row["ratio20_best_primary"],
            }
        )
    write_csv(
        out_path,
        rows,
        [
            "task",
            "task_label",
            "primary_metric",
            "official_ep399",
            "weakaug_best",
            "dinov2crop_bs4096_best",
            "ratio20_dinov2crop_bs4096_best",
        ],
    )


def plot_dotline_summary(
    summary: List[Dict[str, object]],
    out_path: Path,
    styles: Dict[str, Dict[str, str]] = MODEL_STYLES,
) -> None:
    import matplotlib.pyplot as plt

    rows = list(reversed(summary))
    y = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(12, 8.2))

    value_keys = (
        (OFFICIAL_LABEL, "official_primary"),
        (WEAKAUG_LABEL, "weakaug_best_primary"),
        (DINOV2_LABEL, "dinov2_best_primary"),
        (RATIO20_LABEL, "ratio20_best_primary"),
    )

    for yi, row in zip(y, rows):
        values = [float(row[key]) for _, key in value_keys if row[key] != ""]
        if values:
            ax.hlines(yi, min(values), max(values), color="#d8d8d8", linewidth=1.4, zorder=1)

    for model, key in value_keys:
        style = styles[model]
        points = [(float(row[key]), yi) for yi, row in zip(y, rows) if row[key] != ""]
        if not points:
            continue
        xs, ys = zip(*points)
        ax.scatter(
            xs,
            ys,
            s=58,
            marker=str(style["marker"]),
            color=str(style["color"]),
            edgecolor="white",
            linewidth=0.7,
            label=str(style["label"]),
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([display_task(row["task"]) for row in rows], fontsize=10)
    ax.set_xlim(40, 100)
    ax.set_xlabel("Primary metric (%)")
    ax.set_title("KNN Best Checkpoint Horizontal Dot-Line Comparison")
    ax.grid(axis="x", color="#e3e3e3", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def task_curve_rows(rows: List[Dict[str, object]], task: str) -> List[Dict[str, object]]:
    return [r for r in rows if r["task"] == task]


def plot_task_curve(rows: List[Dict[str, object]], task: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    task_rows = task_curve_rows(rows, task)
    if not task_rows:
        return
    metric = str(task_rows[0]["primary_metric"])
    plt.figure(figsize=(11, 5.5))
    colors = {
        WEAKAUG_LABEL: MODEL_STYLES[WEAKAUG_LABEL]["color"],
        DINOV2_LABEL: MODEL_STYLES[DINOV2_LABEL]["color"],
        RATIO20_LABEL: MODEL_STYLES[RATIO20_LABEL]["color"],
    }
    for model in CURVE_MODEL_ORDER:
        series = sorted(
            [r for r in task_rows if r["model"] == model and r["epoch"] is not None],
            key=lambda r: int(r["epoch"]),
        )
        if not series:
            continue
        best = max(series, key=lambda r: float(r["primary"]))
        plt.plot(
            [int(r["epoch"]) for r in series],
            [float(r["primary"]) for r in series],
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            label=f"{model} best {float(best['primary']):.2f}@{int(best['epoch'])}",
            color=colors[model],
        )
    official = best_for(rows, task, OFFICIAL_LABEL)
    if official:
        plt.axhline(
            float(official["primary"]),
            color=MODEL_STYLES[OFFICIAL_LABEL]["color"],
            linestyle="--",
            linewidth=1.7,
            label=f"official ep399 ({float(official['primary']):.2f})",
        )
    plt.title(f"KNN Epoch Curve - {display_task(task)}")
    plt.xlabel("Epoch")
    plt.ylabel(f"{metric} (%)")
    plt.grid(color="#e3e3e3", linewidth=0.8)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_all_curves(rows: List[Dict[str, object]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 4, figsize=(22, 16))
    axes = axes.ravel()
    colors = {
        WEAKAUG_LABEL: MODEL_STYLES[WEAKAUG_LABEL]["color"],
        DINOV2_LABEL: MODEL_STYLES[DINOV2_LABEL]["color"],
        RATIO20_LABEL: MODEL_STYLES[RATIO20_LABEL]["color"],
    }
    for ax, task in zip(axes, TASKS):
        task_rows = task_curve_rows(rows, task)
        metric = str(task_rows[0]["primary_metric"]) if task_rows else "primary"
        for model in CURVE_MODEL_ORDER:
            series = sorted(
                [r for r in task_rows if r["model"] == model and r["epoch"] is not None],
                key=lambda r: int(r["epoch"]),
            )
            if not series:
                continue
            ax.plot(
                [int(r["epoch"]) for r in series],
                [float(r["primary"]) for r in series],
                marker="o",
                markersize=2.2,
                linewidth=1.2,
                color=colors[model],
                label=model,
            )
        official = best_for(rows, task, OFFICIAL_LABEL)
        if official:
            ax.axhline(
                float(official["primary"]),
                color=MODEL_STYLES[OFFICIAL_LABEL]["color"],
                linestyle="--",
                linewidth=1.0,
            )
        ax.set_title(f"{display_task(task)} ({metric})")
        ax.grid(color="#e3e3e3", linewidth=0.7)
    for ax in axes[len(TASKS) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("KNN Primary Metric Epoch Curves", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def clean_eval_results(eval_root: Path, keep_dir: Path) -> None:
    keep = keep_dir.resolve()
    for path in eval_root.iterdir():
        if path.resolve() == keep:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", default="eval_results/knn_final_summary")
    parser.add_argument("--clean-eval-results", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    fig_dir = out_dir / "fig"
    data_dir = out_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(repo_root)
    detail_fields = [
        "model",
        "task",
        "ckpt_id",
        "epoch",
        "primary_metric",
        "primary",
        "secondary_metric",
        "secondary",
        "source_csv",
    ]
    write_csv(data_dir / "all_knn_metrics.csv", rows, detail_fields)

    best_summary = build_best_summary(rows)
    best_fields = [
        "task",
        "primary_metric",
        "secondary_metric",
        "dinov2_best_ckpt",
        "dinov2_best_epoch",
        "dinov2_best_primary",
        "dinov2_best_secondary",
        "ratio20_best_ckpt",
        "ratio20_best_epoch",
        "ratio20_best_primary",
        "ratio20_best_secondary",
        "weakaug_best_ckpt",
        "weakaug_best_epoch",
        "weakaug_best_primary",
        "weakaug_best_secondary",
        "official_primary",
        "official_secondary",
        "dinov2_minus_official",
        "ratio20_minus_official",
        "ratio20_minus_dinov2",
        "weakaug_minus_official",
        "dinov2_minus_weakaug",
    ]
    write_csv(data_dir / "best_checkpoint_comparison.csv", best_summary, best_fields)
    write_radar_data(best_summary, data_dir / "radar_best_checkpoint_comparison.csv")
    write_dotline_data(best_summary, data_dir / "dotline_best_checkpoint_comparison.csv")

    for task in TASKS:
        per_task = task_curve_rows(rows, task)
        write_csv(data_dir / f"epoch_curve_{task}.csv", per_task, detail_fields)
        plot_task_curve(rows, task, fig_dir / f"epoch_curve_{task}.png")

    plot_best_summary(best_summary, fig_dir / "best_checkpoint_comparison.png")
    plot_radar_summary(best_summary, fig_dir / "radar_best_checkpoint_comparison.png")
    plot_dotline_summary(best_summary, fig_dir / "best_checkpoint_dotline_comparison.png")
    for palette_name, styles in COLOR_VARIANTS.items():
        plot_radar_summary(
            best_summary,
            fig_dir / f"radar_best_checkpoint_comparison_{palette_name}.png",
            styles=styles,
        )
        plot_dotline_summary(
            best_summary,
            fig_dir / f"best_checkpoint_dotline_comparison_{palette_name}.png",
            styles=styles,
        )
    plot_all_curves(rows, fig_dir / "epoch_curves_all_tasks.png")

    (data_dir / "README.txt").write_text(
        "all_knn_metrics.csv: all collected KNN rows for dinov2crop, ratio20 dinov2crop, weakaug, and official.\n"
        "best_checkpoint_comparison.csv: best checkpoint per model and task by primary metric.\n"
        "radar_best_checkpoint_comparison.csv: compact table used by the radar comparison plot.\n"
        "dotline_best_checkpoint_comparison.csv: compact table used by the horizontal dot-line comparison plot.\n"
        "fig/*_gray_red_blue.png, *_gray_orange_blue.png, *_gray_magenta_green.png, *_gray_vermillion_teal.png: alternate color palettes for radar and dot-line plots.\n"
        "epoch_curve_<task>.csv: per-task rows used for each epoch curve plot.\n",
        encoding="utf-8",
    )

    if args.clean_eval_results:
        clean_eval_results(repo_root / "eval_results", out_dir)

    print(f"Wrote final summary to {out_dir}")
    print(f"Figures: {fig_dir}")
    print(f"Data: {data_dir}")


if __name__ == "__main__":
    main()
