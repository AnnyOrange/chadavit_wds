import argparse
import csv
import math
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

OFFICIAL = "official_ep399"
WEAKAUG = "weakaug"
DINOV2 = "dinov2crop_bs4096"
RATIO10 = "ratio10_dinov2crop_bs4096"
RATIO20 = "ratio20_dinov2crop_bs4096"
SEMDEDUP_RATIO10 = "semdedup_ratio10_dinov2crop_bs4096"
SEMDEDUP_RATIO20 = "semdedup_ratio20_dinov2crop_bs4096"

MODEL_ORDER = (OFFICIAL, WEAKAUG, DINOV2, RATIO10, RATIO20, SEMDEDUP_RATIO10, SEMDEDUP_RATIO20)
CURVE_MODEL_ORDER = (WEAKAUG, DINOV2, RATIO10, RATIO20, SEMDEDUP_RATIO10, SEMDEDUP_RATIO20)
MODEL_KEYS = {
    OFFICIAL: "official",
    WEAKAUG: "weakaug",
    DINOV2: "dinov2",
    RATIO10: "ratio10",
    RATIO20: "ratio20",
    SEMDEDUP_RATIO10: "semdedup_ratio10",
    SEMDEDUP_RATIO20: "semdedup_ratio20",
}
MODEL_STYLES = {
    OFFICIAL: {"label": "official ep399", "color": "#6f6f6f", "marker": "o"},
    WEAKAUG: {"label": "weakaug best", "color": "#cc6677", "marker": "s"},
    DINOV2: {"label": "dinov2crop bs4096 best", "color": "#0072b2", "marker": "^"},
    RATIO10: {"label": "ratio10 dinov2crop best", "color": "#f0a202", "marker": "v"},
    RATIO20: {"label": "ratio20 dinov2crop best", "color": "#009e73", "marker": "P"},
    SEMDEDUP_RATIO10: {"label": "semdedup ratio10 best", "color": "#984ea3", "marker": "D"},
    SEMDEDUP_RATIO20: {"label": "semdedup ratio20 best", "color": "#7570b3", "marker": "X"},
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
    suffix = "_knn_offline_eval.csv"
    name = path.name
    if not name.endswith(suffix):
        return None
    for task in TASKS:
        prefix = f"knn_{task}_"
        if name.startswith(prefix):
            return {"task": task, "ckpt_id": name[len(prefix) : -len(suffix)]}
    return None


def model_from_ckpt(ckpt_id: str, source_path: Optional[Path] = None) -> Optional[str]:
    if ckpt_id == OFFICIAL:
        return OFFICIAL
    text = f"{ckpt_id} {'' if source_path is None else str(source_path)}".lower()
    if "semdedup_ratio10" in text or "semdedup-ratio10" in text:
        return SEMDEDUP_RATIO10
    if "semdedup_ratio20" in text or "semdedup-ratio20" in text:
        return SEMDEDUP_RATIO20
    if "semdedup" in text:
        return SEMDEDUP_RATIO20
    if RATIO10 in text or "ratio10" in text:
        return RATIO10
    if RATIO20 in text or "ratio20" in text:
        return RATIO20
    if DINOV2 in text or "webds_shuffle_dinov2crop" in text:
        return DINOV2
    if WEAKAUG in text:
        return WEAKAUG
    if re.fullmatch(r"ep_\d+_run\d+", ckpt_id):
        return DINOV2
    if re.fullmatch(r"ep_\d+", ckpt_id):
        return WEAKAUG
    return None


def collect_rows(repo_root: Path) -> List[Dict[str, object]]:
    paths = list(repo_root.glob("knn_*_knn_offline_eval.csv"))
    paths.extend((repo_root / "csv_archive" / "knn_offline_eval").glob("**/knn_*_knn_offline_eval.csv"))
    paths.extend((repo_root / "eval_results" / "knn_final_summary" / "data" / "raw_knn_csvs").glob("**/knn_*_knn_offline_eval.csv"))
    rows: List[Dict[str, object]] = []
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
    rows.sort(key=lambda r: (str(r["task"]), MODEL_ORDER.index(str(r["model"])), int(r["epoch"] or 10**9), str(r["ckpt_id"])))
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
        best = {model: best_for(rows, task, model) for model in MODEL_ORDER}
        if not any(best.values()):
            continue
        first = next(r for r in best.values() if r)
        row: Dict[str, object] = {
            "task": task,
            "primary_metric": first["primary_metric"],
            "secondary_metric": first["secondary_metric"],
        }
        official_primary = float(best[OFFICIAL]["primary"]) if best[OFFICIAL] else None
        dinov2_primary = float(best[DINOV2]["primary"]) if best[DINOV2] else None
        weakaug_primary = float(best[WEAKAUG]["primary"]) if best[WEAKAUG] else None
        for model in MODEL_ORDER:
            key = MODEL_KEYS[model]
            b = best[model]
            if model == OFFICIAL:
                row["official_ckpt"] = "" if b is None else b["ckpt_id"]
                row["official_primary"] = "" if b is None else f"{float(b['primary']):.8f}"
                row["official_secondary"] = "" if b is None else f"{float(b['secondary']):.8f}"
                continue
            row[f"{key}_best_ckpt"] = "" if b is None else b["ckpt_id"]
            row[f"{key}_best_epoch"] = "" if b is None else b["epoch"]
            row[f"{key}_best_primary"] = "" if b is None else f"{float(b['primary']):.8f}"
            row[f"{key}_best_secondary"] = "" if b is None else f"{float(b['secondary']):.8f}"
            primary = float(b["primary"]) if b else None
            row[f"{key}_minus_official"] = "" if primary is None or official_primary is None else f"{primary - official_primary:.8f}"
            row[f"{key}_minus_dinov2"] = "" if primary is None or dinov2_primary is None else f"{primary - dinov2_primary:.8f}"
        row["dinov2_minus_weakaug"] = "" if dinov2_primary is None or weakaug_primary is None else f"{dinov2_primary - weakaug_primary:.8f}"
        out.append(row)
    return out


def best_field(model: str) -> str:
    return "official_primary" if model == OFFICIAL else f"{MODEL_KEYS[model]}_best_primary"


def compact_best_rows(summary: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for row in summary:
        compact = {"task": row["task"], "task_label": display_task(row["task"]), "primary_metric": row["primary_metric"]}
        for model in MODEL_ORDER:
            compact[model] = row.get(best_field(model), "")
        rows.append(compact)
    return rows


def plot_best_summary(summary: List[Dict[str, object]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    tasks = [display_task(r["task"]) for r in summary]
    x = list(range(len(tasks)))
    width = min(0.12, 0.78 / len(MODEL_ORDER))
    offsets = [(i - (len(MODEL_ORDER) - 1) / 2) * width for i in range(len(MODEL_ORDER))]

    fig, ax = plt.subplots(figsize=(20, 7.5))
    for offset, model in zip(offsets, MODEL_ORDER):
        vals = [float("nan") if r.get(best_field(model), "") == "" else float(r[best_field(model)]) for r in summary]
        ax.bar(
            [i + offset for i in x],
            vals,
            width=width,
            label=MODEL_STYLES[model]["label"],
            color=MODEL_STYLES[model]["color"],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=35, ha="right")
    ax.set_ylabel("Primary metric (%)")
    ax.set_title("KNN Best Checkpoint Comparison")
    ax.grid(axis="y", color="#e3e3e3", linewidth=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_radar_summary(summary: List[Dict[str, object]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [display_task(r["task"]) for r in summary]
    n = len(labels)
    angles = [2 * math.pi * i / n for i in range(n)] + [0]
    fig = plt.figure(figsize=(12, 12))
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
    for model in MODEL_ORDER:
        values = [float("nan") if r.get(best_field(model), "") == "" else float(r[best_field(model)]) for r in summary]
        values += values[:1]
        style = MODEL_STYLES[model]
        ax.plot(
            angles,
            values,
            linewidth=1.8,
            color=style["color"],
            marker=style["marker"],
            markersize=4.5,
            label=style["label"],
        )
    ax.set_title("KNN Best Checkpoint Radar Comparison", pad=30, fontsize=14)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=3, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.90, bottom=0.14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_dotline_summary(summary: List[Dict[str, object]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = list(reversed(summary))
    y = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(13, 8.5))
    for yi, row in zip(y, rows):
        values = [float(row[best_field(model)]) for model in MODEL_ORDER if row.get(best_field(model), "") != ""]
        if values:
            ax.hlines(yi, min(values), max(values), color="#d8d8d8", linewidth=1.4, zorder=1)
    for model in MODEL_ORDER:
        style = MODEL_STYLES[model]
        points = [(float(row[best_field(model)]), yi) for yi, row in zip(y, rows) if row.get(best_field(model), "") != ""]
        if not points:
            continue
        xs, ys = zip(*points)
        ax.scatter(xs, ys, s=58, marker=style["marker"], color=style["color"], edgecolor="white", linewidth=0.7, label=style["label"], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([display_task(row["task"]) for row in rows], fontsize=10)
    ax.set_xlim(40, 100)
    ax.set_xlabel("Primary metric (%)")
    ax.set_title("KNN Best Checkpoint Horizontal Dot-Line Comparison")
    ax.grid(axis="x", color="#e3e3e3", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)
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
    fig, ax = plt.subplots(figsize=(12, 5.8))
    for model in CURVE_MODEL_ORDER:
        series = sorted([r for r in task_rows if r["model"] == model and r["epoch"] is not None], key=lambda r: int(r["epoch"]))
        if not series:
            continue
        best = max(series, key=lambda r: float(r["primary"]))
        style = MODEL_STYLES[model]
        ax.plot(
            [int(r["epoch"]) for r in series],
            [float(r["primary"]) for r in series],
            marker=style["marker"],
            markersize=3.2,
            linewidth=1.5,
            color=style["color"],
            label=f"{MODEL_KEYS[model]} best {float(best['primary']):.2f}@{int(best['epoch'])}",
        )
    official = best_for(rows, task, OFFICIAL)
    if official:
        ax.axhline(float(official["primary"]), color=MODEL_STYLES[OFFICIAL]["color"], linestyle="--", linewidth=1.5, label=f"official ({float(official['primary']):.2f})")
    ax.set_title(f"KNN Epoch Curve - {display_task(task)}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{metric} (%)")
    ax.grid(color="#e3e3e3", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_all_curves(rows: List[Dict[str, object]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 4, figsize=(23, 16))
    axes = axes.ravel()
    for ax, task in zip(axes, TASKS):
        task_rows = task_curve_rows(rows, task)
        metric = str(task_rows[0]["primary_metric"]) if task_rows else "primary"
        for model in CURVE_MODEL_ORDER:
            series = sorted([r for r in task_rows if r["model"] == model and r["epoch"] is not None], key=lambda r: int(r["epoch"]))
            if not series:
                continue
            style = MODEL_STYLES[model]
            ax.plot([int(r["epoch"]) for r in series], [float(r["primary"]) for r in series], marker=style["marker"], markersize=2.0, linewidth=1.1, color=style["color"], label=MODEL_KEYS[model])
        official = best_for(rows, task, OFFICIAL)
        if official:
            ax.axhline(float(official["primary"]), color=MODEL_STYLES[OFFICIAL]["color"], linestyle="--", linewidth=1.0)
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
    detail_fields = ["model", "task", "ckpt_id", "epoch", "primary_metric", "primary", "secondary_metric", "secondary", "source_csv"]
    write_csv(data_dir / "all_knn_metrics.csv", rows, detail_fields)

    best_summary = build_best_summary(rows)
    best_fields = ["task", "primary_metric", "secondary_metric"]
    for model in MODEL_ORDER:
        key = MODEL_KEYS[model]
        if model == OFFICIAL:
            best_fields.extend(["official_ckpt", "official_primary", "official_secondary"])
        else:
            best_fields.extend([
                f"{key}_best_ckpt",
                f"{key}_best_epoch",
                f"{key}_best_primary",
                f"{key}_best_secondary",
                f"{key}_minus_official",
                f"{key}_minus_dinov2",
            ])
    best_fields.append("dinov2_minus_weakaug")
    write_csv(data_dir / "best_checkpoint_comparison.csv", best_summary, best_fields)

    compact_rows = compact_best_rows(best_summary)
    compact_fields = ["task", "task_label", "primary_metric", *MODEL_ORDER]
    write_csv(data_dir / "radar_best_checkpoint_comparison.csv", compact_rows, compact_fields)
    write_csv(data_dir / "dotline_best_checkpoint_comparison.csv", compact_rows, compact_fields)

    for task in TASKS:
        per_task = task_curve_rows(rows, task)
        write_csv(data_dir / f"epoch_curve_{task}.csv", per_task, detail_fields)
        plot_task_curve(rows, task, fig_dir / f"epoch_curve_{task}.png")

    plot_best_summary(best_summary, fig_dir / "best_checkpoint_comparison.png")
    plot_radar_summary(best_summary, fig_dir / "radar_best_checkpoint_comparison.png")
    plot_dotline_summary(best_summary, fig_dir / "best_checkpoint_dotline_comparison.png")
    plot_all_curves(rows, fig_dir / "epoch_curves_all_tasks.png")

    (data_dir / "README.txt").write_text(
        "all_knn_metrics.csv: all collected KNN rows for official, weakaug, full dinov2crop, ratio10, ratio20, semdedup ratio10, and semdedup ratio20.\n"
        "best_checkpoint_comparison.csv: best checkpoint per model and task by primary metric.\n"
        "radar_best_checkpoint_comparison.csv and dotline_best_checkpoint_comparison.csv: compact tables for summary plots.\n"
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
