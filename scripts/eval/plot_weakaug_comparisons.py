import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


TASKS = ("cyclops", "bbbc048")
METRICS = ("acc1", "acc5")
KNN_RE = re.compile(r"knn_(?P<task>[^_]+)_ep_(?P<epoch>\d+)_knn_offline_eval\.csv$")
LINEAR_RE = re.compile(r"ep_(?P<epoch>\d+)$")


@dataclass(frozen=True)
class KnnSpec:
    label: str
    output_dir: Path


@dataclass(frozen=True)
class LinearSpec:
    label: str
    summary_csv: Path


KNN_SPECS = (
    KnnSpec("trained_models_full1TB", Path("trained_models_full1TB/output")),
    KnnSpec("trained_models_full1TB_np65536", Path("trained_models_full1TB_np65536/outputs")),
    KnnSpec("trained_models_full1TB_np65536_in", Path("trained_models_full1TB_np65536_in/outputs")),
    KnnSpec("trained_models_full1TB_np65536_n1", Path("trained_models_full1TB_np65536_n1/outputs")),
    KnnSpec("trained_models_full1TB_np65536_weakaug", Path("trained_models_full1TB_np65536_weakaug/outputs")),
)

LINEAR_SPECS = (
    LinearSpec("trained_models_full1TB_e1", Path("eval_results/linear_sweep_cls_e1_cyclops/linear_sweep_summary.csv")),
    LinearSpec("trained_models_full1TB_np65536_weakaug_e1", Path("eval_results/linear_sweep_weakaug_e1/linear_sweep_summary.csv")),
)


def read_one_row_csv(path: Path) -> Dict[str, str]:
    with path.open("r", newline="") as f:
        return next(csv.DictReader(f))


def read_knn_rows(repo_root: Path) -> Tuple[List[Dict[str, object]], Dict[Tuple[str, str], float]]:
    rows: List[Dict[str, object]] = []
    baselines: Dict[Tuple[str, str], float] = {}
    for spec in KNN_SPECS:
        out_dir = repo_root / spec.output_dir
        if not out_dir.exists():
            continue
        for path in sorted(out_dir.glob("knn_*_ep_*_knn_offline_eval.csv")):
            m = KNN_RE.match(path.name)
            if not m:
                continue
            task = m.group("task")
            if task not in TASKS:
                continue
            row = read_one_row_csv(path)
            rows.append(
                {
                    "model": spec.label,
                    "task": task,
                    "epoch": int(m.group("epoch")),
                    "acc1": float(row["acc@1"]),
                    "acc5": float(row["acc@5"]),
                    "source": str(path),
                }
            )

        for task in TASKS:
            for candidate in (
                out_dir / f"knn_{task}_official_ep399_knn_offline_eval.csv",
                out_dir / f"knn_{task}_official_ep399_fix_knn_offline_eval.csv",
            ):
                if candidate.exists():
                    row = read_one_row_csv(candidate)
                    baselines[(task, "acc1")] = float(row["acc@1"])
                    baselines[(task, "acc5")] = float(row["acc@5"])
                    break
    return rows, baselines


def read_linear_rows(repo_root: Path) -> Tuple[List[Dict[str, object]], Dict[Tuple[str, str], float]]:
    rows: List[Dict[str, object]] = []
    baselines: Dict[Tuple[str, str], float] = {}
    for spec in LINEAR_SPECS:
        path = repo_root / spec.summary_csv
        if not path.exists():
            continue
        with path.open("r", newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] not in {"ok", "cached"}:
                    continue
                task = row["task"]
                if task not in TASKS:
                    continue
                ckpt_id = row["ckpt_id"]
                if "official" in ckpt_id:
                    if row["val_acc1"]:
                        baselines[(task, "acc1")] = float(row["val_acc1"])
                    if row["val_acc5"]:
                        baselines[(task, "acc5")] = float(row["val_acc5"])
                    continue
                m = LINEAR_RE.match(ckpt_id)
                if not m:
                    continue
                rows.append(
                    {
                        "model": spec.label,
                        "task": task,
                        "epoch": int(m.group("epoch")),
                        "acc1": float(row["val_acc1"]),
                        "acc5": float(row["val_acc5"]),
                        "source": str(path),
                    }
                )
    return rows, baselines


def write_rows(rows: Iterable[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "task", "epoch", "acc1", "acc5", "source"])
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["task"], r["model"], r["epoch"])):
            writer.writerow(row)


def plot_rows(
    rows: List[Dict[str, object]],
    baselines: Dict[Tuple[str, str], float],
    specs: Iterable,
    task: str,
    metric: str,
    title_prefix: str,
    out_png: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    colors = ["#1b6ca8", "#d17a22", "#258a5b", "#7b3fb2", "#b3284d", "#6a6a6a"]
    for spec, color in zip(specs, colors):
        series = sorted(
            [r for r in rows if r["task"] == task and r["model"] == spec.label],
            key=lambda r: r["epoch"],
        )
        if not series:
            continue
        plt.plot(
            [r["epoch"] for r in series],
            [r[metric] for r in series],
            marker="o",
            linewidth=2.0,
            markersize=4,
            color=color,
            label=spec.label,
        )

    baseline = baselines.get((task, metric))
    if baseline is not None:
        plt.axhline(
            baseline,
            color="#222222",
            linestyle="--",
            linewidth=1.7,
            alpha=0.85,
            label=f"official ep399 ({baseline:.2f})",
        )

    metric_label = "acc@1" if metric == "acc1" else "acc@5"
    plt.title(f"{title_prefix} {metric_label} - {task}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(alpha=0.28)
    plt.legend(frameon=True)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


def best_lines(rows: List[Dict[str, object]]) -> List[str]:
    lines: List[str] = []
    for task in TASKS:
        task_rows = [r for r in rows if r["task"] == task]
        for model in sorted({r["model"] for r in task_rows}):
            model_rows = [r for r in task_rows if r["model"] == model]
            best = max(model_rows, key=lambda r: float(r["acc1"]))
            lines.append(
                f"{task},{model},best_acc1_epoch={best['epoch']},best_acc1={best['acc1']:.8f}"
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", default="eval_results/weakaug_comparison")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    knn_rows, knn_baselines = read_knn_rows(repo_root)
    write_rows(knn_rows, out_dir / "knn_comparison_metrics.csv")
    for task in TASKS:
        for metric in METRICS:
            plot_rows(
                knn_rows,
                knn_baselines,
                KNN_SPECS,
                task,
                metric,
                "KNN Comparison",
                out_dir / f"knn_comparison_{task}_{metric}.png",
            )

    linear_rows, linear_baselines = read_linear_rows(repo_root)
    write_rows(linear_rows, out_dir / "linear_e1_comparison_metrics.csv")
    for task in TASKS:
        for metric in METRICS:
            plot_rows(
                linear_rows,
                linear_baselines,
                LINEAR_SPECS,
                task,
                metric,
                "Linear Probe e1 Comparison",
                out_dir / f"linear_e1_comparison_{task}_{metric}.png",
            )

    summary_path = out_dir / "best_acc1_summary.txt"
    summary_path.write_text(
        "\n".join(["[KNN]", *best_lines(knn_rows), "", "[Linear e1]", *best_lines(linear_rows), ""])
    )
    print(f"Wrote comparison plots and summaries to {out_dir}")


if __name__ == "__main__":
    main()
