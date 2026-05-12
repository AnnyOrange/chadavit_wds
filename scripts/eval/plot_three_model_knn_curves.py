import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ModelSpec:
    label: str
    output_dir: Path


TASKS = ("cyclops", "bbbc048")
MODEL_SPECS = (
    ModelSpec("trained_models_full1TB", Path("trained_models_full1TB/output")),
    ModelSpec("trained_models_full1TB_np65536", Path("trained_models_full1TB_np65536/outputs")),
    ModelSpec("trained_models_full1TB_np65536_in", Path("trained_models_full1TB_np65536_in/outputs")),
)
N1_MODEL_SPEC = ModelSpec("trained_models_full1TB_np65536_n1", Path("trained_models_full1TB_np65536_n1/outputs"))
WEAKAUG_MODEL_SPEC = ModelSpec("trained_models_full1TB_np65536_weakaug", Path("trained_models_full1TB_np65536_weakaug/outputs"))
CSV_RE = re.compile(r"knn_(?P<task>[^_]+)_ep_(?P<epoch>\d+)_knn_offline_eval\.csv$")


def read_knn_csv(path: Path) -> Tuple[float, float]:
    with path.open("r", newline="") as f:
        row = next(csv.DictReader(f))
    return float(row["acc@1"]), float(row["acc@5"])


def iter_epoch_rows(repo_root: Path, spec: ModelSpec) -> Iterable[Dict[str, object]]:
    out_dir = repo_root / spec.output_dir
    for path in sorted(out_dir.glob("knn_*_ep_*_knn_offline_eval.csv")):
        m = CSV_RE.match(path.name)
        if not m:
            continue
        task = m.group("task")
        if task not in TASKS:
            continue
        acc1, acc5 = read_knn_csv(path)
        yield {
            "model": spec.label,
            "task": task,
            "epoch": int(m.group("epoch")),
            "acc1": acc1,
            "acc5": acc5,
            "csv_path": str(path),
        }


def find_baseline(repo_root: Path, task: str, metric: str, model_specs: Tuple[ModelSpec, ...]) -> Optional[float]:
    # Prefer the newest/in-distribution output because it was produced by the current eval code.
    search_dirs = [spec.output_dir for spec in reversed(model_specs)]
    for rel_dir in search_dirs:
        out_dir = repo_root / rel_dir
        candidates = [
            out_dir / f"knn_{task}_official_ep399_knn_offline_eval.csv",
            out_dir / f"knn_{task}_official_ep399_fix_knn_offline_eval.csv",
        ]
        for path in candidates:
            if path.exists():
                acc1, acc5 = read_knn_csv(path)
                return acc1 if metric == "acc1" else acc5
    return None


def write_summary(rows: List[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        fieldnames = ["model", "task", "epoch", "acc1", "acc5", "csv_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["task"], r["model"], r["epoch"])):
            writer.writerow(row)


def plot_task(
    rows: List[Dict[str, object]],
    repo_root: Path,
    task: str,
    metric: str,
    out_png: Path,
    model_specs: Tuple[ModelSpec, ...],
) -> None:
    import matplotlib.pyplot as plt

    task_rows = [r for r in rows if r["task"] == task]
    if not task_rows:
        return

    plt.figure(figsize=(11, 6))
    colors = ["#1b6ca8", "#d17a22", "#258a5b", "#7b3fb2", "#b3284d"]
    for spec, color in zip(model_specs, colors):
        series = sorted([r for r in task_rows if r["model"] == spec.label], key=lambda r: r["epoch"])
        if not series:
            continue
        xs = [r["epoch"] for r in series]
        ys = [r[metric] for r in series]
        plt.plot(xs, ys, marker="o", linewidth=2.2, markersize=4.5, color=color, label=spec.label)

    baseline = find_baseline(repo_root, task, metric, model_specs)
    if baseline is not None:
        plt.axhline(
            baseline,
            color="#222222",
            linestyle="--",
            linewidth=1.8,
            alpha=0.85,
            label=f"official ep399 baseline ({baseline:.2f})",
        )

    metric_label = "acc@1" if metric == "acc1" else "acc@5"
    plt.title(f"KNN {metric_label} Curves - {task}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(alpha=0.28)
    plt.legend(frameon=True)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", default="eval_results/knn_three_model_curves")
    parser.add_argument("--metrics", default="acc1,acc5", help="Comma-separated subset of acc1,acc5")
    parser.add_argument("--include-n1", action="store_true", help="Include trained_models_full1TB_np65536_n1 outputs")
    parser.add_argument("--include-weakaug", action="store_true", help="Include trained_models_full1TB_np65536_weakaug outputs")
    parser.add_argument("--prefix", default=None, help="Output filename prefix")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = repo_root / args.out_dir
    model_specs = MODEL_SPECS
    if args.include_n1:
        model_specs = model_specs + (N1_MODEL_SPEC,)
    if args.include_weakaug:
        model_specs = model_specs + (WEAKAUG_MODEL_SPEC,)
    if args.prefix:
        prefix = args.prefix
    elif args.include_n1 and args.include_weakaug:
        prefix = "knn_five_model"
    elif args.include_n1:
        prefix = "knn_four_model"
    else:
        prefix = "knn_three_model"
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    bad = [m for m in metrics if m not in {"acc1", "acc5"}]
    if bad:
        raise ValueError(f"Unsupported metrics: {bad}")

    rows: List[Dict[str, object]] = []
    for spec in model_specs:
        rows.extend(iter_epoch_rows(repo_root, spec))

    write_summary(rows, out_dir / f"{prefix}_metrics.csv")
    for task in TASKS:
        for metric in metrics:
            plot_task(rows, repo_root, task, metric, out_dir / f"{prefix}_{task}_{metric}.png", model_specs)

    print(f"Wrote {len(rows)} metric rows to {out_dir}")


if __name__ == "__main__":
    main()
