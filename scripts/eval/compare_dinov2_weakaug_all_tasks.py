import argparse
import csv
import re
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

DINOV2_LABEL = "trained_models_webds_shuffle_dinov2crop_bs4096"
WEAKAUG_LABEL = "trained_models_webds_shuffle_weakaug"
OFFICIAL_LABEL = "official_ep399"


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
    if not match:
        return None
    return int(match.group(1))


def parse_task_ckpt(path: Path) -> Optional[Dict[str, str]]:
    name = path.name
    for task in TASKS:
        prefix = f"knn_{task}_"
        suffix = "_knn_offline_eval.csv"
        if name.startswith(prefix) and name.endswith(suffix):
            return {"task": task, "ckpt_id": name[len(prefix) : -len(suffix)]}
    return None


def model_from_ckpt(ckpt_id: str) -> Optional[str]:
    lowered = ckpt_id.lower()
    if lowered == OFFICIAL_LABEL:
        return OFFICIAL_LABEL
    if "dinov2" in lowered:
        return DINOV2_LABEL
    if re.fullmatch(r"ep_\d+", ckpt_id):
        return WEAKAUG_LABEL
    return None


def collect_rows(repo_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(repo_root.glob("knn_*_knn_offline_eval.csv")):
        parsed = parse_task_ckpt(path)
        if not parsed:
            continue
        model = model_from_ckpt(parsed["ckpt_id"])
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
    if not candidates:
        return None
    return max(candidates, key=lambda r: float(r["primary"]))


def build_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    for task in TASKS:
        dinov2 = best_for(rows, task, DINOV2_LABEL)
        weakaug = best_for(rows, task, WEAKAUG_LABEL)
        official = best_for(rows, task, OFFICIAL_LABEL)
        if not dinov2 and not weakaug and not official:
            continue

        primary_metric = "acc@1"
        secondary_metric = "acc@5"
        for item in (dinov2, weakaug, official):
            if item:
                primary_metric = str(item["primary_metric"])
                secondary_metric = str(item["secondary_metric"])
                break

        def value(item: Optional[Dict[str, object]], key: str) -> object:
            return "" if item is None else item[key]

        dinov2_primary = float(dinov2["primary"]) if dinov2 else None
        weakaug_primary = float(weakaug["primary"]) if weakaug else None
        official_primary = float(official["primary"]) if official else None
        summary.append(
            {
                "task": task,
                "primary_metric": primary_metric,
                "secondary_metric": secondary_metric,
                "dinov2_best_ckpt": value(dinov2, "ckpt_id"),
                "dinov2_best_epoch": value(dinov2, "epoch"),
                "dinov2_best_primary": "" if dinov2_primary is None else f"{dinov2_primary:.8f}",
                "dinov2_best_secondary": "" if dinov2 is None else f"{float(dinov2['secondary']):.8f}",
                "weakaug_best_ckpt": value(weakaug, "ckpt_id"),
                "weakaug_best_epoch": value(weakaug, "epoch"),
                "weakaug_best_primary": "" if weakaug_primary is None else f"{weakaug_primary:.8f}",
                "weakaug_best_secondary": "" if weakaug is None else f"{float(weakaug['secondary']):.8f}",
                "official_primary": "" if official_primary is None else f"{official_primary:.8f}",
                "official_secondary": "" if official is None else f"{float(official['secondary']):.8f}",
                "dinov2_minus_official": ""
                if dinov2_primary is None or official_primary is None
                else f"{dinov2_primary - official_primary:.8f}",
                "weakaug_minus_official": ""
                if weakaug_primary is None or official_primary is None
                else f"{weakaug_primary - official_primary:.8f}",
                "dinov2_minus_weakaug": ""
                if dinov2_primary is None or weakaug_primary is None
                else f"{dinov2_primary - weakaug_primary:.8f}",
            }
        )
    return summary


def plot_summary(summary: List[Dict[str, object]], out_png: Path) -> None:
    import matplotlib.pyplot as plt

    tasks = [str(r["task"]) for r in summary]
    x = list(range(len(tasks)))
    width = 0.26

    def as_float(row: Dict[str, object], key: str) -> float:
        value = row[key]
        return float("nan") if value == "" else float(value)

    official = [as_float(r, "official_primary") for r in summary]
    weakaug = [as_float(r, "weakaug_best_primary") for r in summary]
    dinov2 = [as_float(r, "dinov2_best_primary") for r in summary]

    plt.figure(figsize=(18, 7))
    plt.bar([i - width for i in x], official, width=width, label="official ep399", color="#555555")
    plt.bar(x, weakaug, width=width, label="weakaug best", color="#b3284d")
    plt.bar([i + width for i in x], dinov2, width=width, label="dinov2crop bs4096 best", color="#2a7f62")
    plt.xticks(x, tasks, rotation=35, ha="right")
    plt.ylabel("Primary metric (%)")
    plt.title("KNN Best Checkpoint Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", default="eval_results/weakaug_comparison")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = repo_root / args.out_dir
    rows = collect_rows(repo_root)
    rows = sorted(rows, key=lambda r: (str(r["task"]), str(r["model"]), r["epoch"] or 10**9))

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
    write_csv(out_dir / "dinov2_weakaug_official_all_knn_metrics.csv", rows, detail_fields)

    summary = build_summary(rows)
    summary_fields = [
        "task",
        "primary_metric",
        "secondary_metric",
        "dinov2_best_ckpt",
        "dinov2_best_epoch",
        "dinov2_best_primary",
        "dinov2_best_secondary",
        "weakaug_best_ckpt",
        "weakaug_best_epoch",
        "weakaug_best_primary",
        "weakaug_best_secondary",
        "official_primary",
        "official_secondary",
        "dinov2_minus_official",
        "weakaug_minus_official",
        "dinov2_minus_weakaug",
    ]
    write_csv(out_dir / "dinov2_vs_weakaug_vs_official_best.csv", summary, summary_fields)
    plot_summary(summary, out_dir / "dinov2_vs_weakaug_vs_official_best.png")
    print(f"Wrote {out_dir / 'dinov2_weakaug_official_all_knn_metrics.csv'}")
    print(f"Wrote {out_dir / 'dinov2_vs_weakaug_vs_official_best.csv'}")
    print(f"Wrote {out_dir / 'dinov2_vs_weakaug_vs_official_best.png'}")


if __name__ == "__main__":
    main()
