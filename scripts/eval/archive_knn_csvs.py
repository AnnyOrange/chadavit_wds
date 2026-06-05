import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Set


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


def parse_epoch(ckpt_id: str) -> str:
    match = re.search(r"ep_(\d+)", ckpt_id)
    return match.group(1) if match else ""


def model_group(ckpt_id: str, forced_group: Optional[str] = None) -> str:
    if ckpt_id == "official_ep399":
        return "official_ep399"
    if forced_group:
        return forced_group
    if "dinov2" in ckpt_id.lower():
        return "dinov2crop_bs4096"
    if re.fullmatch(r"ep_\d+_run\d+", ckpt_id):
        return "dinov2crop_bs4096"
    if re.fullmatch(r"ep_\d+", ckpt_id):
        return "weakaug"
    return "legacy_or_other"


def read_metrics(path: Path) -> Dict[str, str]:
    try:
        with path.open("r", newline="") as f:
            row = next(csv.DictReader(f))
    except Exception:
        return {
            "primary_metric": "unreadable",
            "primary": "",
            "secondary_metric": "unreadable",
            "secondary": "",
        }
    if "mean_auroc" in row:
        return {
            "primary_metric": "mean_auroc",
            "primary": row.get("mean_auroc", ""),
            "secondary_metric": "mean_ap",
            "secondary": row.get("mean_ap", ""),
        }
    return {
        "primary_metric": "acc@1",
        "primary": row.get("acc@1", ""),
        "secondary_metric": "acc@5",
        "secondary": row.get("acc@5", ""),
    }


def read_summary_outputs(path: Path, repo_root: Path) -> Set[Path]:
    outputs: Set[Path] = set()
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            output = row.get("output_csv", "")
            if not output:
                continue
            output_path = Path(output)
            if not output_path.is_absolute():
                output_path = repo_root / output_path
            outputs.add(output_path.resolve())
    return outputs


def write_readme(path: Path) -> None:
    path.write_text(
        """# CSV Archive

This folder stores raw KNN CSV outputs moved out of the repository root.

## Layout

- `knn_offline_eval/dinov2crop_bs4096/<task>/`: KNN CSVs for `trained_models_webds_shuffle_dinov2crop_bs4096`.
- `knn_offline_eval/weakaug/<task>/`: KNN CSVs for `trained_models_webds_shuffle_weakaug`.
- `knn_offline_eval/official_ep399/<task>/`: official checkpoint baseline CSVs.
- `knn_offline_eval/legacy_or_other/<task>/`: older one-off or legacy CSV names that do not match the current naming convention.
- `knn_offline_eval/manifest.csv`: index of every archived CSV with task, checkpoint, epoch, metric names, metric values, and archived path.

## Naming Convention

Raw KNN CSVs are produced by `main_knn.py` in the repository root using:

```text
knn_<task>_<ckpt_id>_knn_offline_eval.csv
```

Current checkpoint IDs:

- `ep_<epoch>_run<run>`: dinov2crop bs4096 checkpoints.
- `ep_<epoch>`: weakaug checkpoints.
- `ep_<epoch>` archived with `--model-group ratio20_dinov2crop_bs4096`: ratio20 dinov2crop checkpoints.
- `official_ep399`: official baseline checkpoint.

## Future Cleanup Workflow

After running new KNN sweeps, new raw CSVs will appear in the repo root again. To archive them into this structure, run:

```bash
python scripts/eval/archive_knn_csvs.py --repo-root .
```

For checkpoint IDs that are ambiguous, archive immediately after each sweep with an explicit model group:

```bash
python scripts/eval/archive_knn_csvs.py --repo-root . --model-group ratio20_dinov2crop_bs4096
```

To avoid archiving unrelated root CSVs, pass the sweep summary:

```bash
python scripts/eval/archive_knn_csvs.py --repo-root . --summary-csv eval_results/knn_sweep/knn_sweep_summary.csv --model-group ratio20_dinov2crop_bs4096
```

Then regenerate the final summary plots/tables if needed:

```bash
MPLCONFIGDIR=/tmp/mpl python scripts/eval/finalize_knn_summary.py --repo-root . --out-dir eval_results/knn_final_summary
```

The final user-facing summary remains:

```text
eval_results/knn_final_summary/
  fig/
  data/
```
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--archive-dir", default="csv_archive/knn_offline_eval")
    parser.add_argument(
        "--model-group",
        choices=[
            "dinov2crop_bs4096",
            "ratio10_dinov2crop_bs4096",
            "ratio20_dinov2crop_bs4096",
            "semdedup_ratio20_dinov2crop_bs4096",
            "weakaug",
            "legacy_or_other",
        ],
        help="Assign all non-official root CSVs to this group. Useful when ckpt IDs like ep_0 are ambiguous.",
    )
    parser.add_argument(
        "--summary-csv",
        help="Only archive OK output_csv paths listed in this run_knn_sweep.py summary.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    archive_dir = (repo_root / args.archive_dir).resolve()
    archive_dir.mkdir(parents=True, exist_ok=True)

    existing_manifest = archive_dir / "manifest.csv"
    manifest_rows = []
    if existing_manifest.exists():
        with existing_manifest.open("r", newline="") as f:
            manifest_rows.extend(csv.DictReader(f))

    allowed_paths: Optional[Set[Path]] = None
    if args.summary_csv:
        summary_csv = Path(args.summary_csv)
        if not summary_csv.is_absolute():
            summary_csv = repo_root / summary_csv
        allowed_paths = read_summary_outputs(summary_csv, repo_root)

    root_csvs = sorted(repo_root.glob("knn_*_knn_offline_eval.csv"))
    moved = 0
    skipped = 0
    for src in root_csvs:
        if allowed_paths is not None and src.resolve() not in allowed_paths:
            skipped += 1
            continue
        parsed = parse_task_ckpt(src)
        if not parsed:
            skipped += 1
            continue
        group = model_group(parsed["ckpt_id"], args.model_group)
        dst_dir = archive_dir / group / parsed["task"]
        dst = dst_dir / src.name
        if dst.exists():
            skipped += 1
            if not args.dry_run:
                src.unlink()
            continue
        metrics = read_metrics(src)
        row = {
            "model_group": group,
            "task": parsed["task"],
            "ckpt_id": parsed["ckpt_id"],
            "epoch": parse_epoch(parsed["ckpt_id"]),
            **metrics,
            "archived_path": str(dst.relative_to(repo_root)),
        }
        manifest_rows.append(row)
        moved += 1
        if not args.dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    if not args.dry_run:
        fieldnames = [
            "model_group",
            "task",
            "ckpt_id",
            "epoch",
            "primary_metric",
            "primary",
            "secondary_metric",
            "secondary",
            "archived_path",
        ]
        manifest_rows.sort(key=lambda r: (r["model_group"], r["task"], r["ckpt_id"]))
        with existing_manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
        write_readme(archive_dir.parent / "README.md")

    print(f"root_csvs={len(root_csvs)} moved={moved} skipped={skipped} archive_dir={archive_dir}")


if __name__ == "__main__":
    main()
