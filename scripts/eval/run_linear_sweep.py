import argparse
import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt


@dataclass
class TaskSpec:
    name: str
    config_path: str
    config_name: str
    train_path: str
    val_path: str
    img_channels: int
    max_img_channels: int


@dataclass
class RunResult:
    task: str
    ckpt_id: str
    ckpt_path: str
    status: str
    val_acc1: Optional[float]
    val_acc5: Optional[float]
    log_path: str
    stderr_tail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=str, required=True)
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="cyclops,bbbc048")
    parser.add_argument("--gpus", type=str, default="0,1,2")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--out-dir", type=str, default="eval_results/linear_sweep")
    return parser.parse_args()


def build_tasks(args: argparse.Namespace) -> List[TaskSpec]:
    selected = {x.strip().lower() for x in args.tasks.split(",") if x.strip()}
    tasks: List[TaskSpec] = []
    if "cyclops" in selected:
        tasks.append(
            TaskSpec(
                name="cyclops",
                config_path="scripts/linear/cyclops",
                config_name="dino_chada_vit_moyen.yaml",
                train_path=f"{args.repo_root}/eval_data/cyclops",
                val_path=f"{args.repo_root}/eval_data/cyclops",
                img_channels=2,
                max_img_channels=2,
            )
        )
    if "bbbc048" in selected:
        tasks.append(
            TaskSpec(
                name="bbbc048",
                config_path="scripts/linear/bbbc048",
                config_name="dino_chada_vit_moyen.yaml",
                train_path=f"{args.repo_root}/eval_data/bbbc048",
                val_path=f"{args.repo_root}/eval_data/bbbc048",
                img_channels=3,
                max_img_channels=3,
            )
        )
    return tasks


def parse_epoch(ckpt_id: str) -> Optional[int]:
    m = re.search(r"ep_(\d+)", ckpt_id)
    if m:
        return int(m.group(1))
    return None


def parse_final_metrics(log_text: str) -> Optional[tuple]:
    m = re.search(r"FINAL_METRICS val_acc1=([0-9.]+) val_acc5=([0-9.]+)", log_text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def run_one(
    repo_root: Path,
    task: TaskSpec,
    ckpt_id: str,
    ckpt_path: str,
    gpu: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> RunResult:
    run_name = f"linear_{task.name}_{ckpt_id}"
    logs_dir = out_dir / "logs" / task.name
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{ckpt_id}.log"

    if log_path.exists():
        existing = log_path.read_text(errors="ignore")
        cached_metrics = parse_final_metrics(existing)
        if cached_metrics is not None:
            return RunResult(
                task=task.name,
                ckpt_id=ckpt_id,
                ckpt_path=ckpt_path,
                status="cached",
                val_acc1=cached_metrics[0],
                val_acc5=cached_metrics[1],
                log_path=str(log_path),
                stderr_tail="",
            )

    backbone_max_channels = "10" if "official" in ckpt_id else "8"

    cmd = [
        sys.executable,
        "main_linear.py",
        "--config-path",
        task.config_path,
        "--config-name",
        task.config_name,
        f"name={run_name}",
        f"pretrained_feature_extractor={ckpt_path}",
        "pretrain_method=dino",
        "backbone.name=vit_channels",
        "backbone.kwargs.embed_dim=192",
        "backbone.kwargs.patch_size=16",
        "backbone.kwargs.return_all_tokens=False",
        f"backbone.kwargs.max_number_channels={backbone_max_channels}",
        "channels_strategy=multi_channels",
        "strategy=auto",
        "accelerator=gpu",
        "devices=[0]",
        f"max_epochs={args.epochs}",
        f"optimizer.batch_size={args.batch_size}",
        f"data.num_workers={args.num_workers}",
        "checkpoint.enabled=False",
        "auto_resume.enabled=False",
        "wandb.enabled=False",
        f"data.train_path={task.train_path}",
        f"data.val_path={task.val_path}",
        f"data.img_channels={task.img_channels}",
        f"data.max_img_channels={task.max_img_channels}",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["CHADAVIT_DISABLE_CUDNN"] = "1"
    env["WANDB_ERROR_REPORTING"] = "false"
    env["WANDB_SILENT"] = "true"
    env["MPLCONFIGDIR"] = "/tmp/mpl"
    env["NUMBA_CACHE_DIR"] = "/tmp/numba"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    with log_path.open("w") as lf:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=lf,
            stderr=lf,
            text=True,
        )

    log_text = log_path.read_text(errors="ignore")
    metrics = parse_final_metrics(log_text)
    if proc.returncode != 0 or metrics is None:
        return RunResult(
            task=task.name,
            ckpt_id=ckpt_id,
            ckpt_path=ckpt_path,
            status="failed",
            val_acc1=None,
            val_acc5=None,
            log_path=str(log_path),
            stderr_tail=log_text[-5000:],
        )

    return RunResult(
        task=task.name,
        ckpt_id=ckpt_id,
        ckpt_path=ckpt_path,
        status="ok",
        val_acc1=metrics[0],
        val_acc5=metrics[1],
        log_path=str(log_path),
        stderr_tail="",
    )


def write_summary(results: List[RunResult], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(
            [
                "task",
                "ckpt_id",
                "ckpt_path",
                "status",
                "val_acc1",
                "val_acc5",
                "log_path",
                "stderr_tail",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.task,
                    r.ckpt_id,
                    r.ckpt_path,
                    r.status,
                    "" if r.val_acc1 is None else f"{r.val_acc1:.8f}",
                    "" if r.val_acc5 is None else f"{r.val_acc5:.8f}",
                    r.log_path,
                    r.stderr_tail,
                ]
            )


def plot_task(results: List[RunResult], task: str, out_png: Path):
    rows = [r for r in results if r.task == task and r.status in ("ok", "cached")]
    if not rows:
        return
    official_rows = [r for r in rows if "official" in r.ckpt_id.lower()]
    curve_rows = [r for r in rows if "official" not in r.ckpt_id.lower()]
    if not curve_rows:
        return

    curve_rows.sort(
        key=lambda x: (10**9 if parse_epoch(x.ckpt_id) is None else parse_epoch(x.ckpt_id))
    )
    xs = [parse_epoch(r.ckpt_id) for r in curve_rows]
    acc1 = [r.val_acc1 for r in curve_rows]
    acc5 = [r.val_acc5 for r in curve_rows]
    plt.figure(figsize=(12, 5))
    line1, = plt.plot(xs, acc1, marker="o", label="val_acc1")
    line5, = plt.plot(xs, acc5, marker="o", label="val_acc5")
    if official_rows:
        # Use the first official checkpoint as a reference baseline.
        o = official_rows[0]
        plt.axhline(
            y=o.val_acc1,
            linestyle="--",
            linewidth=1.5,
            color=line1.get_color(),
            alpha=0.8,
            label=f"{o.ckpt_id} acc1 baseline",
        )
        plt.axhline(
            y=o.val_acc5,
            linestyle="--",
            linewidth=1.5,
            color=line5.get_color(),
            alpha=0.8,
            label=f"{o.ckpt_id} acc5 baseline",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Linear Probe Sweep - {task}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_items = [(p.stem, str(p)) for p in sorted(ckpt_dir.glob("ep_*.ckpt"))]

    tasks = build_tasks(args)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise ValueError("No GPUs provided.")

    assignments = {g: [] for g in gpus}
    idx = 0
    for task in tasks:
        for ckpt_id, ckpt_path in ckpt_items:
            gpu = gpus[idx % len(gpus)]
            assignments[gpu].append((task, ckpt_id, ckpt_path))
            idx += 1

    def run_gpu_queue(gpu: str) -> List[RunResult]:
        out: List[RunResult] = []
        for task, ckpt_id, ckpt_path in assignments[gpu]:
            r = run_one(repo_root, task, ckpt_id, ckpt_path, gpu, out_dir, args)
            out.append(r)
            print(
                f"[gpu {gpu}] [{r.status}] task={r.task} ckpt={r.ckpt_id} "
                f"val_acc1={r.val_acc1} val_acc5={r.val_acc5}",
                flush=True,
            )
        return out

    results: List[RunResult] = []
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(gpus))) as ex:
        futures = [ex.submit(run_gpu_queue, g) for g in gpus]
        for fut in as_completed(futures):
            results.extend(fut.result())

    summary_csv = out_dir / "linear_sweep_summary.csv"
    write_summary(results, summary_csv)
    print(f"Summary saved to {summary_csv}")
    for task in {r.task for r in results}:
        p = out_dir / f"linear_sweep_{task}.png"
        plot_task(results, task, p)
        print(f"Plot saved to {p}")


if __name__ == "__main__":
    main()
