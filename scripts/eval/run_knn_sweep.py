import argparse
import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import List, Optional


@dataclass
class TaskSpec:
    name: str
    dataset: str
    config_path: str
    config_name: str
    train_path: str
    val_path: str
    sample_ratio: float
    batch_size: int
    num_workers: int


@dataclass
class RunResult:
    task: str
    ckpt_id: str
    ckpt_path: str
    output_csv: str
    status: str
    acc1: Optional[float]
    acc5: Optional[float]
    stderr_tail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=str, required=True)
    parser.add_argument("--benchmark-root", type=str, default="/mnt/huawei_deepcad/benchmark")
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--official-ckpt", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="cyclops,bbbc048")
    parser.add_argument("--gpus", type=str, default="0,1,2,5,6,7")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="eval_results/knn_sweep")
    return parser.parse_args()


def build_tasks(args: argparse.Namespace) -> List[TaskSpec]:
    tasks: List[TaskSpec] = []
    selected = {t.strip().lower() for t in args.tasks.split(",") if t.strip()}
    medmnist_root = f"{args.benchmark_root}/Classification/MedMNIST"
    if "cyclops" in selected:
        tasks.append(
            TaskSpec(
                name="cyclops",
                dataset="cyclops",
                config_path="scripts/knn/cyclops",
                config_name="dino_chada_vit_moyen.yaml",
                train_path=f"{args.repo_root}/eval_data/cyclops",
                val_path=f"{args.repo_root}/eval_data/cyclops",
                sample_ratio=args.sample_ratio,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        )
    if "bbbc048" in selected:
        tasks.append(
            TaskSpec(
                name="bbbc048",
                dataset="bbbc048",
                config_path="scripts/knn/bbbc048",
                config_name="dino_chada_vit_moyen.yaml",
                train_path=f"{args.repo_root}/eval_data/bbbc048",
                val_path=f"{args.repo_root}/eval_data/bbbc048",
                sample_ratio=args.sample_ratio,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        )
    medmnist_tasks = {
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
    }
    for task_name in sorted(selected & medmnist_tasks):
        tasks.append(
            TaskSpec(
                name=task_name,
                dataset=task_name,
                config_path="scripts/knn/bbbc048",
                config_name="dino_chada_vit_moyen.yaml",
                train_path=medmnist_root,
                val_path=medmnist_root,
                sample_ratio=args.sample_ratio,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        )
    return tasks


def parse_epoch_from_name(name: str) -> Optional[int]:
    m = re.search(r"ep_(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def parse_knn_csv(csv_path: Path) -> (float, float):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    if "acc@1" in row:
        return float(row["acc@1"]), float(row["acc@5"])
    return float(row["mean_auroc"]), float(row["mean_ap"])


def run_one(
    repo_root: Path,
    task: TaskSpec,
    ckpt_id: str,
    ckpt_path: str,
    gpu: str,
    out_dir: Path,
) -> RunResult:
    run_name = f"knn_{task.name}_{ckpt_id}"
    output_csv = repo_root / f"{run_name}_knn_offline_eval.csv"
    if output_csv.exists():
        try:
            acc1, acc5 = parse_knn_csv(output_csv)
            return RunResult(
                task=task.name,
                ckpt_id=ckpt_id,
                ckpt_path=ckpt_path,
                output_csv=str(output_csv),
                status="ok",
                acc1=acc1,
                acc5=acc5,
                stderr_tail="cached",
            )
        except Exception:
            # Corrupt/partial cached CSV (e.g. truncated by a killed run) -> drop and recompute
            # instead of letting parse errors crash the whole sweep.
            try:
                output_csv.unlink()
            except OSError:
                pass
    backbone_max_channels = "10" if "official" in ckpt_id.lower() else "8"

    cmd = [
        sys.executable,
        "main_knn.py",
        "--config-path",
        task.config_path,
        "--config-name",
        task.config_name,
        f"name={run_name}",
        f"weights_init={ckpt_path}",
        f"data.dataset={task.dataset}",
        f"data.train_path={task.train_path}",
        f"data.val_path={task.val_path}",
        f"+data.sample_ratio={task.sample_ratio}",
        f"optimizer.batch_size={task.batch_size}",
        f"data.num_workers={task.num_workers}",
        f"backbone.kwargs.max_number_channels={backbone_max_channels}",
        "devices=[0]",
        "knn_eval_offline.k=[20]",
        "knn_eval_offline.temperature=[0.07]",
        "knn_eval_offline.distance_function=[cosine]",
        "knn_eval_offline.feature_type=[backbone]",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["CHADAVIT_DISABLE_CUDNN"] = "1"
    env["WANDB_ERROR_REPORTING"] = "false"
    env["WANDB_SILENT"] = "true"
    env["MPLCONFIGDIR"] = "/tmp/mpl"
    env["NUMBA_CACHE_DIR"] = "/tmp/numba"

    logs_dir = out_dir / "logs" / task.name
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{ckpt_id}.log"
    with log_path.open("w") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=logf,
            stderr=logf,
            text=True,
        )

    log_tail = ""
    try:
        log_tail = log_path.read_text(errors="ignore")[-5000:]
    except Exception:
        pass

    if proc.returncode != 0:
        return RunResult(
            task=task.name,
            ckpt_id=ckpt_id,
            ckpt_path=ckpt_path,
            output_csv=str(output_csv),
            status="failed",
            acc1=None,
            acc5=None,
            stderr_tail=log_tail,
        )

    if not output_csv.exists():
        return RunResult(
            task=task.name,
            ckpt_id=ckpt_id,
            ckpt_path=ckpt_path,
            output_csv=str(output_csv),
            status="failed_missing_csv",
            acc1=None,
            acc5=None,
            stderr_tail=log_tail,
        )

    acc1, acc5 = parse_knn_csv(output_csv)
    return RunResult(
        task=task.name,
        ckpt_id=ckpt_id,
        ckpt_path=ckpt_path,
        output_csv=str(output_csv),
        status="ok",
        acc1=acc1,
        acc5=acc5,
        stderr_tail="",
    )


def write_summary(results: List[RunResult], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [
                "task",
                "ckpt_id",
                "ckpt_path",
                "status",
                "acc1",
                "acc5",
                "output_csv",
                "stderr_tail",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.task,
                    r.ckpt_id,
                    r.ckpt_path,
                    r.status,
                    "" if r.acc1 is None else f"{r.acc1:.8f}",
                    "" if r.acc5 is None else f"{r.acc5:.8f}",
                    r.output_csv,
                    r.stderr_tail,
                ]
            )


def plot_task(results: List[RunResult], task: str, out_png: Path):
    import matplotlib.pyplot as plt

    ok = [r for r in results if r.task == task and r.status == "ok"]
    if not ok:
        return
    official_rows = [r for r in ok if "official" in r.ckpt_id.lower()]
    curve_rows = [r for r in ok if "official" not in r.ckpt_id.lower()]
    if not curve_rows:
        return

    def sort_key(r: RunResult):
        ep = parse_epoch_from_name(r.ckpt_id)
        if ep is None:
            return (10**9, r.ckpt_id)
        return (ep, r.ckpt_id)

    curve_rows = sorted(curve_rows, key=sort_key)
    x = list(range(len(curve_rows)))
    labels = [r.ckpt_id for r in curve_rows]
    acc1 = [r.acc1 for r in curve_rows]
    acc5 = [r.acc5 for r in curve_rows]

    plt.figure(figsize=(14, 5))
    line1, = plt.plot(x, acc1, marker="o", label="acc@1")
    line5, = plt.plot(x, acc5, marker="o", label="acc@5")
    if official_rows:
        # Use the first official checkpoint as a reference baseline.
        o = official_rows[0]
        plt.axhline(
            y=o.acc1,
            linestyle="--",
            linewidth=1.5,
            color=line1.get_color(),
            alpha=0.8,
            label=f"{o.ckpt_id} acc1 baseline",
        )
        plt.axhline(
            y=o.acc5,
            linestyle="--",
            linewidth=1.5,
            color=line5.get_color(),
            alpha=0.8,
            label=f"{o.ckpt_id} acc5 baseline",
        )
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title(f"KNN Sweep - {task}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def main():
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_items = []
    for p in sorted(ckpt_dir.glob("ep_*.ckpt")):
        ckpt_id = p.stem
        ckpt_items.append((ckpt_id, str(p)))
    ckpt_items.append(("official_ep399", args.official_ckpt))

    tasks = build_tasks(args)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise ValueError("No GPUs provided.")

    results: List[RunResult] = []
    # Shared work queue -> dynamic load balancing: each GPU worker pulls the next
    # checkpoint as soon as it is free, so all GPUs stay busy until the queue drains
    # (avoids the end-of-task tail where statically-assigned GPUs sit idle).
    work: "Queue" = Queue()
    for task in tasks:
        for ckpt_id, ckpt_path in ckpt_items:
            work.put((task, ckpt_id, ckpt_path))

    def gpu_worker(gpu: str) -> List[RunResult]:
        gpu_results: List[RunResult] = []
        while True:
            try:
                task, ckpt_id, ckpt_path = work.get_nowait()
            except Empty:
                break
            r = run_one(repo_root, task, ckpt_id, ckpt_path, gpu, out_dir)
            gpu_results.append(r)
            print(
                f"[gpu {gpu}] [{r.status}] task={r.task} ckpt={r.ckpt_id} "
                f"acc1={r.acc1} acc5={r.acc5}",
                flush=True,
            )
        return gpu_results

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futures = [ex.submit(gpu_worker, gpu) for gpu in gpus]
        for fut in as_completed(futures):
            results.extend(fut.result())

    summary_csv = out_dir / "knn_sweep_summary.csv"
    write_summary(results, summary_csv)
    print(f"Summary saved to {summary_csv}")

    for task in {r.task for r in results}:
        plot_path = out_dir / f"knn_sweep_{task}.png"
        plot_task(results, task, plot_path)
        print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
