import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


A_MODES = [
    "independent_real_decay",
    "real_rotation2x2",
]
SEEDS = [1234, 2345, 3456]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the real-valued SSM comparison matrix.")
    parser.add_argument(
        "--task",
        default="spiking-heidelberg-digits",
        help="Hydra task configuration to train (default: SHD medium benchmark).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--a-modes", nargs="+", default=A_MODES)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--jax-visible-devices",
        default="0",
        help="CUDA device IDs visible to each training subprocess (default: 0).",
    )
    parser.add_argument("--output-root", default="outputs/shd_medium_real_ssm_comparison")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def metric_value(metrics, key):
    return metrics.get(key, metrics.get(key.replace(" ", "_")))


def load_metrics(run_dir):
    metrics_dir = run_dir / "metrics"
    best_eval_path = metrics_dir / "best_eval.json"
    test_path = metrics_dir / "test.json"
    best_eval = json.loads(best_eval_path.read_text()) if best_eval_path.exists() else {}
    test = json.loads(test_path.read_text()) if test_path.exists() else {}
    return {
        "best_val_accuracy": metric_value(best_eval, "Performance/Validation accuracy"),
        "best_val_loss": metric_value(best_eval, "Performance/Validation loss"),
        "test_accuracy": metric_value(test, "Performance/Test accuracy"),
        "test_loss": metric_value(test, "Performance/Test loss"),
    }


def flatten_params(tree, prefix=()):
    if hasattr(tree, "items"):
        for key, value in tree.items():
            yield from flatten_params(value, prefix + (key,))
    else:
        yield prefix, tree


def describe_values(values, prefix):
    if not values:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
        }

    flat = [float(value) for array in values for value in array.reshape(-1)]
    return {
        f"{prefix}_mean": statistics.mean(flat),
        f"{prefix}_std": statistics.stdev(flat) if len(flat) > 1 else 0.0,
        f"{prefix}_min": min(flat),
        f"{prefix}_max": max(flat),
    }


def load_parameter_stats(run_dir):
    import jax
    import numpy as np
    from flax.training import checkpoints

    state = checkpoints.restore_checkpoint(run_dir / "checkpoints", target=None)
    params = state["params"] if isinstance(state, dict) else state.params
    alphas = []
    omegas = []
    param_count = 0

    for path, value in flatten_params(params):
        name = path[-1]
        array = np.asarray(value)
        param_count += array.size * (2 if np.iscomplexobj(array) else 1)
        if name == "raw_alpha":
            alphas.append(np.asarray(jax.nn.softplus(value) + 1e-6))
        elif name == "omega":
            omegas.append(array)

    stats = {"param_count": param_count}
    stats.update(describe_values(alphas, "alpha"))
    stats.update(describe_values(omegas, "omega"))
    return stats


def summarize(rows):
    summary = {}
    for mode in sorted({row["a_mode"] for row in rows}):
        mode_rows = [row for row in rows if row["a_mode"] == mode]
        summary[mode] = {}
        for key in [
            "best_val_accuracy",
            "best_val_loss",
            "test_accuracy",
            "test_loss",
            "alpha_mean",
            "alpha_std",
            "omega_mean",
            "omega_std",
            "param_count",
        ]:
            values = [row[key] for row in mode_rows if row[key] is not None]
            if values:
                summary[mode][f"{key}_mean"] = statistics.mean(values)
                summary[mode][f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return summary


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = repo_root / args.output_root
    rows = []

    for a_mode in args.a_modes:
        for seed in args.seeds:
            run_dir = output_root / a_mode / f"seed_{seed}"
            cmd = [
                sys.executable,
                "run_training.py",
                f"task={args.task}",
                f"seed={seed}",
                f"training.num_epochs={args.epochs}",
                f"training.num_workers={args.num_workers}",
                f"model.ssm.a_mode={a_mode}",
                f"output_dir={run_dir}",
                "logging.interval=1000",
            ]
            env = os.environ.copy()
            if args.jax_visible_devices:
                env["CUDA_VISIBLE_DEVICES"] = args.jax_visible_devices
            print(" ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=repo_root, check=True, env=env)

            row = {
                "a_mode": a_mode,
                "seed": seed,
                "run_dir": str(run_dir),
            }
            row.update(load_metrics(run_dir) if not args.dry_run else {
                "best_val_accuracy": None,
                "best_val_loss": None,
                "test_accuracy": None,
                "test_loss": None,
            })
            row.update(load_parameter_stats(run_dir) if not args.dry_run else {
                "alpha_mean": None,
                "alpha_std": None,
                "alpha_min": None,
                "alpha_max": None,
                "omega_mean": None,
                "omega_std": None,
                "omega_min": None,
                "omega_max": None,
                "param_count": None,
            })
            rows.append(row)

    if args.dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    rows_path = output_root / "real_ssm_comparison_rows.csv"
    with rows_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    (output_root / "real_ssm_comparison_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"Wrote {rows_path}")
    print(f"Wrote {output_root / 'real_ssm_comparison_summary.json'}")


if __name__ == "__main__":
    main()
