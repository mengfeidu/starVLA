#!/usr/bin/env python3
"""Plot training loss curves from local wandb offline logs.

Reads metrics from wandb offline run directories produced by starVLA training
(WANDB_MODE=offline). Logs are typically saved under:

    {output_dir}/wandb/offline-run-*/

Usage:
    # Plot from a training output directory (auto-finds latest offline run)
    python examples/plot_training_loss.py \\
        /mnt/hdfs/data/.../Checkpoints/qwenfast_libero_all_qwen3vl4b_action

    # Plot a specific offline run, save PNG/CSV, apply smoothing
    python examples/plot_training_loss.py \\
        /path/to/wandb/offline-run-20250608_120000-xxx \\
        --output loss_curve.png --csv loss_history.csv --smooth 50

    # Compare multiple runs
    python examples/plot_training_loss.py run_a/wandb run_b/wandb --labels exp_a exp_b
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

LOSS_METRIC_PATTERNS = (
    re.compile(r"loss", re.IGNORECASE),
)
EVAL_METRIC_KEYS = ("mse_score",)
SKIP_KEYS = {"_step", "_runtime", "_timestamp", "epoch"}


def find_history_files(path: Path) -> List[Path]:
    """Resolve wandb-history.jsonl files from a user-provided path."""
    path = path.expanduser().resolve()

    if path.is_file() and path.name == "wandb-history.jsonl":
        return [path]

    candidates: List[Path] = []

    search_roots: List[Path] = [path]
    if path.is_dir():
        wandb_dir = path / "wandb"
        if wandb_dir.is_dir():
            search_roots.append(wandb_dir)

    for root in search_roots:
        if not root.is_dir():
            continue

        for pattern in ("offline-run-*", "run-*"):
            for run_dir in sorted(root.glob(pattern)):
                for rel in ("wandb-history.jsonl", "files/wandb-history.jsonl"):
                    history_file = run_dir / rel
                    if history_file.is_file():
                        candidates.append(history_file)

    # Deduplicate while preserving order (latest run last for default selection).
    seen = set()
    unique_candidates: List[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(resolved)

    return unique_candidates


def load_history(history_file: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Load wandb-history.jsonl into {metric_name: [(step, value), ...]}."""
    series: Dict[str, List[Tuple[int, float]]] = {}

    with history_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {history_file}:{line_no}") from exc

            step = record.get("_step")
            if step is None:
                continue

            for key, value in record.items():
                if key in SKIP_KEYS or key.startswith("_"):
                    continue
                if not isinstance(value, (int, float)):
                    continue
                series.setdefault(key, []).append((int(step), float(value)))

    for key in series:
        series[key].sort(key=lambda item: item[0])

    return series


def smooth_values(values: Sequence[float], window: int) -> List[float]:
    """Simple moving average; window=1 returns the original values."""
    if window <= 1:
        return list(values)
    if len(values) < window:
        return list(values)

    smoothed: List[float] = []
    running_sum = sum(values[:window])
    smoothed.append(running_sum / window)
    for idx in range(window, len(values)):
        running_sum += values[idx] - values[idx - window]
        smoothed.append(running_sum / window)
    return smoothed


def classify_metrics(metric_names: Iterable[str]) -> Tuple[List[str], List[str], List[str]]:
    """Split metrics into loss, eval, and other groups."""
    loss_metrics: List[str] = []
    eval_metrics: List[str] = []
    other_metrics: List[str] = []

    for name in sorted(metric_names):
        if any(pattern.search(name) for pattern in LOSS_METRIC_PATTERNS):
            loss_metrics.append(name)
        elif name in EVAL_METRIC_KEYS:
            eval_metrics.append(name)
        elif name.startswith("learning_rate/"):
            other_metrics.append(name)
        elif name.startswith("timing/"):
            other_metrics.append(name)
        else:
            other_metrics.append(name)

    return loss_metrics, eval_metrics, other_metrics


def export_csv(
    series: Dict[str, List[Tuple[int, float]]],
    metrics: Sequence[str],
    output_csv: Path,
) -> None:
    """Export selected metrics to a wide-format CSV indexed by step."""
    steps = sorted({step for metric in metrics for step, _ in series.get(metric, [])})
    rows = ["step," + ",".join(metrics)]
    for step in steps:
        row_values = [str(step)]
        for metric in metrics:
            value_map = dict(series.get(metric, []))
            value = value_map.get(step)
            row_values.append("" if value is None else f"{value:.8g}")
        rows.append(",".join(row_values))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")


def plot_series(
    run_series: List[Tuple[str, Dict[str, List[Tuple[int, float]]]]],
    metrics: Sequence[str],
    title: str,
    ylabel: str,
    output_path: Optional[Path],
    smooth_window: int,
) -> None:
    if not metrics:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, series in run_series:
        for metric in metrics:
            points = series.get(metric, [])
            if not points:
                continue
            steps = [step for step, _ in points]
            values = [value for _, value in points]
            if smooth_window > 1:
                values = smooth_values(values, smooth_window)
            plot_label = f"{label} / {metric}" if len(run_series) > 1 else metric
            ax.plot(steps, values, label=plot_label, linewidth=1.8)

    ax.set_title(title)
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        print(f"Saved plot: {output_path}")

    if output_path is None:
        plt.show()
    else:
        plt.close(fig)


def resolve_runs(
    input_paths: Sequence[str],
    labels: Optional[Sequence[str]],
    use_latest_only: bool,
) -> List[Tuple[str, Path, Dict[str, List[Tuple[int, float]]]]]:
    runs: List[Tuple[str, Path, Dict[str, List[Tuple[int, float]]]]] = []

    for idx, raw_path in enumerate(input_paths):
        history_files = find_history_files(Path(raw_path))
        if not history_files:
            raise FileNotFoundError(
                f"No wandb-history.jsonl found under `{raw_path}`. "
                "Expected an output dir, wandb dir, offline-run dir, or history file."
            )

        selected_files = [history_files[-1]] if use_latest_only else history_files
        for file_idx, history_file in enumerate(selected_files):
            if labels and idx < len(labels):
                label = labels[idx]
                if len(selected_files) > 1:
                    label = f"{label} ({file_idx + 1})"
            elif len(input_paths) == 1 and len(selected_files) == 1:
                label = history_file.parent.name
            else:
                label = history_file.parent.name

            series = load_history(history_file)
            if not series:
                raise ValueError(f"No metrics found in {history_file}")
            runs.append((label, history_file, series))
            print(f"Loaded `{label}` from {history_file}")

    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training loss curves from local wandb offline logs.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Training output dir, wandb dir, offline-run dir, or wandb-history.jsonl",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Legend labels for each input path (same length as inputs)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output PNG path. Default: <first_input>/training_loss.png",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV export path for loss metrics",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving-average window for loss curves (default: 1, no smoothing)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Explicit metric names to plot (default: auto-detect loss metrics)",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Plot every offline run under each input instead of only the latest one",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.labels and len(args.labels) != len(args.inputs):
        raise SystemExit("--labels must have the same length as input paths")

    runs = resolve_runs(args.inputs, args.labels, use_latest_only=not args.all_runs)
    run_series = [(label, series) for label, _, series in runs]

    all_metric_names = sorted({name for _, _, series in runs for name in series})
    loss_metrics, eval_metrics, other_metrics = classify_metrics(all_metric_names)

    if args.metrics:
        plot_metrics = list(args.metrics)
    else:
        plot_metrics = loss_metrics or [name for name in all_metric_names if name not in SKIP_KEYS]

    if not plot_metrics:
        raise SystemExit("No plottable metrics found in wandb history.")

    first_input = Path(args.inputs[0]).expanduser().resolve()
    default_output = first_input / "training_loss.png" if first_input.is_dir() else first_input.with_suffix(".png")
    output_path = Path(args.output).expanduser() if args.output else default_output

    plot_series(
        run_series=run_series,
        metrics=plot_metrics,
        title="Training Loss",
        ylabel="Loss",
        output_path=output_path,
        smooth_window=max(1, args.smooth),
    )

    if eval_metrics:
        eval_output = output_path.with_name(output_path.stem + "_eval" + output_path.suffix)
        plot_series(
            run_series=run_series,
            metrics=eval_metrics,
            title="Evaluation Metrics",
            ylabel="Score",
            output_path=eval_output,
            smooth_window=1,
        )

    lr_metrics = [name for name in other_metrics if name.startswith("learning_rate/")]
    if lr_metrics:
        lr_output = output_path.with_name(output_path.stem + "_lr" + output_path.suffix)
        plot_series(
            run_series=run_series,
            metrics=lr_metrics,
            title="Learning Rate",
            ylabel="LR",
            output_path=lr_output,
            smooth_window=1,
        )

    if args.csv:
        csv_path = Path(args.csv).expanduser()
        export_csv(runs[0][2], plot_metrics, csv_path)
        print(f"Saved CSV: {csv_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
