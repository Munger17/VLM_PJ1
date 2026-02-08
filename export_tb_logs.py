"""
export_tb_logs.py — Export TensorBoard logs to plots and CSV
Works offline without launching a TensorBoard server.

Usage:
    python export_tb_logs.py
    python export_tb_logs.py --logdir ./output/qwen3vl-4b-fashion-lora/logs
"""

import argparse
import os
import csv
from collections import defaultdict

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("Warning: matplotlib not installed, skipping plot generation")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description="Export TensorBoard logs")
    parser.add_argument(
        "--logdir",
        type=str,
        default="./output/qwen3vl-4b-fashion-lora/tb_logs",
    )
    parser.add_argument("--outdir", type=str, default="./output/reports")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load events
    ea = EventAccumulator(args.logdir)
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if not tags:
        print(f"No scalar logs found in {args.logdir}")
        return

    print(f"Found {len(tags)} metrics: {tags}")

    # Group tags by prefix (e.g., "hw/", "train/", "eval/")
    groups = defaultdict(list)
    for tag in tags:
        prefix = tag.split("/")[0] if "/" in tag else "train"
        groups[prefix].append(tag)

    # Export CSV
    csv_path = os.path.join(args.outdir, "training_logs.csv")
    all_data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        all_data[tag] = {e.step: e.value for e in events}

    # Get all unique steps
    all_steps = sorted(set(s for d in all_data.values() for s in d))

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + tags)
        for step in all_steps:
            row = [step] + [all_data[tag].get(step, "") for tag in tags]
            writer.writerow(row)

    print(f"CSV saved to {csv_path}")

    # Export plots
    if HAS_PLT:
        for group_name, group_tags in groups.items():
            fig, axes = plt.subplots(
                len(group_tags), 1, figsize=(10, 4 * len(group_tags)), squeeze=False
            )
            fig.suptitle(f"{group_name.upper()} Metrics", fontsize=14, fontweight="bold")

            for i, tag in enumerate(group_tags):
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                axes[i][0].plot(steps, values, linewidth=1.5)
                axes[i][0].set_title(tag, fontsize=11)
                axes[i][0].set_xlabel("Step")
                axes[i][0].set_ylabel("Value")
                axes[i][0].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = os.path.join(args.outdir, f"{group_name}_metrics.png")
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Plot saved to {plot_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for tag in tags:
        events = ea.Scalars(tag)
        if events:
            values = [e.value for e in events]
            print(f"  {tag:40s}  last={values[-1]:.4f}  min={min(values):.4f}  max={max(values):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()