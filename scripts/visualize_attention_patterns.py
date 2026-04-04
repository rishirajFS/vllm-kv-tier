#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Attention Pattern Visualization - Priority 4 from final_plan.md

Generates three key visualizations from eviction data:

1. Attention Heatmap: Shows which blocks LRU evicts vs Attention-aware evicts
   - X-axis: Block ID
   - Y-axis: Cumulative attention score
   - Color: Red = evicted by LRU, Green = kept, Blue = evicted by Attention

2. Score Distribution: Histogram comparing kept vs evicted blocks
   - Shows clear separation between high-score (kept) and low-score (evicted)

3. Access Pattern Comparison: Side-by-side heatmaps across workloads
   - ShareGPT: Sequential (diagonal pattern)
   - MS-MARCO: Clustered (document hotspots)
   - HumanEval: Local hotspots (function context)

Usage:
    # Generate all visualizations
    python scripts/visualize_attention_patterns.py \
        --data eviction_data.json \
        --output-dir ./visualizations

    # Generate specific visualization
    python scripts/visualize_attention_patterns.py \
        --data eviction_data.json \
        --viz-type heatmap \
        --output attention_heatmap.png
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_eviction_data(data_path: Path) -> list[dict[str, Any]]:
    """Load eviction data from JSON."""
    with open(data_path) as f:
        return json.load(f)


def plot_attention_heatmap(
    lru_data: dict[str, Any],
    attention_data: dict[str, Any],
    output_path: Path,
    sample_idx: int = 0,
):
    """
    Visualization 1: Attention Heatmap.

    Shows which blocks are evicted by LRU vs Attention policies.
    Color coding:
    - Red: Evicted by LRU only (high attention but old)
    - Blue: Evicted by Attention only (low attention but recent)
    - Purple: Evicted by both
    - Green: Kept by both

    Args:
        lru_data: Eviction dataset for LRU policy
        attention_data: Eviction dataset for Attention policy
        output_path: Where to save the figure
        sample_idx: Which sample to visualize (0-based)
    """
    lru_req = lru_data["requests"][sample_idx]
    attn_req = attention_data["requests"][sample_idx]

    # Extract block info
    block_ids = sorted(lru_req["block_scores"].keys())
    scores = [lru_req["block_scores"][str(b)] for b in block_ids]

    lru_evicted = set(lru_req["evicted_blocks"])
    attn_evicted = set(attn_req["evicted_blocks"])

    # Categorize blocks
    colors = []
    labels = []
    for b in block_ids:
        if b in lru_evicted and b in attn_evicted:
            colors.append("purple")
            labels.append("Both evicted")
        elif b in lru_evicted:
            colors.append("red")
            labels.append("LRU evicted")
        elif b in attn_evicted:
            colors.append("blue")
            labels.append("Attention evicted")
        else:
            colors.append("green")
            labels.append("Both kept")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Scatter plot with colors
    ax.scatter(block_ids, scores, c=colors, alpha=0.6, s=30)

    ax.set_xlabel("Block ID", fontsize=12)
    ax.set_ylabel("Attention Score", fontsize=12)
    ax.set_title(
        f"Attention Heatmap: LRU vs Attention-Aware Eviction\n"
        f"Workload: {lru_data['workload']}, Request: {sample_idx}",
        fontsize=14,
        fontweight="bold",
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="red", alpha=0.6, label="LRU evicted (high attn!)"),
        Patch(facecolor="blue", alpha=0.6, label="Attention evicted (low attn)"),
        Patch(facecolor="purple", alpha=0.6, label="Both evicted"),
        Patch(facecolor="green", alpha=0.6, label="Both kept"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved attention heatmap to: {output_path}")
    plt.close()


def plot_score_distribution(
    datasets: list[dict[str, Any]],
    output_path: Path,
):
    """
    Visualization 2: Score Distribution Histogram.

    Shows histogram of attention scores for kept vs evicted blocks.
    Should show clear separation: kept blocks have higher scores.

    Args:
        datasets: List of eviction datasets (different policies)
        output_path: Where to save the figure
    """
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))

    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        policy = dataset["policy"]

        # Collect all scores
        evicted_scores = []
        kept_scores = []

        for req in dataset["requests"]:
            evicted_scores.extend([req["block_scores"][str(b)] for b in req["evicted_blocks"] if str(b) in req["block_scores"]])
            kept_scores.extend([req["block_scores"][str(b)] for b in req["kept_blocks"] if str(b) in req["block_scores"]])

        # Plot histograms
        bins = np.linspace(0, 1, 30)
        ax.hist(evicted_scores, bins=bins, alpha=0.6, color="red", label="Evicted blocks", edgecolor="black")
        ax.hist(kept_scores, bins=bins, alpha=0.6, color="green", label="Kept blocks", edgecolor="black")

        # Statistics
        mean_evicted = np.mean(evicted_scores) if evicted_scores else 0
        mean_kept = np.mean(kept_scores) if kept_scores else 0

        ax.axvline(mean_evicted, color="red", linestyle="--", linewidth=2, label=f"Mean evicted: {mean_evicted:.3f}")
        ax.axvline(mean_kept, color="green", linestyle="--", linewidth=2, label=f"Mean kept: {mean_kept:.3f}")

        ax.set_xlabel("Attention Score", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{policy.upper()} Policy\nScore Separation: {mean_kept - mean_evicted:.3f}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Score Distribution: Kept vs Evicted Blocks\n"
        f"Workload: {datasets[0]['workload']}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved score distribution to: {output_path}")
    plt.close()


def plot_access_pattern_comparison(
    datasets_by_workload: dict[str, dict[str, Any]],
    output_path: Path,
    policy: str = "attention",
):
    """
    Visualization 3: Access Pattern Comparison.

    Side-by-side heatmaps showing attention patterns across workloads.
    - ShareGPT: Sequential (diagonal pattern)
    - MS-MARCO: Clustered (document sections)
    - HumanEval: Local hotspots (function context)

    Args:
        datasets_by_workload: Dict mapping workload name to eviction dataset
        output_path: Where to save the figure
        policy: Which policy to visualize
    """
    workloads = ["sharegpt", "msmarco", "humaneval"]
    available_workloads = [w for w in workloads if w in datasets_by_workload]

    if not available_workloads:
        print(f"⚠️  No workload data available for access pattern comparison")
        return

    num_workloads = len(available_workloads)
    fig, axes = plt.subplots(1, num_workloads, figsize=(6 * num_workloads, 5))

    if num_workloads == 1:
        axes = [axes]

    for ax, workload in zip(axes, available_workloads):
        dataset = datasets_by_workload[workload]

        # Average attention patterns across all requests
        max_blocks = max(len(req["block_scores"]) for req in dataset["requests"])

        # Create attention matrix (requests × blocks)
        attention_matrix = np.zeros((len(dataset["requests"]), max_blocks))

        for i, req in enumerate(dataset["requests"]):
            for block_id, score in req["block_scores"].items():
                block_id = int(block_id)
                if block_id < max_blocks:
                    attention_matrix[i, block_id] = score

        # Plot heatmap
        im = ax.imshow(attention_matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")

        ax.set_xlabel("Block ID", fontsize=11)
        ax.set_ylabel("Request ID", fontsize=11)
        ax.set_title(f"{workload.upper()}", fontsize=12, fontweight="bold")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Attention Score", fontsize=10)

    plt.suptitle(
        f"Access Pattern Comparison: {policy.upper()} Policy\n"
        f"(Darker = Higher Attention)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved access pattern comparison to: {output_path}")
    plt.close()


def plot_overlap_analysis(
    lru_data: dict[str, Any],
    attention_data: dict[str, Any],
    output_path: Path,
):
    """
    Bonus visualization: Eviction overlap analysis.

    Shows what % of evictions are shared between LRU and Attention policies.

    Args:
        lru_data: Eviction dataset for LRU
        attention_data: Eviction dataset for Attention
        output_path: Where to save the figure
    """
    overlaps = []
    lru_only = []
    attn_only = []

    for lru_req, attn_req in zip(lru_data["requests"], attention_data["requests"]):
        lru_set = set(lru_req["evicted_blocks"])
        attn_set = set(attn_req["evicted_blocks"])

        if len(lru_set) == 0 or len(attn_set) == 0:
            continue

        overlap = len(lru_set & attn_set)
        overlap_pct = overlap / len(lru_set) * 100

        overlaps.append(overlap_pct)
        lru_only.append(len(lru_set - attn_set) / len(lru_set) * 100)
        attn_only.append(len(attn_set - lru_set) / len(attn_set) * 100)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(overlaps))
    width = 0.25

    ax.bar(x - width, overlaps, width, label="Overlap", color="purple", alpha=0.7)
    ax.bar(x, lru_only, width, label="LRU only", color="red", alpha=0.7)
    ax.bar(x + width, attn_only, width, label="Attention only", color="blue", alpha=0.7)

    ax.set_xlabel("Request ID", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title(
        f"Eviction Overlap: LRU vs Attention-Aware\n"
        f"Workload: {lru_data['workload']} | "
        f"Mean Overlap: {np.mean(overlaps):.1f}%",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved overlap analysis to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate attention pattern visualizations"
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Eviction data JSON file (from collect_eviction_data.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./visualizations"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--viz-type",
        type=str,
        choices=["all", "heatmap", "distribution", "patterns", "overlap"],
        default="all",
        help="Which visualization to generate",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Sample index for heatmap visualization",
    )

    args = parser.parse_args()

    # Load data
    print(f"\nLoading eviction data from: {args.data}")
    datasets = load_eviction_data(args.data)

    if not datasets:
        print("Error: No data loaded")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Generating Attention Pattern Visualizations")
    print(f"{'='*70}")
    print(f"Data file: {args.data}")
    print(f"Output dir: {args.output_dir}")
    print(f"Workload: {datasets[0]['workload']}")
    print(f"Policies: {', '.join([d['policy'] for d in datasets])}")
    print(f"{'='*70}\n")

    # Separate datasets by policy
    datasets_by_policy = {d["policy"]: d for d in datasets}

    # Visualization 1: Attention Heatmap
    if args.viz_type in ["all", "heatmap"]:
        if "lru" in datasets_by_policy and "attention" in datasets_by_policy:
            plot_attention_heatmap(
                datasets_by_policy["lru"],
                datasets_by_policy["attention"],
                args.output_dir / "attention_heatmap.png",
                args.sample_idx,
            )
        else:
            print("⚠️  Heatmap requires both LRU and Attention data")

    # Visualization 2: Score Distribution
    if args.viz_type in ["all", "distribution"]:
        plot_score_distribution(
            datasets,
            args.output_dir / "score_distribution.png",
        )

    # Visualization 3: Access Pattern Comparison (requires multiple workloads)
    if args.viz_type in ["all", "patterns"]:
        print("⚠️  Access pattern comparison requires data from multiple workloads")
        print("Run collect_eviction_data.py for sharegpt, msmarco, and humaneval separately,")
        print("then use a script to combine them for this visualization.")

    # Bonus: Overlap Analysis
    if args.viz_type in ["all", "overlap"]:
        if "lru" in datasets_by_policy and "attention" in datasets_by_policy:
            plot_overlap_analysis(
                datasets_by_policy["lru"],
                datasets_by_policy["attention"],
                args.output_dir / "overlap_analysis.png",
            )
        else:
            print("⚠️  Overlap analysis requires both LRU and Attention data")

    print(f"\n{'='*70}")
    print(f"Visualizations saved to: {args.output_dir}")
    print(f"{'='*70}\n")

    print("Key insights to extract:")
    print("1. Do LRU and Attention evict different blocks? (RED blocks in heatmap)")
    print("2. Is there clear score separation? (Distribution histograms)")
    print("3. What % of evictions overlap? (Overlap analysis)")
    print("")
    print("Expected findings:")
    print("- LRU evicts ~35% high-attention blocks (RED in heatmap)")
    print("- Attention evicts only ~8% high-attention blocks")
    print("- Score separation: 0.3-0.5 between kept and evicted blocks")
    print("- Overlap: 50-65% (some blocks are obvious to evict)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
