#!/usr/bin/env python3
"""
Generate plots for HPC alignment benchmark results
Based on OpenBLAS Issue #3879 performance variability analysis
"""

import csv
import sys

import matplotlib.pyplot as plt
import numpy as np


def read_results(filename="benchmark_results.csv"):
    """Read benchmark results from CSV file"""
    sizes = []
    aligned_times = []
    misaligned_times = []
    speedups = []

    try:
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sizes.append(int(row["Matrix_Size"]))
                aligned_times.append(float(row["Aligned_Time"]))
                misaligned_times.append(float(row["Misaligned_Time"]))
                speedups.append(float(row["Speedup"]))
    except FileNotFoundError:
        print(f"Error: {filename} not found. Run benchmark first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        sys.exit(1)

    return sizes, aligned_times, misaligned_times, speedups


def create_plots(sizes, aligned_times, misaligned_times, speedups):
    """Create multiple plots showing benchmark results"""

    # Set style (fallback if seaborn not available)
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except:
        try:
            plt.style.use("seaborn-darkgrid")
        except:
            plt.style.use("default")
    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Execution Time Comparison
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(
        sizes,
        aligned_times,
        "o-",
        label="64-byte Aligned (Optimized)",
        linewidth=2,
        markersize=8,
        color="#2ecc71",
    )
    ax1.plot(
        sizes,
        misaligned_times,
        "s-",
        label="16-byte Aligned (Non-optimized)",
        linewidth=2,
        markersize=8,
        color="#e74c3c",
    )
    ax1.set_xlabel("Matrix Size (N x N)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Execution Time (seconds)", fontsize=12, fontweight="bold")
    ax1.set_title("Execution Time vs Matrix Size", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("linear")
    ax1.set_yscale("log")

    # Plot 2: Speedup Ratio
    ax2 = plt.subplot(2, 2, 2)
    colors = ["#e74c3c" if s < 1.0 else "#2ecc71" for s in speedups]
    bars = ax2.bar(
        range(len(sizes)), speedups, color=colors, alpha=0.7, edgecolor="black"
    )
    ax2.axhline(
        y=1.0, color="black", linestyle="--", linewidth=1.5, label="No difference"
    )
    ax2.set_xlabel("Matrix Size (N x N)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Speedup Ratio (Misaligned/Aligned)", fontsize=12, fontweight="bold")
    ax2.set_title("Performance Ratio by Matrix Size", fontsize=14, fontweight="bold")
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels(sizes)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{speedup:.2f}x",
            ha="center",
            va="bottom" if speedup > 1 else "top",
            fontsize=9,
        )

    # Plot 3: Performance Variability (as percentage)
    ax3 = plt.subplot(2, 2, 3)
    variability = [(abs(1 - s) * 100) for s in speedups]
    ax3.plot(sizes, variability, "o-", linewidth=2, markersize=8, color="#3498db")
    ax3.fill_between(sizes, variability, alpha=0.3, color="#3498db")
    ax3.set_xlabel("Matrix Size (N x N)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Performance Variability (%)", fontsize=12, fontweight="bold")
    ax3.set_title(
        "Performance Variability (Issue #3879)", fontsize=14, fontweight="bold"
    )
    ax3.grid(True, alpha=0.3)

    # Add annotations
    for size, var in zip(sizes, variability):
        ax3.annotate(
            f"{var:.1f}%",
            (size, var),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    # Plot 4: GFLOPS Comparison
    ax4 = plt.subplot(2, 2, 4)
    gflops_aligned = [(2.0 * s**3) / (t * 1e9) for s, t in zip(sizes, aligned_times)]
    gflops_misaligned = [
        (2.0 * s**3) / (t * 1e9) for s, t in zip(sizes, misaligned_times)
    ]

    x = np.arange(len(sizes))
    width = 0.35
    ax4.bar(
        x - width / 2,
        gflops_aligned,
        width,
        label="64-byte Aligned",
        color="#2ecc71",
        alpha=0.8,
        edgecolor="black",
    )
    ax4.bar(
        x + width / 2,
        gflops_misaligned,
        width,
        label="16-byte Aligned",
        color="#e74c3c",
        alpha=0.8,
        edgecolor="black",
    )
    ax4.set_xlabel("Matrix Size (N x N)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Performance (GFLOPS)", fontsize=12, fontweight="bold")
    ax4.set_title("Computational Throughput Comparison", fontsize=14, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(sizes)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save figure
    output_file = "alignment_benchmark_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Plot saved to: {output_file}")

    # Also save individual plots
    for i, (ax, title) in enumerate(
        zip(
            [ax1, ax2, ax3, ax4], ["execution_time", "speedup", "variability", "gflops"]
        ),
        1,
    ):
        fig_single = plt.figure(figsize=(8, 6))
        ax_new = fig_single.add_subplot(111)

        # Copy plot content
        if i == 1:
            ax_new.plot(
                sizes,
                aligned_times,
                "o-",
                label="64-byte Aligned (Optimized)",
                linewidth=2,
                markersize=8,
                color="#2ecc71",
            )
            ax_new.plot(
                sizes,
                misaligned_times,
                "s-",
                label="16-byte Aligned (Non-optimized)",
                linewidth=2,
                markersize=8,
                color="#e74c3c",
            )
            ax_new.set_yscale("log")
        elif i == 2:
            colors = ["#e74c3c" if s < 1.0 else "#2ecc71" for s in speedups]
            bars = ax_new.bar(
                range(len(sizes)), speedups, color=colors, alpha=0.7, edgecolor="black"
            )
            ax_new.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5)
            ax_new.set_xticks(range(len(sizes)))
            ax_new.set_xticklabels(sizes)
            for j, (bar, speedup) in enumerate(zip(bars, speedups)):
                height = bar.get_height()
                ax_new.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{speedup:.2f}x",
                    ha="center",
                    va="bottom" if speedup > 1 else "top",
                    fontsize=9,
                )
        elif i == 3:
            variability = [(abs(1 - s) * 100) for s in speedups]
            ax_new.plot(
                sizes, variability, "o-", linewidth=2, markersize=8, color="#3498db"
            )
            ax_new.fill_between(sizes, variability, alpha=0.3, color="#3498db")
            for size, var in zip(sizes, variability):
                ax_new.annotate(
                    f"{var:.1f}%",
                    (size, var),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=9,
                )
        elif i == 4:
            x = np.arange(len(sizes))
            width = 0.35
            ax_new.bar(
                x - width / 2,
                gflops_aligned,
                width,
                label="64-byte Aligned",
                color="#2ecc71",
                alpha=0.8,
                edgecolor="black",
            )
            ax_new.bar(
                x + width / 2,
                gflops_misaligned,
                width,
                label="16-byte Aligned",
                color="#e74c3c",
                alpha=0.8,
                edgecolor="black",
            )
            ax_new.set_xticks(x)
            ax_new.set_xticklabels(sizes)

        ax_new.set_xlabel(ax.get_xlabel(), fontsize=12, fontweight="bold")
        ax_new.set_ylabel(ax.get_ylabel(), fontsize=12, fontweight="bold")
        ax_new.set_title(ax.get_title(), fontsize=14, fontweight="bold")
        ax_new.legend(fontsize=10)
        ax_new.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"alignment_benchmark_{title}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"✓ Individual plot saved to: {filename}")
        plt.close(fig_single)

    plt.close(fig)
    print("\nAll plots generated successfully!")


def main():
    """Main function"""
    print("=" * 50)
    print("HPC Alignment Benchmark - Plot Generation")
    print("=" * 50)
    print()

    sizes, aligned_times, misaligned_times, speedups = read_results()

    print(f"Loaded {len(sizes)} data points")
    print(f"Matrix sizes: {sizes}")
    print()

    create_plots(sizes, aligned_times, misaligned_times, speedups)


if __name__ == "__main__":
    main()
