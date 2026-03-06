"""
Visualization Module for Airbag Recall Analysis

Generates charts and visual outputs:
    - Pareto chart of root causes
    - Bar chart of most common failing components
    - Manufacturer vs defect heatmap
    - Yearly recall trend line chart

All charts are saved as PNG files in the output directory.
Uses matplotlib with the Agg backend for headless environments.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"

# Chart styling
plt.rcParams.update(
    {
        "figure.figsize": (12, 7),
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    }
)


def create_pareto_chart(
    pareto_df: pd.DataFrame, output_path: Path | None = None
) -> Path:
    """Generate a Pareto chart showing root cause frequency and cumulative %.

    Args:
        pareto_df: Pareto table with category, count, cumulative_percentage.
        output_path: File path for the saved chart.

    Returns:
        Path to the saved chart file.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "pareto_chart.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Bar chart for counts
    x = range(len(pareto_df))
    colors = sns.color_palette("viridis", len(pareto_df))
    bars = ax1.bar(x, pareto_df["count"], color=colors, alpha=0.8, zorder=2)
    ax1.set_xlabel("Root Cause Category")
    ax1.set_ylabel("Number of Recalls", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(pareto_df["category"], rotation=45, ha="right")

    # Add count labels on bars
    for bar, count in zip(bars, pareto_df["count"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Cumulative line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        pareto_df["cumulative_percentage"],
        color="red",
        marker="o",
        linewidth=2,
        markersize=6,
        zorder=3,
    )
    ax2.set_ylabel("Cumulative Percentage (%)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 105)

    # 80% threshold line
    ax2.axhline(y=80, color="gray", linestyle="--", alpha=0.7, label="80% threshold")
    ax2.legend(loc="center right")

    plt.title("Pareto Analysis: Root Causes of Airbag Recalls")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Pareto chart saved to %s", output_path)
    return output_path


def create_component_bar_chart(
    component_df: pd.DataFrame, top_n: int = 10, output_path: Path | None = None
) -> Path:
    """Generate a horizontal bar chart of the most common failing components.

    Args:
        component_df: DataFrame with failing_component and count columns.
        top_n: Number of top components to show.
        output_path: File path for the saved chart.

    Returns:
        Path to the saved chart file.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "component_bar_chart.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    top = component_df.head(top_n).iloc[::-1]  # Reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = sns.color_palette("coolwarm", len(top))
    bars = ax.barh(top["failing_component"], top["count"], color=colors)

    # Add count labels
    for bar, count in zip(bars, top["count"]):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Number of Recalls")
    ax.set_title("Most Common Failing Components in Airbag Recalls")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Component bar chart saved to %s", output_path)
    return output_path


def create_manufacturer_heatmap(
    mfr_matrix: pd.DataFrame, top_n: int = 15, output_path: Path | None = None
) -> Path:
    """Generate a heatmap of manufacturer vs. defect category.

    Args:
        mfr_matrix: Cross-tabulation DataFrame.
        top_n: Number of top manufacturers to include.
        output_path: File path for the saved chart.

    Returns:
        Path to the saved chart file.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "manufacturer_heatmap.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select top manufacturers by total recalls
    top_mfrs = mfr_matrix.sum(axis=1).nlargest(top_n).index
    subset = mfr_matrix.loc[top_mfrs]

    # Remove columns with all zeros
    subset = subset.loc[:, (subset != 0).any(axis=0)]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        subset,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Number of Recalls"},
    )
    ax.set_title("Manufacturer vs. Defect Category Heatmap")
    ax.set_xlabel("Defect Category")
    ax.set_ylabel("Manufacturer")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Manufacturer heatmap saved to %s", output_path)
    return output_path


def create_yearly_trend_chart(
    yearly_df: pd.DataFrame, output_path: Path | None = None
) -> Path:
    """Generate a line chart showing recall trends over time.

    Args:
        yearly_df: DataFrame with recall_year and count columns.
        output_path: File path for the saved chart.

    Returns:
        Path to the saved chart file.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "yearly_trend.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        yearly_df["recall_year"],
        yearly_df["count"],
        marker="o",
        linewidth=2,
        color="steelblue",
        markersize=8,
    )
    ax.fill_between(
        yearly_df["recall_year"],
        yearly_df["count"],
        alpha=0.2,
        color="steelblue",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Airbag Recalls")
    ax.set_title("Airbag Recall Trends Over Time")
    ax.grid(alpha=0.3)

    # Annotate peak year
    if len(yearly_df) > 0:
        peak_idx = yearly_df["count"].idxmax()
        peak_year = yearly_df.loc[peak_idx, "recall_year"]
        peak_count = yearly_df.loc[peak_idx, "count"]
        ax.annotate(
            f"Peak: {peak_count} recalls\n({int(peak_year)})",
            xy=(peak_year, peak_count),
            xytext=(peak_year + 1, peak_count + 2),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10,
            color="red",
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Yearly trend chart saved to %s", output_path)
    return output_path


def generate_all_charts(analysis_results: dict) -> dict[str, Path]:
    """Generate all visualization charts.

    Args:
        analysis_results: Dict from pareto_analysis.run_pareto_analysis().

    Returns:
        Dict mapping chart names to their file paths.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    charts = {}

    charts["pareto"] = create_pareto_chart(analysis_results["pareto_table"])
    charts["components"] = create_component_bar_chart(
        analysis_results["component_freq"]
    )

    if not analysis_results["mfr_defect_matrix"].empty:
        charts["heatmap"] = create_manufacturer_heatmap(
            analysis_results["mfr_defect_matrix"]
        )

    if not analysis_results["yearly_trends"].empty:
        charts["trends"] = create_yearly_trend_chart(
            analysis_results["yearly_trends"]
        )

    logger.info("Generated %d charts", len(charts))
    return charts
