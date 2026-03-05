"""
Pareto Analysis for Airbag Recall Root Causes

Performs frequency analysis of root cause categories and generates
Pareto tables showing cause category, number of recalls, percentage,
and cumulative percentage.

The Pareto principle (80/20 rule) is applied to identify the small
number of causes responsible for the majority of recalls.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_pareto(
    df: pd.DataFrame, column: str = "cause_category"
) -> pd.DataFrame:
    """Compute a Pareto table for the given categorical column.

    Args:
        df: DataFrame with classified recall data.
        column: Column name to analyze.

    Returns:
        DataFrame with columns: category, count, percentage,
        cumulative_count, cumulative_percentage.
    """
    counts = df[column].value_counts().reset_index()
    counts.columns = ["category", "count"]
    counts = counts.sort_values("count", ascending=False).reset_index(drop=True)

    total = counts["count"].sum()
    counts["percentage"] = (counts["count"] / total * 100).round(2)
    counts["cumulative_count"] = counts["count"].cumsum()
    counts["cumulative_percentage"] = (
        counts["cumulative_count"] / total * 100
    ).round(2)

    return counts


def identify_vital_few(
    pareto_df: pd.DataFrame, threshold: float = 80.0
) -> pd.DataFrame:
    """Identify the 'vital few' causes that account for a given threshold.

    Args:
        pareto_df: Pareto table from compute_pareto().
        threshold: Cumulative percentage threshold (default 80%).

    Returns:
        Subset of the Pareto table containing the vital few causes.
    """
    mask = pareto_df["cumulative_percentage"] <= threshold
    # Include the first row that crosses the threshold
    first_over = pareto_df[~mask].head(1)
    vital_few = pd.concat([pareto_df[mask], first_over]).drop_duplicates()
    return vital_few


def component_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Count frequency of each failing component.

    Args:
        df: DataFrame with 'failing_component' column.

    Returns:
        DataFrame with component names and counts, sorted descending.
    """
    counts = df["failing_component"].value_counts().reset_index()
    counts.columns = ["failing_component", "count"]
    return counts


def manufacturer_defect_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create a manufacturer vs. defect category cross-tabulation.

    Args:
        df: DataFrame with 'manufacturer' and 'cause_category' columns.

    Returns:
        Cross-tabulation DataFrame (manufacturers as rows, causes as columns).
    """
    return pd.crosstab(df["manufacturer"], df["cause_category"])


def yearly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Count recalls per year.

    Args:
        df: DataFrame with 'recall_year' column.

    Returns:
        DataFrame with year and count columns.
    """
    if "recall_year" not in df.columns:
        logger.warning("No 'recall_year' column found")
        return pd.DataFrame(columns=["recall_year", "count"])

    counts = (
        df["recall_year"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .reset_index()
    )
    counts.columns = ["recall_year", "count"]
    return counts


def run_pareto_analysis(df: pd.DataFrame) -> dict:
    """Run the complete Pareto analysis suite.

    Args:
        df: Classified recall DataFrame.

    Returns:
        Dict containing all analysis DataFrames:
        - pareto_table: Full Pareto analysis
        - vital_few: Top causes covering ~80% of recalls
        - component_freq: Component frequency counts
        - mfr_defect_matrix: Manufacturer vs defect cross-tab
        - yearly_trends: Yearly recall counts
    """
    logger.info("Running Pareto analysis on %d records", len(df))

    pareto_table = compute_pareto(df, "cause_category")
    vital_few = identify_vital_few(pareto_table)
    comp_freq = component_frequency(df)
    mfr_matrix = manufacturer_defect_matrix(df)
    trends = yearly_trend(df)

    logger.info(
        "Vital few causes (%d of %d categories) cover %.1f%% of recalls",
        len(vital_few),
        len(pareto_table),
        vital_few["cumulative_percentage"].max() if len(vital_few) > 0 else 0,
    )

    return {
        "pareto_table": pareto_table,
        "vital_few": vital_few,
        "component_freq": comp_freq,
        "mfr_defect_matrix": mfr_matrix,
        "yearly_trends": trends,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Demo with synthetic data
    demo_data = pd.DataFrame(
        {
            "cause_category": ["Inflator Rupture"] * 40
            + ["Propellant Degradation"] * 25
            + ["Sensor Malfunction"] * 15
            + ["Electrical Wiring Failure"] * 10
            + ["Manufacturing Defect"] * 5
            + ["Other"] * 5,
            "failing_component": ["Inflator"] * 40
            + ["Propellant Charge"] * 25
            + ["Crash Sensor"] * 15
            + ["Wiring Harness"] * 10
            + ["Various"] * 5
            + ["Unknown"] * 5,
            "manufacturer": ["Honda"] * 30
            + ["Toyota"] * 20
            + ["Ford"] * 15
            + ["BMW"] * 10
            + ["Nissan"] * 10
            + ["Subaru"] * 10
            + ["Chrysler"] * 5,
            "recall_year": list(range(2010, 2020)) * 10,
        }
    )

    results = run_pareto_analysis(demo_data)
    print("\nPareto Table:")
    print(results["pareto_table"].to_string(index=False))
    print("\nVital Few (80% threshold):")
    print(results["vital_few"].to_string(index=False))
