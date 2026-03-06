"""
Aggregate trial counts by country and year.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Explode helpers
# ---------------------------------------------------------------------------

def explode_by_recruitment(df: pd.DataFrame, year_range: tuple[int, int]) -> pd.DataFrame:
    """
    Explode the recruitment_countries list column so every
    (trial_id, country) pair becomes its own row.
    Filtered to [year_min, year_max].
    """
    year_min, year_max = year_range
    df_f = df[df["year"].between(year_min, year_max)].copy()
    df_f = df_f.explode("recruitment_countries")
    df_f = df_f.rename(columns={"recruitment_countries": "country"})
    df_f = df_f[df_f["country"].notna() & (df_f["country"] != "")]
    return df_f


def explode_by_sponsor(df: pd.DataFrame, year_range: tuple[int, int]) -> pd.DataFrame:
    """
    Use sponsor_country as the country dimension.
    Filtered to [year_min, year_max].
    """
    year_min, year_max = year_range
    df_f = df[df["year"].between(year_min, year_max)].copy()
    df_f = df_f.rename(columns={"sponsor_country": "country"})
    df_f = df_f[df_f["country"].notna() & (df_f["country"] != "")]
    return df_f


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def country_year_counts(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Count unique trial_ids per (country, year).
    Returns a tidy DataFrame with columns [country, year, count].
    """
    counts = (
        df_long.groupby(["country", "year"])["trial_id"]
        .nunique()
        .reset_index(name="count")
    )
    counts["year"] = counts["year"].astype(int)
    return counts


def top_countries_in_year(counts: pd.DataFrame, year: int, n: int = 10) -> list[str]:
    """Return the top-N country names by count in `year`."""
    subset = counts[counts["year"] == year]
    return subset.nlargest(n, "count")["country"].tolist()


# ---------------------------------------------------------------------------
# Growth metrics
# ---------------------------------------------------------------------------

def cagr(start: float, end: float, periods: int) -> float:
    """
    Compound Annual Growth Rate.
    Returns NaN when start is zero (undefined) or inputs are invalid.
    """
    if start <= 0 or end < 0 or periods <= 0:
        return math.nan
    return (end / start) ** (1.0 / periods) - 1.0


def build_growth_table(
    counts: pd.DataFrame,
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    """
    For every country that appears in both year_start and year_end,
    compute absolute growth and CAGR.
    Returns columns: [country, count_start, count_end, abs_growth, cagr_pct]
    """
    periods = year_end - year_start
    # Only compute CAGR for countries present in at least one boundary year
    countries_start = set(counts[counts["year"] == year_start]["country"])
    countries_end   = set(counts[counts["year"] == year_end  ]["country"])
    all_countries   = countries_start | countries_end
    rows = []
    for c in all_countries:
        cd = counts[counts["country"] == c]
        v_start = int(cd[cd["year"] == year_start]["count"].sum())
        v_end   = int(cd[cd["year"] == year_end  ]["count"].sum())
        r = cagr(v_start, v_end, periods)
        rows.append({
            "country":     c,
            "count_start": v_start,
            "count_end":   v_end,
            "abs_growth":  v_end - v_start,
            "cagr_pct":    r * 100 if not math.isnan(r) else math.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_counts(counts: pd.DataFrame, out_dir: Path) -> None:
    """Save the aggregated counts table to CSV and Parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out_dir / "ictrp_counts.csv", index=False)
    counts.to_parquet(out_dir / "ictrp_counts.parquet", index=False)
