"""
Chart generation — Plotly (interactive).

Five charts:
  A. World map    — choropleth by trial volume in the end year
  B. Volume leaders — horizontal bar, top N countries in end year
  C. Time series  — registrations over time for top N countries
  D. Slope        — start → end comparison for top N countries
  E. CAGR         — fastest-growing countries (bar)
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.aggregate import build_growth_table, top_countries_in_year

# ── Shared style ──────────────────────────────────────────────────────────────

_LAYOUT = dict(
    font=dict(family="Inter, Arial, sans-serif", size=12),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=10, r=10, t=50, b=10),
    hoverlabel=dict(bgcolor="white", font_size=12),
)

def _layout(**overrides) -> dict:
    """Merge _LAYOUT with per-chart overrides (overrides win on conflicts)."""
    return {**_LAYOUT, **overrides}


_BLUE       = "#2166ac"
_LIGHT_BLUE = "#6baed6"
_ORANGE     = "#f4a582"

# Plotly's qualitative palette for multi-series charts
_PALETTE = (
    px.colors.qualitative.Set2
    + px.colors.qualitative.Pastel
    + px.colors.qualitative.Safe
)

# US name variants (for "always include" logic)
_US_VARIANTS = {
    "United States of America",
    "United States",
    "USA",
    "U.S.A.",
    "U.S.",
}

# ISO 3166-1 alpha-3 lookup for common country names.
# Used by chart_world_map to switch from 'country names' mode (unreliable)
# to 'ISO-3' mode (unambiguous).
_ISO3: dict[str, str] = {
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "Argentina": "ARG",
    "Armenia": "ARM", "Australia": "AUS", "Austria": "AUT", "Azerbaijan": "AZE",
    "Bahrain": "BHR", "Bangladesh": "BGD", "Belarus": "BLR", "Belgium": "BEL",
    "Bolivia": "BOL", "Bosnia and Herzegovina": "BIH", "Brazil": "BRA",
    "Bulgaria": "BGR", "Cambodia": "KHM", "Cameroon": "CMR", "Canada": "CAN",
    "Chile": "CHL", "China": "CHN", "Colombia": "COL", "Costa Rica": "CRI",
    "Croatia": "HRV", "Cuba": "CUB", "Cyprus": "CYP", "Czech Republic": "CZE",
    "Democratic Republic of the Congo": "COD", "Denmark": "DNK", "Ecuador": "ECU",
    "Egypt": "EGY", "Ethiopia": "ETH", "Finland": "FIN", "France": "FRA",
    "Georgia": "GEO", "Germany": "DEU", "Ghana": "GHA", "Greece": "GRC",
    "Guatemala": "GTM", "Hong Kong": "HKG", "Hungary": "HUN", "India": "IND",
    "Indonesia": "IDN", "Iran": "IRN", "Iraq": "IRQ", "Ireland": "IRL",
    "Israel": "ISR", "Italy": "ITA", "Japan": "JPN", "Jordan": "JOR",
    "Kazakhstan": "KAZ", "Kenya": "KEN", "Kuwait": "KWT", "Latvia": "LVA",
    "Lebanon": "LBN", "Libya": "LBY", "Lithuania": "LTU", "Luxembourg": "LUX",
    "Malaysia": "MYS", "Mexico": "MEX", "Morocco": "MAR", "Mozambique": "MOZ",
    "Netherlands": "NLD", "New Zealand": "NZL", "Nigeria": "NGA", "Norway": "NOR",
    "Oman": "OMN", "Pakistan": "PAK", "Palestine": "PSE", "Panama": "PAN",
    "Peru": "PER", "Philippines": "PHL", "Poland": "POL", "Portugal": "PRT",
    "Qatar": "QAT", "Romania": "ROU", "Russia": "RUS", "Saudi Arabia": "SAU",
    "Senegal": "SEN", "Serbia": "SRB", "Singapore": "SGP", "Slovakia": "SVK",
    "Slovenia": "SVN", "South Africa": "ZAF", "South Korea": "KOR", "Spain": "ESP",
    "Sudan": "SDN", "Sweden": "SWE", "Switzerland": "CHE", "Syria": "SYR",
    "Taiwan": "TWN", "Tanzania": "TZA", "Thailand": "THA", "Tunisia": "TUN",
    "Turkey": "TUR", "Uganda": "UGA", "Ukraine": "UKR",
    "United Arab Emirates": "ARE", "United Kingdom": "GBR",
    "United States": "USA", "United States of America": "USA",
    "Uruguay": "URY", "Uzbekistan": "UZB", "Venezuela": "VEN",
    "Vietnam": "VNM", "Yemen": "YEM", "Zimbabwe": "ZWE",
    "Côte d'Ivoire": "CIV", "Cameroon": "CMR",
}


# ── Chart A — World map ───────────────────────────────────────────────────────

def chart_world_map(counts: pd.DataFrame, year: int) -> go.Figure:
    """Choropleth world map: trial volume per country in `year`.

    Uses a log1p colour scale so that mid-tier countries show meaningful
    colour differences instead of all appearing near-white against a few
    dominant countries.  The colorbar tick labels are converted back to
    real trial counts so they remain readable.
    """
    import numpy as np

    subset = counts[counts["year"] == year].copy()

    if subset.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No data for {year}", **_LAYOUT)
        return fig

    # Log-transform for colour mapping; keep raw count for hover
    subset = subset.copy()
    subset["log_count"] = np.log1p(subset["count"])
    max_log = float(subset["log_count"].max())

    # Build readable colorbar ticks in original trial-count space
    raw_max   = int(subset["count"].max())
    tick_vals = [np.log1p(v) for v in [1, 5, 20, 50, 100, 200, 500, 1000, 2000] if v <= raw_max]
    tick_text = [str(int(np.expm1(v))) for v in tick_vals]

    # Map to ISO-3 codes — far more reliable than locationmode='country names'
    subset = subset.copy()
    subset["iso3"] = subset["country"].map(_ISO3)
    mapped = subset.dropna(subset=["iso3"])

    fig = go.Figure(go.Choropleth(
        locations=mapped["iso3"],
        locationmode="ISO-3",
        z=mapped["log_count"],
        zmin=0,
        zmax=max_log,
        zauto=False,
        colorscale="Blues",
        text=mapped["country"],
        customdata=mapped["count"],
        hovertemplate="<b>%{text}</b><br>Trials: %{customdata:,}<extra></extra>",
        colorbar=dict(
            title=dict(text="Trials", side="right"),
            tickvals=tick_vals,
            ticktext=tick_text,
            thickness=15,
        ),
    ))
    fig.update_layout(title=f"Trial Registrations by Country — {year}")
    fig.update_layout(
        **_layout(margin=dict(l=0, r=0, t=50, b=0)),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="lightgrey",
            showland=True,
            landcolor="#f7f7f7",
            showocean=True,
            oceancolor="#eaf3fb",
            projection_type="natural earth",
        ),
    )
    return fig


# ── Chart B — Volume leaders ──────────────────────────────────────────────────

def chart_volume_leaders(counts: pd.DataFrame, year: int, n: int = 10) -> go.Figure:
    """Horizontal bar: top N countries by trial count in `year`."""
    subset = (
        counts[counts["year"] == year]
        .nlargest(n, "count")
        .sort_values("count")
    )

    fig = go.Figure(go.Bar(
        x=subset["count"],
        y=subset["country"],
        orientation="h",
        marker_color=_BLUE,
        text=subset["count"].apply(lambda v: f"{v:,}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Trials: %{x:,}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT,
        title=f"Top {n} Countries — Trial Volume ({year})",
        xaxis=dict(title="Registered Trials", showgrid=True, gridcolor="#eeeeee"),
        yaxis=dict(title=""),
        xaxis_range=[0, subset["count"].max() * 1.18],
    )
    return fig


# ── Chart C — Time series ─────────────────────────────────────────────────────

def chart_time_series(
    counts: pd.DataFrame,
    top_countries: list[str],
    year_start: int,
    year_end: int,
) -> go.Figure:
    """Line chart: trial registrations over time for top N countries."""
    subset = counts[
        counts["country"].isin(top_countries)
        & counts["year"].between(year_start, year_end)
    ].copy()

    if subset.empty:
        fig = go.Figure()
        fig.update_layout(title="No time-series data for selection", **_LAYOUT)
        return fig

    fig = go.Figure()
    for i, country in enumerate(top_countries):
        cd = subset[subset["country"] == country].sort_values("year")
        if cd.empty:
            continue
        color = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(go.Scatter(
            x=cd["year"],
            y=cd["count"],
            mode="lines+markers",
            name=country,
            line=dict(color=color, width=2),
            marker=dict(size=5),
            hovertemplate=f"<b>{country}</b><br>Year: %{{x}}<br>Trials: %{{y:,}}<extra></extra>",
        ))

    fig.update_layout(
        **_LAYOUT,
        title=f"Registration Trends — Top {len(top_countries)} Countries ({year_start}–{year_end})",
        xaxis=dict(title="Year", showgrid=True, gridcolor="#eeeeee", dtick=1),
        yaxis=dict(title="Registered Trials", showgrid=True, gridcolor="#eeeeee"),
        legend=dict(
            orientation="v",
            x=1.01, y=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
        hovermode="x unified",
    )
    return fig


# ── Chart D — Slope chart ─────────────────────────────────────────────────────

def chart_slope(
    counts: pd.DataFrame,
    countries: list[str],
    year_start: int,
    year_end: int,
) -> go.Figure:
    """Slope chart: one line per country connecting year_start → year_end counts."""
    growth = build_growth_table(counts, year_start, year_end)
    subset = (
        growth[growth["country"].isin(countries)]
        .sort_values("count_end", ascending=False)
    )

    if subset.empty:
        fig = go.Figure()
        fig.update_layout(title="No slope data for selection", **_LAYOUT)
        return fig

    fig = go.Figure()
    for i, (_, row) in enumerate(subset.iterrows()):
        color = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(go.Scatter(
            x=[year_start, year_end],
            y=[row["count_start"], row["count_end"]],
            mode="lines+markers",
            name=row["country"],
            line=dict(color=color, width=2.5),
            marker=dict(size=9, symbol="circle"),
            hovertemplate=(
                f"<b>{row['country']}</b><br>"
                f"{year_start}: {row['count_start']:,}<br>"
                f"{year_end}: {row['count_end']:,}<br>"
                f"Change: {row['abs_growth']:+,}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_LAYOUT,
        title=f"Trial Count: {year_start} → {year_end}",
        xaxis=dict(
            tickvals=[year_start, year_end],
            ticktext=[str(year_start), str(year_end)],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(title="Registered Trials", showgrid=True, gridcolor="#eeeeee"),
        legend=dict(
            orientation="v",
            x=1.01, y=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
    )
    return fig


# ── Chart E — CAGR accelerators ───────────────────────────────────────────────

def chart_cagr_accelerators(
    counts: pd.DataFrame,
    year_start: int,
    year_end: int,
    n: int = 10,
    always_include: set[str] | None = None,
) -> go.Figure:
    """Horizontal bar: top N countries by CAGR; US always included."""
    if always_include is None:
        always_include = _US_VARIANTS

    growth = build_growth_table(counts, year_start, year_end)
    # Require at least 1 trial in year_start (CAGR undefined from zero)
    growth = growth[growth["count_start"] > 0]
    growth = growth[growth["cagr_pct"].apply(
        lambda v: isinstance(v, float) and not math.isnan(v) and not math.isinf(v)
    )]

    if growth.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=(
                f"No CAGR data for {year_start}→{year_end}.<br>"
                f"CAGR requires countries to have ≥1 trial in {year_start}.<br>"
                f"Try moving the start year forward (e.g. 2023+)."
            ),
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13, color="grey"),
            align="center",
        )
        fig.update_layout(title=f"Fastest Growing Countries — CAGR {year_start}→{year_end}", **_LAYOUT)
        return fig

    top = growth.nlargest(n, "cagr_pct").copy()
    top["highlighted"] = False

    for us_name in always_include:
        if (
            us_name in growth["country"].values
            and us_name not in top["country"].values
        ):
            row = growth[growth["country"] == us_name].copy()
            row["highlighted"] = True
            top = pd.concat([top, row], ignore_index=True)

    top = top[top["cagr_pct"].apply(lambda v: not math.isnan(v) and not math.isinf(v))]
    top = top.sort_values("cagr_pct")

    colors = [_ORANGE if h else _BLUE for h in top["highlighted"]]

    fig = go.Figure(go.Bar(
        x=top["cagr_pct"],
        y=top["country"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in top["cagr_pct"]],
        textposition="outside",
        customdata=top[["count_start", "count_end"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "CAGR: %{x:.1f}%<br>"
            f"{year_start}: %{{customdata[0]:,.0f}}<br>"
            f"{year_end}: %{{customdata[1]:,.0f}}<extra></extra>"
        ),
    ))

    max_cagr = top["cagr_pct"].max()
    fig.update_layout(
        **_LAYOUT,
        title=f"Fastest Growing Countries — CAGR {year_start}→{year_end}",
        xaxis=dict(
            title=f"CAGR (%) over {year_end - year_start} years",
            ticksuffix="%",
            showgrid=True,
            gridcolor="#eeeeee",
            range=[0, max_cagr * 1.22 if max_cagr > 0 else 10],
        ),
        yaxis=dict(title=""),
    )

    if top["highlighted"].any():
        fig.add_annotation(
            text=f"<i>{next(iter(always_include & set(top['country'])))} always included</i>",
            xref="paper", yref="paper",
            x=1, y=0, xanchor="right", yanchor="bottom",
            showarrow=False, font=dict(size=9, color="grey"),
        )

    return fig
