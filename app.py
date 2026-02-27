"""
ClinicalTrials.gov AI/LLM Trial Explorer — Streamlit app.

Queries ClinicalTrials.gov for AI & machine-learning trials (2016–2026),
visualises country-level registration trends, and exports aggregated data.

Usage:
    streamlit run app.py
    # or, with uv:
    uv run streamlit run app.py
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from src.aggregate import (
    country_year_counts,
    explode_by_recruitment,
    save_counts,
    top_countries_in_year,
)
from src.charts import (
    chart_cagr_accelerators,
    chart_slope,
    chart_time_series,
    chart_volume_leaders,
    chart_world_map,
)
from src.fetch import fetch_ai_trials

logging.basicConfig(level=logging.INFO)

_CACHE_PATH = Path("data/cache/ctgov_ai_trials.parquet")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ClinicalTrials.gov AI Explorer",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 ClinicalTrials.gov — AI/LLM Trial Explorer")
st.caption(
    "Queries ClinicalTrials.gov for *artificial intelligence* and "
    "*machine learning* trials (2016–2026) and maps country-level "
    "registration trends."
)


# ── Cache status helper ───────────────────────────────────────────────────────

def _cache_age_str() -> str:
    if not _CACHE_PATH.exists():
        return "No cache"
    age_sec = (
        datetime.now(timezone.utc)
        - datetime.fromtimestamp(_CACHE_PATH.stat().st_mtime, tz=timezone.utc)
    ).total_seconds()
    if age_sec < 120:
        return "Just fetched"
    if age_sec < 3600:
        return f"Last fetched: {int(age_sec // 60)} min ago"
    if age_sec < 86400:
        return f"Last fetched: {int(age_sec // 3600)} h ago"
    return f"Last fetched: {int(age_sec // 86400)} day(s) ago"


# ── Year slider initialisation ────────────────────────────────────────────────

def _year_bounds(df: pd.DataFrame) -> tuple[int, int]:
    valid = df["year"].dropna()
    if valid.empty:
        return 2016, 2026
    return int(valid.min()), int(valid.max())


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Data")

    cache_status = _cache_age_str()
    st.caption(f"Cache: {cache_status}")

    fetch_btn = st.button(
        "Fetch / Refresh Data",
        type="primary",
        use_container_width=True,
        help="Download all matching trials from ClinicalTrials.gov (may take 1–3 min).",
    )

    st.divider()
    st.subheader("Chart controls")

    # Year range — computed once data is available
    df_loaded: pd.DataFrame | None = st.session_state.get("df")
    if df_loaded is not None:
        _mn, _mx = _year_bounds(df_loaded)
    else:
        _mn, _mx = 2016, 2026

    year_range = st.slider(
        "Year range",
        min_value=_mn,
        max_value=_mx,
        value=(_mn, _mx),
        step=1,
        key="year_slider",
    )
    year_start, year_end = year_range

    top_n = st.number_input("Top N countries", min_value=5, max_value=20, value=10, step=1)


# ── Fetch handler ─────────────────────────────────────────────────────────────

if fetch_btn:
    status_el = st.empty()
    with st.spinner("Fetching trials from ClinicalTrials.gov…"):
        try:
            df_new = fetch_ai_trials(
                cache_path=_CACHE_PATH,
                force_refresh=True,
                status_placeholder=status_el,
            )
            st.session_state["df"] = df_new
            # Reset slider bounds after new data
            st.session_state.pop("year_slider", None)
            status_el.empty()
            st.success(
                f"Loaded **{len(df_new):,}** unique AI/LLM trials "
                f"from ClinicalTrials.gov."
            )
            st.rerun()
        except Exception as exc:
            status_el.empty()
            st.error(f"Fetch failed: {exc}")
            st.stop()

# ── Auto-load from cache on first visit ──────────────────────────────────────

if st.session_state.get("df") is None and _CACHE_PATH.exists():
    with st.spinner("Loading from cache…"):
        st.session_state["df"] = fetch_ai_trials(
            cache_path=_CACHE_PATH,
            force_refresh=False,
        )

# ── Guard — need data ─────────────────────────────────────────────────────────

df: pd.DataFrame | None = st.session_state.get("df")

if df is None or df.empty:
    st.info(
        "Click **Fetch / Refresh Data** in the sidebar to download trials "
        "from ClinicalTrials.gov. This takes 1–3 minutes and results are "
        "cached locally for 24 hours."
    )
    st.stop()


# ── Dashboard ─────────────────────────────────────────────────────────────────

df_long = explode_by_recruitment(df, year_range)

if df_long.empty:
    st.warning(f"No trials with recruitment country data for {year_start}–{year_end}.")
    st.stop()

counts        = country_year_counts(df_long)
top_countries = top_countries_in_year(counts, year_end, n=int(top_n))
total_trials  = int(df[df["year"].between(year_start, year_end)]["trial_id"].nunique())
total_ctries  = counts["country"].nunique()

# Metrics row
m1, m2, m3, m4 = st.columns(4)
m1.metric("AI/LLM trials in range", f"{total_trials:,}")
m2.metric("Countries represented",  f"{total_ctries:,}")
m3.metric("Year range",             f"{year_start}–{year_end}")
m4.metric("Data source",            "ClinicalTrials.gov")

st.divider()

# Chart A — World map
st.subheader(f"Trial Registrations by Country — {year_end}")
st.plotly_chart(chart_world_map(counts, year_end), use_container_width=True)

st.divider()

# Charts B + C
col_b, col_c = st.columns(2)
with col_b:
    st.subheader(f"Top {top_n} Countries — {year_end} Volume")
    st.plotly_chart(
        chart_volume_leaders(counts, year_end, n=int(top_n)),
        use_container_width=True,
    )
with col_c:
    st.subheader(f"Registration Trends — {year_start} to {year_end}")
    st.plotly_chart(
        chart_time_series(counts, top_countries, year_start, year_end),
        use_container_width=True,
    )

st.divider()

# Charts D + E
col_d, col_e = st.columns(2)
with col_d:
    st.subheader(f"Start → End: {year_start} vs {year_end}")
    st.plotly_chart(
        chart_slope(counts, top_countries, year_start, year_end),
        use_container_width=True,
    )
with col_e:
    st.subheader(f"Fastest Growing — CAGR {year_start}→{year_end}")
    st.plotly_chart(
        chart_cagr_accelerators(counts, year_start, year_end, n=int(top_n)),
        use_container_width=True,
    )

st.divider()

# Export
st.subheader("Export")
col_save, col_dl1, col_dl2, col_dl3 = st.columns([2, 1, 1, 1])

with col_save:
    if st.button("💾  Save counts to data/cache/", use_container_width=True):
        save_counts(counts, Path("data/cache"))
        st.success("Saved to `data/cache/`")

with col_dl1:
    csv_buf = io.StringIO()
    counts.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇ Counts CSV",
        data=csv_buf.getvalue(),
        file_name="ctgov_ai_counts.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_dl2:
    parquet_buf = io.BytesIO()
    counts.to_parquet(parquet_buf, index=False)
    st.download_button(
        "⬇ Counts Parquet",
        data=parquet_buf.getvalue(),
        file_name="ctgov_ai_counts.parquet",
        mime="application/octet-stream",
        use_container_width=True,
    )

with col_dl3:
    raw_csv = io.StringIO()
    df_export = df.copy()
    df_export["recruitment_countries"] = df_export["recruitment_countries"].apply(
        lambda v: "; ".join(v) if isinstance(v, list) else ""
    )
    df_export["conditions"] = df_export["conditions"].apply(
        lambda v: "; ".join(v) if isinstance(v, list) else ""
    )
    df_export.to_csv(raw_csv, index=False)
    st.download_button(
        "⬇ Raw trials CSV",
        data=raw_csv.getvalue(),
        file_name="ctgov_ai_trials.csv",
        mime="text/csv",
        use_container_width=True,
    )

with st.expander("Preview aggregated counts table"):
    st.dataframe(counts, use_container_width=True, height=300)
