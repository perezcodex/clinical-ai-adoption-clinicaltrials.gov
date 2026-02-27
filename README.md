# ClinicalTrials.gov — AI/LLM Trial Explorer

A Streamlit app that queries the [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api) for clinical trials involving artificial intelligence and machine learning, then visualises global country-level registration trends from 2016 to 2026.

**Live app:** https://clinical-ai-adoption-clinicaltrialsgov.streamlit.app/

---

## What it does

- Searches ClinicalTrials.gov for `"artificial intelligence"` and `"machine learning"` trials with a start date between 2016 and 2026
- Paginates the full result set via the v2 API (no API key required)
- Deduplicates on NCT ID and normalised title
- Aggregates trial counts by **recruitment country** and year
- Renders five interactive charts:
  - World map — choropleth of trial volume in the selected end year
  - Volume leaders — top N countries by raw count
  - Time series — registration trends over the selected year range
  - Slope chart — start vs end count comparison
  - CAGR accelerators — fastest-growing countries by compound annual growth rate
- Caches results locally as Parquet for 24 hours (one-click refresh available)
- Exports aggregated counts and raw trial data as CSV or Parquet

---

## Quickstart

```bash
git clone https://github.com/perezcodex/clinical-ai-adoption-clinicaltrials.gov.git
cd clinical-ai-adoption-clinicaltrials.gov

pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), click **Fetch / Refresh Data** in the sidebar, and wait ~1–3 minutes for the initial API pull. Subsequent visits load from the local cache instantly.

---

## Project structure

```
├── app.py                  # Streamlit app
├── src/
│   ├── fetch.py            # ClinicalTrials.gov API v2 client + caching
│   ├── aggregate.py        # Country/year count aggregation + CAGR helpers
│   └── charts.py           # Plotly chart builders (5 chart types)
├── data/
│   └── cache/              # Parquet cache (git-ignored)
├── requirements.txt
└── pyproject.toml
```

---

## API details

| Parameter | Value |
|---|---|
| Base URL | `https://clinicaltrials.gov/api/v2/studies` |
| Auth | None required |
| Search terms | `"artificial intelligence"`, `"machine learning"` |
| Date filter | `AREA[StartDate]RANGE[01/01/2016,12/31/2026]` |
| Page size | 1000 (maximum) |
| Rate limit | ~50 req/min (1.2 s delay between pages) |

Key fields extracted per trial: NCT ID, title, start date, recruitment countries (from locations), lead sponsor name, and conditions.

---

## Requirements

- Python ≥ 3.11
- `streamlit`, `pandas`, `plotly`, `requests`, `pyarrow`

No API keys or credentials needed.

---

## Notes

- **Country attribution** uses recruitment location countries, not sponsor country. A trial recruiting in both the US and China is counted once for each.
- **Deduplication** is applied in two passes: first on NCT ID (removes trials matching both search terms), then on normalised title (removes any cross-registered duplicates).
- **Cache** is stored at `data/cache/ctgov_ai_trials.parquet` and expires after 24 hours. On Streamlit Cloud the filesystem is ephemeral, so the fetch runs on each cold start.
