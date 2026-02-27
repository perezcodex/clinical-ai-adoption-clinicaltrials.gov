"""
ClinicalTrials.gov API v2 client for AI/LLM trial retrieval.

Searches for "artificial intelligence" and "machine learning" trials with a
start date between 2016 and 2026, paginates all results, deduplicates, and
caches to Parquet (<24 h TTL).
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
_SEARCH_TERMS = ["artificial intelligence", "machine learning"]
_DATE_FILTER = "AREA[StartDate]RANGE[01/01/2016,12/31/2026]"
_PAGE_SIZE = 1000
_RATE_LIMIT_DELAY = 1.2  # seconds between requests (~50 req/min)

# Returned fields (reduces payload size)
_FIELDS = ",".join([
    "protocolSection.identificationModule",
    "protocolSection.statusModule.startDateStruct",
    "protocolSection.contactsLocationsModule.locations",
    "protocolSection.sponsorCollaboratorsModule.leadSponsor",
    "protocolSection.conditionsModule",
])

# ---------------------------------------------------------------------------
# Country normalisation (mirrors ingest.py from who-ictrp-explorer)
# ---------------------------------------------------------------------------

_COUNTRY_ALIASES: dict[str, str | None] = {
    "United States of America": "United States",
    "USA":                      "United States",
    "U.S.A.":                   "United States",
    "U.S.":                     "United States",
    "America":                  "United States",
    "Viet Nam":                 "Vietnam",
    "Korea, Republic of":       "South Korea",
    "Korea, South":             "South Korea",
    "Republic of Korea":        "South Korea",
    "Iran (Islamic Republic of)":   "Iran",
    "Iran, Islamic Republic of":    "Iran",
    "People's Republic of China":   "China",
    "The Netherlands":          "Netherlands",
    "Russian Federation":       "Russia",
    "Czechia":                  "Czech Republic",
    "Türkiye":                  "Turkey",
    "Turkey (Türkiye)":         "Turkey",
    "England":                  "United Kingdom",
    "Scotland":                 "United Kingdom",
    "Wales":                    "United Kingdom",
    "Northern Ireland":         "United Kingdom",
}

_BAD_CAPS = re.compile(r"[A-Z]{2,}(?=[a-z])")


def _normalize_country(name: str) -> str | None:
    name = name.strip()
    if not name:
        return None
    if name in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[name]
    if name[0].islower():
        return name.title()
    if _BAD_CAPS.search(name):
        return name.title()
    return name


# ---------------------------------------------------------------------------
# Dedup by title (mirrors ingest.py)
# ---------------------------------------------------------------------------

def _dedup_by_title(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    before = len(df)
    df = df.copy()
    df["_title_key"] = df["title"].str.strip().str.lower()
    df = df.sort_values("registration_date", ascending=True, na_position="last")
    mask_has_title = df["_title_key"].notna() & (df["_title_key"] != "")
    deduped = pd.concat([
        df[mask_has_title].drop_duplicates(subset="_title_key", keep="first"),
        df[~mask_has_title],
    ], ignore_index=True)
    deduped = deduped.drop(columns="_title_key")
    dropped = before - len(deduped)
    if dropped:
        log.info("  Title dedup: removed %d duplicate(s)", dropped)
    return deduped


# ---------------------------------------------------------------------------
# API extraction
# ---------------------------------------------------------------------------

def _extract_trial(study: dict) -> dict | None:
    ps = study.get("protocolSection", {})

    id_mod = ps.get("identificationModule", {})
    trial_id = id_mod.get("nctId")
    if not trial_id:
        return None
    title = id_mod.get("briefTitle", "")

    status_mod = ps.get("statusModule", {})
    raw_date = status_mod.get("startDateStruct", {}).get("date", "")
    try:
        reg_date = pd.to_datetime(raw_date)
    except Exception:
        reg_date = pd.NaT
    year = reg_date.year if pd.notna(reg_date) else None

    sponsor_mod = ps.get("sponsorCollaboratorsModule", {})
    primary_sponsor = sponsor_mod.get("leadSponsor", {}).get("name", "")

    locations = ps.get("contactsLocationsModule", {}).get("locations", [])
    seen: set[str] = set()
    countries: list[str] = []
    for loc in locations:
        raw_country = loc.get("country", "")
        if raw_country:
            norm = _normalize_country(raw_country)
            if norm and norm not in seen:
                seen.add(norm)
                countries.append(norm)

    conditions = ps.get("conditionsModule", {}).get("conditions", [])

    return {
        "trial_id":              trial_id,
        "registration_date":     reg_date,
        "year":                  year,
        "title":                 title,
        "primary_sponsor":       primary_sponsor,
        "recruitment_countries": countries,
        "conditions":            conditions,
        "source":                "ClinicalTrials.gov",
    }


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

def _fetch_term(term: str, status_placeholder=None) -> list[dict]:
    """Paginate all results for a single search term."""
    params: dict = {
        "query.term":      term,
        "filter.advanced": _DATE_FILTER,
        "pageSize":        _PAGE_SIZE,
        "fields":          _FIELDS,
    }

    records: list[dict] = []
    page_num = 0

    while True:
        page_num += 1
        if status_placeholder is not None:
            status_placeholder.markdown(
                f"Fetching **{term}** — page {page_num} "
                f"({len(records):,} trials so far)…"
            )

        resp = requests.get(_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for study in data.get("studies", []):
            rec = _extract_trial(study)
            if rec:
                records.append(rec)

        next_token = data.get("nextPageToken")
        if not next_token:
            break

        params["pageToken"] = next_token
        time.sleep(_RATE_LIMIT_DELAY)

    return records


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_ai_trials(
    cache_path: Path,
    force_refresh: bool = False,
    status_placeholder=None,
) -> pd.DataFrame:
    """
    Return AI/LLM trials from ClinicalTrials.gov.

    Uses a Parquet cache at `cache_path`; bypasses it when `force_refresh`
    is True or the file is older than 24 hours.

    `status_placeholder` is an optional st.empty() element for live progress.
    """
    _EMPTY_COLS = [
        "trial_id", "registration_date", "year", "title",
        "primary_sponsor", "recruitment_countries", "conditions", "source",
    ]

    # ── Cache hit ────────────────────────────────────────────────────────────
    if not force_refresh and cache_path.exists():
        age = (
            datetime.now(timezone.utc)
            - datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
        ).total_seconds()
        if age < 86_400:
            df = pd.read_parquet(cache_path)
            # Deserialise semicolon strings → lists
            for col in ("recruitment_countries", "conditions"):
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda v: [x for x in str(v).split(";") if x] if isinstance(v, str) else (v if isinstance(v, list) else [])
                    )
            log.info("Loaded %d trials from cache (%s)", len(df), cache_path)
            return df

    # ── Fetch from API ───────────────────────────────────────────────────────
    all_records: list[dict] = []
    for term in _SEARCH_TERMS:
        records = _fetch_term(term, status_placeholder=status_placeholder)
        all_records.extend(records)
        log.info("Fetched %d trials for term '%s'", len(records), term)

    if not all_records:
        return pd.DataFrame(columns=_EMPTY_COLS)

    df = pd.DataFrame(all_records)

    # Dedup on trial_id first (same trial matching both search terms)
    before = len(df)
    df = df.drop_duplicates(subset="trial_id", keep="first")
    log.info("Dedup by trial_id: %d → %d", before, len(df))

    # Dedup on title (cross-registry duplicates)
    df = _dedup_by_title(df)
    log.info("Final: %d unique trials", len(df))

    # ── Write cache ──────────────────────────────────────────────────────────
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_flat = df.copy()
    df_flat["recruitment_countries"] = df_flat["recruitment_countries"].apply(
        lambda v: ";".join(v) if isinstance(v, list) else ""
    )
    df_flat["conditions"] = df_flat["conditions"].apply(
        lambda v: ";".join(v) if isinstance(v, list) else ""
    )
    df_flat.to_parquet(cache_path, index=False)
    log.info("Cached %d trials → %s", len(df), cache_path)

    return df
