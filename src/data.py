from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
import re
import zipfile

import pandas as pd
import requests

FRENCH_DAILY_25_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "25_Portfolios_5x5_Daily_CSV.zip"
)


def _parse_french_daily_csv(text: str) -> pd.DataFrame:
    """Parse the daily value-weighted 25 size/BM portfolio table.

    The Kenneth French file contains descriptive text plus more than one table.
    We locate the first 25-column daily table and retain rows whose index is an
    eight-digit YYYYMMDD date.
    """
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        # Header of the first 25-portfolio table.
        if "SMALL LoBM" in line and line.count(",") >= 24:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not locate the 25-portfolio daily table header.")

    table_lines = [lines[header_idx]]
    for line in lines[header_idx + 1:]:
        first = line.split(",", 1)[0].strip()
        if re.fullmatch(r"\d{8}", first):
            table_lines.append(line)
        elif len(table_lines) > 1:
            break

    frame = pd.read_csv(StringIO("\n".join(table_lines)), index_col=0)
    if frame.shape[1] != 25:
        raise ValueError(f"Expected 25 portfolios, found {frame.shape[1]}.")

    frame.index = pd.to_datetime(frame.index.astype(str), format="%Y%m%d")
    frame = frame.apply(pd.to_numeric, errors="coerce") / 100.0
    frame = frame.sort_index().dropna(how="any")
    frame.index.name = "date"
    return frame


def download_fama_french_25_daily(
    cache_path: str | Path = "data/fama_french_25_daily.csv",
    force: bool = False,
) -> pd.DataFrame:
    """Download/cache daily 25 Fama-French size-value portfolio returns."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        out = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        out.index.name = "date"
        return out

    response = requests.get(FRENCH_DAILY_25_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("Downloaded ZIP does not contain a CSV file.")
        text = zf.read(csv_names[0]).decode("utf-8", errors="replace")

    out = _parse_french_daily_csv(text)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path)
    return out


def load_local_returns(path: str | Path) -> pd.DataFrame:
    """Load a local date-indexed return CSV (returns expressed as decimals)."""
    out = pd.read_csv(path, index_col=0, parse_dates=True)
    out = out.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    out.index.name = "date"
    return out.sort_index()
