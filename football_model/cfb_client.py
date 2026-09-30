"""CollegeFootballData.com API access (plain HTTPS + bearer token).

The official `cfbd` package pins pydantic v1, which breaks nflreadpy (pydantic v2), so the REST API is called directly.
The API key is read from the CFBD_API_KEY environment variable or cfbd_key.txt. Free tier: 1,000 calls per month,
so callers should cache whole seasons (see cfb_data.py).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests

KEY_FILE = Path(__file__).parent / "cfbd_key.txt"
BASE = "https://api.collegefootballdata.com"
_session = requests.Session()


def api_key() -> str:
    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        lines = [l.strip() for l in KEY_FILE.read_text().splitlines() if l.strip() and not l.startswith("#")]
        key = lines[0] if lines else ""
    if not key:
        raise SystemExit("No CFBD API key found. Paste it into cfbd_key.txt (get one free at collegefootballdata.com).")
    return key


def get(path: str, **params):
    """GET an endpoint, e.g. get('/games', year=2025). Retries transient failures; raises on auth/limit errors."""
    params = {k: v for k, v in params.items() if v is not None}
    for attempt in range(4):
        r = _session.get(BASE + path, params=params, timeout=60,
                         headers={"Authorization": f"Bearer {api_key()}", "Accept": "application/json"})
        if r.status_code == 200:
            return r.json()
        if r.status_code in (401, 403):
            raise SystemExit(f"CFBD rejected the API key ({r.status_code}). Check cfbd_key.txt.")
        if r.status_code == 429:
            raise SystemExit("CFBD monthly call limit reached. Cached seasons still work; new data resumes next month.")
        time.sleep(2 * (attempt + 1))
    r.raise_for_status()


if __name__ == "__main__":
    info = get("/info")
    print(f"CFBD key works. Tier: {info.get('tierName', info.get('patronLevel'))}. "
          f"Calls left this month: {info.get('remainingCalls')} of {info.get('monthlyLimit')}")
