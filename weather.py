"""Game-time weather forecasts for outdoor NFL games (Open-Meteo: free, no API key).

The model's totals use strong wind (mph above 10) and cold (degrees below 40F). Past games use the recorded weather;
upcoming games use the forecast for the stadium's city at kickoff. Indoor / closed-roof games are calm and mild.
"""
from __future__ import annotations

import requests
import pandas as pd

GEO = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"
_geo_cache: dict[str, tuple[float, float] | None] = {}


def _coords(city: str, state: str | None = None):
    key = f"{city}|{state}"
    if key in _geo_cache:
        return _geo_cache[key]
    try:
        params = {"name": city, "count": 10}
        if state:                                   # "City, ST" = a US venue; no state (e.g. London) = search worldwide
            params["countryCode"] = "US"
        js = requests.get(GEO, params=params, timeout=15).json()
        res = js.get("results") or []
        if state:
            res = [r for r in res if state.lower() in str(r.get("admin1", "")).lower() or
                   state.upper() == str(r.get("admin1_code", "")).upper()] or res
        _geo_cache[key] = (res[0]["latitude"], res[0]["longitude"]) if res else None
    except (requests.RequestException, ValueError, KeyError):
        _geo_cache[key] = None
    return _geo_cache[key]


STATE_NAMES = {"AZ": "Arizona", "GA": "Georgia", "MD": "Maryland", "NY": "New York", "NC": "North Carolina", "IL": "Illinois",
               "OH": "Ohio", "TX": "Texas", "CO": "Colorado", "MI": "Michigan", "WI": "Wisconsin", "IN": "Indiana",
               "FL": "Florida", "MO": "Missouri", "NV": "Nevada", "CA": "California", "MA": "Massachusetts",
               "LA": "Louisiana", "NJ": "New Jersey", "PA": "Pennsylvania", "WA": "Washington", "TN": "Tennessee",
               "MN": "Minnesota", "DC": "District of Columbia", "VA": "Virginia"}


def forecast(city_state: str, kickoff_utc) -> dict:
    """{'temp_f', 'wind_mph'} at kickoff for 'City, ST'; {} if unavailable (more than ~2 weeks out, or no match)."""
    if not isinstance(city_state, str) or not city_state.strip():
        return {}
    parts = [p.strip() for p in city_state.split(",")]
    city, st = parts[0], (parts[1] if len(parts) > 1 else None)
    c = _coords(city, STATE_NAMES.get(st, st))
    if c is None:
        return {}
    k = pd.Timestamp(kickoff_utc).tz_convert("UTC") if pd.Timestamp(kickoff_utc).tzinfo else pd.Timestamp(kickoff_utc, tz="UTC")
    try:
        js = requests.get(FORECAST, params={"latitude": c[0], "longitude": c[1], "hourly": "temperature_2m,wind_speed_10m",
                                            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "timezone": "UTC",
                                            "forecast_days": 16}, timeout=15).json()
        h = pd.DataFrame({"t": pd.to_datetime(js["hourly"]["time"], utc=True), "temp": js["hourly"]["temperature_2m"],
                          "wind": js["hourly"]["wind_speed_10m"]})
    except (requests.RequestException, ValueError, KeyError):
        return {}
    h = h[(h["t"] >= k - pd.Timedelta(hours=1)) & (h["t"] <= k + pd.Timedelta(hours=3))]
    if h.empty:
        return {}
    return {"temp_f": float(h["temp"].mean()), "wind_mph": float(h["wind"].mean())}


def fill_forecasts(wk: pd.DataFrame) -> pd.DataFrame:
    """For upcoming outdoor games with no recorded weather, fill temp_f / wind_mph from the forecast and rebuild the
    model's weather features (wind_strong, cold)."""
    out = wk.copy()
    outdoor = ~out.get("roof", pd.Series("outdoors", index=out.index)).isin(["dome", "closed"])
    need = outdoor & out["wind_mph"].isna() & out.get("city", pd.Series(index=out.index)).notna()
    for i in out.index[need]:
        f = forecast(out.at[i, "city"], out.at[i, "kickoff"])
        if f:
            out.at[i, "temp_f"], out.at[i, "wind_mph"] = f["temp_f"], f["wind_mph"]
    out["wind_strong"] = (out["wind_mph"].fillna(10) - 10).clip(lower=0)
    out["cold"] = (40 - out["temp_f"].fillna(60)).clip(lower=0)
    return out
