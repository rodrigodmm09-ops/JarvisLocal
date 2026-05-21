from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Optional, Tuple

_GEO_API_URL = "http://ip-api.com/json?fields=lat,lon,city,regionName,status"
_CACHE_TTL = 600  # 10 minutes

_cache: Optional[Tuple[float, float]] = None
_cache_ts: float = 0.0


def get_location(
    lat_fallback: float,
    lon_fallback: float,
    auto: bool = True,
    timeout: float = 3.0,
) -> Tuple[float, float]:
    """Returns (latitude, longitude).

    When auto=True, tries IP-based geolocation first (ip-api.com, no key needed),
    caches the result for 10 minutes, and falls back to the static config values
    if the request fails or there is no internet connection.
    """
    if not auto:
        return lat_fallback, lon_fallback

    global _cache, _cache_ts
    now = time.monotonic()
    if _cache is not None and (now - _cache_ts) < _CACHE_TTL:
        return _cache

    try:
        with urllib.request.urlopen(_GEO_API_URL, timeout=timeout) as resp:
            data = json.loads(resp.read())
        if data.get("status") == "success":
            lat = float(data["lat"])
            lon = float(data["lon"])
            city = data.get("city", "")
            region = data.get("regionName", "")
            logging.info(
                f"[Location] Ubicación detectada: {city}, {region} ({lat:.4f}, {lon:.4f})"
            )
            _cache = (lat, lon)
            _cache_ts = now
            return _cache
    except urllib.error.URLError as e:
        logging.warning(f"[Location] Sin conexión para detectar ubicación: {e}")
    except Exception as e:
        logging.warning(f"[Location] Error al obtener ubicación automática: {e}")

    logging.info(
        f"[Location] Usando coordenadas del config: ({lat_fallback}, {lon_fallback})"
    )
    return lat_fallback, lon_fallback
