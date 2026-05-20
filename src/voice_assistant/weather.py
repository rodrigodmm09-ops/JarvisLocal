from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Optional

# WMO weather interpretation codes → description in Spanish
_WMO = {
    0: "cielo despejado",
    1: "mayormente despejado",
    2: "parcialmente nublado",
    3: "nublado",
    45: "niebla",
    48: "niebla con escarcha",
    51: "llovizna ligera",
    53: "llovizna moderada",
    55: "llovizna densa",
    61: "lluvia ligera",
    63: "lluvia moderada",
    65: "lluvia intensa",
    71: "nieve ligera",
    73: "nieve moderada",
    75: "nieve intensa",
    77: "granizo",
    80: "chubascos ligeros",
    81: "chubascos moderados",
    82: "chubascos violentos",
    85: "nevada ligera",
    86: "nevada intensa",
    95: "tormenta",
    96: "tormenta con granizo",
    99: "tormenta con granizo intenso",
}


class WeatherClient:
    """Fetches current weather from Open-Meteo (free, no API key)."""

    def __init__(self, args):
        self.latitude = getattr(args, "location_latitude", 40.4168)
        self.longitude = getattr(args, "location_longitude", -3.7038)
        self.timeout = 5.0

    def get_current(self) -> Optional[dict]:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={self.latitude}&longitude={self.longitude}"
            "&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
            "&timezone=auto"
        )
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
            c = data.get("current", {})
            code = c.get("weather_code", 0)
            return {
                "temperature": round(c.get("temperature_2m", 0)),
                "humidity": round(c.get("relative_humidity_2m", 0)),
                "wind_speed": round(c.get("wind_speed_10m", 0)),
                "description": _WMO.get(code, "tiempo desconocido"),
            }
        except urllib.error.URLError as e:
            logging.error(f"[Weather] Sin conexión a Open-Meteo: {e}")
        except Exception as e:
            logging.error(f"[Weather] Error obteniendo el tiempo: {e}")
        return None

    def format_response(self, data: dict) -> str:
        temp = data["temperature"]
        desc = data["description"]
        wind = data["wind_speed"]
        humidity = data["humidity"]
        return (
            f"{temp} grados, {desc}. "
            f"Viento a {wind} kilómetros por hora y humedad del {humidity} por ciento."
        )
