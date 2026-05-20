import json
import logging
import urllib.error
import urllib.request


class ShellyController:
    """Controls a Shelly relay via local HTTP API. Supports Gen1 and Gen2/3.

    Gen1: GET /relay/{id}?turn=on|off|toggle
    Gen2/3 (Plus, Mini Gen3): POST /rpc/Switch.Set  {"id": N, "on": true|false}
    """

    def __init__(self, args):
        self.enabled = getattr(args, "shelly_enabled", False)
        self.device_ip = getattr(args, "shelly_device_ip", "")
        self.relay_id = getattr(args, "shelly_relay_id", 0)
        self.generation = getattr(args, "shelly_generation", 2)
        self.timeout = getattr(args, "shelly_timeout", 3.0)

        if self.enabled and not self.device_ip:
            logging.warning("Shelly habilitado pero sin IP configurada. Deshabilitando.")
            self.enabled = False

        if self.enabled:
            logging.info(
                f"ShellyController: Gen{self.generation} en {self.device_ip}, relay {self.relay_id}"
            )
        else:
            logging.info("ShellyController: deshabilitado (shelly_enabled = false en config.ini)")

    def available(self) -> bool:
        return self.enabled and bool(self.device_ip)

    def _request(self, url: str, data: bytes | None = None) -> dict | None:
        headers = {"Content-Type": "application/json"} if data else {}
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as e:
            logging.error(f"Shelly no alcanzable en {self.device_ip}: {e}")
        except Exception as e:
            logging.error(f"Error comunicando con Shelly: {e}")
        return None

    def turn_on(self) -> bool:
        if not self.available():
            return False
        if self.generation == 1:
            result = self._request(
                f"http://{self.device_ip}/relay/{self.relay_id}?turn=on"
            )
        else:
            body = json.dumps({"id": self.relay_id, "on": True}).encode()
            result = self._request(f"http://{self.device_ip}/rpc/Switch.Set", data=body)
        if result is not None:
            logging.info("Shelly: luz encendida")
            return True
        return False

    def turn_off(self) -> bool:
        if not self.available():
            return False
        if self.generation == 1:
            result = self._request(
                f"http://{self.device_ip}/relay/{self.relay_id}?turn=off"
            )
        else:
            body = json.dumps({"id": self.relay_id, "on": False}).encode()
            result = self._request(f"http://{self.device_ip}/rpc/Switch.Set", data=body)
        if result is not None:
            logging.info("Shelly: luz apagada")
            return True
        return False

    def toggle(self) -> bool:
        if not self.available():
            return False
        if self.generation == 1:
            result = self._request(
                f"http://{self.device_ip}/relay/{self.relay_id}?turn=toggle"
            )
        else:
            body = json.dumps({"id": self.relay_id}).encode()
            result = self._request(
                f"http://{self.device_ip}/rpc/Switch.Toggle", data=body
            )
        if result is not None:
            logging.info("Shelly: luz conmutada")
            return True
        return False

    def get_status(self) -> bool | None:
        """Returns True if on, False if off, None if unreachable."""
        if not self.available():
            return None
        if self.generation == 1:
            result = self._request(
                f"http://{self.device_ip}/relay/{self.relay_id}"
            )
            if result is not None:
                return result.get("ison", False)
        else:
            body = json.dumps({"id": self.relay_id}).encode()
            result = self._request(
                f"http://{self.device_ip}/rpc/Switch.GetStatus", data=body
            )
            if result is not None:
                return result.get("output", False)
        return None
