"""Synchronous native WebSocket session; failures close the session, never retry motion."""
from urllib.parse import urlparse
from .codec import MAX_MESSAGE, packb, unpackb
from .protocol import CHECKPOINT_REVISION


class GalaxeaSession:
    def __init__(self, endpoint="ws://127.0.0.1:8765", *, timeout_s=120):
        address = urlparse(endpoint)
        if (address.scheme != "ws" or address.hostname != "127.0.0.1"
                or address.username or address.query or address.path not in ("", "/")):
            raise ValueError("Use a loopback endpoint, including a tunnel for EC2")
        self.endpoint, self.timeout_s = endpoint, timeout_s
        self.socket = None
        self.greeting = None

    def connect(self):
        if self.socket is not None:
            return
        from websockets.sync.client import connect
        try:
            self.socket = connect(self.endpoint, open_timeout=self.timeout_s,
                                  close_timeout=2, max_size=MAX_MESSAGE)
            self.greeting = unpackb(self.socket.recv(timeout=self.timeout_s))
            if self.greeting.get("action_steps") != 32:
                raise ValueError("SO101 server must advertise32 action steps")
            health = self.greeting.get("health", {})
            if (health.get("profile") != "g05-so101"
                    or health.get("checkpoint_revision") != CHECKPOINT_REVISION
                    or health.get("status") != "ready"):
                raise ValueError("Unverified SO101 checkpoint/server identity")
        except Exception:
            self.close()
            raise

    def request(self, value):
        try:
            self.connect()
            self.socket.send(packb(value))
            result = unpackb(self.socket.recv(timeout=self.timeout_s))
            if not isinstance(result, dict):
                raise ValueError("Server response must be a map")
            if "error" in result:
                raise RuntimeError(f"Galaxea server error: {result['error']}")
            return result
        except Exception:
            self.close()
            raise

    def reset(self):
        result = self.request({"__reset__": True})
        if result != {"__reset__": True}:
            self.close()
            raise ValueError("Missing reset acknowledgement")

    def close(self):
        socket, self.socket = self.socket, None
        if socket is not None:
            socket.close()
