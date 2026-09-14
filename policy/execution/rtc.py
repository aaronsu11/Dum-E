"""RTC prefix queue and deadline validation; hardware projection is injected."""
import numpy as np

class BoundedRTCQueue:
    """Keep guidance identical to queued commands, or fail closed."""
    def __init__(self, origin, limits, tracking=3.75, *, bound_command):
        self.bound_command = bound_command
        self.origin = np.asarray(origin, dtype=float).copy()
        self.previous = self.origin.copy()
        self.limits = limits
        self.tracking = tracking
        self.items = []

    def install(self, raw, observed, elapsed=0, chunk_index=0):
        raw = np.asarray(raw)
        if (raw.shape != (50, 6) or raw.dtype.kind not in "fiu"
                or not np.isfinite(raw).all()):
            raise ValueError("Invalid complete RTC chunk")
        if type(elapsed) is not int or not 0 <= elapsed < 25:
            raise ValueError("Stale RTC chunk")
        previous = self.previous.copy()
        items = []
        for index in range(elapsed, 50):
            command = self.bound_command(
                raw[index], observed, previous, self.origin, self.limits,
                max_tracking_error=self.tracking)
            items.append({"raw": raw[index].astype(float).copy(), "command": command,
                          "chunk_index": chunk_index, "action_index": index})
            previous = command
        # Install atomically after validating/projecting the entire replacement.
        self.items = items

    def prefix(self):
        if len(self.items) != 25:
            raise ValueError("RTC request must start with exactly 25 queued actions")
        return np.stack([item["command"] for item in self.items]).copy()

    def consume(self, observed):
        if not self.items:
            raise RuntimeError("RTC action buffer exhausted")
        item = self.items[0]
        checked = self.bound_command(
            item["command"], observed, self.previous, self.origin, self.limits,
            max_tracking_error=self.tracking)
        if not np.allclose(checked, item["command"], rtol=0, atol=1e-8):
            raise RuntimeError("Queued RTC target no longer safe; invalidate prefix and hold")
        self.items.pop(0)
        self.previous = item["command"].copy()
        return item


def accept_reply(pending, now, epoch, elapsed):
    """Deadline applies even if a late future is already marked done."""
    if pending["epoch"] != epoch:
        raise RuntimeError("Old RTC instruction epoch")
    if now - pending["started"] >= pending["deadline"]:
        raise RuntimeError("RTC inference deadline missed; hold")
    if elapsed >= 25:
        raise RuntimeError("RTC prefix exhausted; hold")
