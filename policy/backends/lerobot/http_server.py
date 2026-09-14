"""Start one pinned evaluation profile. No controller, hardware or task execution."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import traceback

from policy.checkpoints import PROFILES
from policy.backends.lerobot.transports.http import MAX_BODY_BYTES, decode_request, decode_rtc_request
from .runtime import ModelRuntime


def handler(runtime):
    gate = threading.BoundedSemaphore(1)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def respond(self, code, result):
            raw = json.dumps(result, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path != "/health":
                return self.respond(404, {"error": "Not found"})
            return self.respond(200, runtime.health())

        def do_POST(self):
            if self.path not in ("/infer", "/infer/rtc"):
                return self.respond(404, {"error": "Not found"})
            if not gate.acquire(blocking=False):
                return self.respond(409, {"error": "One inference request at a time"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY_BYTES:
                    return self.respond(413, {"error": "Invalid request size"})
                data = json.loads(self.rfile.read(length))
                if self.path == "/infer/rtc":
                    request, rtc = decode_rtc_request(data)
                    result = runtime.infer_rtc(request, rtc)
                else:
                    request = decode_request(data)
                    result = runtime.infer(*request)
                self.respond(200, result)
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(400, {"error": str(exc)})
            except Exception as exc:
                traceback.print_exc()
                self.respond(503, {"error": f"{type(exc).__name__}: {exc}", "physical_ready": False})
            finally:
                gate.release()
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1"])
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    try:
        runtime = ModelRuntime(args.profile)
        result = runtime.health()
        if args.evidence_dir:
            args.evidence_dir.mkdir(parents=True, exist_ok=True)
            (args.evidence_dir / "startup.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        ThreadingHTTPServer((args.host, args.port), handler(runtime)).serve_forever()
    except Exception as exc:
        if args.evidence_dir:
            args.evidence_dir.mkdir(parents=True, exist_ok=True)
            (args.evidence_dir / "startup-failure.json").write_text(json.dumps({
                "profile": args.profile, "error_type": type(exc).__name__, "error": str(exc),
                "physical_ready": False,
            }, indent=2))
        raise


if __name__ == "__main__":
    main()
