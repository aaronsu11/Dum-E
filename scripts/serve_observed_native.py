"""Compatibility command for policy.backends.isaac_groot.server."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy.backends.isaac_groot.server import main

if __name__ == "__main__":
    raise SystemExit(main())
