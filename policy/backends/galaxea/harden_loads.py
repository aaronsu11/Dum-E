"""Two reviewed loader edits against a pinned checkout; no permissive fallback."""
from pathlib import Path
import sys

root = Path(sys.argv[1])
for relative in (
    "src/g05/utils/checkpoint/checkpoint_utils.py",
    "src/g05/tokenizer/models/actioncodec2_v2/wrapper.py",
):
    path = root / relative
    source = path.read_text()
    if source.count("weights_only=False") != 1:
        raise RuntimeError(f"Upstream loader changed: {relative}")
    path.write_text(source.replace("weights_only=False", "weights_only=True"))
