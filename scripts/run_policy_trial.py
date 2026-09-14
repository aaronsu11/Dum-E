"""Bounded SO101 trial command. Implementation is owned by the embodiment."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embodiment.so_arm10x.trial import main

if __name__ == "__main__":
    main()
