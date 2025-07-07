#!/usr/bin/env python3
"""
Wrapper to apply our SPR‐only patch, then launch the original training script
using the same arguments.
"""

# 0) force the parent directory (project root) onto sys.path so that
#    "import rde" will pick up your local rde/ folder:
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 1) import the patch module so it applies the monkey‐patch
import rde.datasets.skempi_parallel_spr

# 2) re‐execute train_DDAffinity.py as __main__, forwarding argv
from runpy import run_path

# locate your train script (adjust if its name or location differs)
SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../train_DDAffinity.py'))

if __name__ == "__main__":
    run_path(SCRIPT, run_name="__main__")
