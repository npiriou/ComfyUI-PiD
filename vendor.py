from __future__ import annotations

import sys
from pathlib import Path


def get_vendor_root() -> Path:
    return Path(__file__).resolve().parent / "third_party" / "PiD"


def ensure_pid_on_path() -> Path:
    vendor_root = get_vendor_root()
    vendor_str = str(vendor_root)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)
    return vendor_root
