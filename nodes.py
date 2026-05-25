from __future__ import annotations

from .pid_decode import PiDDecode
from .pid_loader import PiDModelLoader


NODE_CLASS_MAPPINGS = {
    "PiD Model Loader": PiDModelLoader,
    "PiD Decode": PiDDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiD Model Loader": "PiD Model Loader",
    "PiD Decode": "PiD Decode",
}
