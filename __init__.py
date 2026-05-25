from __future__ import annotations

from pathlib import Path

import folder_paths


def _register_pid_model_folder() -> None:
    pid_dir = Path(folder_paths.models_dir) / "pid"
    pid_dir.mkdir(parents=True, exist_ok=True)

    supported_exts = set(getattr(folder_paths, "supported_pt_extensions", {".pth"}))
    if "pid" in folder_paths.folder_names_and_paths:
        paths, exts = folder_paths.folder_names_and_paths["pid"]
        if str(pid_dir) not in paths:
            paths.insert(0, str(pid_dir))
        exts.update(supported_exts)
    else:
        folder_paths.folder_names_and_paths["pid"] = ([str(pid_dir)], supported_exts)


_register_pid_model_folder()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
