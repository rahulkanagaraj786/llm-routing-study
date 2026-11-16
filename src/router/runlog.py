"""Record the environment a result was produced in."""
from __future__ import annotations

import datetime as dt
import importlib
import platform
import subprocess

from . import data as D


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10, cwd=D.ROOT).stdout.strip()
    except Exception:
        return None


def env_info(device: str | None = None) -> dict:
    versions = {}
    for m in ("numpy", "scipy", "sklearn", "threadpoolctl", "torch", "transformers", "tokenizers"):
        try:
            versions[m] = importlib.import_module(m).__version__
        except Exception:
            versions[m] = None
    meta_path = D.ROOT / "manifests" / "split_manifest_v1.meta.json"
    import json
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return dict(
        timestamp=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        python=platform.python_version(), platform=platform.platform(), machine=platform.machine(),
        cpu=_cmd(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
        cpu_count=_cmd(["sysctl", "-n", "hw.ncpu"]),
        ram_gb=(int(_cmd(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30) or None,
        device=device, packages=versions,
        git_commit=_cmd(["git", "rev-parse", "HEAD"]), git_dirty=bool(_cmd(["git", "status", "--porcelain"])),
        paired_sha256=meta.get("paired_sha256"), manifest_sha256=meta.get("manifest_sha256"),
    )


def protocol_sha256() -> str:
    return D.sha256_file(D.ROOT / "docs" / "PROTOCOL.md")
