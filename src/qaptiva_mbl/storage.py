"""Per-run persistence; no overwriting disorder data or credentials."""
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np

from .config import RunConfig
from .models import SpinSample, make_sample


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_json(path: Path, obj: dict):
    """Atomically replace a small JSON record after flushing it to disk."""
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def prepare_run(cfg: RunConfig, root="runs") -> Path:
    """Prepare and persist a realization BEFORE any remote submission."""
    sample = make_sample(cfg)
    run_dir = Path(root) / f"{cfg.model}_L{cfg.L}_{uuid4().hex[:12]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(run_dir / "sample.npz", deltas=sample.deltas,
                        fields=sample.fields, spins=sample.spins)
    save_json(run_dir / "run.json", {
        "format_version": 1,
        "run_id": run_dir.name,
        "created_utc": utc_now(),
        "config": cfg.to_dict(),
        "bitstring": sample.bitstring,
        "status": "prepared",
        "job_id": None,
    })
    return run_dir


def load_run(run_dir) -> tuple[RunConfig, SpinSample, dict]:
    run_dir = Path(run_dir)
    with (run_dir / "run.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    cfg = RunConfig.from_dict(metadata["config"])
    with np.load(run_dir / "sample.npz", allow_pickle=False) as arr:
        sample = SpinSample(arr["deltas"].copy(), arr["fields"].copy(),
                            arr["spins"].copy())
    if sample.bitstring != metadata["bitstring"]:
        raise ValueError("Initial state metadata does not match sample.npz")
    return cfg, sample, metadata


def update_run(run_dir, **fields):
    run_dir = Path(run_dir)
    with (run_dir / "run.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    metadata.update(fields)
    save_json(run_dir / "run.json", metadata)
