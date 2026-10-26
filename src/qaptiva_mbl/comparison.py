"""Deterministic ED - AnalogQPU comparisons; never interpolate silently."""
from pathlib import Path
import numpy as np


def compare_run(run_dir):
    """Persist error arrays and return a compact numeric summary."""
    run_dir = Path(run_dir)
    with np.load(run_dir / "qpu_results.npz", allow_pickle=False) as q, \
         np.load(run_dir / "ed_results.npz", allow_pickle=False) as e:
        qt, et = q["times"], e["times"]
        if qt.shape != et.shape or not np.allclose(qt, et, rtol=1e-12, atol=1e-12):
            raise ValueError("ED and QPU time grids differ; rerun ED at QPU times")
        qc, ec = q["correlator"], e["correlator"]
        if qc.shape != ec.shape:
            raise ValueError("Incompatible site/time dimensions")
        diff = qc - ec
    abs_error = np.abs(diff)
    rms = np.sqrt(np.mean(diff * diff, axis=0))
    mean_abs = np.mean(abs_error, axis=0)
    max_abs = np.max(abs_error, axis=0)
    avg_diff = np.mean(diff, axis=0)  # signed error in site-averaged observable
    np.savez_compressed(run_dir / "comparison.npz", times=qt,
                        signed_error=diff, absolute_error=abs_error,
                        rmse=rms, mean_absolute_error=mean_abs,
                        maximum_absolute_error=max_abs,
                        averaged_correlator_difference=avg_diff)
    return {
        "global_max_absolute_error": float(abs_error.max()),
        "global_rmse": float(np.sqrt(np.mean(diff * diff))),
        "global_mean_absolute_error": float(abs_error.mean()),
        "per_site_max_absolute_error": abs_error.max(axis=1).tolist(),
    }
