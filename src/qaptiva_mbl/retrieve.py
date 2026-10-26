"""Download time-resolved local Z expectation values using a job ID."""
from collections.abc import Mapping
from pathlib import Path
import numpy as np

from .storage import load_run, update_run, utc_now


def _real_value(obj) -> float:
    """QLMaaS result values often expose `.re`, unlike Python complex."""
    number = obj.re if hasattr(obj, "re") else (obj.real if hasattr(obj, "real") else obj)
    val = float(number)
    if not np.isfinite(val):
        raise ValueError("Nonfinite observable value")
    return val


def _one_site(series):
    if not isinstance(series, Mapping) or not series:
        raise ValueError("Expected a nonempty time->value dictionary from QLMaaS")
    ordered = sorted(series.items(), key=lambda pair: float(pair[0]))
    times = np.array([float(key) for key, _ in ordered], dtype=float)
    values = np.array([_real_value(val) for _, val in ordered], dtype=float)
    if not (np.all(np.isfinite(times)) and np.all(np.diff(times) > 0)):
        raise ValueError("Remote times must be finite and strictly increasing")
    return times, values


def extract_correlators(result, initial_spins):
    """Return times, magnetization[L,T], and C[L,T] from a QLM result.

    Uses the exact `observable=Z0`, `observables=[Z1,...]` order used
    during submission. Site indexing and time alignment are checked.
    """
    spins = np.asarray(initial_spins)
    L = len(spins)
    times, site0 = _one_site(result.value_data)
    others = list(result.values_data) if result.values_data is not None else []
    if len(others) != L - 1:
        raise ValueError(f"Expected {L-1} secondary observables, got {len(others)}")
    magnetization = np.empty((L, len(times)), dtype=float)
    magnetization[0] = site0
    for site, data in enumerate(others, start=1):
        site_times, site_values = _one_site(data)
        if not np.array_equal(site_times, times):
            if site_times.shape != times.shape or not np.allclose(
                site_times, times, rtol=1e-12, atol=1e-12
            ):
                raise ValueError(f"Time grid mismatch for site {site}")
        magnetization[site] = site_values
    if not np.all(np.isin(spins, [-1, 1])):
        raise ValueError("Initial spins must be Z eigenvalues +/-1")
    return times, magnetization, spins[:, None] * magnetization


def retrieve_run(run_dir, *, connection=None, host=None, port=443,
                 authentication="password", result=None):
    """Retrieve a submitted job. Supply a connected QLMaaSConnection or host.

    `result` is an optional injected test object, never needed by end users.
    Credentials are neither accepted here nor persisted by the package.
    """
    run_dir = Path(run_dir)
    _, sample, metadata = load_run(run_dir)
    job_id = metadata.get("job_id")
    if not job_id:
        raise ValueError("No job_id stored: submit_run() must complete first")
    if result is None:
        if connection is None:
            if not host:
                raise ValueError("Provide connection=... or host='your Qaptiva server'")
            try:
                from qat.qlmaas import QLMaaSConnection
            except ImportError as exc:
                raise ImportError("Qaptiva `qat.qlmaas` is needed to retrieve jobs") from exc
            connection = QLMaaSConnection(host=host, port=port,
                                         authentication=authentication)
        result = connection.get_result(job_id)
    times, mag, corr = extract_correlators(result, sample.spins)
    np.savez_compressed(run_dir / "qpu_results.npz", times=times,
                        magnetization=mag, correlator=corr, average=corr.mean(axis=0))
    update_run(run_dir, status="retrieved", retrieved_utc=utc_now(),
               n_output_times=int(len(times)))
    return times, corr
