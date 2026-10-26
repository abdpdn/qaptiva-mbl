"""Qaptiva job submission. Imports are deliberately lazy for ED-only installs."""
from pathlib import Path
import time

from .models import pauli_terms
from .storage import load_run, update_run, utc_now


def build_job(cfg, sample):
    """Measure raw Z_i, not s_i Z_i; multiply by s_i during retrieval."""
    try:
        from qat.core import Observable, Term, Schedule
    except ImportError as exc:
        raise ImportError("Qaptiva `qat` is required for submission; use its supplied environment.") from exc
    hamiltonian = Observable(cfg.L)
    for coefficient, operator, sites in pauli_terms(cfg, sample):
        if coefficient != 0:
            hamiltonian.add_term(Term(coefficient, operator, list(sites)))
    schedule = Schedule(drive=[(1.0, hamiltonian)], tmax=cfg.tmax)
    # Matches the user's proven site-indexed result convention:
    # value_data -> Z_0; values_data[k] -> Z_(k+1).
    return schedule.to_job(
        job_type="OBS", psi_0=sample.bitstring,
        observable=Observable.z(0, cfg.L),
        observables=[Observable.z(i, cfg.L) for i in range(1, cfg.L)],
    )


def create_qpu(cfg):
    try:
        from qlmaas.qpus import AnalogQPU
    except ImportError as exc:
        raise ImportError("`qlmaas` with AnalogQPU is needed to submit remote jobs.") from exc
    kwargs = {"error_control": cfg.error_control, "use_GPU": cfg.use_gpu}
    if cfg.error_control:
        # Adaptive Dormand-Prince integration: no `nsteps` argument.
        kwargs.update(abs_tol=cfg.abs_tol, rel_tol=cfg.rel_tol)
    else:
        # Explicit fixed-step count: don't pass adaptive tolerances.
        kwargs["nsteps"] = cfg.nsteps
    return AnalogQPU(**kwargs)


def submit_run(run_dir, qpu=None) -> str:
    """Submit a prepared run and persist its remote ID.

    `qpu` may be injected for local unit testing. Submission can take time;
    the returned object is expected to expose `job_id`, as in user examples.
    """
    run_dir = Path(run_dir)
    cfg, sample, metadata = load_run(run_dir)
    if metadata.get("job_id") is not None:
        raise ValueError("This run already has a job ID; prepare a new run to resubmit")
    job = build_job(cfg, sample)
    backend = qpu if qpu is not None else create_qpu(cfg)
    update_run(run_dir, status="submitting", submission_started_utc=utc_now())
    start = time.perf_counter()
    try:
        result = backend.submit(job)
        job_id = getattr(result, "job_id", None)
        if not job_id:
            raise RuntimeError("Submission returned no job_id; check remote account/job history")
    except Exception as exc:
        update_run(run_dir, status="submission_unconfirmed", submission_error=str(exc))
        raise
    update_run(run_dir, status="submitted", job_id=str(job_id),
               submitted_utc=utc_now(), submission_seconds=time.perf_counter() - start)
    return str(job_id)
