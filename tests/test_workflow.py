from types import SimpleNamespace
import json
import sys
import types
import numpy as np
import pytest

from qaptiva_mbl import RunConfig, prepare_run, load_run, retrieve_run, run_ed, compare_run, plot_run
from qaptiva_mbl.submit import submit_run, create_qpu, build_job
from qaptiva_mbl.ed import evolve_ed


class CloudValue:
    def __init__(self, value):
        self.re = float(value)


def fake_remote_result(times, correlator, spins):
    # QPU stores raw Z magnetizations rather than s_i*Z_i.
    mags = correlator * spins[:, None]
    series = [{str(t): CloudValue(mags[j, k]) for k, t in enumerate(times)}
              for j in range(len(spins))]
    return SimpleNamespace(value_data=series[0], values_data=series[1:])


def test_local_retrieval_comparison_and_plots(tmp_path):
    cfg = RunConfig(model="imbrie", L=4, seed=12, tmax=1.0,
                    boundary="obc", initial_state="random")
    folder = prepare_run(cfg, root=tmp_path)
    cfg2, sample, meta = load_run(folder)
    assert meta["status"] == "prepared"
    assert meta["job_id"] is None
    times = np.array([0, 0.12, 0.5, 1.0])
    correlator = evolve_ed(cfg2, sample, times, method="dense")
    result = fake_remote_result(times, correlator, sample.spins)
    # Inject a fake job ID, as though submit_run had returned it.
    from qaptiva_mbl.storage import update_run
    update_run(folder, job_id="JobTest001", status="submitted")
    rt, qc = retrieve_run(folder, result=result)
    np.testing.assert_array_equal(rt, times)
    np.testing.assert_allclose(qc, correlator)
    et, ec = run_ed(folder)
    np.testing.assert_array_equal(et, times)
    np.testing.assert_allclose(ec, qc, atol=1e-13)
    summary = compare_run(folder)
    assert summary["global_max_absolute_error"] < 1e-12
    fig = plot_run(folder, kind="sites", sites=[0, 2], save=folder / "sites.png")
    assert (folder / "sites.png").exists()
    import matplotlib.pyplot as plt
    plt.close(fig)
    fig = plot_run(folder, kind="average")
    plt.close(fig)
    fig = plot_run(folder, kind="error")
    plt.close(fig)
    fig = plot_run(folder, kind="site_error", sites=[0, 2])
    plt.close(fig)
    assert load_run(folder)[2]["job_id"] == "JobTest001"


def test_mismatch_detection(tmp_path):
    cfg = RunConfig(L=3)
    folder = prepare_run(cfg, root=tmp_path)
    from qaptiva_mbl.storage import update_run
    update_run(folder, job_id="Job2")
    sample = load_run(folder)[1]
    res = fake_remote_result(np.array([0., .2]), np.ones((3, 2)), sample.spins)
    res.values_data[0] = {"0.0": CloudValue(1), "0.3": CloudValue(1)}
    with pytest.raises(ValueError, match="Time grid mismatch"):
        retrieve_run(folder, result=res)


def test_integration_configuration_forwarded_exactly(monkeypatch):
    calls = []
    class FakeAnalogQPU:
        def __init__(self, **kwargs):
            calls.append(kwargs)
    pkg = types.ModuleType("qlmaas")
    submodule = types.ModuleType("qlmaas.qpus")
    submodule.AnalogQPU = FakeAnalogQPU
    monkeypatch.setitem(sys.modules, "qlmaas", pkg)
    monkeypatch.setitem(sys.modules, "qlmaas.qpus", submodule)
    create_qpu(RunConfig(error_control=True, nsteps=None))
    assert calls[-1] == {"error_control": True, "use_GPU": False,
                          "abs_tol": 1e-8, "rel_tol": 1e-8}
    create_qpu(RunConfig(error_control=False, nsteps=250, use_gpu=True))
    assert calls[-1] == {"error_control": False, "use_GPU": True, "nsteps": 250}


def test_adapter_site_observables_and_submission(tmp_path, monkeypatch):
    class FakeObservable:
        def __init__(self, L):
            self.terms = []
            self.L = L
        def add_term(self, term):
            self.terms.append(term)
        @staticmethod
        def z(site, n):
            return f"Z{site}"
    class FakeTerm:
        def __init__(self, coeff, pauli, sites):
            self.coeff, self.pauli, self.sites = coeff, pauli, sites
    class FakeSchedule:
        def __init__(self, drive, tmax):
            self.drive, self.tmax = drive, tmax
        def to_job(self, **kwargs):
            return kwargs
    qat = types.ModuleType("qat")
    core = types.ModuleType("qat.core")
    core.Observable, core.Term, core.Schedule = FakeObservable, FakeTerm, FakeSchedule
    monkeypatch.setitem(sys.modules, "qat", qat)
    monkeypatch.setitem(sys.modules, "qat.core", core)
    cfg = RunConfig(model="xxz", L=3, boundary="pbc")
    folder = prepare_run(cfg, root=tmp_path)
    job = build_job(cfg, load_run(folder)[1])
    assert job["observable"] == "Z0"
    assert job["observables"] == ["Z1", "Z2"]
    assert job["psi_0"] == load_run(folder)[1].bitstring
    class FakeQPU:
        def submit(self, job):
            return SimpleNamespace(job_id="JobABC")
    assert submit_run(folder, qpu=FakeQPU()) == "JobABC"
    assert load_run(folder)[2]["job_id"] == "JobABC"
    with pytest.raises(ValueError, match="already"):
        submit_run(folder, qpu=FakeQPU())
