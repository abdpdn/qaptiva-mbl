# qaptiva-mbl

**Minimal, reproducible spin-chain MBL dynamics with Qaptiva AnalogQPU and independent exact diagonalization.**

A small Python package designed for researchers working on the Imbrie (random transverse-field Ising) and random-field XXZ chains. Its primary interface is the Jupyter notebook at [`notebooks/MBL_Simulation.ipynb`](notebooks/MBL_Simulation.ipynb). A short PDF manual is included at [`docs/User_Guide.pdf`](docs/User_Guide.pdf).

## Scope (v0.1.0)

- Imbrie and random-field XXZ spin-1/2 chains; open or periodic boundaries.
- User-selected parameters, disorder seed, and Néel or random **Z-basis product states**.
- Qaptiva `AnalogQPU` jobs; saved job IDs and full-precision disorder/initial-state arrays.
- Adaptive `error_control=True` (tolerances only) **or** fixed `error_control=False` (required `nsteps`).
- Independent manual NumPy/SciPy ED, with dense diagonalization or sparse Krylov propagation.
- `C_j(t) = s_j <Z_j(t)>`, the site average, and QPU-vs-ED numerical errors.
- Retrieve once, then plot/replot without connecting to Qaptiva.

This first release handles **one disorder realization per run**. It does not exploit U(1) symmetry for XXZ; both backends represent the full `2**L` spin Hilbert space. It does not implement entropy, eigenstate observables, automated disorder averaging, or distributed job orchestration.

## Hamiltonian conventions

All indices are **0-based**, `Z|0> = +|0>`, and site 0 is the leftmost character of `psi_0`.

**Imbrie (Pauli operators):**

```text
H = sum_<i,j> Delta_ij Z_i Z_j + sum_i [h_i Z_i + Gamma X_i]
Delta_ij ~ Uniform(delta_min, delta_max)
h_i ~ Uniform(-h0, h0)
```

**Random-field XXZ (spin-1/2 operators `S^alpha = sigma^alpha / 2`):**

```text
H = sum_<i,j> [Jxy (Sx_i Sx_j + Sy_i Sy_j) + Jz Sz_i Sz_j]
    + sum_i h_i Sz_i
h_i ~ Uniform(-h0, h0)
```

For PBC, one closing bond `(L-1, 0)` is added; minimum `L=3`. For OBC, bonds are `(0,1), ..., (L-2,L-1)`. The random bond array in the Imbrie model and site fields are stored in `sample.npz` at their original floating-point precision. A random initial state uses a random stream independent of disorder, so changing `initial_state` does not change the Hamiltonian for a fixed seed.

## Installation

**Python 3.10+** and NumPy, SciPy, and Matplotlib are required. From the repository root:

```bash
python -m pip install -e ".[notebook,test]"
python -m pytest -q
jupyter notebook notebooks/MBL_Simulation.ipynb
```

The proprietary **Qaptiva / QLMaaS** packages (`qat`, `qlmaas`) and access credentials are **not** bundled or installed by pip. Remote job submission and retrieval must run within an environment where those packages and your server access are already configured. The ED-only example and tests work without them.

If you don't have Qaptiva access, try:

```bash
python examples/offline_ed.py
```

## Typical workflow

```python
from qaptiva_mbl import (
    RunConfig, prepare_run, submit_run, retrieve_run,
    run_ed, compare_run, plot_run
)

# 1. Prepare a reproducible realization; no remote calls yet.
cfg = RunConfig(
    model="imbrie", L=8, boundary="pbc", seed=4011652654,
    initial_state="neel", h0=3.5, gamma=1.0,
    delta_min=0.8, delta_max=1.2, tmax=100,
    error_control=True, nsteps=None, abs_tol=1e-8, rel_tol=1e-8,
    use_gpu=False,
)
run_dir = prepare_run(cfg, root="runs")

# 2. Submit once on a Qaptiva installation, recording the returned job_id.
job_id = submit_run(run_dir)
print(job_id, run_dir)

# 3. Retrieve now or days later using this saved run directory.
#    The password login is performed by QLMaaS, never stored in run.json.
times, correlators = retrieve_run(run_dir, host="YOUR-QLMAAS-HOST")

# 4. Compare against independent ED on precisely those same times.
run_ed(run_dir, max_L=12)
print(compare_run(run_dir))

# 5. Plot saved observables (no remote call).
plot_run(run_dir, kind="sites", sites=[0, 1, 2, 3], save="sites.png")
plot_run(run_dir, kind="average", save="average.png")
plot_run(run_dir, kind="error", save="errors.png")
plot_run(run_dir, kind="site_error", sites=[0, 2], save="site_errors.png")
```

**Fixed-step integration:** replace the integrator arguments with `error_control=False, nsteps=1000`. Do not supply `nsteps` in adaptive mode. The adaptive solver is the Dormand-Prince Runge-Kutta DP5 implementation described by the user's Qaptiva setup; the QPU chooses internal integration steps using `abs_tol` / `rel_tol`. A fixed-step job requires `nsteps`. **Returned observable timestamps** are loaded from Qaptiva, not guessed from the solver step count.

**ED without Qaptiva:** prepare a run, then specify `times` explicitly:

```python
import numpy as np
run_dir = prepare_run(RunConfig(L=6, tmax=10))
times, C = run_ed(run_dir, times=np.linspace(0, 10, 101))
```

## Saved run structure

Every `prepare_run` creates a unique run folder. All the files below are written into **one folder** (no external registry is needed):

```text
runs/imbrie_L8_<unique-id>/
    run.json          # config, timestamp, status, remote job ID, bitstring
    sample.npz        # full precision delta bonds, h_i fields, initial Z spins
    qpu_results.npz   # times, magnetization[L,T], correlator[L,T], average[T]
    ed_results.npz    # times, correlator[L,T], average[T]
    comparison.npz    # signed/absolute error[L,T], RMSE[T], etc.
```

Only files relevant to steps already completed are present. The `run.json` path is portable: it does **not** embed user-specific machine paths, credentials, or hardcoded server addresses. `submission_unconfirmed` indicates an exception during submission; check the server job list before starting a new run to avoid duplicate remote jobs.

## ED comparison

For an initial Z-basis product state with spin `s_j=+1` or `-1`,

```text
C_j(t) = s_j <psi(t)| Z_j |psi(t)>
C(t)   = (1/L) sum_j C_j(t)
Delta_j(t) = C_j^AnalogQPU(t) - C_j^ED(t)
```

The comparison additionally computes `|Delta_j|`, site-mean absolute error, site-RMSE, site-maximum absolute error as functions of time, and global error statistics. **Do not compare solutions built from independently regenerated random numbers.** The package always reads the identical saved disorder arrays and bitstring in both backends. The error shown is a *solver difference*, not an error bound against the thermodynamic limit.

Use `run_ed(..., method="dense")` to diagonalize once, or `method="krylov"` for sparse propagation. The default selects dense when `L<=9`. ED defaults to a conservative `max_L=12`; the user may increase this explicitly at their own memory/time cost. Dense diagonalization scales much worse than Krylov evolution.

## Qaptiva API compatibility

The remote adapter follows the **working user-provided QLMaaS examples**:

- `from qat.core import Observable, Term, Schedule`
- `from qlmaas.qpus import AnalogQPU`
- `schedule.to_job(job_type="OBS", psi_0=..., observable=Observable.z(0,L), observables=[Observable.z(i,L) for i in range(1,L)])`
- `result = QLMaaSConnection(...).get_result(job_id)`
- `result.value_data` is site 0; `result.values_data[k]` is site `k+1`.

API references: [Qaptiva analog overview](https://qlm.bull.com/qlm/notebooks/tutorials/analog_simulations/overview.html), [analog getting started](https://qlm.bull.com/qlm/notebooks/tutorials/analog_simulations/aqpu_getting_started.html), [schedule basics](https://qlm.bull.com/qlm/notebooks/tutorials/schedule_arithmetic.html).

**Validation status:** Python tests check Hamiltonians against explicit Kronecker matrices, Hermiticity, XXZ magnetization conservation, dense vs Krylov time evolution, adaptive/fixed-step keyword routing, job-observable order, local result parsing, plotting, and ED/QPU comparisons using simulated QLMaaS-shaped results. Actual remote submission and retrieval require an accessible Qaptiva server and must be smoke-tested there.

## Development and contribution

Run `python -m pytest -q` before opening a pull request. Keep new physics in `models.py` and `ed.py`, Qaptiva-specific imports in `submit.py`/`retrieve.py`, and notebook cells short. Do not commit job data, access tokens, or passwords. Review [`LICENSE`](LICENSE) (MIT) before public release if a different license is desired.
