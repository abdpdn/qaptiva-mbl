# qaptiva-mbl

**Reproducible many-body-localization dynamics with Qaptiva AnalogQPU and independent exact diagonalization (ED).**

`qaptiva-mbl` is a minimal Python package for the **Imbrie** and **random-field XXZ** chains. It prepares and saves disorder realizations, submits analog-simulation jobs, retrieves site-resolved magnetizations, reconstructs autocorrelators, and compares them with independently computed NumPy/SciPy ED results.

The primary interface is [`notebooks/MBL_Simulation.ipynb`](notebooks/MBL_Simulation.ipynb). The [four-page PDF user guide](docs/User_Guide.pdf) contains professionally typeset equations; its editable [LaTeX source](docs/User_Guide.tex) is also provided.

> **Version 0.1.0.** Local numerical tests and simulated QLMaaS result parsing are checked. Actual remote submission and retrieval still require validation against the intended Qaptiva server.

## Features

- Two Hamiltonians: Imbrie and random-field XXZ; OBC or PBC.
- User-selected length, couplings, disorder strength and seed, and Néel or random $Z$-basis product states.
- Adaptive AnalogQPU integration using accuracy tolerances **without** `nsteps`, or fixed-step integration **with** `nsteps`.
- Full-precision storage of disorder arrays, initial state, job ID and run metadata.
- Independent manual ED via dense diagonalization or sparse Krylov propagation.
- Site-resolved autocorrelators, site averages, ED–QPU differences and error plots.
- Separate submission, retrieval, analysis and plotting, so existing jobs can be replotted without rerunning them.

For now, each run contains **one disorder realization**. The full $2^L$ Hilbert space is used; $U(1)$-resolved XXZ computations, automated disorder averaging, and additional observables are not part of this release.

## Hamiltonians and conventions

We use $\hbar=1$, sites $j=0,\ldots,L-1$, and Pauli matrices $X_j=\sigma_j^x$, $Y_j=\sigma_j^y$, $Z_j=\sigma_j^z$. The set of nearest-neighbor bonds is

$$
\mathcal{B}_{\mathrm{OBC}}=\{(j,j+1):0\le j<L-1\},\qquad
\mathcal{B}_{\mathrm{PBC}}=\mathcal{B}_{\mathrm{OBC}}\cup\{(L-1,0)\}.
$$

OBC supports $L\ge2$ and PBC supports $L\ge3$ in this implementation.

### Imbrie model (Pauli normalization)

$$
\boxed{H_{\mathrm I}=\sum_{(i,j)\in\mathcal B}\Delta_{ij}Z_iZ_j+
\sum_{j=0}^{L-1}\big(h_j Z_j+\Gamma X_j\big)}
$$

with independent random parameters

$$
\Delta_{ij}\sim\mathcal U(\Delta_{\min},\Delta_{\max}),\qquad
h_j\sim\mathcal U(-h_0,h_0).
$$

The parameters are `gamma`, `delta_min`, `delta_max` and `h0`. **No $1/2$ factors** multiply the Pauli operators in this model.

### Random-field XXZ (spin-$\tfrac12$ normalization)

Using $S_j^\alpha=\sigma_j^\alpha/2$ and $h_j\sim\mathcal U(-h_0,h_0)$,

$$
\boxed{H_{\mathrm{XXZ}}=\sum_{(i,j)\in\mathcal B}
\left[J_{xy}(S_i^xS_j^x+S_i^yS_j^y)+J_zS_i^zS_j^z\right]
+\sum_j h_jS_j^z}.
$$

The equivalent Pauli representation used internally is

$$
H_{\mathrm{XXZ}}=\sum_{(i,j)\in\mathcal B}
\left[\frac{J_{xy}}4(X_iX_j+Y_iY_j)+\frac{J_z}4 Z_iZ_j\right]
+\frac12\sum_j h_j Z_j.
$$

Set `jxy`, `jz` and `h0` in `RunConfig`. Both the QPU job and ED Hamiltonian use the **same coefficients and saved disorder arrays**.

## Initial states and observables

The initial state $|\psi_0\rangle$ is a computational-basis product state with $s_j\in\{+1,-1\}$,

$$
Z_j|\psi_0\rangle=s_j|\psi_0\rangle.
$$

Choose `initial_state="neel"` for $s_j=(-1)^j$, or `initial_state="random"` for independent random spins $s_j=\pm1$. In the submitted bitstring, $s_j=+1$ corresponds to `0` and $s_j=-1$ to `1`; **site 0 is the leftmost bit**. Independent random streams preserve the same disorder realization when switching the initial-state type at fixed seed.

For $|\psi(t)\rangle=e^{-iHt}|\psi_0\rangle$ and $Z_j(t)=e^{iHt}Z_je^{-iHt}$, the site-resolved product-state autocorrelator is

$$
\boxed{C_j(t)=\langle\psi_0|Z_j(t)Z_j|\psi_0\rangle
=s_j\langle\psi(t)|Z_j|\psi(t)\rangle},\qquad
\boxed{C(t)=\frac1L\sum_{j=0}^{L-1}C_j(t)}.
$$

The package submits **unweighted** local $Z_j$ observables, then multiplies their returned expectation values by the saved $s_j$. For ideal evolution, $C_j(0)=1$. The single-state correlator is not generally the infinite-temperature trace correlator $2^{-L}\mathrm{Tr}[Z_j(t)Z_j]$.

## Installation

Python 3.10+ is required. From the repository root:

```bash
python -m pip install -e ".[notebook,test]"
python -m pytest -q
jupyter notebook notebooks/MBL_Simulation.ipynb
```

The proprietary Qaptiva components (`qat`, `qlmaas`) and authenticated server access must be available in your environment **only for remote submission/retrieval**. They are not bundled with the package. To run an offline ED example without any remote connection:

```bash
python examples/offline_ed.py
```

## Example: prepare, submit, retrieve, compare and plot

```python
from qaptiva_mbl import (
    RunConfig, prepare_run, submit_run, retrieve_run,
    run_ed, compare_run, plot_run,
)

# Configure one Hamiltonian realization and initial state.
cfg = RunConfig(
    model="imbrie",                # "imbrie" or "xxz"
    L=8, boundary="pbc", seed=12345,
    initial_state="neel",          # "neel" or "random"
    gamma=1.0, delta_min=0.8, delta_max=1.2, h0=3.5,
    # For XXZ, set jxy and jz as desired.
    tmax=100.0,
    error_control=True, nsteps=None,
    abs_tol=1e-8, rel_tol=1e-8, use_gpu=False,
)
run_dir = prepare_run(cfg, root="runs")

# Remote stages (run within your configured Qaptiva environment).
job_id = submit_run(run_dir)
times, C_qpu = retrieve_run(run_dir, host="YOUR-QLMAAS-HOST")

# Independent ED at the exact returned QPU timestamps.
times, C_ed = run_ed(run_dir, max_L=12)
metrics = compare_run(run_dir)
print(metrics["global_max_absolute_error"])

# Analysis and plotting use saved local data only.
plot_run(run_dir, kind="sites", sites=[0, 1, 2, 3], save="sites.pdf")
plot_run(run_dir, kind="average", save="average.pdf")
plot_run(run_dir, kind="error", save="errors.pdf")
plot_run(run_dir, kind="site_error", sites=[0, 1], save="site_errors.pdf")
```

For a **purely offline** ED run, skip both remote stages and provide an explicit time grid:

```python
import numpy as np
from qaptiva_mbl import RunConfig, prepare_run, run_ed
run_dir = prepare_run(RunConfig(model="xxz", L=8, initial_state="neel"))
times, C_ed = run_ed(run_dir, times=np.linspace(0, 100, 201))
```

### Integrator options

| Mode | Input parameters | Meaning |
|---|---|---|
| Adaptive | `error_control=True`, `nsteps=None`, `abs_tol`, `rel_tol` | Adaptive Dormand–Prince Runge–Kutta (DP5); internal step count is selected using error estimates. |
| Fixed-step | `error_control=False`, `nsteps=1000` (example) | Explicit number of integration steps; tolerances are not forwarded. |

**Never supply `nsteps` for adaptive integration.** Internal solver steps and output timestamps are distinct; retrieval reads the actual time keys in the QPU result. `use_gpu` controls the optional QPU GPU setting.

## ED benchmarking and numerical errors

Manual ED represents $H$ in the full $2^L$ basis. `method="dense"` performs Hermitian diagonalization; `method="krylov"` propagates the state with sparse matrix exponentials; `method="auto"` selects dense for $L\le9$ and Krylov otherwise. `run_ed()` uses `max_L=12` by default to limit resources, but it may be changed explicitly.

For matched sites, disorder, initial state, and times $t_k$, the signed and absolute differences are

$$
\delta C_j(t_k)=C_j^{\mathrm{QPU}}(t_k)-C_j^{\mathrm{ED}}(t_k),\qquad
\epsilon_j(t_k)=|\delta C_j(t_k)|.
$$

The package saves the time-dependent site averages and error metrics

$$
\begin{aligned}
\delta C(t_k)&=\frac1L\sum_j\delta C_j(t_k),\\
\mathrm{MAE}(t_k)&=\frac1L\sum_j |\delta C_j(t_k)|,\\
\mathrm{RMSE}(t_k)&=\sqrt{\frac1L\sum_j[\delta C_j(t_k)]^2},\\
\epsilon_{\max}(t_k)&=\max_j |\delta C_j(t_k)|.
\end{aligned}
$$

It also reports global and per-site maximum absolute errors and global RMSE. Signed deviations may cancel in $\delta C(t)$, so it differs from the average absolute error. The difference between ED and QPU is a **numerical solver discrepancy**, not an error bar for the thermodynamic limit.

## Data files and plotting

Each `prepare_run()` creates one uniquely named folder. Files appear as the respective stages complete:

```text
runs/imbrie_L8_<run-id>/
├── run.json          # model parameters, initial bitstring, job ID, status
├── sample.npz        # full-precision disorder couplings, fields and spins
├── qpu_results.npz   # times, magnetization[L,T], correlator[L,T], average[T]
├── ed_results.npz    # times, correlator[L,T], average[T]
└── comparison.npz    # signed/absolute errors and aggregate metrics
```

The retrieval adapter follows the submitted order: site 0 is in `result.value_data` and sites $1,\ldots,L-1$ are in `result.values_data`. Time-grid consistency is checked. Never generate new random arrays independently when benchmarking the same remote job.

`plot_run()` supports `kind="sites"`, `"average"`, `"error"`, and `"site_error"`, site selection (`sites=[0,2]`), logarithmic time (`log_time=True`) and PNG/PDF/SVG output via `save=`. Plots depend on locally saved arrays, not a Qaptiva connection. In ED-only mode, the notebook demonstrates plots directly from the ED arrays.

## Repository structure

```text
qaptiva-mbl/
├── src/qaptiva_mbl/       # physical models, QPU interface, ED, storage, plots
├── notebooks/             # MBL_Simulation.ipynb
├── docs/                  # User_Guide.pdf and User_Guide.tex
├── examples/              # offline_ed.py
├── tests/                 # unit and workflow tests
├── .github/workflows/     # automated tests
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## Validation and references

Offline tests check Hamiltonian matrices, Hermiticity, XXZ conservation of total $S^z$, dense/Krylov agreement, adaptive/fixed-step configuration, QPU observable order, mocked result parsing, plotting, and numerical error metrics. Run `python -m pytest -q` after installation. **Live AnalogQPU behavior requires testing in the actual connected environment.**

Qaptiva documentation: [analog overview](https://qlm.bull.com/qlm/notebooks/tutorials/analog_simulations/overview.html), [getting started](https://qlm.bull.com/qlm/notebooks/tutorials/analog_simulations/aqpu_getting_started.html), and [schedules](https://qlm.bull.com/qlm/notebooks/tutorials/schedule_arithmetic.html).

Contributions are welcome. Never commit credentials, tokens, or generated run data. See the [MIT License](LICENSE).
