"""Independent manual ED using sparse Pauli Hamiltonians.

    Computational basis convention: site 0 is the leftmost bit.
    |0> has sigma_z=+1 and |1> has sigma_z=-1.
"""
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import expm_multiply

from .models import pauli_terms
from .storage import load_run


def _site_mask(L, site):
    return 1 << (L - 1 - site)


def manual_hamiltonian(cfg, sample) -> csr_matrix:
    """Construct H as a CSR matrix by applying Pauli strings to basis states.

    This is deliberately independent of Qaptiva's Observable machinery.
    """
    dimension = 1 << cfg.L
    cols = np.arange(dimension, dtype=np.int64)
    diagonal = np.zeros(dimension, dtype=np.float64)
    row_blocks = []
    col_blocks = []
    val_blocks = []
    for coefficient, ops, sites in pauli_terms(cfg, sample):
        if coefficient == 0:
            continue
        row = cols.copy()
        multiplier = np.full(dimension, complex(coefficient), dtype=np.complex128)
        for axis, site in zip(ops, sites):
            mask = _site_mask(cfg.L, site)
            # +1 on |0>, -1 on |1>.
            sign = 1 - 2 * ((cols & mask) != 0).astype(np.int8)
            if axis == "Z":
                multiplier *= sign
            elif axis == "X":
                row ^= mask
            elif axis == "Y":
                row ^= mask
                multiplier *= 1j * sign
            else:
                raise ValueError(f"Unsupported Pauli operator: {axis}")
        if np.array_equal(row, cols):
            diagonal += multiplier.real
            if np.any(np.abs(multiplier.imag) > 1e-12):
                raise ValueError("Hamiltonian diagonal must be real")
        else:
            row_blocks.append(row)
            col_blocks.append(cols)
            val_blocks.append(multiplier)
    if row_blocks:
        rows = np.concatenate(row_blocks)
        columns = np.concatenate(col_blocks)
        values = np.concatenate(val_blocks)
        matrix = coo_matrix((values, (rows, columns)),
                            shape=(dimension, dimension)).tocsr()
    else:
        matrix = csr_matrix((dimension, dimension), dtype=complex)
    matrix.setdiag(matrix.diagonal() + diagonal)
    matrix.eliminate_zeros()
    return matrix


def state_vector(sample) -> np.ndarray:
    dimension = 1 << len(sample.spins)
    index = int(sample.bitstring, 2)
    psi = np.zeros(dimension, dtype=np.complex128)
    psi[index] = 1.0
    return psi


def evolve_ed(cfg, sample, times, method="auto"):
    """Return shape (L,T) autocorrelators at the requested nonnegative times.

    dense: diagonalize once; fast for modest L and many times.
    krylov: successive sparse expm_multiply steps, with no eigendecomposition.
    The default switches at L=9 to avoid expensive dense diagonalization.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)):
        raise ValueError("times must be a nonempty finite 1D array")
    if times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("times must be nonnegative and strictly increasing")
    if method not in {"auto", "dense", "krylov"}:
        raise ValueError("method must be auto, dense, or krylov")
    if method == "auto":
        method = "dense" if cfg.L <= 9 else "krylov"
    H = manual_hamiltonian(cfg, sample)
    psi0 = state_vector(sample)
    dimension = len(psi0)
    indices = np.arange(dimension, dtype=np.int64)
    z_signs = np.column_stack([
        1 - 2 * ((indices & _site_mask(cfg.L, site)) != 0).astype(float)
        for site in range(cfg.L)
    ])
    corr = np.empty((cfg.L, len(times)), dtype=float)

    def observe(psi, it):
        mz = np.abs(psi)**2 @ z_signs
        corr[:, it] = sample.spins * mz

    if method == "dense":
        energies, eigenvectors = np.linalg.eigh(H.toarray())
        amplitudes = eigenvectors.conj().T @ psi0
        for k, t in enumerate(times):
            psi = eigenvectors @ (np.exp(-1j * energies * t) * amplitudes)
            observe(psi, k)
    else:
        psi = psi0.copy()
        prev = 0.0
        for k, t in enumerate(times):
            dt = float(t - prev)
            if dt > 0:
                # Every Pauli string is traceless, so trace(H)=0.
                psi = expm_multiply((-1j * dt) * H, psi, traceA=0.0)
            observe(psi, k)
            prev = float(t)
    return corr


def run_ed(run_dir, *, times=None, method="auto", max_L=12, overwrite=False):
    """Run ED on saved disorder and saved initial product state.

    When QPU data exist, use their exact time grid automatically.
    For ED-only demonstrations, explicitly supply `times` instead.
    """
    run_dir = Path(run_dir)
    cfg, sample, _ = load_run(run_dir)
    if cfg.L > max_L:
        raise ValueError(f"ED refused for L={cfg.L}; increase max_L explicitly if feasible")
    target = run_dir / "ed_results.npz"
    if target.exists() and not overwrite:
        raise FileExistsError(f"ED results already exist: {target}; set overwrite=True")
    if times is None:
        qpu_file = run_dir / "qpu_results.npz"
        if not qpu_file.exists():
            raise ValueError("Pass times=... for standalone ED or retrieve QPU data first")
        with np.load(qpu_file, allow_pickle=False) as data:
            times = data["times"].copy()
    times = np.asarray(times, dtype=float)
    corr = evolve_ed(cfg, sample, times, method=method)
    np.savez_compressed(target, times=times, correlator=corr,
                        average=corr.mean(axis=0))
    return times, corr
