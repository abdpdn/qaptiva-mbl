"""Local tests do not require proprietary Qaptiva packages or network access."""
import numpy as np
import pytest
from scipy.linalg import expm

from qaptiva_mbl import RunConfig
from qaptiva_mbl.models import make_sample, pauli_terms
from qaptiva_mbl.ed import manual_hamiltonian, state_vector, evolve_ed


PAULI = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]),
}


def explicit_kron(cfg, sample):
    H = np.zeros((2**cfg.L, 2**cfg.L), dtype=complex)
    for coeff, op, sites in pauli_terms(cfg, sample):
        symbols = ["I"] * cfg.L
        for letter, site in zip(op, sites):
            symbols[site] = letter
        mat = PAULI[symbols[0]]
        for symbol in symbols[1:]:
            mat = np.kron(mat, PAULI[symbol])
        H += coeff * mat
    return H


@pytest.mark.parametrize("model", ["imbrie", "xxz"])
@pytest.mark.parametrize("boundary", ["pbc", "obc"])
def test_manual_ed_matches_pauli_matrix(model, boundary):
    cfg = RunConfig(model=model, L=4, boundary=boundary, seed=44,
                    h0=0.9, initial_state="random")
    sample = make_sample(cfg)
    actual = manual_hamiltonian(cfg, sample).toarray()
    np.testing.assert_allclose(actual, explicit_kron(cfg, sample), atol=1e-13)
    np.testing.assert_allclose(actual, actual.conj().T, atol=1e-13)


def test_ed_dense_krylov_and_direct_propagation():
    cfg = RunConfig(model="xxz", L=4, seed=131, boundary="obc", h0=1.3)
    sample = make_sample(cfg)
    times = np.array([0, 0.2, 0.7, 1.25])
    dense = evolve_ed(cfg, sample, times, method="dense")
    krylov = evolve_ed(cfg, sample, times, method="krylov")
    np.testing.assert_allclose(dense, krylov, atol=5e-13)
    H = manual_hamiltonian(cfg, sample).toarray()
    psi = expm(-1j * H * times[-1]) @ state_vector(sample)
    spins = np.array([np.abs(psi)**2 @ np.diag(explicit_z(cfg.L, i))
                      for i in range(cfg.L)])
    np.testing.assert_allclose(dense[:, -1], sample.spins * spins, atol=2e-13)
    np.testing.assert_allclose(dense[:, 0], np.ones(cfg.L), atol=2e-13)


def explicit_z(L, site):
    m = PAULI["Z"] if site == 0 else PAULI["I"]
    for i in range(1, L):
        m = np.kron(m, PAULI["Z"] if site == i else PAULI["I"])
    return m


def test_disorder_does_not_depend_on_initial_state_choice():
    base = dict(model="imbrie", L=6, seed=199, boundary="pbc")
    neel = make_sample(RunConfig(**base, initial_state="neel"))
    random_state = make_sample(RunConfig(**base, initial_state="random"))
    np.testing.assert_array_equal(neel.deltas, random_state.deltas)
    np.testing.assert_array_equal(neel.fields, random_state.fields)
    assert neel.bitstring == "010101"
    assert len(random_state.bitstring) == 6


def test_no_double_count_periodic_l2():
    with pytest.raises(ValueError, match="PBC"):
        RunConfig(L=2, boundary="pbc")


def test_adaptive_and_fixed_step_are_exclusive():
    with pytest.raises(ValueError, match="omit nsteps"):
        RunConfig(error_control=True, nsteps=200)
    with pytest.raises(ValueError, match="requires"):
        RunConfig(error_control=False, nsteps=None)
    assert RunConfig(error_control=False, nsteps=50).nsteps == 50


def test_xxz_conserves_total_z():
    cfg = RunConfig(model="xxz", L=4, seed=8)
    H = manual_hamiltonian(cfg, make_sample(cfg)).toarray()
    M = sum(explicit_z(cfg.L, i) for i in range(cfg.L))
    np.testing.assert_allclose(H @ M - M @ H, 0, atol=1e-13)
