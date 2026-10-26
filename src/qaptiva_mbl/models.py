"""Shared Pauli-term Hamiltonian specification for Qaptiva and manual ED."""
from dataclasses import dataclass
import numpy as np

from .config import RunConfig


@dataclass(frozen=True)
class SpinSample:
    """Arrays are the authoritative, lossless specification of one sample."""
    deltas: np.ndarray  # Imbrie ZZ bond couplings; empty for XXZ
    fields: np.ndarray  # h_i longitudinal fields
    spins: np.ndarray   # eigenvalues of Z_i at t=0, each +1/-1

    @property
    def bitstring(self):
        # Matches Qaptiva psi_0: |0> is Z=+1; site 0 is LEFTMOST bit.
        return "".join("0" if spin == 1 else "1" for spin in self.spins)


def bonds(L: int, boundary: str) -> list[tuple[int, int]]:
    pairs = [(i, i + 1) for i in range(L - 1)]
    if boundary == "pbc":
        pairs.append((L - 1, 0))
    return pairs


def make_sample(cfg: RunConfig) -> SpinSample:
    """Separate deterministic random streams for disorder and state choice."""
    disorder_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 0]))
    state_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 1]))
    n_bonds = len(bonds(cfg.L, cfg.boundary))
    deltas = (disorder_rng.uniform(cfg.delta_min, cfg.delta_max, n_bonds)
              if cfg.model == "imbrie" else np.empty(0, dtype=float))
    fields = disorder_rng.uniform(-cfg.h0, cfg.h0, cfg.L)
    if cfg.initial_state == "neel":
        spins = np.where(np.arange(cfg.L) % 2 == 0, 1, -1)
    else:
        spins = state_rng.choice(np.array([1, -1]), cfg.L)
    return SpinSample(np.asarray(deltas, dtype=float),
                      np.asarray(fields, dtype=float),
                      np.asarray(spins, dtype=np.int8))


def pauli_terms(cfg: RunConfig, sample: SpinSample) -> list[tuple[float, str, tuple[int, ...]]]:
    """Canonical coefficients and operators used by BOTH backends.

    No implicit 1/2 factors are inserted outside this function.
    """
    assert sample.fields.shape == (cfg.L,)
    assert sample.spins.shape == (cfg.L,)
    pairs = bonds(cfg.L, cfg.boundary)
    terms: list[tuple[float, str, tuple[int, ...]]] = []
    if cfg.model == "imbrie":
        if sample.deltas.shape != (len(pairs),):
            raise ValueError("Inconsistent Imbrie bond-coupling count")
        for k, pair in enumerate(pairs):
            terms.append((float(sample.deltas[k]), "ZZ", pair))
        for i in range(cfg.L):
            terms.extend([(float(sample.fields[i]), "Z", (i,)),
                          (float(cfg.gamma), "X", (i,))])
    else:
        for pair in pairs:
            terms.extend([(float(cfg.jxy / 4), "XX", pair),
                          (float(cfg.jxy / 4), "YY", pair),
                          (float(cfg.jz / 4), "ZZ", pair)])
        for i in range(cfg.L):
            terms.append((float(sample.fields[i] / 2), "Z", (i,)))
    return terms
