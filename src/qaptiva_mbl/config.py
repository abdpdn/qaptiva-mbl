"""Validated user-facing configuration. No Qaptiva imports are required."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class RunConfig:
    """One Hamiltonian, one disorder realization, and one initial product state.

    Hamiltonian conventions (site labels 0..L-1):
      imbrie: H = sum_b delta[b] Z_i Z_j + sum_i (h[i] Z_i + gamma X_i)
      xxz:    H = sum_b [jxy/4 (X_i X_j + Y_i Y_j) + jz/4 Z_i Z_j]
                   + sum_i h[i]/2 Z_i
    Thus XXZ uses S^a = sigma^a / 2, whereas Imbrie uses Pauli matrices.

    `seed` controls disorder; initial-state randomness uses an independent
    deterministic stream derived from the same seed.
    """
    model: str = "imbrie"
    L: int = 8
    boundary: str = "pbc"
    seed: int = 12345
    initial_state: str = "neel"  # neel or random

    # Imbrie couplings (actual array saved for each realization).
    gamma: float = 1.0
    delta_min: float = 0.8
    delta_max: float = 1.2

    # XXZ couplings; local random longitudinal fields for both models.
    jxy: float = 1.0
    jz: float = 1.0
    h0: float = 3.5  # h_i ~ Uniform(-h0, h0)

    tmax: float = 100.0
    error_control: bool = True
    nsteps: int | None = None
    abs_tol: float = 1e-8
    rel_tol: float = 1e-8
    use_gpu: bool = False

    def __post_init__(self):
        if self.model not in {"imbrie", "xxz"}:
            raise ValueError("model must be 'imbrie' or 'xxz'")
        if type(self.L) is not int or self.L < 2:
            raise ValueError("L must be an integer >= 2")
        if self.boundary not in {"pbc", "obc"}:
            raise ValueError("boundary must be 'pbc' or 'obc'")
        if self.boundary == "pbc" and self.L < 3:
            raise ValueError("PBC requires L>=3 to avoid duplicate L=2 bonds")
        if self.initial_state not in {"neel", "random"}:
            raise ValueError("initial_state must be 'neel' or 'random'")
        if type(self.seed) is not int or not (0 <= self.seed < 2**32):
            raise ValueError("seed must be an integer in [0, 2**32)")
        if type(self.error_control) is not bool or type(self.use_gpu) is not bool:
            raise ValueError("error_control and use_gpu must be boolean")
        for name in ("gamma", "delta_min", "delta_max", "jxy", "jz", "h0", "tmax"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.h0 < 0 or self.delta_min > self.delta_max or self.tmax <= 0:
            raise ValueError("Require h0>=0, delta_min<=delta_max, and tmax>0")
        if self.error_control:
            if self.nsteps is not None:
                raise ValueError("Adaptive mode: omit nsteps (use nsteps=None)")
            if not (math.isfinite(self.abs_tol) and math.isfinite(self.rel_tol)
                    and self.abs_tol > 0 and self.rel_tol > 0):
                raise ValueError("Adaptive mode requires positive finite tolerances")
        else:
            if type(self.nsteps) is not int or self.nsteps < 1:
                raise ValueError("Fixed-step mode requires a positive integer nsteps")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)
