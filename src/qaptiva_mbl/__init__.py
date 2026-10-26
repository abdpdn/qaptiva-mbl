"""Minimal, reproducible spin-chain dynamics and AnalogQPU/ED comparisons."""

from .config import RunConfig
from .storage import prepare_run, load_run
from .submit import submit_run
from .retrieve import retrieve_run
from .ed import run_ed
from .comparison import compare_run
from .plotting import plot_run

__all__ = [
    "RunConfig", "prepare_run", "load_run", "submit_run", "retrieve_run",
    "run_ed", "compare_run", "plot_run",
]
__version__ = "0.1.0"
