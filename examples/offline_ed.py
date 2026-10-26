"""Run without Qaptiva, e.g. python examples/offline_ed.py after pip install -e ."""
import numpy as np
from qaptiva_mbl import RunConfig, prepare_run, run_ed

cfg = RunConfig(model="imbrie", L=6, boundary="obc", seed=101,
                initial_state="neel", tmax=5)
folder = prepare_run(cfg)
times, correlators = run_ed(folder, times=np.linspace(0, cfg.tmax, 51))
print(f"Results written to: {folder}")
print("First and last site-averaged C(t):", correlators.mean(axis=0)[[0, -1]])
