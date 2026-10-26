"""Plots computed solely from locally saved arrays (no Qaptiva required)."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_run(run_dir, *, kind="sites", sites=None, overlay_ed=True,
             log_time=False, save=None):
    """Plot 'sites', 'average', 'error', or 'site_error' from stored results.

    Return the matplotlib Figure so notebook users can customize/save it.
    Use `sites=[0, 3]` or None for all sites in kind='sites'.
    """
    run_dir = Path(run_dir)
    if kind not in {"sites", "average", "error", "site_error"}:
        raise ValueError("kind must be sites, average, error, or site_error")
    with np.load(run_dir / "qpu_results.npz", allow_pickle=False) as q:
        times = q["times"].copy()
        corr = q["correlator"].copy()
    L = corr.shape[0]
    edfile = run_dir / "ed_results.npz"
    ed = None
    if edfile.exists() and overlay_ed:
        with np.load(edfile, allow_pickle=False) as d:
            if d["times"].shape != times.shape or not np.allclose(d["times"], times):
                raise ValueError("ED grid does not match the retrieved QPU grid")
            ed = d["correlator"].copy()
    if kind in {"sites", "site_error"}:
        local_error = None
        if kind == "site_error":
            error_file = run_dir / "comparison.npz"
            if not error_file.exists():
                raise FileNotFoundError("Run run_ed() and compare_run() before plotting site errors")
            with np.load(error_file, allow_pickle=False) as error_data:
                local_error = error_data["absolute_error"].copy()
        selected = list(range(L)) if sites is None else [int(s) for s in sites]
        if not selected or len(set(selected)) != len(selected) or any(s < 0 or s >= L for s in selected):
            raise ValueError(f"sites must be distinct integers in [0, {L-1}]")
        ncols = min(4, len(selected))
        nrows = (len(selected) + ncols - 1) // ncols
        fig, axs = plt.subplots(nrows, ncols, figsize=(3.6*ncols, 2.8*nrows), squeeze=False)
        for ax in axs.ravel():
            ax.set_visible(False)
        for ax, site in zip(axs.ravel(), selected):
            ax.set_visible(True)
            if kind == "site_error":
                ax.plot(times, local_error[site], "-", lw=1.25, label="|QPU - ED|")
                ax.set(title=f"Site {site}", xlabel="Time", ylabel="Absolute error")
            else:
                ax.plot(times, corr[site], ".", ms=3, label="AnalogQPU")
                if ed is not None:
                    ax.plot(times, ed[site], "-", lw=1.25, label="Manual ED")
                ax.set(title=f"Site {site}", xlabel="Time", ylabel=r"$C_j(t)$")
            ax.grid(alpha=0.2)
            if log_time:
                ax.set_xscale("log")
        axs.ravel()[0].legend(fontsize=8)
    else:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        if kind == "average":
            ax.plot(times, corr.mean(axis=0), ".", ms=4, label="AnalogQPU")
            if ed is not None:
                ax.plot(times, ed.mean(axis=0), "-", lw=1.5, label="Manual ED")
            ax.set(ylabel=r"$C(t)=L^{-1}\sum_j C_j(t)$")
        else:
            error_file = run_dir / "comparison.npz"
            if not error_file.exists():
                raise FileNotFoundError("Run run_ed() and compare_run() before plotting errors")
            with np.load(error_file, allow_pickle=False) as comp:
                ax.plot(times, comp["mean_absolute_error"], label="Mean |error|")
                ax.plot(times, comp["maximum_absolute_error"], label="Max |error|")
                ax.plot(times, comp["rmse"], label="RMSE")
            ax.set(ylabel="ED - AnalogQPU error magnitude")
        ax.set_xlabel("Time")
        ax.grid(alpha=0.2)
        ax.legend()
        if log_time:
            ax.set_xscale("log")
    fig.tight_layout()
    if save is not None:
        path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight")
    return fig
