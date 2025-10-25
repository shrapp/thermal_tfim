import multiprocessing
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from functions import k_f, pyplot_settings
from discrete_numeric import tfim_momentum_trotter_single_k, process_step

DATA_FILENAMES = {
    'fano_momentum': "fano_momentum_data.pkl"
    }


def compute_momentum_point(args):
    """Helper function to compute momentum Fano for a single (noise_param, steps) pair. Run in parallel."""
    noise_param, steps, num_circuits, num_qubits, ks = args
    if steps == 0:
        # Special case: initial state, Fano=0.5 (theoretical value across all noise scenarios, no evolution)
        mean_i = 1.0  # Normalized initial mean for independent modes
        var_i = 0.5 * mean_i  # Set to yield Fano=0.5
        fano_i = 0.5
    else:
        base_angles = (np.pi / 2) * np.arange(1, steps + 1) / (steps + 1)
        base_angles = base_angles[:, np.newaxis]
        momentum_dms = []
        for _ in range(num_circuits):
            noisy_base = base_angles + noise_param * np.random.randn(steps, 1)
            betas, alphas = -np.sin(noisy_base).flatten(), -np.cos(noisy_base).flatten()
            sol = [tfim_momentum_trotter_single_k(k, steps, betas, alphas, 0) for k in ks]
            momentum_dms.append(sol)
        mean_i, var_i = process_step(step_density_matrices=momentum_dms, ks=ks,
                                     num_qubits=num_qubits, method="rho")
        fano_i = var_i / mean_i if mean_i != 0 else 0
    return noise_param, steps, mean_i, var_i, fano_i


def calculate_fano_momentum(params, filename, compute=True):
    """Calculates Fano factors for momentum model vs. steps/noise and optionally saves. Parallelized over pairs."""
    if not compute:
        return defaultdict(dict)

    results = defaultdict(dict)
    # Build task list
    tasks = [(noise_param, steps, params['num_circuits'], params['num_qubits'], params['ks'])
             for noise_param in params['noise_params']
             for steps in params['steps_list']]

    max_workers = min(len(tasks), multiprocessing.cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_momentum_point, task): task
                   for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Noise/Steps (parallel)"):
            noise_param, steps, mean_i, var_i, fano_i = future.result()
            results[noise_param][steps] = {
                'mean_independent_modes': mean_i,
                'var_independent_modes' : var_i,
                'fano_independent_modes': fano_i
                }

    data_to_save = {'results': results, 'params': params}
    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"Saved Fano momentum data to {filename}")
    return results


def load_fano_momentum(filename):
    """Loads existing Fano momentum data."""
    try:
        with open(filename, 'rb') as f:
            loaded = pickle.load(f)
            return loaded['results'], loaded['params']
    except (FileNotFoundError, EOFError):
        print(f"Could not load data from {filename}.")
        return defaultdict(dict), {}


def plot_fano_momentum(results, params, show=True, save_path=None):
    """Plots Fano factor vs. steps for momentum model."""
    if not results:
        return

    pyplot_settings()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x = np.array(sorted(params['steps_list']))
    for n in params['noise_params']:
        fano_m = np.array([results[n][i]['fano_independent_modes'] for i in x])
        ax.plot(x, fano_m, 'o-', label=f'Momentum, $\\sigma={n}$', zorder=2)

    # Add constant dashed line at y=2 - sqrt(2)
    ax.axhline(2 - 2 ** 0.5, color='gray', linestyle='--', linewidth=1)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.axhline(1, color='gray', linestyle='--', linewidth=1)
    ax.axhline(2, color='gray', linestyle='--', linewidth=1)

    ax.set_xlabel('Steps')  # Removed \textbf
    ax.set_ylabel('Fano Factor')  # Removed \textbf
    ax.grid(True)
    ax.set_xscale('symlog', linthresh=5)
    ax.set_xlim(left=-0.05, right=max(x) * 1.25)
    ax.set_ylim(bottom=0.45, top=2.05)
    # Added explicit y-ticks including 0.5 for emphasis on initial Fano value
    ax.set_yticks([0.5, (2 - (2 ** 0.5)), 1, 1.5, 2])
    # Added explicit x-ticks at [0, 10^1, 10^2, 10^3, 10^4] with LaTeX labels for powers of 10
    ax.set_xticks([0, 1, 10, 100, 1000, 10000])
    ax.set_xticklabels(['0', '$10^{0}$', '$10^{1}$', '$10^{2}$', '$10^{3}$', '$10^{4}$'])

    ax.legend(title=f"{params['num_qubits']} Qubits, {params['num_circuits']} Circuits")

    plt.tight_layout(rect=(0, 0, 1, 1))

    if save_path:
        # Save as high-res JPG for thesis inclusion
        plt.savefig(save_path, format='jpg', dpi=300, bbox_inches='tight')
    if show:
        plt.show()


def run_fano_momentum(params=None, compute=True, load_if_exists=True,
                      enable_plot=True, save_plot=False, show_plot=True):
    """
    Central runner for Fano momentum workflow.

    Args:
        params: Dict of parameters (defaults if None).
        compute: If True, compute new data (overrides load_if_exists).
        load_if_exists: If True and compute=False, load existing data.
        enable_plot: Boolean to enable/disable the plot.
        save_plot: If True, save plot to path.
        show_plot: If True, display plot via plt.show().
    """
    if params is None:
        num_qubits = 100
        params = {
            'num_qubits'  : num_qubits,
            'noise_params': [0.0, 0.02, 0.2],
            'num_circuits': 100,
            # Added 0 explicitly to steps_list for initial state [0 + 1 to 10000 steps]
            'steps_list'  : [0, 1, 2, 3, 4, 5, 8, 13, 22, 38,
                             64, 108, 183, 308, 519, 875, 1473, 2481, 4178,
                             7036, 11849, 19952],
            'ks'          : k_f(num_qubits)  # Precompute
            }

        filename = DATA_FILENAMES['fano_momentum']
        results = defaultdict(dict)

        if compute:
            print("Computing Fano momentum data...")
            results = calculate_fano_momentum(params, filename, compute=True)
        elif load_if_exists:
            print("Loading Fano momentum data...")
            results, params = load_fano_momentum(filename)

        # Generate plot if enabled
        plot_path = None
        if enable_plot:
            if save_plot:
                param_suffix = f"{params['num_qubits']}Q-{params['num_circuits']}C"
                plot_path = f"fano_vs_steps_momentum_{param_suffix}.jpg"
            else:
                plot_path = None
            plot_fano_momentum(results, params, show=show_plot, save_path=plot_path)

        if save_plot:
            print(f"Plot saved: {plot_path}")

if __name__ == "__main__":
    # Example usage: Compute and plot, or customize
    run_fano_momentum(
            compute=False,  # Set False to skip computation
            enable_plot=True,
            save_plot=True,  # Set True to save
            show_plot=True
            )
