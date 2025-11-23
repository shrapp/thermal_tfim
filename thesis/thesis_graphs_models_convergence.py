import multiprocessing
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from functions import k_f, pyplot_settings
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from tqdm import tqdm

from discrete_numeric import tfim_momentum_trotter_single_k, process_step, count_kinks

DATA_FILENAMES = {
    'graph4': "graph4_data.pkl"
    }


def compute_circuits_data(num_circuits, noise_param, params):
    """Helper function to compute data for a single (num_circuits, noise_param) pair. Run in parallel."""
    num_qubits = params['num_qubits']
    steps = params['steps']
    num_shots = params['num_shots']

    base_angles = (np.pi / 2) * np.arange(1, steps + 1) / (steps + 1)
    base_angles = base_angles[:, np.newaxis]
    momentum_dms, qiskit_circs = [], []

    ks = k_f(num_qubits)

    for _ in range(num_circuits):
        noisy_base = base_angles + noise_param * np.random.randn(steps, 1)
        betas, alphas = -np.sin(noisy_base).flatten(), -np.cos(noisy_base).flatten()
        sol = [tfim_momentum_trotter_single_k(k, steps, betas, alphas, 0) for k in ks]
        momentum_dms.append(sol)

        circuit = QuantumCircuit(num_qubits, num_qubits)
        circuit.h(range(num_qubits))
        for step in range(steps):
            for i in range(0, num_qubits, 2):
                circuit.rzz(betas[step], i, (i + 1) % num_qubits)
            for i in range(1, num_qubits, 2):
                circuit.rzz(betas[step], i, (i + 1) % num_qubits)
            circuit.rx(alphas[step], range(num_qubits))
        circuit.measure(range(num_qubits), range(num_qubits))
        qiskit_circs.append(circuit)

    # Momentum calculation (averaged over circuits)
    mean_i, var_i = process_step(step_density_matrices=momentum_dms, ks=ks,
                                 num_qubits=num_qubits, method="rho")
    ratio_i = var_i / mean_i if mean_i != 0 else 0

    # Qiskit calculation
    simulator = AerSimulator()
    transpiled_circuits = transpile(qiskit_circs, simulator)
    job_result = simulator.run(transpiled_circuits, shots=num_shots, memory=True).result()

    kink_counts_matrix = np.zeros((len(qiskit_circs), num_shots))
    for i in range(len(qiskit_circs)):
        outcomes = job_result.get_memory(i)
        kink_counts_matrix[i, :] = [count_kinks(state) for state in outcomes]

    # Per-shot diffs
    diffs_per_shot = _compute_diff_stats(kink_counts_matrix, ratio_i)
    diff_mean = np.mean(diffs_per_shot) if diffs_per_shot else 0
    diff_std = np.std(diffs_per_shot) if diffs_per_shot else 0

    result = {
        'diff_mean': diff_mean,
        'diff_std' : diff_std,
        }

    return num_circuits, noise_param, result


def _compute_diff_stats(kink_counts_matrix, ratio_i):
    """Helper to compute per-shot absolute differences in ratios."""
    num_shots = kink_counts_matrix.shape[1]
    diffs_per_shot = []
    for j in range(num_shots):
        kinks_for_shot_j = kink_counts_matrix[:, j]
        mean_kinks_j = np.mean(kinks_for_shot_j)
        var_kinks_j = np.var(kinks_for_shot_j)
        if mean_kinks_j != 0:
            ratio_r_j = var_kinks_j / mean_kinks_j
            diffs_per_shot.append(np.abs(ratio_i - ratio_r_j))
    return diffs_per_shot


def calculate_graph4_data(params, filename, compute=True):
    """Calculates data for Graph 4 and optionally saves it. Parallelized over num_circuits x noise pairs."""
    if not compute:
        return defaultdict(dict)

    results = defaultdict(dict)
    # Flatten pairs for parallelization
    pairs = [(nc, np) for nc in params['num_circuits_list'] for np in params['noise_params_list']]
    max_workers = min(len(pairs), multiprocessing.cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_circuits_data, nc, np, params): (nc, np)
                   for nc, np in pairs}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Circuits/Noise (parallel)"):
            nc, np, nc_result = future.result()
            results[nc][np] = nc_result

    data_to_save = {'results': results, 'params': params}
    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"Saved data to {filename}")
    return results


def load_graph4_data(filename):
    """Loads existing data for Graph 4."""
    try:
        with open(filename, 'rb') as f:
            loaded = pickle.load(f)
            return loaded['results'], loaded['params']
    except (FileNotFoundError, EOFError):
        print(f"Could not load data from {filename}.")
        return defaultdict(dict), {}


def plot_graph4(results, params, show=True, save_path=None):
    """Plots absolute difference in ratios vs. number of circuits for Graph 4."""
    if not results:
        return

    pyplot_settings()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    x = np.array(params['num_circuits_list'])

    for noise_param in params['noise_params_list']:
        y_mean = np.array([results[nc][noise_param]['diff_mean'] for nc in x])
        y_std = np.array([results[nc][noise_param]['diff_std'] for nc in x])
        ax.errorbar(x, y_mean, yerr=y_std, fmt='o-', capsize=3, label=f'$\\sigma={noise_param}$')

    ax.set_xscale('log')
    ax.set_xlabel('Number of Circuits')  # Removed \textbf
    ax.set_ylabel('Abs. Difference in Fano factor')  # Removed \textbf
    # ax.set_title(r'Model Discrepancy vs. Circuit Count')  # Removed \textbf
    ax.grid(True)
    legend_title = f"{params['num_qubits']} Qubits, {params['steps']} Steps, {params['num_shots']} Shots"
    ax.legend(
          # title=legend_title
          )

    # plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=1200)
    if show:
        plt.show()


def run_graph4(params=None, compute=True, load_if_exists=True,
               enable_plot=True, save_plot=False, show_plot=True):
    """
    Central runner for Graph 4 workflow.

    Args:
        params: Dict of parameters (defaults to g4_params if None).
        compute: If True, compute new data (overrides load_if_exists).
        load_if_exists: If True and compute=False, load existing data.
        enable_plot: Boolean to enable/disable the plot.
        save_plot: If True, save plot to path.
        show_plot: If True, display plot via plt.show().
    """
    if params is None:
        params = {
            'num_qubits'       : 10,
            'steps'            : 30,
            'num_circuits_list': [int(i) for i in np.logspace(1, 4, 10)],
            'noise_params_list': [0.2, 0.4, 0.6],
            'num_shots'        : 100
            }

    filename = DATA_FILENAMES['graph4']
    results = defaultdict(dict)

    if compute:
        print("Computing data for Graph 4...")
        results = calculate_graph4_data(params, filename, compute=True)
    elif load_if_exists:
        print("Loading data for Graph 4...")
        results, params = load_graph4_data(filename)

    # Generate plot if enabled
    plot_path = None
    if enable_plot:
        plot_path = f"diff_vs_circuits_{params['num_qubits']}q_{params['steps']}s_{params['num_shots']}shots.png" if save_plot else None
        plot_graph4(results, params, show=show_plot, save_path=plot_path)

    if save_plot:
        print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    # Example usage: Compute and plot, or customize
    run_graph4(
            compute=False,  # Set False to skip computation
            enable_plot=True,
            save_plot=True,  # Set True to save
            show_plot=True
            )