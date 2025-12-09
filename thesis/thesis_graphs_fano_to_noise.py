import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from functions import k_f, pyplot_settings, count_kinks
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from tqdm import tqdm

from discrete_numeric import tfim_momentum_trotter_single_k, process_step

DATA_FILENAMES = {
    'graph3': "graph3_data.pkl"
    }


def compute_noise_data(noise_param, params):
    """Helper function to compute data for a single 'noise_param' value. Run in parallel."""
    num_qubits = params['num_qubits']
    steps = params['steps']
    num_circuits = params['num_circuits']
    num_shots = params['num_shots']

    base_angles = (np.pi / 2) * np.arange(1, steps + 1) / (steps + 1)
    base_angles = base_angles[:, np.newaxis]
    momentum_dms, qiskit_circs, qiskit_circs_local = [], [], []

    ks = k_f(num_qubits)

    for _ in range(num_circuits):
        # Global noise: noisy angles
        noisy_base = base_angles + noise_param * np.random.randn(steps, 1)
        betas, alphas = -np.sin(noisy_base).flatten(), -np.cos(noisy_base).flatten()
        sol = [tfim_momentum_trotter_single_k(k, steps, betas, alphas, 0) for k in ks]
        momentum_dms.append(sol)

        # Qiskit global circuit
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

        # Qiskit local circuit
        betas_local = -np.sin(base_angles).flatten()
        alphas_local = -np.cos(base_angles).flatten()
        local_circuit = QuantumCircuit(num_qubits, num_qubits)
        local_circuit.h(range(num_qubits))
        for step in range(steps):
            for i in range(0, num_qubits, 2):
                local_circuit.rzz(betas_local[step], i, (i + 1) % num_qubits)
            for i in range(1, num_qubits, 2):
                local_circuit.rzz(betas_local[step], i, (i + 1) % num_qubits)
            local_circuit.rx(alphas_local[step], range(num_qubits))
            for q in range(num_qubits):
                theta = noise_param * np.random.randn()
                if np.abs(theta) > 1e-14:
                    local_circuit.rz(theta, q)
        local_circuit.measure(range(num_qubits), range(num_qubits))
        qiskit_circs_local.append(local_circuit)

    # Momentum calculation (deterministic, single value)
    mean_i, var_i = process_step(step_density_matrices=momentum_dms, ks=ks,
                                 num_qubits=num_qubits, method="rho")
    momentum_ratio = var_i / mean_i if mean_i != 0 else 0
    momentum_ratio_std = 0  # No variance in momentum model

    # Qiskit global calculation
    simulator = AerSimulator()
    transpiled_circuits = transpile(qiskit_circs, simulator)
    job_result = simulator.run(transpiled_circuits, shots=num_shots, memory=True).result()

    kink_counts_matrix = np.zeros((len(qiskit_circs), num_shots))
    for i in range(len(qiskit_circs)):
        outcomes = job_result.get_memory(i)
        kink_counts_matrix[i, :] = [count_kinks(state) for state in outcomes]

    ratios_per_shot = _compute_ratio_stats(kink_counts_matrix)
    qiskit_ratio_mean = np.mean(ratios_per_shot) if ratios_per_shot else 0
    qiskit_ratio_std = np.std(ratios_per_shot) if ratios_per_shot else 0

    # Qiskit local calculation
    transpiled_circuits = transpile(qiskit_circs_local, simulator)
    job_result = simulator.run(transpiled_circuits, shots=num_shots, memory=True).result()

    kink_counts_matrix_local = np.zeros((len(qiskit_circs_local), num_shots))
    for i in range(len(qiskit_circs_local)):
        outcomes = job_result.get_memory(i)
        kink_counts_matrix_local[i, :] = [count_kinks(state) for state in outcomes]

    ratios_per_shot_local = _compute_ratio_stats(kink_counts_matrix_local)
    qiskit_local_ratio_mean = np.mean(ratios_per_shot_local) if ratios_per_shot_local else 0
    qiskit_local_ratio_std = np.std(ratios_per_shot_local) if ratios_per_shot_local else 0

    result = {
        'momentum_ratio_mean'    : momentum_ratio,
        'momentum_ratio_std'     : momentum_ratio_std,
        'qiskit_ratio_mean'      : qiskit_ratio_mean,
        'qiskit_ratio_std'       : qiskit_ratio_std,
        'qiskit_local_ratio_mean': qiskit_local_ratio_mean,
        'qiskit_local_ratio_std' : qiskit_local_ratio_std,
        }

    return noise_param, result


def _compute_ratio_stats(kink_counts_matrix):
    """Helper to compute per-shot variance/mean ratios."""
    num_shots = kink_counts_matrix.shape[1]
    ratios_per_shot = []
    for j in range(num_shots):
        kinks_for_shot_j = kink_counts_matrix[:, j]
        mean_kinks_j = np.mean(kinks_for_shot_j)
        var_kinks_j = np.var(kinks_for_shot_j)
        if mean_kinks_j != 0:
            ratios_per_shot.append(var_kinks_j / mean_kinks_j)
    return ratios_per_shot


def calculate_graph3_data(params, filename, compute=True):
    """Calculates data for Graph 3 and optionally saves it. Parallelized over noise params."""
    if not compute:
        return {}

    results = {}
    max_workers = min(len(params['noise_params_list']), multiprocessing.cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_noise_data, noise_param, params): noise_param
                   for noise_param in params['noise_params_list']}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Noise (parallel)"):
            noise_param, noise_result = future.result()
            results[noise_param] = noise_result

    data_to_save = {'results': results, 'params': params}
    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"Saved data to {filename}")
    return data_to_save


def load_graph3_data(filename):
    """Loads existing data for Graph 3."""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        print(f"Could not load data from {filename}.")
        return {}


def plot_graph3(graph_data, show=True, save_path=None):
    """Plots kink statistics ratio vs. noise for Graph 3."""
    if not graph_data:
        return
    results, params = graph_data['results'], graph_data['params']
    noise_params_list = params['noise_params_list']

    # Colors: Qiskit Global (muted black), Momentum (contrasting red, on top), Local (blue)
    color_qiskit = 'black'
    color_momentum = '#d62728'
    color_local = '#1f77b4'

    pyplot_settings()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    x = np.array(noise_params_list)

    # Extract data
    ratio_q_mean = np.array([results[n]['qiskit_ratio_mean'] for n in x])
    ratio_q_std = np.array([results[n]['qiskit_ratio_std'] for n in x])
    ratio_m_mean = np.array([results[n]['momentum_ratio_mean'] for n in x])
    ratio_m_std = np.array([results[n]['momentum_ratio_std'] for n in x])
    ratio_q_local_mean = np.array([results[n]['qiskit_local_ratio_mean'] for n in x])
    ratio_q_local_std = np.array([results[n]['qiskit_local_ratio_std'] for n in x])

    # Plot Qiskit Global first (lower zorder)
    ax.errorbar(x, ratio_q_mean, yerr=ratio_q_std, fmt='o-', capsize=3,
                label='Qiskit Global', zorder=1, color=color_qiskit)
    # Plot Momentum last (higher zorder, contrasting color)
    ax.errorbar(x, ratio_m_mean, yerr=ratio_m_std, fmt='x:', capsize=3,
                label='Momentum', zorder=3, color=color_momentum)
    # Plot Qiskit Local (medium zorder)
    ax.errorbar(x, ratio_q_local_mean, yerr=ratio_q_local_std, fmt='s--', capsize=3,
                label='Qiskit Local', zorder=2, color=color_local)

    ax.set_xscale('log')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.axhline(2.0, color='gray', linestyle='--', linewidth=1)

    ax.set_xlabel('Noise Parameter')
    ax.set_ylabel('Fano factor')
    ax.grid(True)
    legend_title = f"{params['num_circuits']} Circuits, {params['steps']} Steps\n {params['num_shots']} Shots"
    ax.legend(
          # title=legend_title
          )

    plt.tight_layout(rect=(0, 0, 1, 1))

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf', dpi=2400)
    if show:
        plt.show()


def run_graph3(params=None, compute=True, load_if_exists=True,
               enable_plot=True, save_plot=False, show_plot=True):
    """
    Central runner for Graph 3 workflow.

    Args:
        params: Dict of parameters (defaults to g3_params if None).
        compute: If True, compute new data (overrides load_if_exists).
        load_if_exists: If True and compute=False, load existing data.
        enable_plot: Boolean to enable/disable the plot.
        save_plot: If True, save plot to path.
        show_plot: If True, display plot via plt.show().
    """
    if params is None:
        params = {
            'num_qubits'       : 10,
            'steps'            : 100,
            'num_circuits'     : 1000,
            'noise_params_list': np.logspace(-2.2, 0.5, 20).tolist(),
            'num_shots'        : 1000
            }

    filename = DATA_FILENAMES['graph3']
    graph_data = {}

    if compute:
        print("Computing data for Graph 3...")
        graph_data = calculate_graph3_data(params, filename, compute=True)
    elif load_if_exists:
        print("Loading data for Graph 3...")
        graph_data = load_graph3_data(filename)

    # Generate plot if enabled
    plot_path = None
    if enable_plot:
        plot_path = (f"ratio_vs_noise_{params['steps']}-steps_{params['num_qubits']}-qubits_{params['num_circuits']}-circuits_{params['num_shots']}-shots.pdf"
                        if save_plot else None)
        plot_graph3(graph_data, show=show_plot, save_path=plot_path)

    if save_plot:
        print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    # Example usage: Compute and plot, or customize
    run_graph3(
            compute=False,  # Set False to skip computation
            enable_plot=True,
            save_plot=True,  # Set True to save
            show_plot=True
            )