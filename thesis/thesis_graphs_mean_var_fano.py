import multiprocessing
import pickle  # Added for saving/loading
from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor

import matplotlib
import numpy as np
import seaborn as sns

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from functions import k_f, pyplot_settings
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from tqdm import tqdm

from discrete_numeric import tfim_momentum_trotter_single_k, process_step, count_kinks

DATA_FILENAMES = {
    'graph1_2': "graph1_2_data.pkl",
    'graph3'  : "graph3_data.pkl",
    'graph4'  : "graph4_data.pkl",
    'graph5'  : "graph5_data.pkl"
    }


def compute_steps_data(steps, noise_param, params, is_last_noise):
    """Helper function to compute data for a single 'steps' value. Run in parallel."""
    base_angles = (np.pi / 2) * np.arange(1, steps + 1) / (steps + 1)
    base_angles = base_angles[:, np.newaxis]
    momentum_dms, qiskit_circs, local_circuits = [], [], []

    ks = k_f(params['num_qubits'])

    num_qubits = params['num_qubits']
    num_circuits = params['num_circuits']
    num_shots = params['num_shots']

    for _ in range(num_circuits):
        noisy_base = base_angles + noise_param * np.random.randn(steps, 1)
        betas, alphas = -np.sin(noisy_base).flatten(), -np.cos(noisy_base).flatten()
        sol = [tfim_momentum_trotter_single_k(k, steps, betas, alphas, 0) for k in ks]
        momentum_dms.append(sol)

        # Global noise circuit
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

        if is_last_noise:
            # Local noise circuit (only for last noise param)
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
                for qubit in range(num_qubits):
                    local_circuit.rz(np.random.randn() * noise_param, qubit)
            local_circuit.measure(range(num_qubits), range(num_qubits))
            local_circuits.append(local_circuit)

    # Momentum model calculation
    mean_i, var_i = process_step(step_density_matrices=momentum_dms, ks=ks,
                                 num_qubits=num_qubits, method="rho")

    # Qiskit global noise calculation
    simulator = AerSimulator()
    transpiled_circuits = transpile(qiskit_circs, simulator)
    job_result = simulator.run(transpiled_circuits, shots=num_shots, memory=True).result()

    kink_counts_matrix = np.zeros((len(qiskit_circs), num_shots))
    for i in range(len(qiskit_circs)):
        outcomes = job_result.get_memory(i)
        kink_counts_matrix[i, :] = [count_kinks(state) for state in outcomes]

    # Per-shot stats for global
    means_per_shot, vars_per_shot, fano_per_shot = _compute_per_shot_stats(kink_counts_matrix)

    final_mean_kinks = np.mean(means_per_shot) / num_qubits
    std_err_mean_kinks = np.std(means_per_shot) / num_qubits
    final_var_kinks = np.mean(vars_per_shot) / num_qubits
    std_err_var_kinks = np.std(vars_per_shot) / num_qubits
    valid_fano = [f for f in fano_per_shot if f != -1]
    final_fano = np.mean(valid_fano) if valid_fano else -1
    std_err_fano = np.std(valid_fano) if valid_fano else 0

    result = {
        'mean_independent_modes'     : mean_i,
        'qiskit_mean_kinks_r_mean'   : final_mean_kinks,
        'qiskit_mean_kinks_r_std_err': std_err_mean_kinks,
        'var_independent_modes'      : var_i,
        'qiskit_var_kinks_r_mean'    : final_var_kinks,
        'qiskit_var_kinks_r_std_err' : std_err_var_kinks,
        'final_fano_mean'            : final_fano,
        'final_fano_std_err'         : std_err_fano,
        }

    if is_last_noise:
        # Qiskit local noise calculation (duplicated logic for brevity; could extract further)
        transpiled_circuits = transpile(local_circuits, simulator)
        job_result = simulator.run(transpiled_circuits, shots=num_shots, memory=True).result()

        kink_counts_matrix = np.zeros((len(local_circuits), num_shots))
        for i in range(len(local_circuits)):
            outcomes = job_result.get_memory(i)
            kink_counts_matrix[i, :] = [count_kinks(state) for state in outcomes]

        means_per_shot, vars_per_shot, fano_per_shot = _compute_per_shot_stats(kink_counts_matrix)

        final_mean_kinks_local = np.mean(means_per_shot) / num_qubits
        std_err_mean_kinks_local = np.std(means_per_shot) / num_qubits
        final_var_kinks_local = np.mean(vars_per_shot) / num_qubits
        std_err_var_kinks_local = np.std(vars_per_shot) / num_qubits
        valid_fano_local = [f for f in fano_per_shot if f != -1]
        final_fano_local = np.mean(valid_fano_local) if valid_fano_local else -1
        std_err_fano_local = np.std(valid_fano_local) if valid_fano_local else 0

        result.update({
            'qiskit_mean_kinks_r_mean_local'   : final_mean_kinks_local,
            'qiskit_mean_kinks_r_std_err_local': std_err_mean_kinks_local,
            'qiskit_var_kinks_r_mean_local'    : final_var_kinks_local,
            'qiskit_var_kinks_r_std_err_local' : std_err_var_kinks_local,
            'final_fano_mean_local'            : final_fano_local,
            'final_fano_std_err_local'         : std_err_fano_local,
            })

    return steps, result


def _compute_per_shot_stats(kink_counts_matrix):
    """Helper to compute per-shot means, vars, and Fano factors."""
    num_shots = kink_counts_matrix.shape[1]
    means_per_shot, vars_per_shot, fano_per_shot = [], [], []
    for j in range(num_shots):
        kinks_for_shot_j = kink_counts_matrix[:, j]
        means_per_shot.append(np.mean(kinks_for_shot_j))
        vars_per_shot.append(np.var(kinks_for_shot_j))
        mean_kinks = np.mean(kinks_for_shot_j)
        fano_per_shot.append(np.var(kinks_for_shot_j) / mean_kinks if mean_kinks != 0 else -1)
    return means_per_shot, vars_per_shot, fano_per_shot


def calculate_graph_data(params, filename, compute=True):
    """Calculates data for a graph (e.g., graph1_2) and optionally saves it. Parallelized over steps."""
    if not compute:
        return {}

    ks = k_f(params['num_qubits'])
    results = defaultdict(dict)
    last_noise_param = params['noise_params_list'][-1]

    for noise_param in tqdm(params['noise_params_list'], desc="Noise"):
        is_last_noise = (noise_param == last_noise_param)

        max_workers = min(len(params['steps_list']), multiprocessing.cpu_count())
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(compute_steps_data, steps, noise_param, params, is_last_noise): steps
                       for steps in params['steps_list']}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Steps (parallel)", leave=False):
                steps, step_result = future.result()
                results[noise_param][steps] = step_result

    data_to_save = {'results': results, 'params': params}
    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"Saved data to {filename}")
    return data_to_save


def load_graph_data(filename):
    """Loads existing data for a graph."""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        print(f"Could not load data from {filename}.")
        exit(1)


def plot_metric_vs_steps(graph_data, metric_key, ylabel, title, show=True, save_path=None):
    """
    Generic helper to plot a metric (e.g., mean kinks, variance, Fano) vs. steps.

    Args:
        graph_data: Dict with 'results' and 'params'.
        metric_key: String like 'mean_kinks' to select extraction lambdas.
        show, save_path: As in original functions.
    """
    if not graph_data:
        return
    results, params = graph_data['results'], graph_data['params']
    steps_list, noise_params_list = params['steps_list'], params['noise_params_list']

    palette = sns.color_palette("colorblind6", len(noise_params_list))
    colors_qiskit = [sns.desaturate(c, 0.3) for c in palette]  # dark
    colors_qiskit = {n: colors_qiskit[i] for i, n in enumerate(noise_params_list)}
    colors_momentum = [sns.desaturate(c, 0.9) for c in palette]  # very light
    colors_momentum = {n: colors_momentum[i] for i, n in enumerate(noise_params_list)}

    # Metric-specific data extractors (lambdas for flexibility)
    if metric_key == 'mean_kinks':
        q_global_mean = lambda n, i: results[n][i]['qiskit_mean_kinks_r_mean']
        q_global_std = lambda n, i: results[n][i]['qiskit_mean_kinks_r_std_err']
        momentum_val = lambda n, i: results[n][i]['mean_independent_modes']
        q_local_mean = lambda n, i: results[n][i]['qiskit_mean_kinks_r_mean_local']
        q_local_std = lambda n, i: results[n][i]['qiskit_mean_kinks_r_std_err_local']
        marker_global, marker_local = 'o', 'o'
    elif metric_key == 'variance':
        q_global_mean = lambda n, i: results[n][i]['qiskit_var_kinks_r_mean']
        q_global_std = lambda n, i: results[n][i]['qiskit_var_kinks_r_std_err']
        momentum_val = lambda n, i: results[n][i]['var_independent_modes']
        q_local_mean = lambda n, i: results[n][i]['qiskit_var_kinks_r_mean_local']
        q_local_std = lambda n, i: results[n][i]['qiskit_var_kinks_r_std_err_local']
        marker_global, marker_local = 'o', 'o'
    elif metric_key == 'fano':
        q_global_mean = lambda n, i: results[n][i]['final_fano_mean']
        q_global_std = lambda n, i: results[n][i]['final_fano_std_err']
        momentum_val = lambda n, i: results[n][i]['var_independent_modes'] / results[n][i]['mean_independent_modes']
        q_local_mean = lambda n, i: results[n][i]['final_fano_mean_local']
        q_local_std = lambda n, i: results[n][i]['final_fano_std_err_local']
        marker_global, marker_local = 'o', 'o'
    else:
        raise ValueError(f"Unsupported metric_key: {metric_key}")

    pyplot_settings()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x = np.array(steps_list)

    # Collect handles and labels for controlled legend order
    handles = []
    labels = []

    # Plot Qiskit Global and Momentum for each noise level (in loop order)
    for n in noise_params_list:
        color_q = colors_qiskit[n]
        color_m = colors_momentum[n]
        # Extract data using lambdas
        q_vals = np.array([q_global_mean(n, i) for i in x])
        q_errs = np.array([q_global_std(n, i) for i in x])
        m_vals = np.array([momentum_val(n, i) for i in x])
        # Plot Qiskit Global first (lower zorder, muted color)
        label = f'$\\sigma={n}$, Qiskit Global'
        if n == 0.0:
            label = f'$\\sigma={n}$, Qiskit Noiseless'
        h_global = ax.errorbar(x, q_vals, yerr=q_errs, fmt=f'{marker_global}-', capsize=3,
                               zorder=1, color=color_q, label=label)
        handles.append(h_global)
        labels.append(label)
        # Plot Momentum last in loop (higher zorder, contrasting color)
        h_mom, = ax.plot(x, m_vals, 'x:', zorder=3, color=color_m, label=f'$\\sigma={n}$, Momentum')
        handles.append(h_mom)
        labels.append(f'$\\sigma={n}$, Momentum')

    # Local noise (reuse Qiskit color, dashed for distinction, medium zorder)
    # Plot it here, but *do not* append to handles/labels yet—add last for legend
    n = noise_params_list[-1]
    color_local = colors_qiskit[n]
    q_local_vals = np.array([q_local_mean(n, i) for i in x])
    q_local_errs = np.array([q_local_std(n, i) for i in x])
    h_local, = ax.plot(x, q_local_vals, f'{marker_local}--',
                       zorder=2, color=color_local, label=f'$\\sigma={n}$, Qiskit Local', fillstyle='none')

    # Now append Local last to ensure it appears last in legend
    handles.append(h_local)
    labels.append(f'$\\sigma={n}$, Qiskit Local')

    ax.set_xlabel('Steps')
    ax.set_ylabel(ylabel)
    # ax.set_title(title)
    ax.grid(True)

    # Custom legend with controlled order (Local last), but no title displayed
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()

    if save_path:
        # Save as high-res JPG for thesis inclusion
        plt.savefig(save_path, format='pdf', dpi=2400, bbox_inches='tight')
    if show:
        plt.show()


def run_simulation(graph_key='graph1_2', params=None, compute=True, load_if_exists=True,
                   enable_mean_plot=True, enable_variance_plot=True, enable_fano_plot=True,
                   save_plots=False, show_plots=True):
    """
    Central runner for the simulation workflow.

    Args:
        graph_key: Key for DATA_FILENAMES (e.g., 'graph1_2').
        params: Dict of parameters (defaults to g1_2_params if None).
        compute: If True, compute new data (overrides load_if_exists).
        load_if_exists: If True and compute=False, load existing data.
        enable_*_plot: Booleans to enable/disable specific plots.
        save_plots: If True, save plots to dated paths (implement get_dated_plot_path if needed).
        show_plots: If True, display plots via plt.show().
    """
    if params is None:
        params = {
            'num_qubits'       : 10,
            'steps_list'       : [i for i in range(4)] + [i for i in range(4, 51, 2)],
            'num_circuits'     : 1000,
            'noise_params_list': [0.0, 0.2, 0.6],
            'num_shots'        : 1000,
            }

    filename = DATA_FILENAMES[graph_key]
    graph_data = {}

    if compute:
        print(f"Computing data for {graph_key}...")
        graph_data = calculate_graph_data(params, filename, compute=True)
    elif load_if_exists:
        print(f"Loading data for {graph_key}...")
        graph_data = load_graph_data(filename)

    # Helper to build param-suffixed filename in thesis/ subdir
    def _build_save_path(base_name):
        if not save_plots:
            return None
        param_suffix = f"{params['num_qubits']}Q-{params['num_circuits']}C-{params['num_shots']}S"
        return f"{base_name}_{param_suffix}.pdf"

    # Generate plots based on flags
    plot_paths = {}
    if enable_mean_plot:
        path = _build_save_path("mean_kinks_vs_steps")
        plot_mean_kinks(graph_data, show=show_plots, save_path=path)
        plot_paths['mean'] = path
    if enable_variance_plot:
        path = _build_save_path("variance_kinks_vs_steps")
        plot_variance(graph_data, show=show_plots, save_path=path)
        plot_paths['variance'] = path
    if enable_fano_plot:
        path = _build_save_path("fano_factor_vs_steps")
        plot_fano_factor(graph_data, show=show_plots, save_path=path)
        plot_paths['fano'] = path

    if save_plots:
        print(f"Plots saved: {plot_paths}")


# Wrapper functions for backward compatibility (call the helper)
def plot_mean_kinks(graph_data, show=True, save_path=None):
    plot_metric_vs_steps(graph_data, 'mean_kinks',
                         'Mean Kinks / N',
                         'Normalized Mean Kinks vs. Steps',
                         show=show, save_path=save_path)


def plot_variance(graph_data, show=True, save_path=None):
    plot_metric_vs_steps(graph_data, 'variance',
                         'Variance / N',
                         'Normalized Kink Variance vs. Steps',
                         show=show, save_path=save_path)


def plot_fano_factor(graph_data, show=True, save_path=None):
    plot_metric_vs_steps(graph_data, 'fano',
                         'Fano Factor',
                         'Fano Factor vs. Steps',
                         show=show, save_path=save_path)


if __name__ == "__main__":
    # Example usage: Compute and plot all, or customize
    run_simulation(
          graph_key='graph1_2',
          compute=False,  # Set False to skip computation
          enable_mean_plot=True,
          enable_variance_plot=True,
          enable_fano_plot=True,
          save_plots=True,  # Set True to save (add get_dated_plot_path if needed)
          show_plots=True
          )