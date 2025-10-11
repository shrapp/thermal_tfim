from collections import defaultdict

import matplotlib
import numpy as np

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

g1_2_params = {
    'num_qubits'       : 6,
    'steps_list'       : [i for i in range(4)] + [i for i in range(4, 51, 2)],
    'num_circuits'     : 1000,
    'noise_params_list': [0.0, 0.2, 0.6],
    'num_shots'        : 1000,
    'local_noise'      : 0.8
    }


def calculate_graph1_2_data(g1_2_params, filename):
    """Calculates data for Graph 1 and 2 and saves it to a file."""
    ks = k_f(g1_2_params['num_qubits'])
    g1_2_results = defaultdict(dict)

    for noise_param in tqdm(g1_2_params['noise_params_list'], desc="Noise (G1, G2)"):
        for steps in tqdm(g1_2_params['steps_list'], desc="Steps", leave=False):
            base_angles = (np.pi / 2) * np.arange(1, steps + 1) / (steps + 1)
            base_angles = base_angles[:, np.newaxis]
            momentum_dms, qiskit_circs, local_circuits = [], [], []
            for _ in range(g1_2_params['num_circuits']):
                noisy_base = base_angles + noise_param * np.random.randn(steps, 1)
                betas, alphas = -np.sin(noisy_base).flatten(), -np.cos(noisy_base).flatten()
                sol = [tfim_momentum_trotter_single_k(k, steps, betas, alphas, 0) for k in ks]
                momentum_dms.append(sol)
                circuit = QuantumCircuit(g1_2_params['num_qubits'], g1_2_params['num_qubits'])
                circuit.h(range(g1_2_params['num_qubits']))
                for step in range(steps):
                    for i in range(0, g1_2_params['num_qubits'], 2):
                        circuit.rzz(betas[step], i, (i + 1) % g1_2_params['num_qubits'])
                    for i in range(1, g1_2_params['num_qubits'], 2):
                        circuit.rzz(betas[step], i, (i + 1) % g1_2_params['num_qubits'])
                    circuit.rx(alphas[step], range(g1_2_params['num_qubits']))
                circuit.measure(range(g1_2_params['num_qubits']), range(g1_2_params['num_qubits']))
                qiskit_circs.append(circuit)

                if noise_param == g1_2_params['noise_params_list'][-1]:
                    # add local noise
                    betas = -np.sin(base_angles).flatten()
                    alphas = -np.cos(base_angles).flatten()
                    local_circuit = QuantumCircuit(g1_2_params['num_qubits'], g1_2_params['num_qubits'])
                    local_circuit.h(range(g1_2_params['num_qubits']))
                    for step in range(steps):
                        for i in range(0, g1_2_params['num_qubits'], 2):
                            local_circuit.rzz(betas[step], i, (i + 1) % g1_2_params['num_qubits'])
                        for i in range(1, g1_2_params['num_qubits'], 2):
                            local_circuit.rzz(betas[step], i, (i + 1) % g1_2_params['num_qubits'])
                        local_circuit.rx(alphas[step], range(g1_2_params['num_qubits']))
                        for qubit in range(g1_2_params['num_qubits']):
                            local_circuit.rz(np.random.randn() * noise_param, qubit)
                    local_circuit.measure(range(g1_2_params['num_qubits']), range(g1_2_params['num_qubits']))
                    local_circuits.append(local_circuit)

            # Momentum model calculation (once per data point)
            mean_i, var_i = process_step(step_density_matrices=momentum_dms, ks=ks,
                                         num_qubits=g1_2_params['num_qubits'], method="rho")

            # Qiskit calculation with per-shot statistics
            simulator = AerSimulator()

            transpiled_circuits = transpile(qiskit_circs, simulator)
            job_result = simulator.run(transpiled_circuits, shots=g1_2_params['num_shots'], memory=True).result()

            # Create a matrix of kink counts: (num_circuits, num_shots)
            kink_counts_matrix = np.zeros((len(qiskit_circs), g1_2_params['num_shots']))
            for i in range(len(qiskit_circs)):
                outcomes = job_result.get_memory(i)
                kink_counts_matrix[i, :] = [count_kinks(state) for state in outcomes]

            # Calculate stats for each shot index across all circuits
            means_per_shot = []
            vars_per_shot = []
            fano_per_shot = []
            for j in range(g1_2_params['num_shots']):
                kinks_for_shot_j = kink_counts_matrix[:, j]
                means_per_shot.append(np.mean(kinks_for_shot_j))
                vars_per_shot.append(np.var(kinks_for_shot_j))
                fano_per_shot.append(
                        np.var(kinks_for_shot_j) / np.mean(kinks_for_shot_j) if np.mean(kinks_for_shot_j) != 0 else -1)

            # Final statistics are the mean and std of the per-shot statistics
            final_mean_kinks = np.mean(means_per_shot)
            std_err_mean_kinks = np.std(means_per_shot)

            final_var_kinks = np.mean(vars_per_shot)
            std_err_var_kinks = np.std(vars_per_shot)

            fano_per_shot = [f for f in fano_per_shot if f != -1]  # Filter out invalid Fano factors
            final_fano = np.mean(fano_per_shot)
            std_err_fano = np.std(fano_per_shot)

            if noise_param == g1_2_params['noise_params_list'][-1]:
                transpiled_circuits = transpile(local_circuits, simulator)
                job_result = simulator.run(transpiled_circuits, shots=g1_2_params['num_shots'], memory=True).result()

                # Create a matrix of kink counts: (num_circuits, num_shots)
                kink_counts_matrix = np.zeros((len(local_circuits), g1_2_params['num_shots']))
                for i in range(len(local_circuits)):
                    outcomes = job_result.get_memory(i)
                    kink_counts_matrix[i, :] = [count_kinks(state) for state in outcomes]

                # Calculate stats for each shot index across all circuits
                means_per_shot = []
                vars_per_shot = []
                fano_per_shot = []
                for j in range(g1_2_params['num_shots']):
                    kinks_for_shot_j = kink_counts_matrix[:, j]
                    means_per_shot.append(np.mean(kinks_for_shot_j))
                    vars_per_shot.append(np.var(kinks_for_shot_j))
                    fano_per_shot.append(
                            np.var(kinks_for_shot_j) / np.mean(kinks_for_shot_j) if np.mean(
                                    kinks_for_shot_j) != 0 else -1)

                # Final statistics are the mean and std of the per-shot statistics
                final_mean_kinks_local = np.mean(means_per_shot)
                std_err_mean_kinks_local = np.std(means_per_shot)

                final_var_kinks_local = np.mean(vars_per_shot)
                std_err_var_kinks_local = np.std(vars_per_shot)

                fano_per_shot = [f for f in fano_per_shot if f != -1]  # Filter out invalid Fano factors
                final_fano_local = np.mean(fano_per_shot)
                std_err_fano_local = np.std(fano_per_shot)

            g1_2_results[noise_param][steps] = {
                'mean_independent_modes'     : mean_i,
                'qiskit_mean_kinks_r_mean'   : final_mean_kinks / g1_2_params['num_qubits'],
                'qiskit_mean_kinks_r_std_err': std_err_mean_kinks / g1_2_params['num_qubits'],
                'var_independent_modes'      : var_i,
                'qiskit_var_kinks_r_mean'    : final_var_kinks / g1_2_params['num_qubits'],
                'qiskit_var_kinks_r_std_err' : std_err_var_kinks / g1_2_params['num_qubits'],
                'final_fano_mean'            : final_fano,
                'final_fano_std_err'         : std_err_fano,
                }
            if noise_param == g1_2_params['noise_params_list'][-1]:
                g1_2_results[noise_param][steps].update({
                    'qiskit_mean_kinks_r_mean_local'   : final_mean_kinks_local / g1_2_params['num_qubits'],
                    'qiskit_mean_kinks_r_std_err_local': std_err_mean_kinks_local / g1_2_params['num_qubits'],
                    'qiskit_var_kinks_r_mean_local'    : final_var_kinks_local / g1_2_params['num_qubits'],
                    'qiskit_var_kinks_r_std_err_local' : std_err_var_kinks_local / g1_2_params['num_qubits'],
                    'final_fano_mean_local'            : final_fano_local,
                    'final_fano_std_err_local'         : std_err_fano_local,
                    })
    data_to_save = {'results': g1_2_results, 'params': g1_2_params}
    # with open(filename, 'wb') as f:
    #     pickle.dump(data_to_save, f)
    # print(f"Saved Graph 1&2 data to {filename}")
    return data_to_save


def plot_mean(all_data):
    # --- Extract data for Graph 1 ---
    g1_data = all_data['graph1_2']
    results = g1_data['results']
    params = g1_data['params']
    steps_list = params['steps_list']
    noise_params_list = params['noise_params_list']

    # --- Plotting ---
    pyplot_settings()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x = np.array(steps_list)
    for n in noise_params_list:
        # Extract mean and std values for Qiskit and the momentum model
        mean_q_mean = np.array([results[n][i]['qiskit_mean_kinks_r_mean'] for i in x])
        mean_q_std_err = np.array([results[n][i]['qiskit_mean_kinks_r_std_err'] for i in x])
        mean_m = np.array([results[n][i]['mean_independent_modes'] for i in x])

        # Plot lines with error bars for Qiskit, and a simple line for the momentum model
        ax.errorbar(x, mean_q_mean, yerr=mean_q_std_err, fmt='o-', capsize=3, label=f'Qiskit Global, $\\sigma={n}$',
                    zorder=1)
        ax.plot(x, mean_m, 'x:', label=f'Momentum, $\\sigma={n}$', zorder=2)

    n = noise_params_list[-1]
    mean_q_mean = np.array([results[n][i]['qiskit_mean_kinks_r_mean_local'] for i in x])
    mean_q_std_err = np.array([results[n][i]['qiskit_mean_kinks_r_std_err_local'] for i in x])

    # Plot lines with error bars for Qiskit, and a simple line for the momentum model
    ax.errorbar(x, mean_q_mean, yerr=mean_q_std_err, fmt='s', capsize=3, label=f'Qiskit Local, $\\sigma={n}$', zorder=1)

    ax.set_xlabel(r'\textbf{Steps}')
    ax.set_ylabel(r'\textbf{Mean Kinks / N}')
    ax.set_title(r'\textbf{Normalized Mean Kinks vs. Steps}')
    ax.grid(True)
    legend_title = f"{params['num_circuits']} Circuits, {params['num_shots']} Shots"
    ax.legend(title=legend_title, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout(rect=(0, 0, 1, 1))
    # plot_filename = get_dated_plot_path(f"Mean_kinks_vs_steps.svg")
    # plt.savefig(plot_filename, bbox_inches='tight')
    plt.show()


def plot_variance(all_data):
    # --- Extract data for Graph 2 ---
    g2_data = all_data['graph1_2']
    results = g2_data['results']
    params = g2_data['params']
    steps_list = params['steps_list']
    noise_params_list = params['noise_params_list']

    # --- Plotting ---
    pyplot_settings()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x = np.array(steps_list)
    for n in noise_params_list:
        # Extract variance values for Qiskit and the momentum model
        var_q_mean = np.array([results[n][i]['qiskit_var_kinks_r_mean'] for i in x])
        var_q_std_err = np.array([results[n][i]['qiskit_var_kinks_r_std_err'] for i in x])
        var_m = np.array([results[n][i]['var_independent_modes'] for i in x])

        # Plot lines for each noise parameter (with error bars for Qiskit variance)
        ax.errorbar(x, var_q_mean, yerr=var_q_std_err, fmt='o-', capsize=3, label=f'Qiskit Global, $\\sigma={n}$',
                    zorder=1)
        ax.plot(x, var_m, 'x:', label=f'Momentum, $\\sigma={n}$', zorder=2)

    n = noise_params_list[-1]
    var_q_mean = np.array([results[n][i]['qiskit_var_kinks_r_mean_local'] for i in x])
    var_q_std_err = np.array([results[n][i]['qiskit_var_kinks_r_std_err_local'] for i in x])
    ax.errorbar(x, var_q_mean, yerr=var_q_std_err, fmt='s-', capsize=3, label=f'Qiskit Local, $\\sigma={n}$', zorder=1)

    ax.set_xlabel(r'\textbf{Steps}')
    ax.set_ylabel(r'\textbf{Variance / N}')
    ax.set_title(r'\textbf{Normalized Kink Variance vs. Steps}')
    ax.grid(True)
    legend_title = f"{params['num_circuits']} Circuits, {params['num_shots']} Shots"
    ax.legend(title=legend_title, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout(rect=(0, 0, 1, 1))
    # plot_filename = get_dated_plot_path(f"Variance_kinks_vs_steps.svg")
    # plt.savefig(plot_filename, bbox_inches='tight')
    plt.show()


def plot_fano(all_data):
    # --- Extract data for Graph 2 ---
    g2_data = all_data['graph1_2']
    results = g2_data['results']
    params = g2_data['params']
    steps_list = params['steps_list']
    noise_params_list = params['noise_params_list']

    # --- Plotting ---
    pyplot_settings()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x = np.array(steps_list)
    for n in noise_params_list:
        fano_q = np.array([results[n][i]['final_fano_mean'] for i in x])
        fano_q_std_err = np.array([results[n][i]['final_fano_std_err'] for i in x])
        fano_m = np.array([results[n][i]['var_independent_modes'] / results[n][i]['mean_independent_modes'] for i in x])

        # Plot lines for each noise parameter (with error bars for Qiskit variance)
        ax.errorbar(x, fano_q, yerr=fano_q_std_err, fmt='o-', capsize=3, label=f'Qiskit Global, $\\sigma={n}$',
                    zorder=1)
        ax.plot(x, fano_m, 'x:', label=f'Momentum, $\\sigma={n}$', zorder=2)

    n = noise_params_list[-1]
    fano_q = np.array([results[n][i]['final_fano_mean_local'] for i in x])
    fano_q_std_err = np.array([results[n][i]['final_fano_std_err_local'] for i in x])
    ax.errorbar(x, fano_q, yerr=fano_q_std_err, fmt='s-', capsize=3, label=f'Qiskit Local, $\\sigma={n}$', zorder=1)

    ax.set_xlabel(r'\textbf{Steps}')
    ax.set_ylabel(r'\textbf{Fano Factor}')
    ax.set_title(r'\textbf{Fano Factor vs. Steps}')
    ax.grid(True)
    legend_title = f"{params['num_circuits']} Circuits, {params['num_shots']} Shots"
    ax.legend(title=legend_title, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout(rect=(0, 0, 1, 1))
    # plot_filename = get_dated_plot_path(f"Variance_kinks_vs_steps.svg")
    # plt.savefig(plot_filename, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    all_data = {}
    # try:
    #     with open(DATA_FILENAMES['graph1_2'], 'rb') as f:
    #         all_data['graph1_2'] = pickle.load(f)
    #     print(f"Successfully loaded data from {DATA_FILENAMES['graph1_2']}")
    # except (FileNotFoundError, EOFError):
    #     print(f"Could not load data for Graph 1&2. Calculating new data...")
    #     all_data['graph1_2'] = calculate_graph1_2_data(g1_2_params, DATA_FILENAMES['graph1_2'])
    all_data['graph1_2'] = calculate_graph1_2_data(g1_2_params, DATA_FILENAMES['graph1_2'])
    plot_mean(all_data)
    plot_variance(all_data)
    plot_fano(all_data)
