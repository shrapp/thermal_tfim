import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from scipy.special import comb

from discrete_numeric import tfim_momentum_trotter_single_k
from functions import calc_kink_probabilities, k_f, pyplot_settings

DATA_FILENAMES = {
    'kink_distributions': "kink_distributions_data.pkl"
    }


def compute_scenario_data(scenario_idx, scenario, params):
    """Helper function to compute kink distribution for a single scenario. Run in parallel."""
    steps, noise_param = scenario
    num_qubits = params['num_qubits']
    num_circuits = params['num_circuits']
    ks = params['ks']
    kinks_vals = params['kinks_vals']

    if steps == 0:
        # Analytical: Uniform over even kinks (binomial probabilities normalized for even parity)
        probs = np.array([comb(num_qubits, k) / 2 ** num_qubits for k in kinks_vals])
        probs /= np.sum(probs)  # Normalize for even-only
        distribution = dict(zip(kinks_vals, probs))
        # Mean/var for verification (analytical)
        mean_k = sum(k * v for k, v in distribution.items())
        var_k = sum((k - mean_k) ** 2 * v for k, v in distribution.items())
    else:
        # Numerical simulation
        base_angles = (np.pi / 2) * np.arange(1, steps + 1) / (steps + 1)
        base_angles = base_angles[:, np.newaxis]
        momentum_dms = []
        for _ in range(num_circuits):
            noisy_base = base_angles + noise_param * np.random.randn(steps, 1)
            betas, alphas = -np.sin(noisy_base).flatten(), -np.cos(noisy_base).flatten()
            sol = [tfim_momentum_trotter_single_k(k, steps, betas, alphas, 0) for k in ks]
            momentum_dms.append(sol)

        # Average density matrices per k
        avg_solutions = []
        for k_idx in range(len(ks)):
            k_solutions = [solutions[k_idx] for solutions in momentum_dms]
            avg_rho = np.mean(k_solutions, axis=0)
            avg_solutions.append(avg_rho)

        # Compute probabilities
        pks = np.array([
            np.abs(np.dot(
                    np.array([np.sin(k / 2), np.cos(k / 2)]),
                    np.dot(solution, np.array([[np.sin(k / 2)], [np.cos(k / 2)]]))
                    )) for k, solution in zip(ks, avg_solutions)
            ])
        pks = np.where(pks < 1e-10, 0, pks)
        distribution = {k: v for k, v in zip(kinks_vals, calc_kink_probabilities(pks, kinks_vals, parallel=False))}

        # Mean/var for verification
        mean_k = sum(k * v for k, v in distribution.items())
        var_k = sum((k - mean_k) ** 2 * v for k, v in distribution.items())

    return scenario_idx, distribution


def calculate_kink_distributions(params, scenarios, filename, compute=True):
    """Calculates kink distributions for all scenarios and optionally saves. Parallelized over scenarios."""
    if not compute:
        return {}

    results = {}
    max_workers = min(len(scenarios), multiprocessing.cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_scenario_data, idx, scenario, params): idx
                   for idx, scenario in enumerate(scenarios)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Scenarios (parallel)"):
            idx, distribution = future.result()
            results[idx] = distribution

    data_to_save = {'results': results, 'params': params, 'scenarios': scenarios}
    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"Saved kink distributions to {filename}")
    return data_to_save


def load_kink_distributions(filename):
    """Loads existing kink distributions data."""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        print(f"Could not load data from {filename}.")
        return {}


def plot_kink_distributions(data, show=True, base_save_path=None, colors=None, labels=None):
    """Plots bar chart of kink distributions for all scenarios, split into no-noise and noisy subplots."""
    if not data:
        return
    results, params, scenarios = data['results'], data['params'], data['scenarios']
    if colors is None:
        colors = ['black', 'blue', 'red', 'green', 'orange']
    if labels is None:
        # labels = [f"{steps} steps, {noise_param} noise" for steps, noise_param in scenarios]
        labels = [f"{steps} steps" for steps, noise_param in scenarios]

    # No-noise plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    num_qubits = params['num_qubits']
    zero_idx = next(idx for idx in sorted(results) if scenarios[idx][0] == 0)

    no_noise_indices = sorted([idx for idx in sorted(results) if scenarios[idx][1] == 0.0],
                              key=lambda idx: scenarios[idx][0])
    for i, idx in enumerate(no_noise_indices):
        distribution = results[idx]
        x_pos = np.array([k / num_qubits for k in distribution.keys()])
        y_vals = np.array(list(distribution.values()))
        ax.bar(x_pos, y_vals, width=0.02, label=labels[idx], color=colors[i], alpha=0.7)
        # Add dashed line overlay with the same color
        # ax.plot(x_pos, y_vals, color=colors[i], linestyle='--', linewidth=1.5)
    # ax.set_title('No Noise')
    ax.set_xlabel('Kinks / $N$')
    ax.set_ylabel('Probability')
    ax.set_xlim(0, 0.65)
    # legend_title = f"{params['num_qubits']} Qubits, {params['num_circuits']} Circuits"
    # ax.legend(title=legend_title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if base_save_path:
        no_noise_path = f"{base_save_path}_no_noise.jpg"
        plt.savefig(no_noise_path, bbox_inches='tight', dpi=1200)
    if show:
        plt.show()

    # Noisy plot, including zero steps as baseline
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    noisy_indices = [zero_idx] + sorted([idx for idx in sorted(results) if scenarios[idx][1] > 0],
                                        key=lambda idx: scenarios[idx][0])
    num_noisy = len(noisy_indices)
    colors_noisy = colors[:num_noisy]
    for i, idx in enumerate(noisy_indices):
        distribution = results[idx]
        x_pos = np.array([k / num_qubits for k in distribution.keys()])
        y_vals = np.array(list(distribution.values()))
        ax.bar(x_pos, y_vals, width=0.02, label=labels[idx], color=colors_noisy[i], alpha=0.7)
        # Add dashed line overlay with the same color
        ax.plot(x_pos, y_vals, color=colors_noisy[i], linestyle='--', linewidth=1.5)
    ax.set_xlabel('Kinks / $N$')
    ax.set_ylabel('Probability')
    ax.set_xlim(0.25, 0.75)
    # ax.legend(title=legend_title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Shared, centered axis labels (adjusted y-position for tick clearance)
    # fig.text(0.5, 0, 'Kinks / $N$', ha='center', va='top')
    # fig.text(0.04, 0.5, r'Probability', ha='center', va='center', rotation='vertical')

    plt.tight_layout()

    if base_save_path:
        noisy_path = f"{base_save_path}_noisy.jpg"
        plt.savefig(noisy_path, bbox_inches='tight', dpi=1200)
    if show:
        plt.show()


def run_kink_distributions(params=None, scenarios=None, compute=True, load_if_exists=True,
                           enable_plot=True, save_plot=False, show_plot=True,
                           colors=None, labels=None):
    """
    Central runner for kink distributions workflow.

    Args:
        params: Dict of parameters (defaults if None).
        scenarios: List of (steps, noise) tuples (defaults if None).
        compute: If True, compute new data (overrides load_if_exists).
        load_if_exists: If True and compute=False, load existing data.
        enable_plot: Boolean to enable/disable the plot.
        save_plot: If True, save plot to path.
        show_plot: If True, display plot via plt.show().
        colors, labels: Lists for customizing plot aesthetics.
    """
    if params is None:
        num_qubits = 100
        num_circuits = 100
        params = {
            'num_qubits'  : num_qubits,
            'num_circuits': num_circuits,
            'ks'          : k_f(num_qubits),  # Precompute ks
            'kinks_vals'  : np.arange(0, num_qubits + 1, 2)  # Even kinks for periodic chain
            }

    if scenarios is None:
        scenarios = [
            (0, 0.0),  # Initial state, no noise
            (2, 0.0),  # 3 steps, no noise
            (300, 0.0),  # 150 steps, no noise
            (2, 0.8),  # 10 steps, 0.3 noise
            (300, 0.8)  # 1000 steps, 0.1 noise
            ]

    filename = DATA_FILENAMES['kink_distributions']
    data = {}

    if compute:
        data = calculate_kink_distributions(params, scenarios, filename, compute=True)
    elif load_if_exists:
        data = load_kink_distributions(filename)

    # Generate plot if enabled
    base_plot_path = None
    if enable_plot:
        if save_plot:
            base_plot_path = "kink_distributions"
        plot_kink_distributions(data, show=show_plot, base_save_path=base_plot_path,
                                colors=colors, labels=labels)

    if save_plot:
        print(f"Plots saved: kink_distributions_no_noise.jpg and kink_distributions_noisy.jpg")


if __name__ == "__main__":
    pyplot_settings()
    # Example usage: Compute and plot, or customize
    run_kink_distributions(
            compute=False,
            enable_plot=True,
            save_plot=False,
            show_plot=True
            )