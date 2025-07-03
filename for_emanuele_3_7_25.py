from multiprocessing import Pool
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.integrate import quad
from scipy.linalg import expm
from tqdm import tqdm


def pyplot_settings():
    # Global Settings for Matplotlib
    plt.rcParams.update({
        'text.usetex'     : True,  # Enable LaTeX rendering for text
        'font.family'     : 'serif',  # Set font family
        'font.size'       : 18,  # General font size
        'lines.markersize': 10,  # Default marker size
        'legend.fontsize' : 'small',  # Legend font size
        'legend.frameon'  : False,  # Remove frame around legend
        'figure.figsize'  : (6, 5),  # Larger figure size for legend
        'axes.grid'       : True,  # Enable grid for axes
        'grid.alpha'      : 0.1,  # Set grid transparency
        'grid.linestyle'  : '--',  # Set grid line style
        'grid.color'      : 'gray',  # Set grid line color
        'axes.grid.which' : 'both',  # Enable both major and minor gridlines
        'axes.grid.axis'  : 'both',  # Apply grid to both x and y axes
        'axes.labelsize'  : 22,  # Font size for axis labels
        'xtick.labelsize' : 13,  # Font size for x-axis tick labels
        'ytick.labelsize' : 13  # Font size for y-axis tick labels
        })


def calc_kinks_mean(kinks_probability):
    total_kinks = sum(k * v for k, v in kinks_probability.items())
    return total_kinks


def ln_P_tilda_func(theta, pks):
    return np.sum(np.log(1 + pks * (np.exp(2 * 1j * theta) - 1)))


def integrand_func(theta, pks, d):
    return np.exp(ln_P_tilda_func(theta, pks) - 1j * theta * d)


def P_func(pks, d):
    integral, _ = quad(lambda theta: np.real(integrand_func(theta, pks, d)), -np.pi, np.pi, limit=10000)
    return np.abs(integral / (2 * np.pi))


def calc_kink_probabilities(pks, d_vals, parallel=True):
    if parallel:
        with Pool() as pool:
            return np.array(pool.starmap(P_func, [(pks, d) for d in d_vals]))
    else:
        return np.array([P_func(pks, d) for d in d_vals])


def process_step(step_density_matrices, ks, num_qubits, method):
    if method == "observable":
        means_runs, vars_runs = [], []
        for solutions in step_density_matrices:
            # Calculate pks for each k
            pks = np.array([
                np.abs(np.dot(
                        np.array([np.sin(k / 2), np.cos(k / 2)]),
                        np.dot(solution, np.array([[np.sin(k / 2)], [np.cos(k / 2)]]))
                        ))[0] for k, solution in zip(ks, solutions)
                ])
            pks = np.where(pks < 1e-10, 0, pks)
            kinks_vals = np.arange(0, num_qubits + 1, 2)
            distribution = calc_kink_probabilities(pks, kinks_vals, parallel=False)
            kinks_distribution = {k: v for k, v in zip(kinks_vals, distribution)}
            mean_kinks = calc_kinks_mean(kinks_distribution)
            second_moment = sum(k ** 2 * v for k, v in kinks_distribution.items())
            var_kinks = second_moment - mean_kinks ** 2
            means_runs.append(mean_kinks / num_qubits)
            vars_runs.append(var_kinks / num_qubits)
        return np.mean(means_runs), np.mean(vars_runs)
    elif method == "rho":
        avg_solutions = []
        for k_idx in range(len(ks)):
            k_solutions = [solutions[k_idx] for solutions in step_density_matrices]
            avg_rho = np.mean(k_solutions, axis=0)
            avg_solutions.append(avg_rho)
        pks = np.array([
            np.abs(np.dot(
                    np.array([np.sin(k / 2), np.cos(k / 2)]),
                    np.dot(solution, np.array([[np.sin(k / 2)], [np.cos(k / 2)]]))
                    ))[0] for k, solution in zip(ks, avg_solutions)
            ])
        pks = np.where(pks < 1e-10, 0, pks)
        kinks_vals = np.arange(0, num_qubits + 1, 2)
        distribution = calc_kink_probabilities(pks, kinks_vals, parallel=False)
        kinks_distribution = {k: v for k, v in zip(kinks_vals, distribution)}
        mean_kinks = calc_kinks_mean(kinks_distribution)
        second_moment = sum(k ** 2 * v for k, v in kinks_distribution.items())
        var_kinks = second_moment - mean_kinks ** 2
        return mean_kinks / num_qubits, var_kinks / num_qubits


def count_kinks(bitstring: str) -> int:
    """Count the number of kinks in a quantum state string (PBC)."""
    count = 0
    n = len(bitstring)
    if n == 0:  # Handle empty string case
        return 0
    for i in range(n):  # Loop from 0 to N-1
        j = (i + 1) % n  # Use modulo for periodic boundary
        # if j < i: continue # TODO: remove, this is for OPC
        if bitstring[i] != bitstring[j]:
            count += 1
    return count


def k_f(N: int) -> np.ndarray:
    # Filtering out even numbers
    odd_numbers = np.arange(1, N, 2)

    # Multiply each odd number by pi/N
    return odd_numbers * np.pi / N


def tfim_momentum_trotter_single_k(k, tau, beta, alpha, W, dt=1):
    """
    Simulates Trotterized evolution of the TFIM in momentum space for a single momentum k
    with Lindblad noise.

    The evolution is performed in Nambu space, with the initial state chosen as |↓⟩
    (represented as [0, 1]^T, i.e. the ground state for g=0).

    Parameters:
      k     : momentum (in radians)
      tau   : total number of Trotter steps
      beta  : array of angles for ZZ interaction terms
      alpha : array of angles for X field terms
      W     : float, noise strength
      dt    : float, time step per Trotter step

    Returns:
      rho : the final density matrix for momentum k.
    """
    # Initial state |↓⟩ is represented as [0, 1]^T; hence its density matrix:
    rho = np.array([[0, 0], [0, 1]], dtype=complex)

    # Pauli matrices
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)

    # Lindblad parameters from Dutta et al. (2016)
    v_z = -2 * (1 + np.cos(k))
    v_x = 2 * np.sin(k)
    V_norm = np.sqrt(v_z ** 2 + v_x ** 2)
    gamma_dt = (W ** 2 * dt) / 2  # Effective noise strength per step

    # Kraus operators for Lindblad noise
    p = 1 - np.exp(-gamma_dt * V_norm ** 2)  # Simplified probability
    K0 = np.sqrt(1 - p / 2) * np.eye(2)
    K1 = np.sqrt(p / 2) * (v_z / V_norm * sigma_z + v_x / V_norm * sigma_x)

    # Loop over each Trotter step
    for n in range(tau):
        # Get the angles for this step
        h = alpha[n]  # transverse field part
        J = beta[n]  # interaction strength

        # Hamiltonian terms
        H_fm = -1 * J * np.cos(k) * sigma_z + (1 * J * np.sin(k)) * sigma_x
        H_pm = 1 * h * sigma_z

        # Evolution operators
        U_z = expm(-1j * H_fm)
        U_x = expm(-1j * H_pm)

        # Apply the evolution
        rho = U_x @ (U_z @ rho @ U_z.conj().T) @ U_x.conj().T

        # Apply Lindblad noise
        rho = K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T

    return rho


def compare_qiskit_and_momentum_diff(
        num_qubits: int,
        steps: int,
        num_circuits_list: List[int],
        noise_params_list: List[float],
        num_shots: int = 100,
        ) -> None:
    """
    Compare Qiskit results with momentum model for given parameters.
    """
    ks = k_f(num_qubits)

    base_angles = (np.pi / 2) * np.arange(1, steps + 1) / (steps + 1)
    base_angles = base_angles[:, np.newaxis]
    results = {}
    for num_circuits in tqdm(num_circuits_list, desc="Processing number of circuits"):
        results[num_circuits] = {}
        for noise_param in tqdm(noise_params_list, desc="Processing noise parameters"):
            momentum_dms = []
            qiskit_results_list = []
            for _ in range(num_circuits):
                # generate noise for each circuit
                # base angles are the same for all circuits, but noise is different
                noisy_base = base_angles + noise_param * np.random.randn(steps, 1)
                betas = -np.sin(noisy_base).flatten()  # -np.sin(base_angles).flatten()
                alphas = -np.cos(noisy_base).flatten()

                # compute momentum model density matrices
                # calculate density matrices
                sol = [tfim_momentum_trotter_single_k(k, steps, betas, alphas, 0) for k in ks]
                momentum_dms.append(sol)

                # calculate Qiskit results
                # Run the simulation
                circuit = QuantumCircuit(num_qubits, num_qubits)
                circuit.h(range(num_qubits))  # Initial superposition
                for step in range(steps):
                    beta = betas[step]
                    alpha = alphas[step]

                    # Apply RZZ gates in parallel between non-overlapping pairs
                    # First group: even-to-odd pairs
                    for i in range(0, num_qubits, 2):
                        j = (i + 1) % num_qubits
                        # if j < i: continue # TODO: remove, this for OPC
                        circuit.rzz(beta, i, j)

                    # Second group: odd-to-even pairs
                    for i in range(1, num_qubits, 2):
                        j = (i + 1) % num_qubits
                        # if j < i: continue  # TODO: remove, this for OPC
                        circuit.rzz(beta, i, j)

                    # Apply RX gates for transverse field
                    for i in range(num_qubits):
                        circuit.rx(alpha, i)

                circuit.measure(range(num_qubits), range(num_qubits))

                simulator = AerSimulator()
                transpiled_circuits = transpile(circuit, simulator, num_processes=-1)

                job_result = simulator.run(transpiled_circuits, shots=num_shots).result()
                qiskit_results_list.append({
                    'counts': job_result.get_counts(),
                    })

            # Aggregate counts from all circuits
            total_counts = {}
            for result in qiskit_results_list:
                counts = result['counts']  # Access counts from our dictionary
                for state, count in counts.items():
                    total_counts[state] = total_counts.get(state, 0) + count

            # Calculate probabilities
            total_shots = sum(total_counts.values())
            probabilities = {state: count / total_shots for state, count in total_counts.items()}

            # Calculate mean and variance of kinks
            kink_counts = [count_kinks(state) for state in probabilities.keys()]
            mean_kinks0 = sum(k * p for k, p in zip(kink_counts, probabilities.values()))
            var_kinks0 = sum((k - mean_kinks0) ** 2 * p for k, p in zip(kink_counts, probabilities.values()))

            # keep both mean_kinks0 and mean_kinks1 for comparison and give them good names
            qiskit_results = {
                "mean_kinks"   : mean_kinks0,
                "var_kinks"    : var_kinks0,
                "probabilities": probabilities,
                }

            mean_r = qiskit_results['mean_kinks'] / num_qubits
            var_r = qiskit_results['var_kinks'] / num_qubits
            ratio_r = var_r / mean_r
            # independent (rho averaging) method
            mean_i, var_i = process_step(step_density_matrices=momentum_dms, ks=ks, num_qubits=num_qubits, method="rho")
            ratio_i = var_i / mean_i

            results[num_circuits][noise_param] = {
                'momentum_ratio_independent_modes': ratio_i,
                'qiskit_ratio_r'                  : ratio_r,
                'mean_independent_modes'          : mean_i,
                'var_independent_modes'           : var_i,
                'qiskit_mean_kinks_r'             : mean_r,
                'qiskit_var_kinks_r'              : var_r,
                }
    pyplot_settings()
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    x = np.array(num_circuits_list)
    # sum the difference between the models for each noise parameter
    diffs = {}
    for num_circuits in num_circuits_list:
        diffs[num_circuits] = 0
        for noise_param in noise_params_list:
            diffs[num_circuits] += np.abs(results[num_circuits][noise_param]['momentum_ratio_independent_modes'] -
                                          results[num_circuits][noise_param]['qiskit_ratio_r'])
    y = np.array([diffs[nc] for nc in num_circuits_list])

    # plot lines
    ax.plot(x, y)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r'\textbf{Number of Circuits}')
    ax.set_ylabel(r'\textbf{Sum of Differences}')
    ax.grid(True)

    # add title
    super_title = (rf"qubits = {num_qubits}, "
                   rf"steps = {steps}, "
                   rf"noises = {noise_params_list}, ")
    plt.suptitle(super_title, y=0.94)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    noise_params_list = np.logspace(-2, 1, 5).tolist()
    num_qubits = 6
    steps = 30
    num_circuits = 1000
    num_shots = 1000
    num_circuits_list = [int(i) for i in np.logspace(1, 3, 5)]
    compare_qiskit_and_momentum_diff(
            num_qubits=num_qubits,
            steps=steps,
            num_circuits_list=num_circuits_list,
            noise_params_list=noise_params_list,
            num_shots=num_shots
            )
