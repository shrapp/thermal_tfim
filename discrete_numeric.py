import datetime
import logging
import multiprocessing
import os
import random
import sqlite3
from functools import partial
from multiprocessing import Pool
from typing import Tuple, Dict, List

import itertools
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix
from qiskit_aer import AerSimulator
from scipy.linalg import expm
from tqdm import tqdm

from functions import (calc_kink_probabilities, calc_kinks_mean,
                       k_f,
                       pyplot_settings)

# Constants and Configuration
# -------------------------
# Set Qiskit logging level to WARNING to suppress INFO and DEBUG messages
qiskit_logger = logging.getLogger('qiskit')
qiskit_logger.setLevel(logging.WARNING)

# Your existing logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


# Core TFIM Functions
# ------------------
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


def tfim_momentum_trotter(ks, tau, W=0.0, betas=None, alphas=None, parallel=True):
    """
    Parallelize Trotterized density matrix evolution for all k values.
    Uses the provided angles or generates them if not provided.
    If parallel=False, runs serially (for use inside outer multiprocessing).
    """
    if betas is None or alphas is None:
        # Generate angles following Qiskit's convention
        base_angles = (np.pi / 2) * np.arange(1, tau + 1) / (tau + 1)
        base_angles = base_angles[:, np.newaxis]
        noisy_base_angles = base_angles + W * np.random.randn(tau, 1)
        # Calculate beta and alpha arrays
        betas = -np.sin(noisy_base_angles).flatten()
        alphas = -np.cos(noisy_base_angles).flatten()
    if parallel:
        num_workers = multiprocessing.cpu_count()
        with Pool(processes=num_workers) as pool:
            results = pool.starmap(tfim_momentum_trotter_single_k,
                                   [(k, tau, betas, alphas, W) for k in ks])
        return results
    else:
        # Serial version (for use inside outer Pool)
        return [tfim_momentum_trotter_single_k(k, tau, betas, alphas, W) for k in ks]


def process_tfim_momentum_trotter(ks, depth, num_qubits, W, betas=None, alphas=None):
    """Process the TFIM in momentum space with Trotter discretization."""
    solutions = tfim_momentum_trotter(ks, depth, W, betas, alphas)
    tensor_product = solutions[0]
    for rho in solutions[1:]:
        tensor_product = np.kron(tensor_product, rho)
    density_matrix = tensor_product

    # Calculate purity
    try:
        rho2 = density_matrix @ density_matrix
        purity = np.trace(rho2).real
        if not np.isfinite(purity):
            purity = 0.0
    except Exception as e:
        print(f"Warning: Error calculating purity: {str(e)}")
        purity = 0.0

    # Calculate kink probabilities
    pks = np.array([
        np.abs(np.dot(
            np.array([np.sin(k / 2), np.cos(k / 2)]),
            np.dot(solution, np.array([[np.sin(k / 2)], [np.cos(k / 2)]]))
        ))[0] for k, solution in zip(ks, solutions)
    ])
    pks = np.where(pks < 1e-10, 0, pks)

    # Calculate kinks distribution
    kinks_vals = np.arange(0, num_qubits + 1, 2)
    distribution = calc_kink_probabilities(pks, kinks_vals, parallel=False)
    kinks_distribution = {k: v for k, v in zip(kinks_vals, distribution)}

    # Calculate mean and variance
    mean_kinks = calc_kinks_mean(kinks_distribution)
    second_moment = sum(k ** 2 * v for k, v in kinks_distribution.items())
    var_kinks = second_moment - mean_kinks ** 2

    return {
        "solutions": solutions,
        "pks": pks,
        "density_matrix": density_matrix,
        "kinks_distribution": kinks_distribution,
        "mean_kinks": mean_kinks,
        "var_kinks": var_kinks,
        "purity": purity
    }


# Qiskit Circuit Functions
# -----------------------
def generate_single_circuit_parallel(params):
    """Generate a single Qiskit circuit for parallel execution."""
    qubits, steps, circuit_idx, betas, alphas, noisy_betas, noise_method = params
    circuit = QuantumCircuit(qubits, qubits)
    circuit.h(range(qubits))  # Initial superposition
    for step in range(steps):
        beta = betas[step, circuit_idx]
        alpha = alphas[step, circuit_idx]

        # Apply RZZ gates in parallel between non-overlapping pairs
        # First group: even-to-odd pairs
        for i in range(0, qubits, 2):
            j = (i + 1) % qubits
            circuit.rzz(beta, i, j)

        # Second group: odd-to-even pairs
        for i in range(1, qubits, 2):
            j = (i + 1) % qubits
            circuit.rzz(beta, i, j)

        # Apply dephasing noise as RZ gates if needed
        if noise_method == 'dephasing':
            circuit.rz(noisy_betas[step, circuit_idx], range(qubits))

        # Apply RX gates for transverse field
        for i in range(qubits):
            circuit.rx(alpha, i)

    dm = DensityMatrix.from_instruction(circuit)
    circuit.measure(range(qubits), range(qubits))
    return circuit, dm.data


def generate_qiskit_circuits(qubits, steps, num_circuits_per_step, noise_std=0.0, noise_method='global', betas=None,
                             alphas=None):
    """Generate multiple Qiskit circuits in parallel."""
    if betas is None or alphas is None:
        # Generate angles following Qiskit's convention
        base_angles = (np.pi / 2) * np.arange(1, steps + 1) / (steps + 1)
        base_angles = base_angles[:, np.newaxis]
        noisy_base_angles = base_angles + noise_std * np.random.randn(steps, num_circuits_per_step)

        if noise_method == 'global':
            betas = -np.sin(noisy_base_angles)
            alphas = -np.cos(noisy_base_angles)
        else:
            base_angles = np.tile(base_angles, (1, num_circuits_per_step))
            betas = -np.sin(base_angles)
            alphas = -np.cos(base_angles)
            noisy_betas = -np.sin(noisy_base_angles)
    else:
        # Use provided angles
        betas = np.tile(betas[:, np.newaxis], (1, num_circuits_per_step))
        alphas = np.tile(alphas[:, np.newaxis], (1, num_circuits_per_step))
        if noise_method == 'dephasing':
            noisy_base_angles = np.arcsin(-betas) + noise_std * np.random.randn(steps, num_circuits_per_step)
            noisy_betas = -np.sin(noisy_base_angles)

    params = [(qubits, steps, i, betas, alphas, noisy_betas if noise_method == 'dephasing' else None, noise_method)
              for i in range(num_circuits_per_step)]

    with Pool() as pool:
        results = pool.map(generate_single_circuit_parallel, params)

    circuits, density_matrices = zip(*results)
    return list(circuits), list(density_matrices)


def run_qiskit_simulation(qubits, steps, noise_param, noise_type, num_circuits, numshots, betas=None, alphas=None):
    """Run Qiskit simulation and return results."""
    circuits, density_matrices = generate_qiskit_circuits(
        qubits=qubits,
        steps=steps,
        num_circuits_per_step=num_circuits,
        noise_std=noise_param,
        noise_method=noise_type,
        betas=betas,
        alphas=alphas
    )

    simulator = AerSimulator()
    transpiled_circuits = transpile(circuits, simulator, num_processes=-1)

    job_result = simulator.run(transpiled_circuits, shots=numshots).result()

    results = []
    for i in range(len(circuits)):
        results.append({
            'counts': job_result.get_counts(i),
            'density_matrix': density_matrices[i]
        })

    return results


def count_kinks(bitstring: str) -> int:
    """Count the number of kinks in a quantum state string (PBC)."""
    count = 0
    n = len(bitstring)
    if n == 0:  # Handle empty string case
        return 0
    for i in range(n):  # Loop from 0 to N-1
        if bitstring[i] != bitstring[(i + 1) % n]:  # Use modulo for periodic boundary
            count += 1
    return count


def process_qiskit_model(num_qubits, depth, noise_param, noise_type, num_circuits, numshots, betas=None, alphas=None):
    """Process Qiskit simulation results and calculate statistics."""
    try:
        # Run the simulation
        results = run_qiskit_simulation(
            qubits=num_qubits,
            steps=depth,
            noise_param=noise_param,
            noise_type=noise_type,
            num_circuits=num_circuits,
            numshots=numshots,
            betas=betas,
            alphas=alphas
        )

        # Aggregate counts from all circuits
        total_counts = {}
        for result in results:
            counts = result['counts']  # Access counts from our dictionary
            for state, count in counts.items():
                total_counts[state] = total_counts.get(state, 0) + count

        # Calculate probabilities
        total_shots = sum(total_counts.values())
        probabilities = {state: count / total_shots for state, count in total_counts.items()}

        # Calculate mean and variance of kinks
        kink_counts = [count_kinks(state) for state in probabilities.keys()]
        mean_kinks = sum(k * p for k, p in zip(kink_counts, probabilities.values()))
        var_kinks = sum((k - mean_kinks) ** 2 * p for k, p in zip(kink_counts, probabilities.values()))

        return {
            "mean_kinks": mean_kinks,
            "var_kinks": var_kinks,
            "probabilities": probabilities
        }

    except Exception as e:
        print(f"Error in process_qiskit_model: {str(e)}")
        return {
            "mean_kinks": 0.0,
            "var_kinks": 0.0,
            "probabilities": {}
        }


# Plotting Functions
# -----------------
def plot_noise_effects_comparison(num_qubits=4, step_range=range(0, 31, 3),
                                  noise_param=0.0,
                                  num_circuits=2, numshots=100000, interactive=False,
                                  load_data=False, save_data=True):
    """
    Plot comparison of mean kinks and variance between Qiskit and momentum models with noise effects.
    Uses three different methods for calculating mean and variance in the momentum model:
    1. Method 1: Average the final results from multiple runs
    2. Method 2: Average the density matrices first, then calculate mean and variance
    3. Method 3: Direct calculation with Lindblad noise
    
    Parameters:
        noise_param (float): Noise strength for both angle noise and Lindblad evolution
        load_data (bool): If True, load previously saved data instead of running new simulations
        save_data (bool): If True, save the raw data for future use
    """
    if interactive:
        plt.ion()  # Enable interactive mode if requested
    else:
        plt.ioff()  # Disable interactive mode by default

    ks = k_f(num_qubits)

    # Construct data filename
    date_str = datetime.date.today().strftime('%Y%m%d')
    data_filename = f"raw_data_N{num_qubits}_noise{noise_param}_circ{num_circuits}_shots{numshots}_{date_str}.npz"

    if load_data and os.path.exists(data_filename):
        print(f"Loading data from {data_filename}")
        data = np.load(data_filename, allow_pickle=True)
        momentum_results_list = data['momentum_results_list'].tolist()
        qiskit_means = data['qiskit_means'].tolist()
        qiskit_vars = data['qiskit_vars'].tolist()

        # Extract momentum means and variances for each method from loaded data
        momentum_means_method1 = []
        momentum_vars_method1 = []
        momentum_means_method2 = []
        momentum_vars_method2 = []
        momentum_means_method3 = []
        momentum_vars_method3 = []

        # Process each step's results
        for step_results in momentum_results_list:
            # Method 1: Average of final results
            momentum_means_method1.append(np.mean([r["mean_kinks"] for r in step_results]))
            momentum_vars_method1.append(np.mean([r["var_kinks"] for r in step_results]))

            # Method 2: Average density matrices first
            all_solutions = [r["solutions"] for r in step_results]
            avg_solutions = []
            for k_idx in range(len(ks)):
                k_solutions = [solutions[k_idx] for solutions in all_solutions]
                avg_rho = np.mean(k_solutions, axis=0)
                avg_solutions.append(avg_rho)

            # Calculate pks from averaged density matrices
            avg_pks_method2 = np.array([
                np.abs(np.dot(
                    np.array([np.sin(k / 2), np.cos(k / 2)]),
                    np.dot(solution, np.array([[np.sin(k / 2)], [np.cos(k / 2)]]))
                ))[0] for k, solution in zip(ks, avg_solutions)
            ])
            avg_pks_method2 = np.where(avg_pks_method2 < 1e-10, 0, avg_pks_method2)

            # Calculate mean and variance using averaged density matrices
            kinks_vals = np.arange(0, num_qubits + 1, 2)
            distribution_method2 = calc_kink_probabilities(avg_pks_method2, kinks_vals, parallel=False)
            kinks_distribution_method2 = {k: v for k, v in zip(kinks_vals, distribution_method2)}

            mean_kinks_method2 = calc_kinks_mean(kinks_distribution_method2)
            second_moment_method2 = sum(k ** 2 * v for k, v in kinks_distribution_method2.items())
            var_kinks_method2 = second_moment_method2 - mean_kinks_method2 ** 2

            momentum_means_method2.append(mean_kinks_method2)
            momentum_vars_method2.append(var_kinks_method2)

            # Method 3: Use the first result as it's already using Lindblad noise
            result = step_results[0]
            momentum_means_method3.append(result["mean_kinks"])
            momentum_vars_method3.append(result["var_kinks"])
    else:
        momentum_means_method1 = []
        momentum_vars_method1 = []
        momentum_means_method2 = []
        momentum_vars_method2 = []
        momentum_means_method3 = []
        momentum_vars_method3 = []
        qiskit_means = []
        qiskit_vars = []
        momentum_results_list = []

        # Progress bar for steps
        steps_progress = tqdm(step_range, desc="Processing steps")
        for steps in steps_progress:
            steps_progress.set_description(f"Processing {steps} steps")

            # Generate angles once for both models
            base_angles = (np.pi / 2) * np.arange(1, steps + 1) / (steps + 1)
            base_angles = base_angles[:, np.newaxis]
            noisy_base_angles = base_angles + noise_param * np.random.randn(steps, 1)

            # Calculate beta and alpha arrays
            betas = -np.sin(noisy_base_angles).flatten()
            alphas = -np.cos(noisy_base_angles).flatten()

            # Process momentum model multiple times
            step_results = []
            momentum_progress = tqdm(range(num_circuits), desc="Momentum circuits", leave=False)
            for _ in momentum_progress:
                step_results.append(process_tfim_momentum_trotter(ks, steps, num_qubits, 0, betas, alphas))
            momentum_results_list.append(step_results)

            # Method 1: Average the final results
            momentum_mean_method1 = np.mean([r["mean_kinks"] for r in step_results])
            momentum_var_method1 = np.mean([r["var_kinks"] for r in step_results])
            momentum_means_method1.append(momentum_mean_method1)
            momentum_vars_method1.append(momentum_var_method1)

            # Method 2: Average density matrices first
            all_solutions = [r["solutions"] for r in step_results]

            # Average density matrices for each k
            avg_solutions = []
            for k_idx in range(len(ks)):
                k_solutions = [solutions[k_idx] for solutions in all_solutions]
                avg_rho = np.mean(k_solutions, axis=0)
                avg_solutions.append(avg_rho)

            # Calculate pks from averaged density matrices
            avg_pks_method2 = np.array([
                np.abs(np.dot(
                    np.array([np.sin(k / 2), np.cos(k / 2)]),
                    np.dot(solution, np.array([[np.sin(k / 2)], [np.cos(k / 2)]]))
                ))[0] for k, solution in zip(ks, avg_solutions)
            ])
            avg_pks_method2 = np.where(avg_pks_method2 < 1e-10, 0, avg_pks_method2)

            # Calculate mean and variance using averaged density matrices
            kinks_vals = np.arange(0, num_qubits + 1, 2)
            distribution_method2 = calc_kink_probabilities(avg_pks_method2, kinks_vals, parallel=False)
            kinks_distribution_method2 = {k: v for k, v in zip(kinks_vals, distribution_method2)}

            mean_kinks_method2 = calc_kinks_mean(kinks_distribution_method2)
            second_moment_method2 = sum(k ** 2 * v for k, v in kinks_distribution_method2.items())
            var_kinks_method2 = second_moment_method2 - mean_kinks_method2 ** 2

            momentum_means_method2.append(mean_kinks_method2)
            momentum_vars_method2.append(var_kinks_method2)

            # Method 3: Direct calculation with Lindblad noise
            betas = -np.sin(base_angles).flatten()
            alphas = -np.cos(base_angles).flatten()
            result = process_tfim_momentum_trotter(ks, steps, num_qubits, noise_param, betas, alphas)
            momentum_means_method3.append(result["mean_kinks"])
            momentum_vars_method3.append(result["var_kinks"])

            # Process Qiskit model with these angles
            qiskit_progress = tqdm(total=1, desc="Qiskit simulation", leave=False)
            qiskit_results = process_qiskit_model(num_qubits, steps, noise_param,
                                                  'global', num_circuits, numshots, betas, alphas)
            qiskit_progress.update(1)
            qiskit_means.append(qiskit_results["mean_kinks"])
            qiskit_vars.append(qiskit_results["var_kinks"])

        if save_data:
            print(f"Saving data to {data_filename}")
            np.savez(data_filename,
                     momentum_results_list=momentum_results_list,
                     qiskit_means=qiskit_means,
                     qiskit_vars=qiskit_vars,
                     step_range=np.array(list(step_range)),
                     num_qubits=num_qubits,
                     noise_param=noise_param,
                     num_circuits=num_circuits,
                     numshots=numshots)

    # Create two subplots (1x2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot mean kinks - All methods
    ax1.plot(list(step_range), [i / num_qubits for i in momentum_means_method1], 'o-',
             label=f'Average of Final Results (noise={noise_param})')
    ax1.plot(list(step_range), [i / num_qubits for i in momentum_means_method2], 's-',
             label=f'Average of Density Matrices (noise={noise_param})')
    ax1.plot(list(step_range), [i / num_qubits for i in momentum_means_method3], '^-',
             label=f'Lindblad Evolution (noise={noise_param})')
    ax1.plot(list(step_range), [i / num_qubits for i in qiskit_means], 'x-',
             label=f'Qiskit (noise={noise_param})')
    ax1.set_xlabel('Number of Steps')
    ax1.set_ylabel('Mean Kinks/N')
    ax1.set_title(f'Mean Kinks vs Steps ({num_qubits} qubits)')
    ax1.legend()
    ax1.grid(True)

    # Plot variance - All methods
    ax2.plot(list(step_range), [i / num_qubits for i in momentum_vars_method1], 'o-',
             label=f'Average of Final Results (noise={noise_param})')
    ax2.plot(list(step_range), [i / num_qubits for i in momentum_vars_method2], 's-',
             label=f'Average of Density Matrices (noise={noise_param})')
    ax2.plot(list(step_range), [i / num_qubits for i in momentum_vars_method3], '^-',
             label=f'Lindblad Evolution (noise={noise_param})')
    ax2.plot(list(step_range), [i / num_qubits for i in qiskit_vars], 'x-',
             label=f'Qiskit (noise={noise_param})')
    ax2.set_xlabel('Number of Steps')
    ax2.set_ylabel('Variance/N')
    ax2.set_title(f'Variance/N vs Steps ({num_qubits} qubits)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # Construct filename
    plot_filename = f"comparison_N{num_qubits}_noise{noise_param}_circ{num_circuits}_shots{numshots}_{date_str}.png"

    # Save the plot
    handles, labels = ax1.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, -0.05))
    fig.subplots_adjust(bottom=0.25)  # Reserve enough space for the legend
    if plot_filename:
        plt.savefig(plot_filename, bbox_inches='tight', bbox_extra_artists=(legend,))
    # Save as SVG
    svg_filename = plot_filename.rsplit('.', 1)[0] + '.svg'
    plt.savefig(svg_filename, format='svg', bbox_inches='tight', bbox_extra_artists=(legend,))
    plt.show()

    return fig, plot_filename


# --- Modular Momentum Model Analysis Functions ---

def single_circuit_density_matrices(args):
    ks, steps, noise_param = args
    # Generate angles
    base_angles = (np.pi / 2) * np.arange(1, steps + 1) / (steps + 1)
    base_angles = base_angles[:, np.newaxis]
    noisy_base_angles = base_angles + noise_param * np.random.randn(steps, 1)
    betas = -np.sin(noisy_base_angles).flatten()
    alphas = -np.cos(noisy_base_angles).flatten()
    # Compute density matrices for all k (serial, no Pool)
    solutions = tfim_momentum_trotter(ks, steps, 0, betas, alphas, parallel=False)
    return solutions


def compute_and_save_density_matrices_parallel_circuits(ks, steps_list, num_qubits, num_circuits, noise_param,
                                                        filename):
    """
    Parallelized version: For each step in steps_list, run the momentum model num_circuits times in parallel,
    store the resulting density matrices for each run, and save to file.
    This parallelizes over circuits/runs for each step.
    """
    all_density_matrices = []
    for steps in steps_list:
        with Pool(processes=10) as pool:
            # Pass ks, steps, noise_param for each circuit
            step_density_matrices = list(
                pool.map(single_circuit_density_matrices, [(ks, steps, noise_param)] * num_circuits, chunksize=1))
        all_density_matrices.append(step_density_matrices)
    print(f"Saving density matrices to {filename}")
    np.savez(filename, all_density_matrices=all_density_matrices, steps_list=np.array(list(steps_list)),
             num_qubits=num_qubits, noise_param=noise_param, num_circuits=num_circuits)


def load_density_matrices(filename):
    print(f"Loading density matrices from {filename}...")
    data = np.load(filename, allow_pickle=True)
    return data['all_density_matrices'], data['steps_list']


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


def mean_var_from_density_matrices(all_density_matrices, ks, num_qubits, method="observable", parallel=False):
    """
    method: "observable" (average mean/var over runs) or "rho" (average rho, then mean/var)
    Returns: lists of mean_kinks/num_qubits, var_kinks/num_qubits for each step
    If parallel=True, process each step in parallel.
    """
    if all_density_matrices is None or len(all_density_matrices) == 0:
        return [], []
    if parallel:
        with multiprocessing.Pool(processes=10) as pool:
            func = partial(process_step, ks=ks, num_qubits=num_qubits, method=method)
            results = list(tqdm(pool.imap(func, all_density_matrices, chunksize=1), total=len(all_density_matrices),
                                desc="Processing density matrices (parallel)"))
        means, vars_ = zip(*results)
        return list(means), list(vars_)
    else:
        means, vars_ = [], []
        for step_density_matrices in tqdm(all_density_matrices, desc="Processing density matrices"):
            if step_density_matrices is None or len(step_density_matrices) == 0:
                means.append(0.0)
                vars_.append(0.0)
                continue
            mean, var = process_step(step_density_matrices, ks, num_qubits, method)
            means.append(mean)
            vars_.append(var)
        return means, vars_


def plot_momentum_means_vars(steps_list, means_obs, vars_obs, means_rho, vars_rho, num_qubits, filename=None):
    import matplotlib.pyplot as plt

    # Global Settings for Matplotlib
    plt.rcParams.update({
        'text.usetex': True,  # Enable LaTeX rendering for text
        'font.family': 'serif',  # Set font family
        'font.size': 18,  # General font size
        'lines.markersize': 10,  # Default marker size
        'legend.fontsize': 'small',  # Legend font size
        'legend.frameon': False,  # Remove frame around legend
        'figure.figsize': (6, 5),  # Default figure size
        'axes.grid': True,  # Enable grid for axes
        'grid.alpha': 0.1,  # Set grid transparency
        'grid.linestyle': '--',  # Set grid line style
        'grid.color': 'gray',  # Set grid line color
        'axes.grid.which': 'both',  # Enable both major and minor gridlines
        'axes.grid.axis': 'both',  # Apply grid to both x and y axes
        'axes.labelsize': 22,  # Font size for axis labels
        'xtick.labelsize': 13,  # Font size for x-axis tick labels
        'ytick.labelsize': 13  # Font size for y-axis tick labels
    })
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(steps_list, means_obs, 'o-', label='Avg Observable')
    plt.plot(steps_list, means_rho, 's-', label='Avg Rho')
    plt.xlabel(r'\textbf{Steps}')
    plt.ylabel(r'\textbf{Mean Kinks}')
    plt.legend()
    plt.title(r'\textbf{Mean Kinks}')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(steps_list, vars_obs, 'o-', label='Avg Observable')
    plt.plot(steps_list, vars_rho, 's-', label='Avg Rho')
    plt.xlabel(r'\textbf{Steps}')
    plt.ylabel(r'\textbf{Variance}')
    plt.legend()
    plt.title(r'\textbf{Variance}')
    plt.grid(True)

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        # Save as SVG
        svg_filename = filename + '.svg'
        plt.savefig(svg_filename, format='svg')
    plt.show()


def compute_observables_dataframe(all_density_matrices, ks, num_qubits, steps_list):
    """
    Compute mean and variance for both methods and return as a pandas DataFrame.
    Columns: steps, mean_exact, var_exact, mean_independent_modes, var_independent_modes
    """
    means_exact, vars_exact = mean_var_from_density_matrices(
        all_density_matrices, ks, num_qubits, method="observable"
    )
    means_indep, vars_indep = mean_var_from_density_matrices(
        all_density_matrices, ks, num_qubits, method="rho"
    )
    df = pd.DataFrame({
        'steps': list(steps_list),
        'mean_exact': means_exact,
        'var_exact': vars_exact,
        'mean_independent_modes': means_indep,
        'var_independent_modes': vars_indep
    })
    return df


def get_dated_plot_path(filename):
    """
    Returns a path for the given filename inside Plots/YYYYMMDD, creating the directory if needed.
    Usage: get_dated_plot_path('myplot.png')
    """
    date_str = datetime.date.today().strftime('%Y%m%d')
    plots_dir = os.path.join('Plots', date_str)
    os.makedirs(plots_dir, exist_ok=True)
    return os.path.join(plots_dir, filename)


def compute_circuits_for_step(args):
    ks, steps, num_to_compute, noise_param = args
    dms = []
    for _ in range(num_to_compute):
        base_angles = (np.pi / 2) * np.arange(1, steps + 1) / (steps + 1)
        base_angles = base_angles[:, np.newaxis]
        noisy_base_angles = base_angles + noise_param * np.random.randn(steps, 1)
        betas = -np.sin(noisy_base_angles).flatten()
        alphas = -np.cos(noisy_base_angles).flatten()
        solutions = tfim_momentum_trotter(ks, steps, 0, betas, alphas, parallel=False)
        dms.append(solutions)
    return steps, dms


def aggregate_density_matrices_and_save(ks, steps_list, num_qubits, num_circuits, noise_param, filename):
    """
    Aggregate and reuse density matrices for requested steps and num_circuits.
    - Loads existing data if present.
    - For each requested step:
        - If missing, compute all circuits (in parallel over steps).
        - If fewer circuits than requested, compute only the missing ones (in parallel over steps).
        - If more circuits than requested, randomly select the requested number.
        - If exactly the requested number, use as is.
    - Saves the updated data back to the file.
    - Returns (all_density_matrices, steps_list_used)
    """
    # Try to load existing data
    if os.path.exists(filename):
        print(f"Loading and aggregating density matrices from {filename}...")
        data = np.load(filename, allow_pickle=True)
        existing_steps = list(data['steps_list'])
        existing_density_matrices = list(data['all_density_matrices'])
        # Map step -> density_matrices for easy lookup
        step_to_dm = {step: list(dm) for step, dm in zip(existing_steps, existing_density_matrices)}
    else:
        print(f"No existing density matrix file found. Will compute all requested data.")
        step_to_dm = {}

    # Prepare jobs for missing circuits
    jobs = []  # Each job is (ks, steps, num_to_compute, noise_param)
    for steps in steps_list:
        if steps in step_to_dm:
            dms = step_to_dm[steps]
            if len(dms) < num_circuits:
                jobs.append((ks, steps, num_circuits - len(dms), noise_param))
        else:
            jobs.append((ks, steps, num_circuits, noise_param))

    # Run parallel computation for all missing circuits
    if jobs:
        print(f"Computing missing circuits in parallel for steps: {[j[1] for j in jobs]}")
        with Pool(processes=10) as pool:
            results = pool.map(compute_circuits_for_step, jobs, chunksize=1)
        # Insert results into step_to_dm
        for steps, new_dms in results:
            if steps in step_to_dm:
                step_to_dm[steps].extend(new_dms)
            else:
                step_to_dm[steps] = new_dms

    # Now aggregate the final data for all requested steps
    all_density_matrices = []
    steps_list_used = []
    for steps in steps_list:
        dms = step_to_dm[steps]
        if len(dms) > num_circuits:
            selected = random.sample(dms, num_circuits)
            all_density_matrices.append(selected)
        else:
            all_density_matrices.append(dms)
        steps_list_used.append(steps)

    # Save the aggregated data
    # Ensure steps and density matrices are sorted by the requested steps_list order
    print(f"Saving aggregated density matrices to {filename}")
    np.savez(filename, all_density_matrices=all_density_matrices, steps_list=np.array(list(steps_list)),
             num_qubits=num_qubits, noise_param=noise_param, num_circuits=num_circuits)
    return all_density_matrices, list(steps_list)


def aggregate_observables_and_save(all_density_matrices, ks, num_qubits, steps_list, num_circuits, csv_filename):
    """
    Aggregate and reuse observables for requested steps and num_circuits.
    - Loads existing observables if present.
    - For each requested step:
        - If present and has enough circuits, reuse or subsample.
        - If missing or has fewer circuits, compute missing and append.
        - Only reuse if both step and num_circuits match.
    - Saves the updated data back to the CSV.
    - Returns the aggregated DataFrame.
    """

    # Try to load existing data
    if os.path.exists(csv_filename):
        print(f"Loading observables from {csv_filename}")
        df_existing = pd.read_csv(csv_filename)
        # Add num_circuits column if not present
        if 'num_circuits' not in df_existing.columns:
            df_existing['num_circuits'] = num_circuits  # Assume previous runs used the same value
    else:
        df_existing = pd.DataFrame()

    # Compute new observables for all requested steps (with current num_circuits)
    means_exact, vars_exact = mean_var_from_density_matrices(all_density_matrices, ks, num_qubits, method="observable",
                                                             parallel=True)
    means_indep, vars_indep = mean_var_from_density_matrices(all_density_matrices, ks, num_qubits, method="rho",
                                                             parallel=True)
    df_new = pd.DataFrame({
        'steps': list(steps_list),
        'mean_exact': means_exact,
        'var_exact': vars_exact,
        'mean_independent_modes': means_indep,
        'var_independent_modes': vars_indep,
        'num_circuits': num_circuits
    })
    # Only reuse rows where both step and num_circuits match
    if not df_existing.empty:
        mask = (df_existing['num_circuits'] == num_circuits) & (df_existing['steps'].isin(steps_list))
        df_existing = df_existing[~mask]  # Remove rows that will be replaced
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
        df_final = df_final.sort_values('steps').reset_index(drop=True)
    else:
        df_final = df_new
    print(f"Saving observables to {csv_filename}")
    df_final.to_csv(csv_filename, index=False)
    return df_final


def plot_momentum_means_vars_from_df_single(df, num_qubits, num_circuits, noise_param, filename=None):
    pyplot_settings()
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    super_title = (r"\textbf{TFIM Momentum Model:} "
                   + rf"$N$ (qubits) = {num_qubits}, "
                   + rf"$M$ (circuits) = {num_circuits}, "
                   + rf"$\sigma$ (noise variance) = {noise_param}")
    plt.suptitle(super_title, y=0.94)
    # Mean plot
    ax = axs[0]
    ax.plot(df['steps'], df['mean_exact'], 'o-', label='Exact (observable averaging)')
    ax.plot(df['steps'], df['mean_independent_modes'], 's--', label='Independent Modes (rho averaging)')
    ax.set_xlabel(r'\textbf{Steps}')
    ax.set_ylabel(r'\textbf{Mean Kinks/N}')
    # Add extra space at the top for the legend
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.1)
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), borderaxespad=0.)
    ax.grid(True)

    # Variance plot
    ax = axs[1]
    ax.plot(df['steps'], df['var_exact'], 'o-', label='Exact (observable averaging)')
    ax.plot(df['steps'], df['var_independent_modes'], 's--', label='Independent Modes (rho averaging)')
    ax.set_xlabel(r'\textbf{Steps}')
    ax.set_ylabel(r'\textbf{Variance/N}')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.1)
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), borderaxespad=0.)
    ax.grid(True)

    plt.subplots_adjust(left=0.15, right=0.95, wspace=0.25)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_multiple_configs(configs, data_dir='data', plot_filename=None):
    """
    Plot mean and variance for multiple configurations on the same graphs.
    For each config, plot both 'exact' and 'independent modes' (rho) results.
    Place a single legend below both subplots.
    configs: list of dicts, each with keys:
        - num_qubits
        - steps_list
        - noise_param
        - num_circuits
        - label (optional, for legend)
    data_dir: directory where .csv files are stored
    plot_filename: if provided, save the plot to this file
    """

    # Set matplotlib global settings for consistency
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 18,
        'lines.markersize': 10,
        'legend.fontsize': 'small',
        'legend.frameon': False,
        'figure.figsize': (6, 5),
        'axes.grid': True,
        'grid.alpha': 0.1,
        'grid.linestyle': '--',
        'grid.color': 'gray',
        'axes.grid.which': 'both',
        'axes.grid.axis': 'both',
        'axes.labelsize': 22,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13
    })
    fig, axs = plt.subplots(2, 2, figsize=(14, 16))
    colors = plt.get_cmap('viridis', len(configs))
    for idx, config in enumerate(configs):
        num_qubits = config['num_qubits']
        steps_list = config['steps_list']
        noise_param = config['noise_param']
        num_circuits = config['num_circuits']
        label = config.get('label', f"N={num_qubits}, noise={noise_param}, M={num_circuits}")
        # Build CSV filename
        csv_filename = os.path.join(data_dir,
                                    f"momentum_observables_N{num_qubits}_noise{noise_param}_circ{num_circuits}.csv")
        if not os.path.exists(csv_filename):
            # Try to compute if not present
            print(f"CSV {csv_filename} not found, attempting to compute...")
            ks = k_f(num_qubits)
            dm_filename = os.path.join(data_dir,
                                       f"momentum_rhos_N{num_qubits}_noise{noise_param}_circ{num_circuits}.npz")
            if os.path.exists(dm_filename):
                all_density_matrices, steps_list_used = load_density_matrices(dm_filename)
            else:
                all_density_matrices, steps_list_used = aggregate_density_matrices_and_save(
                    ks, steps_list, num_qubits, num_circuits, noise_param, dm_filename
                )
            df = aggregate_observables_and_save(
                all_density_matrices, ks, num_qubits, steps_list, num_circuits, csv_filename
            )
        else:
            df = pd.read_csv(csv_filename)
        # Only plot the requested steps
        df = df[df['steps'].isin(steps_list)]
        color = colors(idx)
        # Top row: exact
        axs[0, 0].plot(df['steps'], df['mean_exact'], marker='o', color=color, linestyle='-', label=label)
        axs[0, 1].plot(df['steps'], df['var_exact'], marker='o', color=color, linestyle='-', label=label)
        # Bottom row: independent
        axs[1, 0].plot(df['steps'], df['mean_independent_modes'], marker='s', color=color, linestyle='--', label=label)
        axs[1, 1].plot(df['steps'], df['var_independent_modes'], marker='s', color=color, linestyle='--', label=label)

    # Axis labels and titles
    axs[0, 0].set_ylabel('Mean Kinks/N')
    axs[1, 0].set_ylabel('Mean Kinks/N')
    axs[1, 0].set_xlabel('Steps')
    axs[1, 1].set_xlabel('Steps')
    axs[0, 0].set_title('Exact: Mean')
    axs[0, 1].set_title('Exact: Variance')
    axs[1, 0].set_title('Independent: Mean')
    axs[1, 1].set_title('Independent: Variance')

    # One legend below all
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.tight_layout()
    # Create a figure-level legend at the bottom
    legend = fig.legend(handles, labels, loc='lower center', ncol=len(configs),
                        bbox_to_anchor=(0.5, 0.02))
    # Add padding at the bottom to make room for the legend
    fig.subplots_adjust(bottom=0.15)

    axs[0, 1].set_yscale('log')  # Exact: Variance
    axs[1, 1].set_yscale('log')  # Independent: Variance

    if plot_filename:
        svg_filename = plot_filename + '.svg'
        plt.savefig(svg_filename, format='svg', bbox_inches='tight', bbox_extra_artists=(legend,))
    plt.show()


def run_single_plot(num_qubits=10, steps_list=range(0, 31, 5), num_circuits=20, noise_param=0.5, data_dir='data',
                    plot_filename=None):
    """
    Run and plot a single configuration of the momentum model.
    Parameters:
        num_qubits (int): Number of qubits
        steps_list (iterable): List or range of steps
        num_circuits (int): Number of circuits
        noise_param (float): Noise parameter
        data_dir (str): Directory for data files
        plot_filename (str or None): If provided, save the plot to this file
    """
    ks = k_f(num_qubits)
    os.makedirs(data_dir, exist_ok=True)
    data_filename = os.path.join(data_dir, f"momentum_rhos_N{num_qubits}_noise{noise_param}_circ{num_circuits}.npz")
    csv_filename = os.path.join(data_dir,
                                f"momentum_observables_N{num_qubits}_noise{noise_param}_circ{num_circuits}.csv")
    if plot_filename is None:
        plot_filename = get_dated_plot_path(
            f"momentum_comparison_N{num_qubits}_noise{noise_param}_circ{num_circuits}.svg")
    all_density_matrices, steps_list_used = aggregate_density_matrices_and_save(
        ks, steps_list, num_qubits, num_circuits, noise_param, data_filename
    )
    df = aggregate_observables_and_save(
        all_density_matrices, ks, num_qubits, steps_list, num_circuits, csv_filename
    )
    plot_momentum_means_vars_from_df_single(df, num_qubits, num_circuits, noise_param, plot_filename)


def plot_var_ratio(num_qubits, steps_list, num_circuits, noise_param, data_dir='data', plot_filename=None):
    """
    Run and plot a single configuration of the momentum model.
    Parameters:
        num_qubits (int): Number of qubits
        steps_list (iterable): List or range of steps
        num_circuits (int): Number of circuits
        noise_param (float): Noise parameter
        data_dir (str): Directory for data files
        plot_filename (str or None): If provided, save the plot to this file
    """
    ks = k_f(num_qubits)
    os.makedirs(data_dir, exist_ok=True)
    data_filename = os.path.join(data_dir, f"momentum_rhos_N{num_qubits}_noise{noise_param}_circ{num_circuits}.npz")
    csv_filename = os.path.join(data_dir,
                                f"momentum_observables_N{num_qubits}_noise{noise_param}_circ{num_circuits}.csv")
    if plot_filename is None:
        plot_filename = get_dated_plot_path(
            f"momentum_comparison_N{num_qubits}_noise{noise_param}_circ{num_circuits}.svg")
    all_density_matrices, steps_list_used = aggregate_density_matrices_and_save(
        ks, steps_list, num_qubits, num_circuits, noise_param, data_filename
    )
    df = aggregate_observables_and_save(
        all_density_matrices, ks, num_qubits, steps_list, num_circuits, csv_filename
    )
    pyplot_settings()
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    super_title = (r"\textbf{TFIM:}"
                   + rf"qubits={num_qubits}, "
                   + rf"circuits={num_circuits}, "
                   + rf"noise var={noise_param}")
    plt.suptitle(super_title, y=0.94)
    ye = df['var_exact'] / (df['mean_exact'])# * (1 - df['mean_exact']))
    ax.plot(df['steps'], ye, 'o-', label='Exact (observable averaging)')
    yi = df['var_independent_modes'] / (df['mean_independent_modes'] )#* (1 - df['mean_independent_modes']))
    ax.plot(df['steps'], yi, 's--', label='Independent Modes (rho averaging)')
    ax.set_xlabel(r'\textbf{Steps}')
    ax.set_ylabel(r'\textbf{Variance/Mean Kinks}')#(1 - Mean Kinks)}')
    # Add extra space at the top for the legend
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.1)
    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), borderaxespad=0.)
    ax.grid(True)

    plt.subplots_adjust(left=0.15, right=0.95, wspace=0.25)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    # if plot_filename:
    #     plt.savefig(plot_filename, bbox_inches='tight')
    plt.show()


def compute_density_matrices(params: Tuple[int, float, int, int]) -> Tuple[
    int, float, int, int, List[Tuple[int, int, npt.NDArray[np.complex128]]]]:
    """
    Compute num_circuits sets of 2x2 density matrices (one per k) for a TFIM simulation with noise.

    Args:
        params: Tuple of (num_qubits, noise_param, step, num_circuits).
            - num_qubits: Number of qubits in the TFIM system.
            - noise_param: Noise strength for angle perturbations.
            - step: Evolution step (integer).
            - num_circuits: Number of circuits to compute.

    Returns:
        Tuple of (num_qubits, noise_param, step, num_circuits, results), where results is a
        list of (circuit_index, k_index, dm) pairs, each dm being a 2x2 complex density matrix.
    """
    num_qubits, noise_param, step, num_circuits = params

    # Compute wavevectors (half the number of qubits)
    ks = k_f(num_qubits)  # Shape: (num_qubits//2,)
    num_ks = num_qubits // 2

    # Validate ks
    if len(ks) != num_ks:
        raise ValueError(f"Expected {num_ks} wavevectors, got {len(ks)}")

    # Store results as list of (circuit_index, k_index, dm)
    results = []

    # Handle step=0 (initial state)
    if step == 0:
        # Return maximally mixed state for each k and circuit
        for circuit_index in range(num_circuits):
            for k_index in range(num_ks):
                dm = np.eye(2) / 2  # Maximally mixed state
                results.append((circuit_index, k_index, dm))
        print(f"Step=0: Generated {len(results)} matrices for N={num_qubits}, sigma={noise_param}")
        return num_qubits, noise_param, step, num_circuits, results

    # Generate angles and compute density matrices
    for circuit_index in range(num_circuits):
        # Base angles for Trotter steps
        base_angles = (np.pi / 2) * np.arange(1, step + 1) / (step + 1)
        base_angles = base_angles[:, np.newaxis]

        # Add noise to angles
        noisy_base_angles = base_angles + noise_param * np.random.randn(step, 1)
        betas = -np.sin(noisy_base_angles).flatten()
        alphas = -np.cos(noisy_base_angles).flatten()

        dm_list = tfim_momentum_trotter(ks, step, 0, betas, alphas, parallel=False)

        for k_index, dm in enumerate(dm_list):
            results.append((circuit_index, k_index, dm))

    print(f"Generated {len(results)} matrices for N={num_qubits}, sigma={noise_param}, step={step}")
    return num_qubits, noise_param, step, num_circuits, results


def compute_observables(
        density_matrices: List[npt.NDArray[np.complex128]],
        ks: npt.NDArray[np.float64],
        num_qubits: int
) -> Dict[str, float]:
    """
    Compute mean and variance of kink observables for TFIM using two methods.

    Args:
        density_matrices: List of arrays, each of shape (num_qubits//2, 2, 2) with complex density matrices.
        ks: Array of wavevectors (length num_qubits//2).
        num_qubits: Number of qubits in the TFIM system.

    Returns:
        Dictionary with keys 'mean_exact', 'variance_exact', 'mean_independent',
        'variance_independent', containing normalized kink observables.
    """
    num_ks = num_qubits // 2

    # Method 1: Exact (process each circuit individually)
    means_runs, vars_runs = [], []
    for solutions in density_matrices:
        if len(solutions) != num_ks:
            raise ValueError(f"Expected {num_ks} matrices per circuit, got {len(solutions)}")
        # Calculate pks for each k using corresponding matrix
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
    mean_exact = np.mean(means_runs)
    variance_exact = np.mean(vars_runs)

    # Method 2: Independent (average density matrices first)
    avg_solutions = np.mean(density_matrices, axis=0)  # Shape: (num_qubits//2, 2, 2)
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
    mean_independent = mean_kinks / num_qubits
    variance_independent = var_kinks / num_qubits

    return {
        "mean_exact": mean_exact,
        "variance_exact": variance_exact,
        "mean_independent": mean_independent,
        "variance_independent": variance_independent
    }


def collect_experiment_data(
        num_qubits_list: List[int],
        noise_param_list: List[float],
        steps_list: List[int],
        num_circuits_list: List[int],
        data_dir: str = "data",
        db_path: str = "experiment.db"
) -> pd.DataFrame:
    """Collect experiment data, reusing or computing density matrices and observables."""
    # Initialize data directory
    os.makedirs(data_dir, exist_ok=True)

    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS density_matrices (
            id INTEGER PRIMARY KEY,
            num_qubits INTEGER NOT NULL,
            noise_param REAL NOT NULL,
            step INTEGER NOT NULL,
            circuit_index INTEGER NOT NULL,
            k_index INTEGER NOT NULL,
            file_path TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(num_qubits, noise_param, step, circuit_index, k_index)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS observables (
            id INTEGER PRIMARY KEY,
            num_qubits INTEGER NOT NULL,
            noise_param REAL NOT NULL,
            step INTEGER NOT NULL,
            num_circuits INTEGER NOT NULL,
            mean_exact REAL NOT NULL,
            variance_exact REAL NOT NULL,
            mean_independent REAL NOT NULL,
            variance_independent REAL NOT NULL,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(num_qubits, noise_param, step, num_circuits),
            FOREIGN KEY (num_qubits, noise_param, step)
            REFERENCES density_matrices (num_qubits, noise_param, step)
        )
    """)
    conn.commit()

    # Check for cached observables
    all_data = []
    missing_combinations = []
    for N, sigma, step, M in itertools.product(num_qubits_list, noise_param_list, steps_list, num_circuits_list):
        cursor.execute("""
            SELECT mean_exact, variance_exact, mean_independent, variance_independent
            FROM observables
            WHERE num_qubits=? AND noise_param=? AND step=? AND num_circuits=?
        """, (N, sigma, step, M))
        result = cursor.fetchone()
        if result:
            all_data.append({
                "num_qubits": N,
                "noise_param": sigma,
                "step": step,
                "num_circuits": M,
                "mean_exact": result[0],
                "variance_exact": result[1],
                "mean_independent": result[2],
                "variance_independent": result[3]
            })
        else:
            missing_combinations.append((N, sigma, step, M))

    # Determine density matrices needed
    groups = {}
    for N, sigma, step, M in missing_combinations:
        key = (N, sigma, step)
        groups[key] = max(groups.get(key, 0), M)

    # Check existing density matrices
    tasks = []
    for (N, sigma, step), max_M in groups.items():
        cursor.execute("""
            SELECT circuit_index, k_index
            FROM density_matrices
            WHERE num_qubits=? AND noise_param=? AND step=?
        """, (N, sigma, step))
        existing = {(row[0], row[1]) for row in cursor.fetchall()}
        expected = {(c, k) for c in range(max_M) for k in range(N // 2)}
        missing = expected - existing
        if missing:
            print(f"Missing {len(missing)} matrices for N={N}, sigma={sigma}, step={step}")
            tasks.append((N, sigma, step, max_M))

    # Compute missing density matrices in parallel
    if tasks:
        with Pool(processes=10) as pool:
            results = pool.map(compute_density_matrices, tasks)
        for N, sigma, step, num_circuits, result_list in results:
            sigma_str = f"{sigma:.4f}".replace(".", "_")
            for circuit_index, k_index, dm in result_list:
                file_name = f"rhos_N{N}_sigma{sigma_str}_step{step}_circuit{circuit_index}_k{k_index}.npy"
                file_path = os.path.join(data_dir, file_name)
                np.save(file_path, dm)
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO density_matrices
                        (num_qubits, noise_param, step, circuit_index, k_index, file_path)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (N, sigma, step, circuit_index, k_index, file_path))
                    print(f"Inserted N={N}, sigma={sigma}, step={step}, circuit={circuit_index}, k={k_index}")
                except sqlite3.Error as e:
                    print(
                        f"Database error for N={N}, sigma={sigma}, step={step}, circuit={circuit_index}, k={k_index}: {e}")
        conn.commit()

    # Compute missing observables
    for N, sigma, step, M in missing_combinations:
        # Load density matrices
        dms = []
        for circuit_index in range(M):
            circuit_dms = []
            for k_index in range(N // 2):
                cursor.execute("""
                    SELECT file_path FROM density_matrices
                    WHERE num_qubits=? AND noise_param=? AND step=? AND circuit_index=? AND k_index=?
                """, (N, sigma, step, circuit_index, k_index))
                result = cursor.fetchone()
                if result:
                    dm = np.load(result[0], allow_pickle=True)
                    circuit_dms.append(dm)
                else:
                    print(f"Missing matrix: N={N}, sigma={sigma}, step={step}, circuit={circuit_index}, k={k_index}")
                    raise ValueError(
                        f"Missing density matrix for N={N}, sigma={sigma}, step={step}, circuit={circuit_index}, k={k_index}")
            dms.append(circuit_dms)

        # Compute observables
        ks = k_f(N)  # Shape: (N//2,)
        observables = compute_observables(dms, ks, N)
        cursor.execute("""
            INSERT INTO observables
            (num_qubits, noise_param, step, num_circuits, mean_exact, variance_exact, mean_independent, variance_independent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (N, sigma, step, M, observables["mean_exact"], observables["variance_exact"],
              observables["mean_independent"], observables["variance_independent"]))
        all_data.append({
            "num_qubits": N,
            "noise_param": sigma,
            "step": step,
            "num_circuits": M,
            "mean_exact": observables["mean_exact"],
            "variance_exact": observables["variance_exact"],
            "mean_independent": observables["mean_independent"],
            "variance_independent": observables["variance_independent"]
        })
    conn.commit()
    conn.close()

    return pd.DataFrame(all_data)


# Main Execution
# -------------
if __name__ == "__main__":
    # Common parameters
    # noise_param = 1
    # steps_list = range(0, 101, 5)
    # num_circuits = 100
    # num_qubits = [10, 100, 1000]
    # configs = []
    # for num_qubit in num_qubits:
    #     configs.append({
    #         'num_qubits': num_qubit,
    #         'steps_list': steps_list,
    #         'noise_param': noise_param,
    #         'num_circuits': num_circuits,
    #         'label': f'N={num_qubit}, noise={noise_param}, M={num_circuits}'
    #     })

    # plot_multiple_configs(configs, data_dir='data',
    #                       plot_filename='Plots/20250510/multi_config_plot')
    # plot_var_ratio(num_qubits=200, steps_list=range(0, 201, 40), num_circuits=50, noise_param=0.05, data_dir='data',
    #                plot_filename=None)
    # data = collect_experiment_data([8], [0.1], range(0, 11, 1), [10])
    # print(data)
    run_single_plot(num_qubits=100, steps_list=range(0, 1001, 50), num_circuits=10, noise_param=0.02)
