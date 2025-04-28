import datetime
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix
from qiskit_aer import AerSimulator
from scipy.linalg import expm
from tqdm import tqdm

from functions import (calc_kink_probabilities, calc_kinks_mean,
                       calc_kinks_probability, epsilon, k_f,
                       noisy_lz_time_evolution)

# """
# I would like to generate data for several models:
# The models are-
#     1. the numeric independent ks model without noise.
#     2. the numeric independent ks model with noise.
#     3. the qiskit dephasing model.
#     4. the qiskit coherent model.
#     5. the qiskit global noise model.
#
#     for each model I would like to generate data for the following cases:
#         1. different number of qubits
#         2. different number of steps/time
#         3. different noise parameters
#
#     the data for each case:
#         1. for the numeric models:
#             1. the density matrix at the end.
#             2. the kinks number probability distribution at the end.
#             3. the average number of kinks.
#             4. the variance of the kinks number.
#             5. the purity of the density matrix.
#
#         2. for the qiskit models:
#             1. the density matrix at the end for each shot.
#             2. the average density matrix for each case.
#             3. the number of kinks for each shot.
#             4. the average number of kinks for each case.
#             5. the variance of the kinks number for each case.
#             6. the purity of the averaged density matrix for each case.
#
# I would like to write a code that generates this data for each case and saves it to files.
# I already have some of the functions that calculate the data for each case bur I need to modify for that purpose.
# """



#
# # Set Qiskit logging level to WARNING to suppress INFO and DEBUG messages
# qiskit_logger = logging.getLogger('qiskit')
# qiskit_logger.setLevel(logging.WARNING)
#
# # Your existing logging configuration
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# logging.getLogger().addHandler(console_handler)
#
# FILE_PATH = "data/for_plot_070225.csv"
#
# def tfim_momentum_trotter_single_k(k, tau, w=0.0, dt=0.1):
#     """
#     Simulates Trotterized evolution of the TFIM in momentum space for a single k,
#     mimicking a Qiskit circuit with fixed time step dt and τ steps.
#
#     Parameters:
#     - k (float): Momentum mode (0 to π).
#     - tau (int): Number of Trotter steps (total time = τ * dt).
#     - w (float): Noise parameter (unused, kept for compatibility).
#     - dt (float): Time step per Trotter step (default 0.1).
#
#     Returns:
#     - rho (2x2 complex array): Final density matrix.
#     """
#     # Initial state: 'down' eigenstate |↓⟩ = [0, 1]^T
#     rho = np.array([[0, 0], [0, 1]], dtype=complex)
#
#     # Perform τ Trotter steps
#     for n in range(1, tau + 1):
#         # Adiabatic parameter schedule
#         theta_n = n * np.pi / (2 * (tau + 1))
#         h = np.cos(theta_n)  # Transverse field term
#         J = np.sin(theta_n)  # Ising coupling (g(t) = J)
#
#         # Hamiltonian terms: H(k) = 2 [h - J cos(k)] σ_z + 2 J sin(k) σ_x
#         h_eff = 2 * (h - J * np.cos(k))  # Coefficient of σ_z
#         delta_k = 2 * J * np.sin(k)  # Coefficient of σ_x
#
#         # Trotter step: Split into σ_z and σ_x terms
#         # U = exp(-i H_z dt) exp(-i H_x dt)
#
#         # Evolve with H_z = h_eff σ_z
#         theta_z = -h_eff * dt  # Rotation angle for σ_z
#         U_z = np.array([
#             [np.exp(-1j * theta_z / 2), 0],
#             [0, np.exp(1j * theta_z / 2)]
#         ])  # exp(-i θ_z σ_z / 2), mimicking R_z gate
#
#         # Evolve with H_x = delta_k σ_x
#         theta_x = -delta_k * dt  # Rotation angle for σ_x
#         U_x = np.cos(theta_x / 2) * np.eye(2) - 1j * np.sin(theta_x / 2) * np.array([[0, 1], [1, 0]])
#         # exp(-i θ_x σ_x / 2), mimicking R_x gate
#
#         # Apply evolution: ρ → U_x U_z ρ U_z† U_x†
#         rho = U_x @ (U_z @ rho @ U_z.conj().T) @ U_x.conj().T
#
#     return rho
#
#
# def tfim_momentum_trotter(ks, tau, w):
#     """
#     Parallelize Trotterized density matrix evolution for all k values, noiseless case with dt = 1.
#     """
#     # Calculate theta_n_values based on steps (tau)
#     with Pool() as pool:
#         results = pool.starmap(tfim_momentum_trotter_single_k, [(k, tau, w) for k in ks])
#     return results
#
#
# def process_tfim_momentum_trotter(ks, depth, num_qubits, noise):
#     """Process the TFIM in momentum space with Trotter discretization, no noise, and dt = 1."""
#     # Evolve each k's density matrix using Trotter steps with dt = 1
#     solutions = tfim_momentum_trotter(ks, depth, noise)  # tau = steps for dt = 1
#
#     # Construct full density matrix by tensor product of final density matrices
#     tensor_product = solutions[0]
#     for rho in solutions[1:]:
#         tensor_product = np.kron(tensor_product, rho)
#
#     density_matrix = tensor_product
#
#     # Calculate purity
#     rho2 = density_matrix @ density_matrix
#     purity = np.trace(rho2).real
#
#     # Calculate kink probabilities in momentum space
#     # Assume p_k is related to magnetization or correlation in momentum space
#     pks = np.array([np.abs(np.dot(np.array([np.sin(k / 2), np.cos(k / 2)]),
#                                   np.dot(solution,
#                                   np.dot(solution,
#                                          np.array([[np.sin(k / 2)], [np.cos(k / 2)]]))))[0]
#                     for k, solution in zip(ks, solutions)])
#     pks = np.where(pks < epsilon / 100, 0, pks)
#
#     # Kink values for periodic chain (0, 2, ..., L)
#     kinks_vals = np.arange(0, num_qubits + 1, 2)
#
#     # Use externally defined calc_kink_probabilities (assumed to be available)
#     distribution = calc_kink_probabilities(pks, kinks_vals)
#     kinks_distribution = {k: v for k, v in zip(kinks_vals, distribution)}
#
#     # Calculate mean and variance of kinks
#     mean_kinks = np.sum(k*v for k, v in kinks_distribution.items())
#     second_moment = np.sum((k**2)*v for k, v in kinks_distribution.items())
#     var_kinks = second_moment - mean_kinks ** 2
#
#     return {
#         "density_matrix": density_matrix,
#         "kinks_distribution": kinks_distribution,
#         "mean_kinks": mean_kinks,
#         "var_kinks": var_kinks,
#         "purity": purity
#     }
#
#
# def generate_single_circuit(params):
#     qubits, steps, circuit_idx, betas, alphas, noisy_betas, noise_method = params
#
#     circuit = QuantumCircuit(qubits, qubits)
#     circuit.h(range(qubits))
#
#     for step in range(steps):
#         beta = betas[step, circuit_idx]
#         alpha = alphas[step, circuit_idx]
#
#         if noise_method == 'dephasing':
#             circuit.rz(noisy_betas[step, circuit_idx], range(qubits))
#         else:
#             circuit.rz(beta, range(qubits))
#
#         for i in range(qubits):
#             j = (i + 1) % qubits
#             circuit.cp(-2 * beta, i, j)
#             circuit.rx(alpha, i)
#
#     dm = DensityMatrix.from_instruction(circuit)
#     circuit.measure(range(qubits), range(qubits))
#
#     return circuit, dm.data
#
#
# def generate_qiskit_circuits(qubits, steps, num_circuits_per_step, noise_std=0.0, noise_method='global'):
#     base_angles = (np.pi / 2) * np.arange(1, steps + 1) / (steps + 1)
#     base_angles = base_angles[:, np.newaxis]
#     noisy_base_angles = base_angles + noise_std * np.random.randn(steps, num_circuits_per_step)
#
#     if noise_method == 'global':
#         betas = -np.sin(noisy_base_angles)
#         alphas = -np.cos(noisy_base_angles)
#     else:
#         base_angles = np.tile(base_angles, (1, num_circuits_per_step))
#         betas = -np.sin(base_angles)
#         alphas = -np.cos(base_angles)
#         noisy_betas = -np.sin(noisy_base_angles)
#
#     params = [(qubits, steps, i, betas, alphas, noisy_betas if noise_method == 'dephasing' else None, noise_method)
#               for i in range(num_circuits_per_step)]
#
#     with Pool() as pool:
#         results = pool.map(generate_single_circuit, params)
#
#     circuits, density_matrices = zip(*results)
#     return list(circuits), list(density_matrices)
#
#
# def save_data(file_path, data, mode='a', header=False):
#     """Save data to a CSV file with a specific column order."""
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#
#     # Define the desired column order
#     column_order = [
#         "model", "qubits", "depth", "noise_param",
#         "density_matrix", "kinks_distribution",
#         "mean_kinks", "var_kinks", "purity"
#     ]
#
#     # Reorder the DataFrame columns
#     data = data[column_order]
#
#     data.to_csv(file_path, mode=mode, header=header, index=False)
#
#
# def data_exists(existing_data, model, num_qubits, depth, noise_param):
#     """Check if data with the given parameters already exists."""
#     if existing_data.empty:
#         return False
#
#     # Ensure the data types match and round floating-point numbers
#     model = str(model)
#     num_qubits = int(num_qubits)
#     depth = round(float(depth), 6)
#     noise_param = round(float(noise_param), 6)
#
#     return ((existing_data['model'] == model) &
#             (existing_data['qubits'] == num_qubits) &
#             (existing_data['depth'].round(6) == depth) &
#             (existing_data['noise_param'].round(6) == noise_param)).any()
#
#
# from scipy.integrate import solve_ivp
#
# ### Pauli matrices
# sigX = np.array([[0, 1],
#                  [1, 0]], dtype=complex)
# sigZ = np.array([[1, 0],
#                  [0, -1]], dtype=complex)
#
#
# def commutator(A, B):
#     return A @ B - B @ A
#
#
# def g0(t):
#     """
#     Dimensionless ramp function: g(t) = t, where t in [0,1].
#     """
#     return t
#
#
# def H0(t, k):
#     """
#     Example Hamiltonian that previously used g = t/tau. Now we just use g(t)= t.
#     Adjust the form to match your original logic:
#       h_z = 2 * (1 - g - g*cos(k))
#       h_x = 2 * g * sin(k)
#     """
#     g_f = g0(t)
#     h_z = 2.0 * (1 - g_f - g_f * np.cos(k))  # same formula, with g_f = t
#     h_x = 2.0 * g_f * np.sin(k)
#     return h_z * sigZ + h_x * sigX
#
#
# def V_func(k):
#     """
#     Same V you used before.
#     If you had dimensionful parameters, keep them as is, or rename if desired.
#     """
#     v_z = -2.0 * (np.cos(k) + 1)
#     v_x = 2.0 * np.sin(k)
#     return v_z * sigZ + v_x * sigX
#
#
# def Dissipator(rho, V, w):
#     """
#     Example dissipator. If w is dimensionful (like a rate),
#     that sets how 'fast' decoherence or dephasing happens in real time.
#     """
#     # Lindblad form w^2 * [ V , [ V , rho ] ]
#     term1 = V @ rho @ V
#     term2 = -0.5 * (V @ V @ rho)
#     term3 = -0.5 * (rho @ V @ V)
#     return w ** 2 * (term1 + term2 + term3)
#
#
# def single_step_ode(t, rho_vec, H, w, V):
#     """
#     ODE for a single short interval where H and V are 'frozen'.
#     H = H0(t_s, k), V = V_func(k), w is your noise/dissipation param.
#     """
#     rho_mat = rho_vec.reshape(2, 2)
#     # Hamiltonian part: -i [H, rho]
#     comm = -1j * commutator(H, rho_mat)
#     # Dissipator
#     dissip = Dissipator(rho_mat, V, w)
#     return (comm + dissip).ravel()
#
#
# def discrete_lz_time_evolution_single_k(k, w, n_steps):
#     """
#     Evolve a single momentum mode k from t=0 to t=1 in n_steps Trotter slices.
#     The Hamiltonian/dissipator is re-evaluated at each slice's start (t_s).
#     """
#     # initial state
#     rho0 = np.array([[0. + 0j, 0. + 0j],
#                      [0. + 0j, 1. + 0j]])  # you can change as needed
#
#     # We'll break [0,1] into n_steps intervals
#     t_points = np.linspace(0, 1, n_steps + 1)
#     rho_current = rho0.copy()
#
#     for i in range(n_steps):
#         t_start = t_points[i]
#         t_end = t_points[i + 1]
#
#         # Freeze H and V at t_start
#         H = H0(t_start, k)
#         V = V_func(k)
#
#         # Integrate from t_start to t_end with those fixed operators
#         sol = solve_ivp(
#             fun=lambda t, y: single_step_ode(t, y, H, w, V),
#             t_span=(t_start, t_end),
#             y0=rho_current.ravel(),
#             method='DOP853',  # or your preferred solver
#             rtol=1e-8, atol=1e-8
#         )
#         rho_current = sol.y[:, -1].reshape(2, 2)
#
#     return rho_current
#
#
# def discrete_lz_time_evolution(ks, w, n_steps):
#     """
#     Parallelize over all k modes.
#     """
#     with Pool() as pool:
#         results = pool.starmap(
#             discrete_lz_time_evolution_single_k,
#             [(k, w, n_steps) for k in ks]
#         )
#     return results
#
#
# def discrete_process_numeric_model(ks, n_steps, w, num_qubits):
#     """
#     1) Solve for each momentum k,
#     2) Tensor them together,
#     3) Compute purity, kinks distribution, etc.
#     """
#     ks_solutions = discrete_lz_time_evolution(ks, w, n_steps)
#
#     # Build the total density matrix as a tensor product
#     rho_total = ks_solutions[0]
#     for rho_k in ks_solutions[1:]:
#         rho_total = np.kron(rho_total, rho_k)
#
#     density_matrix = rho_total
#     rho2 = density_matrix @ density_matrix
#     purity = rho2.trace().real
#
#     # Example "kinks" calculation, analogous to your code.
#     # Replace with your actual routine as needed.
#     pks = []
#     for k, rho_k in zip(ks, ks_solutions):
#         # example state vector = [ sin(k/2), cos(k/2) ]
#         vec = np.array([np.sin(k / 2), np.cos(k / 2)], dtype=complex)
#         amp = vec.conj().T @ (rho_k @ vec)
#         pks.append(np.abs(amp))
#     pks = np.array(pks)
#     pks = np.where(pks < epsilon / 100, 0, pks)
#
#     # Suppose you define 'calc_kink_probabilities' somewhere:
#     kinks_vals = np.arange(0, num_qubits + 1, 2)
#     kinks_distribution = calc_kink_probabilities(pks, kinks_vals)
#     mean_kinks = np.sum(kinks_distribution * kinks_vals)
#     second_moment = np.sum(kinks_distribution * kinks_vals ** 2)
#     var_kinks = second_moment - mean_kinks ** 2
#
#     return {
#         "density_matrix": density_matrix,
#         "kinks_distribution": kinks_distribution,
#         "mean_kinks": mean_kinks,
#         "var_kinks": var_kinks,
#         "purity": purity
#     }
#
#
# def get_matching_rows(df, model, num_qubits, depth, noise_param):
#     """Get all rows from the DataFrame that match the given parameters."""
#     model = str(model)
#     num_qubits = int(num_qubits)
#     depth = round(float(depth), 6)
#     noise_param = round(float(noise_param), 6)
#
#     matching_rows = df[
#         (df['model'] == model) &
#         (df['qubits'] == num_qubits) &
#         (df['depth'].round(6) == depth) &
#         (df['noise_param'].round(6) == noise_param)
#         ]
#     return matching_rows
#
#
# def generate_data():
#     models = [
#         "independent_ks_numeric"
#         ,
#         "qiskit_global_noise",
#         "qiskit_dephasing"
#     ]
#     numshots = 100000
#     num_circuits = 500
#     num_qubits_list = [4]
#     depth_list_numeric = [round(i, 6) for i in range(41)]
#     depth_list_qiskit = [round(i, 6) for i in range(41)]
#     noise_params_qiskit = [round(i, 6) for i in np.linspace(0, 1.5, 10)]
#     noise_params_numeric = [round(i, 6) for i in np.linspace(0, 1.5, 10)]
#
#     total_iterations = (len(models) * len(num_qubits_list) * max(len(noise_params_qiskit), len(noise_params_numeric))
#                         * max(len(depth_list_numeric), len(depth_list_qiskit)))
#     progress_bar = tqdm(total=total_iterations, desc="Generating Data")
#
#     file_path = "data/for_plot_070225.csv"
#     header_written = os.path.exists(file_path)
#
#     if header_written:
#         existing_data = pd.read_csv(file_path)
#     else:
#         existing_data = pd.DataFrame()
#
#     for num_qubits in num_qubits_list:
#         for model in models:
#             ks = k_f(num_qubits)
#             for noise_param in (noise_params_numeric if 'numeric' in model else noise_params_qiskit):
#                 for depth in (depth_list_numeric if 'numeric' in model else depth_list_qiskit):
#                     if data_exists(existing_data, model, num_qubits, depth, noise_param):
#                         logging.info(
#                             f"Skipping: Model={model}, Qubits={num_qubits}, Depth={depth}, Noise={noise_param} (already exists)")
#                         progress_bar.update(1)
#                         continue
#
#                     logging.info(f"Processing: Model={model}, Qubits={num_qubits}, Depth={depth}, Noise={noise_param}")
#                     data_list = []
#                     if "numeric" in model:
#                         data = discrete_process_numeric_model(ks, depth, noise_param, num_qubits)
#                         data.update({
#                             "model": model,
#                             "qubits": num_qubits,
#                             "depth": depth,
#                             "noise_param": noise_param
#                         })
#                         data_list.append(data)
#
#                     elif "qiskit" in model:
#                         noise_type = "dephasing" if "dephasing" in model else "global"
#                         circuits, density_matrices = generate_qiskit_circuits(num_qubits, depth, num_circuits,
#                                                                               noise_param, noise_type)
#                         avg_density_matrix = sum(density_matrices) / num_circuits
#                         rho_squared = avg_density_matrix @ avg_density_matrix
#                         purity = np.trace(rho_squared).real
#                         simulator = AerSimulator()
#                         transpiled_circuits = transpile(circuits, simulator, num_processes=-1)
#                         results = simulator.run(transpiled_circuits, shots=numshots).result().get_counts()
#                         counts = {}
#                         for result in results:
#                             for key, value in result.items():
#                                 if key in counts:
#                                     counts[key] += value
#                                 else:
#                                     counts[key] = value
#                         probs = calc_kinks_probability(counts)
#                         mean = calc_kinks_mean(probs)
#                         variances = sum((k - mean) ** 2 * v for k, v in probs.items())
#                         data = {
#                             "density_matrix": avg_density_matrix,
#                             "kinks_distribution": probs,
#                             "mean_kinks": mean,
#                             "var_kinks": variances,
#                             "purity": purity,
#                             "model": model,
#                             "qubits": num_qubits,
#                             "depth": depth,
#                             "noise_param": noise_param
#                         }
#                         data_list.append(data)
#
#                     df = pd.DataFrame(data_list)
#                     save_data(file_path, df, mode='a', header=not header_written)
#                     header_written = True
#                     progress_bar.update(1)
#
#     progress_bar.close()




def tfim_momentum_trotter_single_k(k, tau, beta, alpha):
    """
    Simulates Trotterized evolution of the TFIM in momentum space for a single momentum k.
    
    The evolution is performed in Nambu space, with the initial state chosen as |↓⟩
    (represented as [0, 1]^T, i.e. the ground state for g=0).

    Parameters:
      k     : momentum (in radians)
      tau   : total number of Trotter steps
      beta  : array of angles for ZZ interaction terms
      alpha : array of angles for X field terms

    Returns:
      rho : the final density matrix for momentum k.
    """
    # Initial state |↓⟩ is represented as [0, 1]^T; hence its density matrix:
    rho = np.array([[0, 0], [0, 1]], dtype=complex)

    # Loop over each Trotter step
    for n in range(tau):
        # Get the angles for this step
        h = alpha[n]  # transverse field part
        J = beta[n]   # interaction strength

        # Hamiltonian terms
        H_fm = -1 * J * np.cos(k) * np.array([[1, 0], [0, -1]]) + (1 * J * np.sin(k)) * np.array([[0, 1], [1, 0]])
        H_pm = 1 * h * np.array([[1, 0], [0, -1]])
        
        # Evolution operators
        U_z = expm(-1j * H_fm)
        U_x = expm(-1j * H_pm)

        # Apply the evolution
        rho = U_x @ (U_z @ rho @ U_z.conj().T) @ U_x.conj().T

    return rho


def tfim_momentum_trotter(ks, tau, w=0.0, betas=None, alphas=None):
    """
    Parallelize Trotterized density matrix evolution for all k values.
    Uses the provided angles or generates them if not provided.
    """
    if betas is None or alphas is None:
        # Generate angles following Qiskit's convention
        base_angles = (np.pi / 2) * np.arange(1, tau + 1) / (tau + 1)
        base_angles = base_angles[:, np.newaxis]
        noisy_base_angles = base_angles + w * np.random.randn(tau, 1)
        
        # Calculate beta and alpha arrays
        betas = -np.sin(noisy_base_angles).flatten()
        alphas = -np.cos(noisy_base_angles).flatten()
    
    with Pool() as pool:
        results = pool.starmap(tfim_momentum_trotter_single_k, 
                             [(k, tau, betas, alphas) for k in ks])
    return results


def process_tfim_momentum_trotter(ks, depth, num_qubits, noise, betas=None, alphas=None):
    """Process the TFIM in momentum space with Trotter discretization."""
    solutions = tfim_momentum_trotter(ks, depth, noise, betas, alphas)
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
    distribution = calc_kink_probabilities(pks, kinks_vals)
    kinks_distribution = {k: v for k, v in zip(kinks_vals, distribution)}
    
    # Calculate mean and variance
    mean_kinks = calc_kinks_mean(kinks_distribution)
    second_moment = sum(k**2 * v for k, v in kinks_distribution.items())
    var_kinks = second_moment - mean_kinks**2

    return {
        "density_matrix": density_matrix,
        "kinks_distribution": kinks_distribution,
        "mean_kinks": mean_kinks,
        "var_kinks": var_kinks,
        "purity": purity
    }


def compute_kink_prob(args):
    """Compute kink probability for a given k and tau value."""
    k, tau, w = args
    # Calculate angles (same as in tfim_momentum_trotter)
    base_angles = (np.pi / 2) * np.arange(1, tau + 1) / (tau + 1)
    noisy_angles = base_angles + w * np.random.randn(tau)
    betas = -np.sin(noisy_angles)
    alphas = -np.cos(noisy_angles)
    
    rho = tfim_momentum_trotter_single_k(k=k, tau=tau, beta=betas, alpha=alphas)
    psi_k = np.array([np.sin(k / 2), np.cos(k / 2)])  # Excited state
    kink_prob = np.abs(np.dot(psi_k, np.dot(rho, psi_k)))
    return kink_prob


def plot_pk_vs_k(m=10):
    k_values = k_f(50)
    k_values = np.append(np.append(k_values, 0), np.pi)
    k_values.sort()
    tau_values = [0, 1, 5, 10, 15, 1000]  # Test different τ values

    for tau in tau_values:
        # Prepare arguments for parallel execution
        args = [(k, tau, m) for k in k_values]

        # Parallel computation of kink probabilities
        with ProcessPoolExecutor() as executor:
            kink_probs = list(executor.map(compute_kink_prob, args))

        plt.plot(k_values, kink_probs, label=f'steps = {tau}')

    # Plot reference: Sudden quench (τ=1) kink probability
    plt.plot(k_values, np.cos(k_values / 2) ** 2, '--', label=r'$\cos^2(k/2)$', color='black')

    plt.xlabel('k')
    plt.ylabel('Kink Probability')
    plt.title(f'M={m}')
    plt.legend()
    plt.grid(True)
    plt.show()


# Qiskit Circuit Simulation Functions
def generate_single_circuit_parallel(params):
    """
    Generate a single Qiskit circuit for parallel execution.
    """
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


def generate_qiskit_circuits(qubits, steps, num_circuits_per_step, noise_std=0.0, noise_method='global', betas=None, alphas=None):
    """
    Generate multiple Qiskit circuits in parallel.
    """
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
    if n == 0: # Handle empty string case
        return 0
    for i in range(n): # Loop from 0 to N-1
        if bitstring[i] != bitstring[(i + 1) % n]: # Use modulo for periodic boundary
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
        probabilities = {state: count/total_shots for state, count in total_counts.items()}
        
        # Calculate mean and variance of kinks
        kink_counts = [count_kinks(state) for state in probabilities.keys()]
        mean_kinks = sum(k * p for k, p in zip(kink_counts, probabilities.values()))
        var_kinks = sum((k - mean_kinks)**2 * p for k, p in zip(kink_counts, probabilities.values()))
        
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


def plot_density_matrices_comparison(k_values, tau_values, w=0.0):
    """Plot density matrices for different k and tau values."""
    fig, axs = plt.subplots(len(tau_values), len(k_values), 
                           figsize=(4*len(k_values), 4*len(tau_values)))
    
    for i, tau in enumerate(tau_values):
        # Generate angles with noise
        base_angles = (np.pi / 2) * np.arange(1, tau + 1) / (tau + 1)
        base_angles = base_angles[:, np.newaxis]
        noisy_base_angles = base_angles + w * np.random.randn(tau, 1)
        betas = -np.sin(noisy_base_angles).flatten()
        alphas = -np.cos(noisy_base_angles).flatten()
        
        for j, k in enumerate(k_values):
            rho = tfim_momentum_trotter_single_k(k, tau, betas, alphas)
            
            if len(tau_values) == 1 and len(k_values) == 1:
                ax = axs
            elif len(tau_values) == 1:
                ax = axs[j]
            elif len(k_values) == 1:
                ax = axs[i]
            else:
                ax = axs[i, j]
                
            im = ax.imshow(np.abs(rho), cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f'k={k:.2f}, τ={tau}')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['|0⟩', '|1⟩'])
            ax.set_yticklabels(['⟨0|', '⟨1|'])
            
            for ii in range(2):
                for jj in range(2):
                    ax.text(jj, ii, f'{rho[ii, jj]:.3f}',
                           ha='center', va='center',
                           color='white' if np.abs(rho[ii, jj]) > 0.5 else 'black')
    
    plt.tight_layout()
    return fig


def plot_noise_effects_comparison(num_qubits=4, step_range=range(0, 31, 3),
                             noise_param=0.0,
                             num_circuits=2, numshots=100000, interactive=False):
    """
    Plot comparison of mean kinks and variance between Qiskit and momentum models with noise effects.
    
    Compares how noise affects the mean number of kinks and variance in both the momentum space
    and Qiskit circuit implementations of the TFIM model, using the same noise parameter
    and angles for both models.

    Parameters:
    -----------
    num_qubits : int
        Number of qubits to simulate (default: 4)
    step_range : range or list
        Range of steps to simulate and plot
    noise_param : float
        Noise parameter for both models
    num_circuits : int
        Number of circuits to average in Qiskit simulation
    numshots : int
        Number of shots per circuit in Qiskit simulation
    interactive : bool
        Whether to enable interactive mode for the plot (default: False)
    """
    if interactive:
        plt.ion()  # Enable interactive mode if requested
    else:
        plt.ioff()  # Disable interactive mode by default

    ks = k_f(num_qubits)

    momentum_means = []
    momentum_vars = []
    qiskit_means = []
    qiskit_vars = []

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
        
        # Process momentum model multiple times and average
        momentum_results_list = []
        # Progress bar for momentum circuits
        momentum_progress = tqdm(range(num_circuits), desc="Momentum circuits", leave=False)
        for _ in momentum_progress:
            results = tfim_momentum_trotter(ks, steps, noise_param, betas, alphas)
            momentum_results_list.append(process_tfim_momentum_trotter(ks, steps, num_qubits, noise_param, betas, alphas))
        
        # Average the momentum results
        momentum_mean = np.mean([r["mean_kinks"] for r in momentum_results_list])
        momentum_var = np.mean([r["var_kinks"] for r in momentum_results_list])
        momentum_means.append(momentum_mean) 
        momentum_vars.append(momentum_var)

        # Process Qiskit model with these angles
        qiskit_progress = tqdm(total=1, desc="Qiskit simulation", leave=False)
        qiskit_results = process_qiskit_model(num_qubits, steps, noise_param,
                                          'global', num_circuits, numshots, betas, alphas)
        qiskit_progress.update(1)
        qiskit_means.append(qiskit_results["mean_kinks"])
        qiskit_vars.append(qiskit_results["var_kinks"])

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot mean kinks
    ax1.plot(list(step_range), [i / num_qubits for i in momentum_means], 'o-',
             label=f'Momentum (noise={noise_param})')
    ax1.plot(list(step_range), [i / num_qubits for i in qiskit_means], 'x-',
             label=f'Qiskit (noise={noise_param})')
    ax1.set_xlabel('Number of Steps')
    ax1.set_ylabel('Mean Kinks/N')
    ax1.set_title(f'Mean Kinks vs Steps Comparison ({num_qubits} qubits)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot variance
    ax2.plot(list(step_range), [i / (num_qubits**2) for i in momentum_vars], 'o-',
             label=f'Momentum (noise={noise_param})')
    ax2.plot(list(step_range), [i / (num_qubits**2) for i in qiskit_vars], 'x-',
             label=f'Qiskit (noise={noise_param})')
    ax2.set_xlabel('Number of Steps')
    ax2.set_ylabel('Variance/N²')
    ax2.set_title(f'Variance vs Steps Comparison ({num_qubits} qubits)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()

    # Construct filenames
    date_str = datetime.date.today().strftime('%Y%m%d')
    mean_filename = f"mean_kinks_N{num_qubits}_noise{noise_param}_circ{num_circuits}_shots{numshots}_{date_str}.png"
    var_filename = f"variance_N{num_qubits}_noise{noise_param}_circ{num_circuits}_shots{numshots}_{date_str}.png"
    
    return fig, mean_filename, var_filename



def plot_momentum_qiskit_comparison(num_qubits=4, step_range=range(0, 31, 3),
                                   noise_param=0.0, num_circuits=10, numshots=100000):
    """
    Plot comparison of mean kinks between Qiskit and momentum models for 4 qubits.
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits (default: 4)
    step_range : range
        Range of steps to simulate
    noise_param : float
        Noise parameter for both models
    num_circuits : int
        Number of circuits for Qiskit simulation
    numshots : int
        Number of shots per circuit
    """
    ks = k_f(num_qubits)
    
    momentum_means = []
    qiskit_means = []
    
    for steps in step_range:
        # Process momentum model
        momentum_results = process_tfim_momentum_trotter(ks, steps, num_qubits, noise_param)
        momentum_means.append(momentum_results["mean_kinks"])
        
        # Process Qiskit model
        qiskit_results = process_qiskit_model(
            num_qubits=num_qubits,
            depth=steps,
            noise_param=noise_param,
            noise_type='global',
            num_circuits=num_circuits,
            numshots=numshots
        )
        qiskit_means.append(qiskit_results["mean_kinks"])
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(list(step_range), [i/num_qubits for i in momentum_means], 'o-', 
             label=f'Momentum (noise={noise_param})')
    plt.plot(list(step_range), [i/num_qubits for i in qiskit_means], 'x-', 
             label=f'Qiskit (noise={noise_param})')
    
    plt.xlabel('Number of Steps')
    plt.ylabel('Mean Kinks/N')
    plt.title(f'Mean Kinks vs Steps Comparison ({num_qubits} qubits)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig('momentum_qiskit_comparison.png')
    plt.close()
    
    return {
        'steps': list(step_range),
        'momentum_means': momentum_means,
        'qiskit_means': qiskit_means
    }


if __name__ == "__main__":
    # Define base parameters
    num_qubits=4
    step_range=range(0, 31, 3)
    numshots=100000  # Reduced number of shots
    num_circuits=2  # Increased number of circuits
    interactive=False  # Default to non-interactive mode
    
    # Define parameter combinations to plot
    parameter_sets = [
        {"noise_param": 0.0, "label": "no_noise"}
        # {"noise_param": 0.05, "label": "noise005"},
        # {"noise_param": 0.1, "label": "noise01"},
        # {"noise_param": 0.2, "label": "noise02"}
    ]
    
    # Generate plots for each parameter set
    for params in parameter_sets:
        print(f"\nGenerating plots for noise={params['noise_param']}")
        
        # Create and show the plot with noise comparison
        fig, mean_filename, var_filename = plot_noise_effects_comparison(
            num_qubits=num_qubits,
            step_range=step_range,
            noise_param=params['noise_param'],
            num_circuits=num_circuits,
            numshots=numshots,
            interactive=interactive
        )
        
        # Save the plots using the generated filenames
        plt.savefig(mean_filename)
        plt.savefig(var_filename)
        print(f"Plots saved as: {mean_filename} and {var_filename}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    print("\nAll plots generated successfully!")