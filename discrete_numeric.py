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
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm
from functions import k_f, epsilon, calc_kink_probabilities, calc_kinks_probability, \
    calc_kinks_mean, noisy_lz_time_evolution
from qiskit.quantum_info import DensityMatrix
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from scipy.linalg import expm
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

import numpy as np
from multiprocessing import Pool
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix

import numpy as np
from scipy.linalg import expm


def tfim_momentum_trotter_single_k(k, tau, w=0.0, M=1):
    """
    Simulates Trotterized evolution of the TFIM in momentum space for a single momentum k,
    with each Trotter step subdivided into M smaller substeps (each of duration dt_sub = 1/M).

    The evolution is performed in Nambu space, with the initial state chosen as |↓⟩
    (represented as [0, 1]^T, i.e. the ground state for g=0).

    Parameters:
      k   : momentum (in radians)
      tau : total number of Trotter steps (each with total time = 1)
      w   : noise parameter (unused in V1)
      M   : number of substeps per Trotter step (thus, dt_sub = 1/M)

    Returns:
      rho : the final density matrix for momentum k.
    """
    # Initial state |↓⟩ is represented as [0, 1]^T; hence its density matrix:
    rho = np.array([[0, 0], [0, 1]], dtype=complex)

    # dt for each substep:
    dt_sub = 1.0 / M

    vs = []
    hs = []
    # Loop over each Trotter step
    for n in range(1, tau + 1):
        # Compute the step-dependent angle theta_n (for the n-th Trotter step)
        theta_n = (np.pi / 2) * n / (tau + 1)
        h = np.cos(theta_n)  # transverse field part (from HPM)
        J = np.sin(theta_n)  # interaction strength (from HFM)

        # Effective parameters in momentum space for this step:
        # h_eff = 2 * (h - J * np.cos(k))  # coefficient in front of σ_z
        # delta_k = 2 * J * np.sin(k)  # coefficient in front of σ_x

        # For each Trotter step, subdivide the evolution into M substeps.
        for m in range(M):
            # For the σ_z part: We want U_z = exp(-i * h_eff * dt_sub * σ_z)
            # Using the standard rotation formula: exp(-i φ σ_z/2)
            # theta_z = h_eff * dt_sub

            H_fm = -1 * J * np.cos(k) * np.array([[1, 0], [0, -1]]) + (1 * J * np.sin(k)) * np.array([[0, 1], [1, 0]])
            H_pm = 1 * h * np.array([[1, 0], [0, -1]])
            U_z = expm(-1j * H_fm)

            # U_z = np.array([[np.exp(-1j * theta_z / 2), 0],
            #                 [0, np.exp(1j * theta_z / 2)]])

            # For the σ_x part: We want U_x = exp(-i * delta_k * dt_sub * σ_x)
            # theta_x = delta_k * dt_sub
            # U_x = np.cos(theta_x / 2) * np.eye(2) - 1j * np.sin(theta_x / 2) * np.array([[0, 1],
            #                                                                              [1, 0]])
            U_x = expm(-1j * H_pm)
            H_s = U_x @ U_z

            com = U_x @ U_z - U_z @ U_x

            # Apply the substep evolution: first U_z then U_x.
            # rho = U_z @ (U_x @ rho @ U_x.conj().T) @ U_z.conj().T
            rho = U_x @ (U_z @ rho @ U_z.conj().T) @ U_x.conj().T

    return rho  # , vs, hs


def tfim_momentum_trotter(ks, tau, w):
    """
    Parallelize Trotterized density matrix evolution for all k values.
    """
    with Pool() as pool:
        results = pool.starmap(tfim_momentum_trotter_single_k, [(k, tau, w) for k in ks])
    return results


def process_tfim_momentum_trotter(ks, depth, num_qubits, noise):
    """
    Process the TFIM in momentum space with Trotter discretization.
    """
    solutions = tfim_momentum_trotter(ks, depth, noise)
    tensor_product = solutions[0]
    for rho in solutions[1:]:
        tensor_product = np.kron(tensor_product, rho)
    density_matrix = tensor_product
    rho2 = density_matrix @ density_matrix
    purity = np.trace(rho2).real
    pks = np.array([
        np.abs(np.dot(
            np.array([np.sin(k / 2), np.cos(k / 2)]),
            np.dot(solution, np.array([[np.sin(k / 2)], [np.cos(k / 2)]]))
        ))[0] for k, solution in zip(ks, solutions)
    ])
    pks = np.where(pks < 1e-10, 0, pks)
    kinks_vals = np.arange(0, num_qubits + 1, 2)
    distribution = calc_kink_probabilities(pks, kinks_vals)
    kinks_distribution = {k: v for k, v in zip(kinks_vals, distribution)}
    mean_kinks = calc_kinks_mean(kinks_distribution)
    return {
        "density_matrix": density_matrix,
        "kinks_distribution": kinks_distribution,
        "mean_kinks": mean_kinks,
        "purity": purity
    }


def compute_kink_prob(args):
    k, tau, m = args
    rho = tfim_momentum_trotter_single_k(k=k, tau=tau, w=0.0, M=m)
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
        if noise_method == 'dephasing':
            circuit.rz(noisy_betas[step, circuit_idx], range(qubits))
        else:
            circuit.rz(beta, range(qubits))
        for i in range(qubits):
            j = (i + 1) % qubits
            circuit.cp(-2 * beta, i, j)  # Controlled phase for interaction
        for i in range(qubits):
            circuit.rx(alpha, i)  # Transverse field
    dm = DensityMatrix.from_instruction(circuit)
    circuit.measure(range(qubits), range(qubits))
    return circuit, dm.data


def generate_qiskit_circuits(qubits, steps, num_circuits_per_step, noise_std=0.0, noise_method='global'):
    """
    Generate multiple Qiskit circuits in parallel.
    """
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
    params = [(qubits, steps, i, betas, alphas, noisy_betas if noise_method == 'dephasing' else None, noise_method)
              for i in range(num_circuits_per_step)]
    with Pool() as pool:
        results = pool.map(generate_single_circuit_parallel, params)
    circuits, density_matrices = zip(*results)
    return list(circuits), list(density_matrices)


def process_qiskit_model(num_qubits, depth, noise_param, noise_type, num_circuits, numshots):
    """
    Process the Qiskit model and return simulation results.
    """
    circuits, density_matrices = generate_qiskit_circuits(num_qubits, depth, num_circuits, noise_param, noise_type)
    avg_density_matrix = sum(density_matrices) / num_circuits
    rho_squared = avg_density_matrix @ avg_density_matrix
    purity = np.trace(rho_squared).real
    simulator = AerSimulator()
    transpiled_circuits = transpile(circuits, simulator)
    results = simulator.run(transpiled_circuits, shots=numshots).result().get_counts()
    counts = {}
    for result in results:
        for key, value in result.items():
            counts[key] = counts.get(key, 0) + value
    probs = calc_kinks_probability(counts)
    mean = calc_kinks_mean(probs)
    return {
        "density_matrix": avg_density_matrix,
        "kinks_distribution": probs,
        "mean_kinks": mean,
        "purity": purity
    }


# def plot_pk_vs_k(m=10):
#     k_values = k_f(50)
#     k_values = np.append(np.append(k_values, 0),np.pi)
#     k_values.sort()
#     tau_values = [0, 1, 100,1000]  # Test different τ values
#
#     for tau in tau_values:
#         kink_probs = []
#         for k in k_values:
#             rho = tfim_momentum_trotter_single_k(k=k, tau=tau, w=0.0, M=m)
#             psi_k = np.array([np.sin(k / 2), np.cos(k / 2)])  # Excited state
#             kink_prob = np.abs(np.dot(psi_k, np.dot(rho, psi_k)))
#             kink_probs.append(kink_prob)
#
#         plt.plot(k_values, kink_probs, label=f'steps = {tau}')
#
#     # Plot reference: Sudden quench (τ=1) kink probability
#     plt.plot(k_values, np.cos(k_values / 2) ** 2, '--', label=r'$\cos^2(k/2)$', color='black')
#
#     plt.xlabel('k')
#     plt.ylabel('Kink Probability')
#     plt.title(f'M={m}')
#     plt.legend()
#     plt.grid(True)
#     plt.show()


def plot_density_matrices_comparison(k_values, tau_values):
    """
    Creates a grid of density matrix plots for different k and tau values.

    Parameters:
    -----------
    k_values : list of float
        List of momentum values to plot
    tau_values : list of int
        List of step values to plot
    """
    fig, axs = plt.subplots(len(tau_values), len(k_values), figsize=(4 * len(k_values), 4 * len(tau_values)))

    for i, tau in enumerate(tau_values):
        for j, k in enumerate(k_values):
            rho = tfim_momentum_trotter_single_k(k, tau, w=0.0)

            if len(tau_values) == 1 and len(k_values) == 1:
                ax = axs
            elif len(tau_values) == 1:
                ax = axs[j]
            elif len(k_values) == 1:
                ax = axs[i]
            else:
                ax = axs[i, j]

            # Display as a 2×2 grid with values
            im = ax.imshow(np.abs(rho), cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f'k={k:.2f}, τ={tau}')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['|0⟩', '|1⟩'])
            ax.set_yticklabels(['⟨0|', '⟨1|'])

            # Show matrix values
            for ii in range(2):
                for jj in range(2):
                    ax.text(jj, ii, f'{rho[ii, jj]:.3f}',
                            ha='center', va='center',
                            color='white' if np.abs(rho[ii, jj]) > 0.5 else 'black')

    plt.tight_layout()
    return fig


def plot_mean_kinks_comparison(num_qubits=4, step_range=range(0, 31, 3),
                               momentum_noise=0.0,
                               qiskit_noise_param=0.0, qiskit_noise_type='global',
                               num_circuits=10, numshots=100000):
    """
    Plot comparison of mean kinks between Qiskit and momentum models across different steps.

    Parameters:
    -----------
    num_qubits : int
        Number of qubits to simulate
    step_range : range or list
        Range of steps to simulate and plot
    momentum_noise : float
        Noise parameter for momentum model
    momentum_dt : int
        Number of substeps per Trotter step in momentum model
    qiskit_noise_param : float
        Noise parameter for Qiskit model
    qiskit_noise_type : str
        Type of noise for Qiskit ('global' or 'dephasing')
    num_circuits : int
        Number of circuits to average in Qiskit simulation
    numshots : int
        Number of shots per circuit in Qiskit simulation
    """
    ks = k_f(num_qubits)

    momentum_means = []
    qiskit_means = []

    for steps in step_range:
        # Process momentum model
        momentum_results = process_tfim_momentum_trotter(ks, steps, num_qubits, momentum_noise)
        momentum_means.append(momentum_results["mean_kinks"])

        # Process Qiskit model
        qiskit_results = process_qiskit_model(num_qubits, steps, qiskit_noise_param,
                                              qiskit_noise_type, num_circuits, numshots)
        qiskit_means.append(qiskit_results["mean_kinks"])

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(list(step_range), [i / num_qubits for i in momentum_means], 'o-',
             label=f'Momentum (noise={momentum_noise})')
    plt.plot(list(step_range), [i / num_qubits for i in qiskit_means], 'x-',
             label=f'Qiskit ({qiskit_noise_type}, noise={qiskit_noise_param})')

    plt.xlabel('Number of Steps')
    plt.ylabel('Mean Kinks/N')
    plt.title(f'Mean Kinks vs Steps Comparison ({num_qubits} qubits)')
    plt.legend()
    plt.grid(True)

    return plt.gcf()


def plot_rho_elements_vs_k(tau_values, k_range=np.linspace(0, np.pi, 100)):
    """
    Plot the [0,0] and [0,1] elements of the density matrix for different k values and tau values.

    Parameters:
    -----------
    tau_values : list of int
        List of step values to plot
    k_range : array
        Range of k values to evaluate
    """
    plt.figure(figsize=(12, 8))

    # First subplot for rho[0,0]
    plt.subplot(2, 1, 1)
    for tau in tau_values:
        rho_00_values = []
        for k in k_range:
            rho = tfim_momentum_trotter_single_k(k, tau, w=0.0)
            rho_00_values.append(rho[0, 0].real)  # Real part of [0,0] element
        plt.plot(k_range, rho_00_values, label=f'τ={tau}')

    plt.xlabel('k')
    plt.ylabel('ρ[0,0]')
    plt.title('Diagonal Element ρ[0,0] vs Momentum k')
    plt.grid(True)
    plt.legend()

    # Second subplot for rho[0,1]
    plt.subplot(2, 1, 2)
    for tau in tau_values:
        rho_01_real_values = []
        rho_01_imag_values = []
        rho_01_abs_values = []

        for k in k_range:
            rho = tfim_momentum_trotter_single_k(k, tau, w=0.0)
            rho_01_real_values.append(rho[0, 1].real)
            rho_01_imag_values.append(rho[0, 1].imag)
            rho_01_abs_values.append(abs(rho[0, 1]))

        # Plot absolute value of off-diagonal element
        plt.plot(k_range, rho_01_abs_values, label=f'|ρ[0,1]|, τ={tau}')

        # Optionally, plot real and imaginary parts
        # plt.plot(k_range, rho_01_real_values, '--', label=f'Re(ρ[0,1]), τ={tau}')
        # plt.plot(k_range, rho_01_imag_values, ':', label=f'Im(ρ[0,1]), τ={tau}')

    plt.xlabel('k')
    plt.ylabel('ρ[0,1]')
    plt.title('Off-Diagonal Element ρ[0,1] vs Momentum k')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    return plt.gcf()


def plot_kinks_vs_steps_for_sizes(system_sizes, step_range, numshots=10000, num_circuits=2):
    """
    Plot the mean number of kinks in Qiskit circuits with no noise as a function of steps
    for multiple system sizes.

    Parameters:
    -----------
    system_sizes : list of int
        List of different qubit counts (system sizes) to simulate
    step_range : list or range
        Range of steps to simulate for each system size
    numshots : int, optional
        Number of shots for each circuit simulation (default: 100000)
    num_circuits : int, optional
        Number of circuits to average for each data point (default: 10)

    Returns:
    --------
    fig : matplotlib Figure
        The generated plot figure
    """
    plt.figure(figsize=(10, 6))

    # Store data for scaling analysis
    all_means = {}

    for size in system_sizes:
        # List to store mean kinks for this system size
        means = []

        for steps in step_range:
            # Process Qiskit model with no noise
            results = process_qiskit_model(
                num_qubits=size,
                depth=steps,
                noise_param=0.0,  # No noise
                noise_type='global',  # Doesn't matter with no noise
                num_circuits=num_circuits,
                numshots=numshots
            )

            mean_kinks = results["mean_kinks"]
            means.append(mean_kinks)

        # Store the data
        all_means[size] = means

        # Plot raw data
        # plt.plot(list(step_range), means, 'o-', label=f'N = {size}')

        # Plot scaled data (kinks/N)
        plt.plot(list(step_range), [m / size for m in means], '--', alpha=0.7, label=f'N = {size} (scaled)')

    plt.xlabel('Number of Steps')
    plt.ylabel('Mean Kinks')
    plt.title('Mean Kinks vs Steps for Different System Sizes (No Noise)')
    plt.grid(True)
    plt.legend()

    # Add annotation explaining the plot
    # plt.figtext(0.5, 0.01,
    #             'Solid lines: absolute number of kinks\nDashed lines: kinks per qubit (kinks/N)',
    #             ha='center', fontsize=10)

    plt.tight_layout()
    return plt.gcf()


def plot_kinks_vs_steps_for_sizes_momentum(system_sizes, step_range, momentum_noise=0.0):
    """
    Plot the mean number of kinks in the discrete momentum model with no noise as a function of steps
    for multiple system sizes.

    Parameters:
    -----------
    system_sizes : list of int
        List of different qubit counts (system sizes) to simulate
    step_range : list or range
        Range of steps to simulate for each system size
    momentum_noise : float, optional
        Noise parameter for the momentum model (default: 0.0)

    Returns:
    --------
    fig : matplotlib Figure
        The generated plot figure
    """
    plt.figure(figsize=(10, 6))

    # Store data for scaling analysis
    all_means = {}

    for size in system_sizes:
        # List to store mean kinks for this system size
        means = []
        ks = k_f(size)  # Get momentum values for this system size

        for steps in step_range:
            # Process momentum model
            results = process_tfim_momentum_trotter(
                ks=ks,
                depth=steps,
                num_qubits=size,
                noise=momentum_noise  # No noise
            )

            mean_kinks = results["mean_kinks"]
            means.append(mean_kinks)

        # Store the data
        all_means[size] = means

        # Plot scaled data (kinks/N)
        plt.plot(list(step_range), [m / size for m in means], 'o-', label=f'N = {size}')

    plt.xlabel('Number of Steps')
    plt.ylabel('Mean Kinks/N')
    plt.title('Mean Kinks vs Steps for Different System Sizes (Momentum Model, No Noise)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    return plt.gcf()


def plot_kinks_vs_steps_lz_evolution(system_sizes, step_range):
    """
    Plot the mean number of kinks using the noisy_lz_time_evolution model with no noise
    as a function of steps for multiple system sizes.

    Parameters:
    -----------
    system_sizes : list of int
        List of different qubit counts (system sizes) to simulate
    step_range : list or range
        Range of steps to simulate for each system size

    Returns:
    --------
    fig : matplotlib Figure
        The generated plot figure
    """
    plt.figure(figsize=(10, 6))

    # Store data for scaling analysis
    all_means = {}

    for size in system_sizes:
        # List to store mean kinks for this system size
        means = []
        ks = k_f(size)  # Get momentum values for this system size

        for steps in step_range:
            # Get solutions from noisy_lz_time_evolution with no noise
            ks_solutions = noisy_lz_time_evolution(ks, steps, 0.0)  # No noise

            # Calculate probabilities for each k
            pks = np.array([
                np.abs(np.dot(
                    np.array([np.sin(k / 2), np.cos(k / 2)]),
                    np.dot(
                        result.y[:, -1].reshape(2, 2),
                        np.array([[np.sin(k / 2)], [np.cos(k / 2)]])
                    )
                ))[0]
                for k, result in zip(ks, ks_solutions)
            ])

            # Clean up small values
            pks = np.where(pks < epsilon / 100, 0, pks)

            # Calculate kinks distribution and mean
            kinks_vals = np.arange(0, size + 1, 2)
            kinks_distribution = calc_kink_probabilities(pks, kinks_vals)
            mean_kinks = np.sum(kinks_distribution * kinks_vals)

            means.append(mean_kinks)

        # Store the data
        all_means[size] = means

        # Plot scaled data (kinks/N)
        plt.plot(list(step_range), [m / size for m in means], 'o-', label=f'N = {size}')

    plt.xlabel('Number of Steps')
    plt.ylabel('Mean Kinks/N')
    plt.title('Mean Kinks vs Steps for Different System Sizes (LZ Time Evolution, No Noise)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    return plt.gcf()


from matplotlib import pyplot as plt

if __name__ == "__main__":
    p = plot_mean_kinks_comparison(num_qubits=8, step_range=range(0, 31,5),
                               momentum_noise=0.0,
                               qiskit_noise_param=0.0, qiskit_noise_type='global',
                               num_circuits=2, numshots=1000)
    p.show()

    # plot_pk_vs_k()

    # Example usage
    tau_values = [1, 20, 1000]
    # fig = plot_rho_elements_vs_k(tau_values)
    # plt.show()

    # # Example usage:
    # k_values = k_f(8)
    #
    # fig = plot_density_matrices_comparison(k_values, tau_values)
    # plt.show()

    # # Example usage:
    # system_sizes = [4, 6, 8, 10]
    # step_range = range(0, 31, 3)
    # # fig = plot_kinks_vs_steps_for_sizes(system_sizes, step_range)
    # # plt.show()
    #
    # # fig = plot_kinks_vs_steps_for_sizes_momentum(system_sizes, step_range)
    # # plt.show()
    #
    # # Example usage:
    # fig = plot_kinks_vs_steps_lz_evolution(system_sizes, step_range)
    # plt.show()

    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # n = 100
    # tau = 101
    #
    # # Parameters for n=1, tau=1
    # theta_n = theta_n = (np.pi / 2) * n / (tau + 1)
    # h = np.cos(theta_n)
    # J = np.sin(theta_n)
    #
    # # Momentum values
    # k_values = [np.pi/3, 2*np.pi/3]
    # prob_down = []
    #
    # for k in k_values:
    #     h_eff = 2 * (h - J * np.cos(k))
    #     delta_k = 2 * J * np.sin(k)
    #     A = h_eff / 2
    #     B = delta_k / 2
    #     H = np.array([[A, B], [B, -A]])
    #     eigenvalues, eigenvectors = np.linalg.eigh(H)
    #     idx = np.argmin(eigenvalues)  # Ground state
    #     psi_g = eigenvectors[:, idx]
    #
    # # Plotting
    # plt.plot(k_values, prob_down, label=r'$|\langle \downarrow | \psi_g \rangle|^2$')
    # plt.xlabel('Momentum \( k \)')
    # plt.ylabel('Probability in \( |downarrow\rangle \)')
    # plt.title('Ground State Probability for Hamiltonian of Step 1')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    # for k in np.linspace(0, np.pi, 20):
    #     rho = tfim_momentum_trotter_single_k(k, 100, 0, 1)
    # rho, vs, hs = tfim_momentum_trotter_single_k(np.pi/8, 100, 0, 1)
    # print(hs[-1])
    # plt.plot(np.angle(np.array(vs)[:, 1]))
    # plt.plot(np.angle(np.array(vs)[:, 0]))
    # plt.show()
    # plot_pk_vs_k(1)

