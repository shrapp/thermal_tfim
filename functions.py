import ast
import logging
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from enum import Enum
from multiprocessing import Lock, Pool, Process

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import (amplitude_damping_error,
                                     depolarizing_error, phase_damping_error)
from scipy.integrate import quad, solve_ivp
from scipy.special import gammaln, logsumexp

logging.basicConfig(filename='logger.log', level=logging.INFO, format='%(asctime)s %(message)s')

class Models(Enum):
    INDEPENDENT_KS_NUMERIC = 'independent_ks_numeric'
    QISKIT_GLOBAL_NOISE = 'qiskit_global_noise'
    QISKIT_DEPHASING = 'qiskit_dephasing'

class Parameters(Enum):
    QUBITS = 'qubits'
    NOISE_PARAM = 'noise_param'
    DEPTH = 'depth'
    MEAN_KINKS = 'mean_kinks'
    VARIANCE_KINKS = 'variance_kinks'
    PURITY = 'purity'
    DENSITY_MATRIX = 'density_matrix'
    KINKS_PROBABILITY = 'kinks_probability'
    KINKS_NUMBER = 'kinks_number'
    AVERAGE_DENSITY_MATRIX = 'average_density_matrix'

def generate_random_density_matrix(N):
    # Step 1: Generate a random diagonal matrix with eigenvalues summing to 1
    eigenvalues = np.random.dirichlet(np.ones(N))
    rho_diag = np.diag(eigenvalues)
    
    # Step 2: Generate a Haar-random unitary matrix using the QR decomposition
    X = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2)
    Q, R = np.linalg.qr(X)
    R = np.diag(np.diag(R) / np.abs(np.diag(R)))
    U = Q @ R
    
    # Step 3: Transform the diagonal matrix into a density matrix
    rho = U @ rho_diag @ np.conjugate(U.T)
    
    return rho

def commutator(A, B):
    """
    Calculate the commutator of two matrices A and B.

    Parameters:
    A (np.ndarray): A square matrix.
    B (np.ndarray): Another square matrix of the same dimension as A.

    Returns:
    np.ndarray: The commutator [A, B] = AB - BA.
    """
    return np.dot(A, B) - np.dot(B, A)

def dag(matrix):
    """
    Calculate the conjugate transpose (dagger) of a complex matrix.

    Parameters:
    matrix (np.ndarray): A complex matrix.

    Returns:
    np.ndarray: The conjugate transpose of the matrix.
    """
    return np.conjugate(matrix).T

# Parameters
hbar = 1.0   # in natural units
epsilon = 1e-7
data_file = 'data/data.csv'
sigX = np.array([[0, 1], [1, 0]])
sigZ = np.array([[1, 0], [0, -1]])

def g0(t, tau):
    return t/tau if tau > 0 else 0

def H0(t, tau, k):
    g_f = g0(t, tau)
    h_z = 2 * (1 - g_f - g_f * np.cos(k))
    h_x = 2 * g_f * np.sin(k)
    H0 = h_z * sigZ + h_x * sigX
    return H0

def V_func(k):
    v_z = -2 * (np.cos(k) + 1)
    v_x = 2 * np.sin(k)
    V_matrix = v_z * sigZ + v_x * sigX
    return V_matrix

def Dissipator(k, rho, w):
    V = V_func(k)
    term1 = V @ rho @ V
    term3 = -0.5 * (V @ V @ rho)
    term2 = -0.5 * (rho @ V @ V)
    D = w**2 * (term1 + term2 + term3)
    return D

def psi_dt(t, psi,  tau, k):
    return -1j * np.dot(H0(t, tau, k), psi)

def rho_dt(t, rho, tau, k, w):
    rho = rho.reshape(2,2)
    H = H0(t, tau, k) # + w[int((t // dt))] * V_func(k)
    D = Dissipator(k, rho, w)
    U = -1j * commutator(H, rho)
    rho_dot = U + D
    return rho_dot.flatten()

def lz_time_evolution_single_k(k, tau):
    # use scipy.integrate.solve_ivp to solve the ODE H(t) psi(t) = i hbar d/dt psi(t)
    psi0 = np.array([0+0j, 1+0j])
    return  solve_ivp(fun=psi_dt, method='DOP853', t_span=(0, tau), y0=psi0, args=(tau, k), rtol = epsilon, atol = epsilon)

def noisy_lz_time_evolution_single_k(k, tau, w):
    # use scipy.integrate.solve_ivp to solve the ODE H(t) psi(t) = i hbar d/dt psi(t)
    rho0 = np.array([[0+0j, 0+0j], [0+0j, 1+0j]])
    return  solve_ivp(fun=rho_dt, method='DOP853', t_span=(0, tau), y0= rho0.flatten(), args=(tau, k, w), rtol = epsilon, atol = epsilon, vectorized=True)

def lz_time_evolution(ks, tau):
    # use pool to parallelize the calculation for each k
    with Pool() as pool:
        results = pool.starmap(lz_time_evolution_single_k, [(k, tau) for k in ks])
    return results

def noisy_lz_time_evolution(ks, tau, w):
    # use pool to parallelize the calculation for each k
    with Pool() as pool:
        results = pool.starmap(noisy_lz_time_evolution_single_k, [(k, tau, w) for k in ks])
    return results

def calc_pk(ks, tau):
    # calculate the time evolution for each k
    results = lz_time_evolution(ks, tau)
    # calculate the probabilities for each k
    pks = np.array([np.abs(np.dot(np.array([np.sin(k/2), np.cos(k/2)]), result.y[:,-1]/np.linalg.norm(result.y[:,-1])))**2 for k ,result in zip(ks, results)])
    # remove valuse that are too small
    return np.where(pks < epsilon/100, 0, pks)

def calk_noisy_pk(ks, tau, w):
    # calculate the time evolution for each k
    results = noisy_lz_time_evolution(ks, tau, w)
    # calculate the probabilities for each k
    pks = np.array([np.abs(np.dot(np.array([np.sin(k/2), np.cos(k/2)]), np.dot(result.y[:,-1].reshape(2,2), np.array([[np.sin(k/2)], [np.cos(k/2)]]))))[0] for k ,result in zip(ks, results)])
    # remove valuse that are too small
    return np.where(pks < epsilon/100, 0, pks)

def k_f(N: int) -> np.ndarray:
    # Filtering out even numbers
    odd_numbers = np.arange(1, N, 2)
    
    # Multiply each odd number by pi/N
    return odd_numbers * np.pi / N

def p_k_analytic(tau, k):
    return np.exp(-2*np.pi*tau*(k**2))*(np.cos(k/2)**2)

def ln_P_tilda_func(theta, pks):
    return np.sum(np.log(1 + pks * (np.exp(2 * 1j * theta) - 1)))

def integrand_func(theta, pks, d):
    return np.exp(ln_P_tilda_func(theta, pks) - 1j * theta * d)

def P_func(pks, d):
    integral, _ = quad(lambda theta: np.real(integrand_func(theta, pks, d)), -np.pi, np.pi, limit=10000)
    return np.abs(integral / (2 * np.pi))

def D_func(P_vals):
    N = (len(P_vals) - 1)*2
    ns = np.arange(0,N + 1,2)
    return (1 / N) * np.sum(ns * P_vals)

def beta(d):
    if d == 0:
        return np.inf
    elif d > 0.5:
        return np.nan
    else:
        return -0.5 * (np.log(d) - np.log(1 - d))

def thermal_prob1(mean, N):
    beta_val = beta(mean)
    if np.isnan(beta_val):
        return np.nan
    if beta_val == np.inf:
        probabilities = np.zeros(N//2 + 1)
        probabilities[0] = 1
        return probabilities

    # Generate the range of n values
    n_values = np.arange(0, N+1, 2)

    # Compute log-probabilities
    log_unnorm_probs = (-2 * n_values * beta_val) - gammaln(n_values + 1) - gammaln(N - n_values + 1)
    
    # Normalize using logsumexp to avoid overflow/underflow
    log_norm = logsumexp(log_unnorm_probs)
    normalized_logprobs = log_unnorm_probs - log_norm
    
    # Exponentiate to get probabilities
    probabilities = np.exp(normalized_logprobs)

    return probabilities

def thermal_prob2(mean, N):
    beta_val = beta(mean)
    if np.isnan(beta_val):
        return np.nan
    if beta_val == np.inf:
        probabilities = np.zeros(N//2 + 1)
        probabilities[0] = 1
        return probabilities

    # Generate the range of n values
    n_values = np.arange(0, N/2+1)

    # Compute log-probabilities
    log_unnorm_probs = (-2 * n_values * beta_val) - gammaln(n_values + 1) - gammaln(N/2 - n_values + 1)
    
    # Normalize using logsumexp to avoid overflow/underflow
    log_norm = logsumexp(log_unnorm_probs)
    normalized_logprobs = log_unnorm_probs - log_norm
    
    # Exponentiate to get probabilities
    probabilities = np.exp(normalized_logprobs)

    return probabilities

def calculate_cumulants(probability_mass_function, values):
    # Calculate the moments
    mean = np.sum(probability_mass_function * values)
    second_moment = np.sum(probability_mass_function * values ** 2)
    third_moment = np.sum(probability_mass_function * values ** 3)
    fourth_moment = np.sum(probability_mass_function * values ** 4)

    # Calculate the cumulants using the moments
    variance = second_moment - mean ** 2
    skewness = (third_moment - 3 * mean * second_moment + 2 * mean ** 3)
    kurtosis = (fourth_moment - 4 * mean * third_moment + 6 * mean ** 2 * second_moment - 3 * mean ** 4)

    cumulants = {'mean': mean, 'second_moment': second_moment, 'third_moment': third_moment, 'fourth_moment': fourth_moment, 'variance': variance, 'skewness': skewness, 'kurtosis': kurtosis}
    return cumulants

def calc_kink_probabilities(pks, d_vals, parallel=True):
    if parallel:
        with Pool() as pool:
            return np.array(pool.starmap(P_func, [(pks, d) for d in d_vals]))
    else:
        return np.array([P_func(pks, d) for d in d_vals])
    
             
def calc_data(Ns, taus, noises):
    processes = []
    lock = Lock()
    for N, tau, noise in [(N, tau, noise) for N in Ns for tau in taus for noise in noises]:
        p = Process(target=calc_data_single, args=(N, tau, lock, noise))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        
def calculate_pk(N, tau, noise):
    ks = k_f(N)
    pks_numeric = calk_noisy_pk(ks, tau, noise)
    return {'probability': str(pks_numeric.tolist())}

def calculate_numeric(N, tau, noise, df):
    pks_numeric = np.array(ast.literal_eval(df[(df['type'] == 'pk') & (df['N'] == N) & (df['tau'] == tau) & (df['noise'] == noise)]['probability'].iloc[0])).flatten()
    d_vals = np.arange(0,N+1,2)
    num_probability_mass_function = calc_kink_probabilities(pks_numeric, d_vals)
    num_probability_mass_function = clean_probabilities(num_probability_mass_function)
    num_cumulants = calculate_cumulants(num_probability_mass_function, d_vals)
    return {**num_cumulants, 'probability': str(num_probability_mass_function.tolist())}

def calculate_thermal1(N, tau, noise, df):
    d_vals = np.arange(0,N+1,2)
    num_probability_mass_function = np.array(ast.literal_eval(df[(df['type'] == 'numeric') & (df['N'] == N) & (df['tau'] == tau) & (df['noise'] == noise)]['probability'].iloc[0])).flatten()
    d_num = D_func(num_probability_mass_function)
    therm_probability_mass_function = thermal_prob1(d_num, N)
    therm_probability_mass_function = clean_probabilities(therm_probability_mass_function)
    therm_cumulants = calculate_cumulants(therm_probability_mass_function, d_vals)
    return {**therm_cumulants, 'probability': str(therm_probability_mass_function.tolist())}

def calculate_thermal2(N, tau, noise, df):
    d_vals = np.arange(0,N+1,2)
    num_probability_mass_function = np.array(ast.literal_eval(df[(df['type'] == 'numeric') & (df['N'] == N) & (df['tau'] == tau) & (df['noise'] == noise)]['probability'].iloc[0])).flatten()
    d_num = D_func(num_probability_mass_function)
    therm_probability_mass_function = thermal_prob2(d_num, N)
    therm_probability_mass_function = clean_probabilities(therm_probability_mass_function)
    therm_cumulants = calculate_cumulants(therm_probability_mass_function, d_vals)
    return {**therm_cumulants, 'probability': str(therm_probability_mass_function.tolist())}

def calculate_analytic(N, tau, noise, df):
    ks = k_f(N)
    pks_analytic = pk_analitic(ks, tau)
    d_vals = np.arange(0,N+1,2)
    analytic_probability_mass_function = calc_kink_probabilities(pks_analytic, d_vals)
    analytic_probability_mass_function = clean_probabilities(analytic_probability_mass_function)
    analytic_cumulants = calculate_cumulants(analytic_probability_mass_function, d_vals)
    return {**analytic_cumulants, 'probability': str(analytic_probability_mass_function.tolist())}
    
def clean_probabilities(probability_mass_function):
    probability_mass_function = np.where(probability_mass_function < epsilon, 0, probability_mass_function)
    mask = (np.roll(probability_mass_function, 0) == 0) & (np.roll(probability_mass_function, -2) == 0)
    mask[-1] = False
    mask[-2] = False
    probability_mass_function[np.roll(mask, 1)] = 0
    return probability_mass_function
    

def pk_analitic(k_values, tau):
    return np.array([np.exp(-2*np.pi*tau*(k**2))*(np.cos(k/2)**2) for k in k_values])


def get_data_in_range(N, tau_min, tau_max, noise_min=0.0, noise_max=0.0):
    # Load the DataFrame
    df = pd.read_csv(data_file)
    df = df.sort_values(['N', 'tau', 'noise'])

    # Apply the conditions and get the corresponding data
    return df[(df['N'] == N) & (df['tau'] >= tau_min) & (df['tau'] <= tau_max) & (df['noise'] >= noise_min) & (df['noise'] <= noise_max)]


def load_data():
    try:
        df = pd.read_csv(data_file)
        return df.sort_values(['N', 'tau', 'noise'])
    except FileNotFoundError:
        return pd.DataFrame(columns=['N', 'tau', 'type', 'noise', 'probability', 'mean', 
                                     'second_moment', 'third_moment', 'fourth_moment',
                                     'variance', 'skewness', 'kurtosis']) 
        

        
def calculate_and_save_type_data(df, N, tau, noise, type_key, calculation_function, lock):
    if df[(df['N'] == N) & (df['tau'] == tau) & (df['noise'] == noise) & (df['type'] == type_key)].empty:
        data = calculation_function(N, tau, noise, df)
        data_df = pd.DataFrame({**data, 'N': [N], 'tau': [tau], 'type': [type_key], 'noise': [noise]})
        with lock:
            df = load_data()
            df = pd.concat([df, data_df])
            df = df.sort_values(['N', 'tau', 'noise'])
            df.to_csv(data_file, index=False)
    return df
    
def calc_data_single(N, tau, lock, noise):
    # os.nice(1) # type: ignore
    tau = round(tau, 6)
    noise = round(noise, 6)
    
    with lock:
        df = load_data()

    # Here, replace calculate_pk, calculate_numeric, etc. with actual function implementations
    df = calculate_and_save_type_data(df, N, tau, noise, 'pk', calculate_pk, lock)
    df = calculate_and_save_type_data(df, N, tau, noise, 'numeric', calculate_numeric, lock)
    
    p1 = Process(target=calculate_and_save_type_data, args=(df, N, tau, noise, 'thermal1', calculate_thermal1, lock))
    p1.start()
    p2 = Process(target=calculate_and_save_type_data, args=(df, N, tau, noise, 'thermal2', calculate_thermal2, lock))
    p2.start()
    if noise == 0:
        p3 = Process(target=calculate_and_save_type_data, args=(df, N, tau, noise, 'analytic', calculate_analytic, lock))
        p3.start()
        p3.join()
    p1.join()
    p2.join()
    
    logging.info(f'Finished calculating data for N={N}, tau={tau}, noise={noise}')


def generate_qiskit_circuits(qubits, steps, num_circuits_per_step, noise_std=0.0, noise_method='global'):
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

def generate_single_circuit_parallel(params):
    qubits, steps, circuit_idx, betas, alphas, noisy_betas, noise_method = params

    circuit = QuantumCircuit(qubits, qubits)
    circuit.h(range(qubits))

    for step in range(steps):
        beta = betas[step, circuit_idx]
        alpha = alphas[step, circuit_idx]

        if noise_method == 'dephasing':
            circuit.rz(noisy_betas[step, circuit_idx], range(qubits))
        else:
            circuit.rz(beta, range(qubits))

        for i in range(qubits):
            j = (i + 1) % qubits
            circuit.cp(-2 * beta, i, j)

        for i in range(qubits):
            circuit.rx(alpha, i)

    dm = DensityMatrix.from_instruction(circuit)
    circuit.measure(range(qubits), range(qubits))

    return circuit, dm.data

def generate_single_circuit(steps, circuit_idx, betas, alphas, qubits):
    # Initialize the quantum circuit
    circuit = QuantumCircuit(qubits, qubits)
    circuit.h(range(qubits))  # Apply Hadamard gate to all qubits to create superposition

    # Apply parameterized gates for each step
    for step in range(steps):
        beta = betas[step, circuit_idx]
        circuit.rz(beta, range(qubits))  # Apply RZ rotation with angle beta to each qubit

        # Apply controlled-phase and RX rotations
        for i in range(qubits):
            j = (i + 1) % qubits  # Neighboring qubit index
            circuit.cp(-2 * beta, i, j)  # Controlled-phase rotation between neighboring qubits
            circuit.rx(alphas[step, circuit_idx], i)  # RX rotation with noisy alpha

    # Measure all qubits
    circuit.measure(range(qubits), range(qubits))
    return circuit

def generate_tfim_circuit(qubits, steps, num_circuits_per_step, angle_noise=0.0):
    circuits = []
    # Generate base angles and add noise
    base_angles = (np.pi / 2) * np.arange(1, steps + 1) / (steps + 1)
    base_angles = base_angles[:, np.newaxis] + angle_noise * np.random.randn(steps, num_circuits_per_step)
    betas = -np.sin(base_angles)
    alphas = -np.cos(base_angles)

    # Generate circuits without parallelism
    for circuit_idx in range(num_circuits_per_step):
        circuit = generate_single_circuit(steps, circuit_idx, betas, alphas, qubits)
        circuits.append(circuit)
    return circuits



def generate_tfim_circuits(qubits, steps_list, num_circuits_per_step, angle_noise=0.0):
    start_time = time.time()
    circuits = []

    for steps in steps_list:
        circuits.extend(generate_tfim_circuit(qubits, steps, num_circuits_per_step, angle_noise))

    # Print the total time taken for circuit generation
    print("Circuit generation time: {:.4f} seconds".format(time.time() - start_time))
    return circuits


def transpile_all_circuits(circuits, simulator):
    start_time = time.time()
    transpiled_circuits = transpile(circuits, simulator, num_processes=-1)
    print("Transpilation time: {:.4f} seconds".format(time.time() - start_time))
    return transpiled_circuits

def simulate_tfim_circuits(qubits, numshots, steps_list, num_circuits_per_step=1, damping=0.0, dephazing=0.0, depolarizing=0.0, angle_noise=0.0):
    circuits = generate_tfim_circuits(qubits, steps_list, num_circuits_per_step, angle_noise)

    start_time = time.time()
    # Noise Model
    noise_model = NoiseModel()
    if damping > 0:
        noise_model.add_all_qubit_quantum_error(amplitude_damping_error(damping), ['rx', 'rz'])
    if dephazing > 0:
        noise_model.add_all_qubit_quantum_error(phase_damping_error(dephazing), ['rx', 'rz'])
    if depolarizing > 0:
        noise_model.add_all_qubit_quantum_error(depolarizing_error(depolarizing, 1), ['rx', 'rz'])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(depolarizing, 2), ['cp'])
    print("Noise model setup time: {:.4f} seconds".format(time.time() - start_time))
    
    # Simulation
    simulator = AerSimulator(noise_model=noise_model)
    transpiled_circuits = transpile_all_circuits(circuits, simulator)
    
    start_time = time.time()
    results = simulator.run(transpiled_circuits, shots=numshots).result().get_counts()
    print("Simulation time: {:.4f} seconds".format(time.time() - start_time))
    
    # Ensure results is a list even if it's a single dictionary
    if isinstance(results, dict):
        results = [results]
    
    # Aggregation of results by number of steps
    start_time = time.time()
    counts_by_steps = {steps: defaultdict(int) for steps in steps_list}
    circuit_index = 0
    for steps in steps_list:
        for _ in range(num_circuits_per_step):
            for key, value in results[circuit_index].items():
                counts_by_steps[steps][key] += value
            circuit_index += 1
    print("Results aggregation time: {:.4f} seconds".format(time.time() - start_time))
    
    # Convert defaultdict to dict
    counts_by_steps = {steps: dict(counts) for steps, counts in counts_by_steps.items()}
    
    return counts_by_steps



def count_kinks(bitstring):
    return sum(1 for i in range(len(bitstring)) if bitstring[i] != bitstring[i-1])

def calc_kinks_probability(counts):
    kinks_count = defaultdict(int)
    for bitstring, count in counts.items():
        kinks_count[count_kinks(bitstring)] += count

    # Normalize the counts to get probabilities
    numshots = sum(kinks_count.values())
    kinks_probability = {k: v/numshots for k, v in kinks_count.items()}
    return kinks_probability

def calc_kinks_mean(kinks_probability):
    total_kinks = sum(k * v for k, v in kinks_probability.items())
    return total_kinks

def calc_kinks_variance(kinks_probability):
    mean_kinks = calc_kinks_mean(kinks_probability)
    total_variance = sum((k - mean_kinks)**2 * v for k, v in kinks_probability.items())
    return total_variance



def plot_model_comparisons(dephasing_param, qubits, total_numsots=100000, steps_max=51, num_circuits_per_step_noisy=50, damping=0.0, depolarizing=0.0, angle_noise=0.0, numeric_noise=0.08, fit_gamma='mean'):
    steps_list = [i for i in range(0, steps_max)]

    # Simulate different models
    results_dephasing, results_global_noise, results_sim = simulate_models(qubits, total_numsots, steps_list, num_circuits_per_step_noisy, dephasing_param, angle_noise)

    # Calculate numeric model values
    means_numeric, variances_numeric, ratios_numeric = calculate_numeric_model_parallel(qubits, numeric_noise, steps_max)

    # Calculate probabilities, means, variances, and ratios for each model
    means_dephasing, variances_dephasing, ratios_dephasing = calculate_model_statistics(results_dephasing, qubits)
    means_global_noisy, variances_global_noisy, ratios_global_noisy = calculate_model_statistics(results_global_noise, qubits)
    means_sim, variances_sim, ratios_sim = calculate_model_statistics(results_sim, qubits)

    # Optimize gamma for depolarizing model
    gamma, means_depolarizing, variances_depolarizing, ratios_depolarizing = optimize_and_calculate(means_dephasing, variances_dephasing, means_sim, variances_sim, fit_gamma, qubits)

    # Plot the results
    individuals = plot_individual_models(steps_list, means_dephasing, variances_dephasing, ratios_dephasing,
                           means_depolarizing, variances_depolarizing, ratios_depolarizing,
                           means_sim, variances_sim, ratios_sim,
                           means_global_noisy, variances_global_noisy, ratios_global_noisy,
                           means_numeric, variances_numeric, ratios_numeric,
                           dephasing_param, angle_noise, numeric_noise, gamma)
    comparrison = plot_combined_models(steps_list, means_dephasing, variances_dephasing, ratios_dephasing,
                         means_depolarizing, variances_depolarizing, ratios_depolarizing,
                         means_sim, variances_sim, ratios_sim,
                         means_global_noisy, variances_global_noisy, ratios_global_noisy,
                         means_numeric, variances_numeric, ratios_numeric,
                         dephasing_param, angle_noise, numeric_noise, gamma)
    
    return individuals, comparrison

def simulate_models(qubits, total_numsots, steps_list, num_circuits_per_step_noisy, dephasing_param, angle_noise):
    numshots = total_numsots // num_circuits_per_step_noisy
    results_dephasing = simulate_tfim_circuits(qubits=qubits, numshots=numshots, steps_list=steps_list, num_circuits_per_step=num_circuits_per_step_noisy, dephazing=dephasing_param)
    results_global_noise = simulate_tfim_circuits(qubits=qubits, numshots=numshots, steps_list=steps_list, num_circuits_per_step=num_circuits_per_step_noisy, angle_noise=angle_noise)
    results_sim = simulate_tfim_circuits(qubits=qubits, numshots=total_numsots, steps_list=steps_list)
    return results_dephasing, results_global_noise, results_sim

def calculate_numeric_model_parallel(qubits, numeric_noise, steps_max):
    ks = k_f(qubits)
    taus = np.linspace(0, 100, steps_max - 1)

    def calculate_for_tau(tau):
        pks_numeric = calk_noisy_pk(ks, tau, numeric_noise)
        d_vals = np.arange(0, qubits + 1, 2)
        num_probability_mass_function = calc_kink_probabilities(pks_numeric, d_vals)
        mean = np.sum(num_probability_mass_function * d_vals)
        second_moment = np.sum(num_probability_mass_function * d_vals ** 2)
        variance = second_moment - mean ** 2
        ratio = variance / (mean + epsilon)
        return tau, mean, variance, ratio  # Replace 'step' with 'tau'

    results = Parallel(n_jobs=-1)(delayed(calculate_for_tau)(tau) for tau in taus)

    means_numeric = {i: mean for i, (_, mean, _, _) in enumerate(results)}
    variances_numeric = {i: variance for i, (_, _, variance, _) in enumerate(results)}
    ratios_numeric = {i: ratio for i, (_, _, _, ratio) in enumerate(results)}

    return means_numeric, variances_numeric, ratios_numeric


def calculate_model_statistics(results, qubits):
    probs = {s: calc_kinks_probability(d) for s, d in results.items()}
    means = {s: calc_kinks_mean(d) for s, d in probs.items()}
    variances = {s: calc_kinks_variance(d) for s, d in probs.items()}
    ratios = {s: variances[s] / (means[s] + epsilon) for s in probs.keys()}
    return means, variances, ratios

def optimize_and_calculate(means_global_noisy, variances_global_noisy, means_sim, variances_sim, fit_gamma, qubits):
    # Helper function: Exponential decay
    exponential_decay = lambda steps, gamma: np.exp(-gamma * steps)

    # Helper function: Depolarizing error model
    def depolarizing_error(gamma, probs, static_prob):
        return {s: (1 - exponential_decay(s, gamma)) * static_prob + exponential_decay(s, gamma) * p for s, p in probs.items()}

    # Objective function for optimization
    def objective_function(gamma, data_noisy, data_sim, static_prob_factor):
        depolarizing_model = depolarizing_error(gamma, data_sim, qubits / static_prob_factor)
        diff = np.sum([(data_noisy[s] - depolarizing_model[s]) ** 2 for s in data_noisy.keys()])
        return diff

    # Optimize gamma
    if fit_gamma == 'mean':
        result = scipy.optimize.minimize_scalar(
            objective_function, 
            bounds=(0, 1), 
            method='bounded', 
            args=(means_global_noisy, means_sim, 2)  # Static prob factor is qubits / 2 for mean
        )
    elif fit_gamma == 'variance':
        result = scipy.optimize.minimize_scalar(
            objective_function, 
            bounds=(0, 1), 
            method='bounded', 
            args=(variances_global_noisy, variances_sim, 4)  # Static prob factor is qubits / 4 for variance
        )
    else:
        raise ValueError("fit_gamma must be either 'mean' or 'variance'")

    gamma = result.x

    # Calculate depolarizing models
    means_depolarizing = depolarizing_error(gamma, means_sim, qubits / 2)
    variances_depolarizing = depolarizing_error(gamma, variances_sim, qubits / 4)
    epsilon = 1e-12  # Small constant to avoid division by zero
    ratios_depolarizing = {s: variances_depolarizing[s] / (means_depolarizing[s] + epsilon) for s in means_sim.keys()}

    return gamma, means_depolarizing, variances_depolarizing, ratios_depolarizing

def plot_individual_models(steps_list, means_dephasing, variances_dephasing, ratios_dephasing,
                           means_depolarizing, variances_depolarizing, ratios_depolarizing,
                           means_sim, variances_sim, ratios_sim,
                           means_global_noisy, variances_global_noisy, ratios_global_noisy,
                           means_numeric, variances_numeric, ratios_numeric,
                           dephasing_param, angle_noise, numeric_noise, gamma):
    x_limits = [min(min(variances_dephasing.keys()), min(variances_depolarizing.keys()), min(variances_global_noisy.keys())), 
                max(max(variances_dephasing.keys()), max(variances_depolarizing.keys()), max(variances_global_noisy.keys()))]

    y_limits_variance = [min(min(variances_dephasing.values()), min(variances_depolarizing.values()), min(variances_sim.values()), min(variances_global_noisy.values()), min(variances_numeric.values())), 
                         max(max(variances_dephasing.values()), max(variances_depolarizing.values()), max(variances_sim.values()), max(variances_global_noisy.values()), max(variances_numeric.values()))]

    y_limits_mean = [min(min(means_dephasing.values()), min(means_depolarizing.values()), min(means_sim.values()), min(means_global_noisy.values()), min(means_numeric.values())), 
                     max(max(means_dephasing.values()), max(means_depolarizing.values()), max(means_sim.values()), max(means_global_noisy.values()), max(means_numeric.values()))]

    y_limits_ratio = [min(min(ratios_dephasing.values()), min(ratios_depolarizing.values()), min(ratios_sim.values()), min(ratios_global_noisy.values()), min(ratios_numeric.values())), 
                      max(max(ratios_dephasing.values()), max(ratios_depolarizing.values()), max(ratios_sim.values()), max(ratios_global_noisy.values()), max(ratios_numeric.values()))]

    fig, axs = plt.subplots(3, 5, figsize=(30, 15))

    models_variance = [
        (variances_dephasing, 'Variance (Dephasing, param={})'.format(dephasing_param)),
        (variances_depolarizing, 'Variance (Depolarizing, gamma={:.4f})'.format(gamma)),
        (variances_sim, 'Variance (Sim)'),
        (variances_global_noisy, 'Variance (Global Noise, param={})'.format(angle_noise)),
        (variances_numeric, 'Variance (Numeric, noise={})'.format(numeric_noise))
    ]

    for idx, (data, label) in enumerate(models_variance):
        axs[1, idx].plot(data.keys(), data.values(), 'o', label=label)
        axs[1, idx].set_title('Variance per Step')
        axs[1, idx].set_xlabel('Steps')
        axs[1, idx].set_ylabel('Variance')
        axs[1, idx].set_xlim(x_limits)
        axs[1, idx].set_ylim(y_limits_variance)
        axs[1, idx].legend()

    models_mean = [
        (means_dephasing, 'Mean (Dephasing, param={})'.format(dephasing_param)),
        (means_depolarizing, 'Mean (Depolarizing, gamma={:.4f})'.format(gamma)),
        (means_sim, 'Mean (Sim)'),
        (means_global_noisy, 'Mean (Global Noise, param={})'.format(angle_noise)),
        (means_numeric, 'Mean (Numeric, noise={})'.format(numeric_noise))
    ]

    for idx, (data, label) in enumerate(models_mean):
        axs[0, idx].plot(data.keys(), data.values(), 'o', label=label)
        axs[0, idx].set_title('Mean per Step')
        axs[0, idx].set_xlabel('Steps')
        axs[0, idx].set_ylabel('Mean')
        axs[0, idx].set_xlim(x_limits)
        axs[0, idx].set_ylim(y_limits_mean)
        axs[0, idx].legend()

    models_ratio = [
        (ratios_dephasing, 'Variance/Mean (Dephasing, param={})'.format(dephasing_param)),
        (ratios_depolarizing, 'Variance/Mean (Depolarizing, gamma={:.4f})'.format(gamma)),
        (ratios_sim, 'Variance/Mean (Sim)'),
        (ratios_global_noisy, 'Variance/Mean (Global Noise, param={})'.format(angle_noise)),
        (ratios_numeric, 'Variance/Mean (Numeric, noise={})'.format(numeric_noise))
    ]

    for idx, (data, label) in enumerate(models_ratio):
        axs[2, idx].plot(data.keys(), data.values(), 'o', label=label)
        axs[2, idx].set_title('Variance/Mean Ratio per Step')
        axs[2, idx].set_xlabel('Steps')
        axs[2, idx].set_ylabel('Variance/Mean Ratio')
        axs[2, idx].set_xlim(x_limits)
        axs[2, idx].set_ylim(y_limits_ratio)
        axs[2, idx].legend()

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()
    return fig

def plot_combined_models(steps_list, means_dephasing, variances_dephasing, ratios_dephasing,
                         means_depolarizing, variances_depolarizing, ratios_depolarizing,
                         means_sim, variances_sim, ratios_sim,
                         means_global_noisy, variances_global_noisy, ratios_global_noisy,
                         means_numeric, variances_numeric, ratios_numeric,
                         dephasing_param, angle_noise, numeric_noise, gamma):
    x_limits = [min(min(variances_dephasing.keys()), min(variances_depolarizing.keys()), min(variances_global_noisy.keys())), 
                max(max(variances_dephasing.keys()), max(variances_depolarizing.keys()), max(variances_global_noisy.keys()))]

    y_limits_variance = [min(min(variances_dephasing.values()), min(variances_depolarizing.values()), min(variances_sim.values()), min(variances_global_noisy.values()), min(variances_numeric.values())), 
                         max(max(variances_dephasing.values()), max(variances_depolarizing.values()), max(variances_sim.values()), max(variances_global_noisy.values()), max(variances_numeric.values()))]

    y_limits_mean = [min(min(means_dephasing.values()), min(means_depolarizing.values()), min(means_sim.values()), min(means_global_noisy.values()), min(means_numeric.values())), 
                     max(max(means_dephasing.values()), max(means_depolarizing.values()), max(means_sim.values()), max(means_global_noisy.values()), max(means_numeric.values()))]

    y_limits_ratio = [min(min(ratios_dephasing.values()), min(ratios_depolarizing.values()), min(ratios_sim.values()), min(ratios_global_noisy.values()), min(ratios_numeric.values())), 
                      max(max(ratios_dephasing.values()), max(ratios_depolarizing.values()), max(ratios_sim.values()), max(ratios_global_noisy.values()), max(ratios_numeric.values()))]

    fig, axs = plt.subplots(3, 1, figsize=(10, 20))
    #fig.suptitle('Comparison of Dephasing, Depolarizing, Sim, Global, and Numeric Models', fontsize=16)

    axs[1].plot(variances_dephasing.keys(), variances_dephasing.values(), 'o', label=f'Dephasing, param={dephasing_param}')
    axs[1].plot(variances_depolarizing.keys(), variances_depolarizing.values(), 'x', label=f'Depolarizing, gamma={gamma:.4f}')
    axs[1].plot(variances_sim.keys(), variances_sim.values(), '^', label='Coherent Simulation')
    axs[1].plot(variances_global_noisy.keys(), variances_global_noisy.values(), 's', label=f'Global tranverse field Noise, param={angle_noise}')
    axs[1].plot(variances_numeric.keys(), variances_numeric.values(), 'd', label=f'Global t-f Noise Numeric model, noise={numeric_noise})')
    axs[1].set_title('Variance per Step')
    axs[1].set_xlabel('Steps')
    axs[1].set_ylabel('Variance')
    axs[1].set_xlim(x_limits)
    axs[1].set_ylim(y_limits_variance)
    axs[1].legend()

    axs[0].plot(means_dephasing.keys(), means_dephasing.values(), 'o', label=f'Dephasing, param={dephasing_param}')
    axs[0].plot(means_depolarizing.keys(), means_depolarizing.values(), 'x', label=f'Depolarizing, gamma={gamma:.4f}')
    axs[0].plot(means_sim.keys(), means_sim.values(), '^', label='Coherent Simulation')
    axs[0].plot(means_global_noisy.keys(), means_global_noisy.values(), 's', label=f'Global tranverse field Noise, param={angle_noise}')
    axs[0].plot(means_numeric.keys(), means_numeric.values(), 'd', label=f'Global t-f Noise Numeric model, noise={numeric_noise})')
    axs[0].set_title('Mean per Step')
    axs[0].set_xlabel('Steps')
    axs[0].set_ylabel('Mean')
    axs[0].set_xlim(x_limits)
    axs[0].set_ylim(y_limits_mean)
    axs[0].legend()

    axs[2].plot(ratios_dephasing.keys(), ratios_dephasing.values(), 'o', label=f'Dephasing, param={dephasing_param}')
    axs[2].plot(ratios_depolarizing.keys(), ratios_depolarizing.values(), 'x', label=f'Depolarizing, gamma={gamma:.4f}')
    axs[2].plot(ratios_sim.keys(), ratios_sim.values(), '^', label='Coherent Simulation')
    axs[2].plot(ratios_global_noisy.keys(), ratios_global_noisy.values(), 's', label=f'Global tranverse field Noise, param={angle_noise}')
    axs[2].plot(ratios_numeric.keys(), ratios_numeric.values(), 'd', label=f'Global t-f Noise Numeric model, noise={numeric_noise}')
    axs[2].set_title('Variance/Mean Ratio per Step')
    axs[2].set_xlabel('Steps')
    axs[2].set_ylabel('Variance/Mean Ratio')
    axs[2].legend()
    
    return fig

def pyplot_settings():
    # Global Settings for Matplotlib
    plt.rcParams.update({
        'text.usetex': True,  # Enable LaTeX rendering for text
        'font.family': 'serif',  # Set font family
        'font.size': 18,  # General font size
        'lines.markersize': 10,  # Default marker size
        'legend.fontsize': 'small',  # Legend font size
        'legend.frameon': False,  # Remove frame around legend
        'figure.figsize': (6, 5),  # Larger figure size for legend
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

    
    
   
