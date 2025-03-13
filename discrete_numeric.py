"""
I would like to generate data for several models:
The models are-
    1. the numeric independent ks model without noise.
    2. the numeric independent ks model with noise.
    3. the qiskit dephasing model.
    4. the qiskit coherent model.
    5. the qiskit global noise model.

    for each model I would like to generate data for the following cases:
        1. different number of qubits
        2. different number of steps/time
        3. different noise parameters

    the data for each case:
        1. for the numeric models:
            1. the density matrix at the end.
            2. the kinks number probability distribution at the end.
            3. the average number of kinks.
            4. the variance of the kinks number.
            5. the purity of the density matrix.

        2. for the qiskit models:
            1. the density matrix at the end for each shot.
            2. the average density matrix for each case.
            3. the number of kinks for each shot.
            4. the average number of kinks for each case.
            5. the variance of the kinks number for each case.
            6. the purity of the averaged density matrix for each case.

I would like to write a code that generates this data for each case and saves it to files.
I already have some of the functions that calculate the data for each case bur I need to modify for that purpose.
"""
import logging
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm
from functions import k_f, noisy_lz_time_evolution, epsilon, calc_kink_probabilities, calc_kinks_probability, \
    calc_kinks_mean
from qiskit.quantum_info import DensityMatrix
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from scipy.linalg import expm

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

FILE_PATH = "data/for_plot_070225.csv"


def tfim_momentum_trotter_single_k(k, tau, w):
    """
    Solve the density matrix evolution for a single momentum k using Trotterized steps, noiseless case.
    Initial state is |+><+| (density matrix for each k).
    """
    # Initial density matrix: |+><+| for each k
    rho0 = np.array([[0+0j, 0+0j], [0+0j, 1+0j]])

    # Set dt = 1, with tau = steps (so dt = tau / steps = 1 if tau = steps)
    dt = 1  # Explicitly set dt = 1

    # Initialize density matrix
    rho = rho0.copy()

    # Trotterize: Apply H_PM then H_FM for each step
    for theta in (np.pi / 2) * np.arange(1, tau + 1) / (tau + 1):
        h_param = np.cos(theta)
        J_param = np.sin(theta)

        # H_FM: Interaction part (diagonal), proportional to σ_z:
        # H_FM = 2*(h - J*cos(k))*σ_z.
        H_FM = 2 * (h_param - J_param * np.cos(k)) * np.array([[1, 0],
                                                               [0, -1]])

        # H_PM: Transverse field part (off-diagonal), proportional to σ_x:
        # H_PM = 2*J*sin(k)*σ_x.
        H_PM = 2 * J_param * np.sin(k) * np.array([[0, 1],
                                                   [1, 0]])

        # Apply Trotter step: first evolve with H_FM then with H_PM
        exp_H_FM = expm(-1j * H_FM * dt)
        rho = exp_H_FM @ rho @ exp_H_FM.conjugate().T
        assert np.allclose(rho, rho.conj().T)  # Check if rho is Hermitian
        assert np.isclose(np.trace(rho), 1)  # Check if rho is trace 1

        exp_H_PM = expm(-1j * H_PM * dt)
        rho = exp_H_PM @ rho @ exp_H_PM.conjugate().T
        assert np.allclose(rho, rho.conj().T)  # Check if rho is Hermitian
        assert np.isclose(np.trace(rho), 1)  # Check if rho is trace 1

    return rho


def tfim_momentum_trotter(ks, tau, w):
    """
    Parallelize Trotterized density matrix evolution for all k values, noiseless case with dt = 1.
    """
    # Calculate theta_n_values based on steps (tau)
    with Pool() as pool:
        results = pool.starmap(tfim_momentum_trotter_single_k, [(k, tau, w) for k in ks])
    return results


def process_tfim_momentum_trotter(ks, depth, num_qubits, noise):
    """Process the TFIM in momentum space with Trotter discretization, no noise, and dt = 1."""
    # Evolve each k's density matrix using Trotter steps with dt = 1
    solutions = tfim_momentum_trotter(ks, depth, noise)  # tau = steps for dt = 1

    # Construct full density matrix by tensor product of final density matrices
    tensor_product = solutions[0]
    for rho in solutions[1:]:
        tensor_product = np.kron(tensor_product, rho)

    density_matrix = tensor_product

    # Calculate purity
    rho2 = density_matrix @ density_matrix
    purity = np.trace(rho2).real

    # Calculate kink probabilities in momentum space
    # Assume p_k is related to magnetization or correlation in momentum space
    pks = np.array([np.abs(np.dot(np.array([np.sin(k / 2), np.cos(k / 2)]),
                                  np.dot(solution,
                                         np.array([[np.sin(k / 2)], [np.cos(k / 2)]]))))[0]
                    for k, solution in zip(ks, solutions)])
    pks = np.where(pks < epsilon / 100, 0, pks)

    # Kink values for periodic chain (0, 2, ..., L)
    kinks_vals = np.arange(0, num_qubits + 1, 2)

    # Use externally defined calc_kink_probabilities (assumed to be available)
    distribution = calc_kink_probabilities(pks, kinks_vals)
    kinks_distribution = {k: v for k, v in zip(kinks_vals, distribution)}

    # Calculate mean and variance of kinks
    mean_kinks = np.sum(k*v for k, v in kinks_distribution.items())
    second_moment = np.sum((k**2)*v for k, v in kinks_distribution.items())
    var_kinks = second_moment - mean_kinks ** 2

    return {
        "density_matrix": density_matrix,
        "kinks_distribution": kinks_distribution,
        "mean_kinks": mean_kinks,
        "var_kinks": var_kinks,
        "purity": purity
    }


def generate_single_circuit(params):
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
            circuit.rx(alpha, i)

    dm = DensityMatrix.from_instruction(circuit)
    circuit.measure(range(qubits), range(qubits))

    return circuit, dm.data


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
        results = pool.map(generate_single_circuit, params)

    circuits, density_matrices = zip(*results)
    return list(circuits), list(density_matrices)


def save_data(file_path, data, mode='a', header=False):
    """Save data to a CSV file with a specific column order."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Define the desired column order
    column_order = [
        "model", "qubits", "depth", "noise_param",
        "density_matrix", "kinks_distribution",
        "mean_kinks", "var_kinks", "purity"
    ]

    # Reorder the DataFrame columns
    data = data[column_order]

    data.to_csv(file_path, mode=mode, header=header, index=False)


def data_exists(existing_data, model, num_qubits, depth, noise_param):
    """Check if data with the given parameters already exists."""
    if existing_data.empty:
        return False

    # Ensure the data types match and round floating-point numbers
    model = str(model)
    num_qubits = int(num_qubits)
    depth = round(float(depth), 6)
    noise_param = round(float(noise_param), 6)

    return ((existing_data['model'] == model) &
            (existing_data['qubits'] == num_qubits) &
            (existing_data['depth'].round(6) == depth) &
            (existing_data['noise_param'].round(6) == noise_param)).any()


from scipy.integrate import solve_ivp

### Pauli matrices
sigX = np.array([[0, 1],
                 [1, 0]], dtype=complex)
sigZ = np.array([[1, 0],
                 [0, -1]], dtype=complex)

epsilon = 1e-9  # small tolerance


def commutator(A, B):
    return A @ B - B @ A


def g0(t):
    """
    Dimensionless ramp function: g(t) = t, where t in [0,1].
    """
    return t


def H0(t, k):
    """
    Example Hamiltonian that previously used g = t/tau. Now we just use g(t)= t.
    Adjust the form to match your original logic:
      h_z = 2 * (1 - g - g*cos(k))
      h_x = 2 * g * sin(k)
    """
    g_f = g0(t)
    h_z = 2.0 * (1 - g_f - g_f * np.cos(k))  # same formula, with g_f = t
    h_x = 2.0 * g_f * np.sin(k)
    return h_z * sigZ + h_x * sigX


def V_func(k):
    """
    Same V you used before.
    If you had dimensionful parameters, keep them as is, or rename if desired.
    """
    v_z = -2.0 * (np.cos(k) + 1)
    v_x = 2.0 * np.sin(k)
    return v_z * sigZ + v_x * sigX


def Dissipator(rho, V, w):
    """
    Example dissipator. If w is dimensionful (like a rate),
    that sets how 'fast' decoherence or dephasing happens in real time.
    """
    # Lindblad form w^2 * [ V , [ V , rho ] ]
    term1 = V @ rho @ V
    term2 = -0.5 * (V @ V @ rho)
    term3 = -0.5 * (rho @ V @ V)
    return w ** 2 * (term1 + term2 + term3)


def single_step_ode(t, rho_vec, H, w, V):
    """
    ODE for a single short interval where H and V are 'frozen'.
    H = H0(t_s, k), V = V_func(k), w is your noise/dissipation param.
    """
    rho_mat = rho_vec.reshape(2, 2)
    # Hamiltonian part: -i [H, rho]
    comm = -1j * commutator(H, rho_mat)
    # Dissipator
    dissip = Dissipator(rho_mat, V, w)
    return (comm + dissip).ravel()


def discrete_lz_time_evolution_single_k(k, w, n_steps):
    """
    Evolve a single momentum mode k from t=0 to t=1 in n_steps Trotter slices.
    The Hamiltonian/dissipator is re-evaluated at each slice's start (t_s).
    """
    # initial state
    rho0 = np.array([[0. + 0j, 0. + 0j],
                     [0. + 0j, 1. + 0j]])  # you can change as needed

    # We'll break [0,1] into n_steps intervals
    t_points = np.linspace(0, 1, n_steps + 1)
    rho_current = rho0.copy()

    for i in range(n_steps):
        t_start = t_points[i]
        t_end = t_points[i + 1]

        # Freeze H and V at t_start
        H = H0(t_start, k)
        V = V_func(k)

        # Integrate from t_start to t_end with those fixed operators
        sol = solve_ivp(
            fun=lambda t, y: single_step_ode(t, y, H, w, V),
            t_span=(t_start, t_end),
            y0=rho_current.ravel(),
            method='DOP853',  # or your preferred solver
            rtol=1e-8, atol=1e-8
        )
        rho_current = sol.y[:, -1].reshape(2, 2)

    return rho_current


def discrete_lz_time_evolution(ks, w, n_steps):
    """
    Parallelize over all k modes.
    """
    with Pool() as pool:
        results = pool.starmap(
            discrete_lz_time_evolution_single_k,
            [(k, w, n_steps) for k in ks]
        )
    return results


def discrete_process_numeric_model(ks, n_steps, w, num_qubits):
    """
    1) Solve for each momentum k,
    2) Tensor them together,
    3) Compute purity, kinks distribution, etc.
    """
    ks_solutions = discrete_lz_time_evolution(ks, w, n_steps)

    # Build the total density matrix as a tensor product
    rho_total = ks_solutions[0]
    for rho_k in ks_solutions[1:]:
        rho_total = np.kron(rho_total, rho_k)

    density_matrix = rho_total
    rho2 = density_matrix @ density_matrix
    purity = rho2.trace().real

    # Example "kinks" calculation, analogous to your code.
    # Replace with your actual routine as needed.
    pks = []
    for k, rho_k in zip(ks, ks_solutions):
        # example state vector = [ sin(k/2), cos(k/2) ]
        vec = np.array([np.sin(k / 2), np.cos(k / 2)], dtype=complex)
        amp = vec.conj().T @ (rho_k @ vec)
        pks.append(np.abs(amp))
    pks = np.array(pks)
    pks = np.where(pks < epsilon / 100, 0, pks)

    # Suppose you define 'calc_kink_probabilities' somewhere:
    kinks_vals = np.arange(0, num_qubits + 1, 2)
    kinks_distribution = calc_kink_probabilities(pks, kinks_vals)
    mean_kinks = np.sum(kinks_distribution * kinks_vals)
    second_moment = np.sum(kinks_distribution * kinks_vals ** 2)
    var_kinks = second_moment - mean_kinks ** 2

    return {
        "density_matrix": density_matrix,
        "kinks_distribution": kinks_distribution,
        "mean_kinks": mean_kinks,
        "var_kinks": var_kinks,
        "purity": purity
    }


def get_matching_rows(df, model, num_qubits, depth, noise_param):
    """Get all rows from the DataFrame that match the given parameters."""
    model = str(model)
    num_qubits = int(num_qubits)
    depth = round(float(depth), 6)
    noise_param = round(float(noise_param), 6)

    matching_rows = df[
        (df['model'] == model) &
        (df['qubits'] == num_qubits) &
        (df['depth'].round(6) == depth) &
        (df['noise_param'].round(6) == noise_param)
        ]
    return matching_rows


def generate_data():
    models = [
        "independent_ks_numeric"
        ,
        "qiskit_global_noise",
        "qiskit_dephasing"
    ]
    numshots = 100000
    num_circuits = 500
    num_qubits_list = [4]
    depth_list_numeric = [round(i, 6) for i in range(41)]
    depth_list_qiskit = [round(i, 6) for i in range(41)]
    noise_params_qiskit = [round(i, 6) for i in np.linspace(0, 1.5, 10)]
    noise_params_numeric = [round(i, 6) for i in np.linspace(0, 1.5, 10)]

    total_iterations = (len(models) * len(num_qubits_list) * max(len(noise_params_qiskit), len(noise_params_numeric))
                        * max(len(depth_list_numeric), len(depth_list_qiskit)))
    progress_bar = tqdm(total=total_iterations, desc="Generating Data")

    file_path = "data/for_plot_070225.csv"
    header_written = os.path.exists(file_path)

    if header_written:
        existing_data = pd.read_csv(file_path)
    else:
        existing_data = pd.DataFrame()

    for num_qubits in num_qubits_list:
        for model in models:
            ks = k_f(num_qubits)
            for noise_param in (noise_params_numeric if 'numeric' in model else noise_params_qiskit):
                for depth in (depth_list_numeric if 'numeric' in model else depth_list_qiskit):
                    if data_exists(existing_data, model, num_qubits, depth, noise_param):
                        logging.info(
                            f"Skipping: Model={model}, Qubits={num_qubits}, Depth={depth}, Noise={noise_param} (already exists)")
                        progress_bar.update(1)
                        continue

                    logging.info(f"Processing: Model={model}, Qubits={num_qubits}, Depth={depth}, Noise={noise_param}")
                    data_list = []
                    if "numeric" in model:
                        data = discrete_process_numeric_model(ks, depth, noise_param, num_qubits)
                        data.update({
                            "model": model,
                            "qubits": num_qubits,
                            "depth": depth,
                            "noise_param": noise_param
                        })
                        data_list.append(data)

                    elif "qiskit" in model:
                        noise_type = "dephasing" if "dephasing" in model else "global"
                        circuits, density_matrices = generate_qiskit_circuits(num_qubits, depth, num_circuits,
                                                                              noise_param, noise_type)
                        avg_density_matrix = sum(density_matrices) / num_circuits
                        rho_squared = avg_density_matrix @ avg_density_matrix
                        purity = np.trace(rho_squared).real
                        simulator = AerSimulator()
                        transpiled_circuits = transpile(circuits, simulator, num_processes=-1)
                        results = simulator.run(transpiled_circuits, shots=numshots).result().get_counts()
                        counts = {}
                        for result in results:
                            for key, value in result.items():
                                if key in counts:
                                    counts[key] += value
                                else:
                                    counts[key] = value
                        probs = calc_kinks_probability(counts)
                        mean = calc_kinks_mean(probs)
                        variances = sum((k - mean) ** 2 * v for k, v in probs.items())
                        data = {
                            "density_matrix": avg_density_matrix,
                            "kinks_distribution": probs,
                            "mean_kinks": mean,
                            "var_kinks": variances,
                            "purity": purity,
                            "model": model,
                            "qubits": num_qubits,
                            "depth": depth,
                            "noise_param": noise_param
                        }
                        data_list.append(data)

                    df = pd.DataFrame(data_list)
                    save_data(file_path, df, mode='a', header=not header_written)
                    header_written = True
                    progress_bar.update(1)

    progress_bar.close()


if __name__ == "__main__":
    generate_data()
