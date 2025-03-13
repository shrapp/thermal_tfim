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
import numpy as np
import pandas as pd
from tqdm import tqdm
from functions import k_f, noisy_lz_time_evolution, epsilon, calc_kink_probabilities, calc_kinks_probability, \
    calc_kinks_mean, calc_kinks_variance, generate_qiskit_circuits
from qiskit.quantum_info import DensityMatrix
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
import matplotlib.pyplot as plt

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



def process_numeric_model(ks, depth, noise_param, num_qubits):
    """Process the numeric model and return the data."""
    ks_solutions = noisy_lz_time_evolution(ks, depth, noise_param)
    tensor_product = ks_solutions[0].y[:, -1].reshape(2, 2)
    for solution in ks_solutions[1:]:
        tensor_product = np.kron(tensor_product, solution.y[:, -1].reshape(2, 2))
    density_matrix = tensor_product
    rho2 = density_matrix @ density_matrix
    purity = rho2.trace().real
    pks = np.array([np.abs(np.dot(np.array([np.sin(k / 2), np.cos(k / 2)]),
                                  np.dot(result.y[:, -1].reshape(2, 2),
                                         np.array([[np.sin(k / 2)], [np.cos(k / 2)]]))))[0] for
                    k, result in zip(ks, ks_solutions)])
    pks = np.where(pks < epsilon / 100, 0, pks)
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
        "independent_ks_numeric",
        "qiskit_global_noise",
        "qiskit_dephasing"
    ]
    numshots = 10000
    num_circuits = 200
    num_qubits_list = [8]
    depth_list_numeric = [round(i, 6) for i in range(0,41, 4)]
    depth_list_qiskit = [round(i, 6) for i in range(0, 41, 4)]
    noise_params_numeric = [round(x, 6) for x in np.linspace(0, 0.3, 3).tolist()]

    total_iterations = (len(models) * len(num_qubits_list) * len(noise_params_numeric)
                        * max(len(depth_list_numeric), len(depth_list_qiskit)))
    progress_bar = tqdm(total=total_iterations, desc="Generating Data")

    file_path = "data/240225.csv"
    header_written = os.path.exists(file_path)

    if header_written:
        existing_data = pd.read_csv(file_path)
    else:
        existing_data = pd.DataFrame()

    for num_qubits in num_qubits_list:
        for model in models:
            ks = k_f(num_qubits)
            for depth in (depth_list_numeric if 'numeric' in model else depth_list_qiskit):
                for noise_param in noise_params_numeric:
                    if "dephasing" in model:
                        noise_type = "dephasing"
                        noise_param = noise_param * 9
                    elif "global" in model:
                        noise_type = "global"
                        noise_param = noise_param * 2.3
                    if data_exists(existing_data, model, num_qubits, depth, noise_param):
                        logging.info(
                            f"Skipping: Model={model}, Qubits={num_qubits}, Depth={depth}, Noise={noise_param} (already exists)")
                        progress_bar.update(1)
                        continue

                    logging.info(f"Processing: Model={model}, Qubits={num_qubits}, Depth={depth}, Noise={noise_param}")
                    data_list = []
                    if "numeric" in model:
                        data = process_numeric_model(ks, depth, noise_param, num_qubits)
                        data.update({
                            "model": model,
                            "qubits": num_qubits,
                            "depth": depth,
                            "noise_param": noise_param
                        })
                        data_list.append(data)

                    elif "qiskit" in model:
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
