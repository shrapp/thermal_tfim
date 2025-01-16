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
i already have some of the functions that calculate the data for each case bur i need to modify for that purpose.
"""

import logging
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from functions import k_f, noisy_lz_time_evolution, epsilon, calc_kink_probabilities

# Utility functions
def save_data(file_path, data, mode='a', header=False):
    """Save data to a CSV file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data.to_csv(file_path, mode=mode, header=header, index=False)

def generate_data():
    models = [
        "independent_ks_numeric",
        "qiskit_dephasing",
        "qiskit_global_noise"
    ]

    num_qubits_list = [4, 6, 8]
    depth_list_numeric = [0] + np.logspace(-4, 2, 50).tolist()
    depth_list_qiskit = [i for i in range(51)]
    noise_params = np.linspace(0, 1, 20).tolist()

    total_iterations = len(models) * len(num_qubits_list) * (len(depth_list_numeric) + len(depth_list_qiskit)) * len(noise_params)
    progress_bar = tqdm(total=total_iterations, desc="Generating Data")

    file_path = "data/new_data_from_2025.csv"
    header_written = False

    a = 0

    for model in models:
        for num_qubits in num_qubits_list:
            ks = k_f(num_qubits)
            kinks_vals = np.arange(0, num_qubits + 1, 2)
            for depth in (depth_list_numeric if 'numeric' in model else depth_list_qiskit):
                for noise_param in noise_params:
                    logging.info(f"Processing: Model={model}, Qubits={num_qubits}, Depth={depth}, Noise={noise_param}")
                    data_list = []
                    if "numeric" in model:
                        ks_solutions = noisy_lz_time_evolution(ks, depth, noise_param)
                        tensor_product = ks_solutions[0].y[:,-1].reshape(2,2)
                        for solution in ks_solutions[1:]:
                            tensor_product = np.kron(tensor_product, solution.y[:,-1].reshape(2,2))
                        density_matrix = tensor_product
                        rho2 = density_matrix @ density_matrix
                        purity = rho2.trace().real
                        pks = np.array([np.abs(np.dot(np.array([np.sin(k / 2), np.cos(k / 2)]),
                                                      np.dot(result.y[:, -1].reshape(2, 2),
                                                             np.array([[np.sin(k / 2)], [np.cos(k / 2)]]))))[0] for
                                        k, result in zip(ks, ks_solutions)])
                        pks = np.where(pks < epsilon / 100, 0, pks)
                        kinks_distribution = calc_kink_probabilities(pks, kinks_vals)
                        mean_kinks = np.sum(kinks_distribution * kinks_vals)
                        second_moment = np.sum(kinks_distribution * kinks_vals ** 2)
                        var_kinks = second_moment - mean_kinks ** 2

                        data_list.append({
                            "model": model,
                            "qubits": num_qubits,
                            "depth": depth,
                            "noise_param": noise_param,
                            "density_matrix": density_matrix,
                            "kinks_distribution": kinks_distribution,
                            "mean_kinks": mean_kinks,
                            "var_kinks": var_kinks,
                            "purity": purity
                        })

                    elif "qiskit" in model:
                        pass
                        # shots = 1024
                        # results_per_shot = simulate_qiskit_model(model, noise_model=None, num_qubits=num_qubits, depth=depth, params={"noise": noise_param}, shots=shots)
                        #
                        # density_matrices = [result["density_matrix"] for result in results_per_shot]
                        # kinks_counts = [result["kinks"] for result in results_per_shot]
                        #
                        # avg_density_matrix = np.mean(density_matrices, axis=0)
                        # mean_kinks = np.mean(kinks_counts)
                        # var_kinks = np.var(kinks_counts)
                        # purity = calculate_purity(avg_density_matrix)
                        #
                        # data_list.append({
                        #     "model": model,
                        #     "num_qubits": num_qubits,
                        #     "depth": depth,
                        #     "noise_param": noise_param,
                        #     "density_matrices": density_matrices,
                        #     "avg_density_matrix": avg_density_matrix,
                        #     "kinks_counts": kinks_counts,
                        #     "mean_kinks": mean_kinks,
                        #     "var_kinks": var_kinks,
                        #     "purity": purity
                        # })

                    df = pd.DataFrame(data_list)
                    save_data(file_path, df, mode='a', header=not header_written)
                    header_written = True
                    progress_bar.update(1)


    progress_bar.close()

if __name__ == "__main__":
    generate_data()