import numpy as np
import matplotlib.pyplot as plt
from qiskit import transpile

from discrete_numeric import process_tfim_momentum_trotter
from functions import generate_qiskit_circuits, calc_kinks_probability, calc_kinks_mean, k_f
from qiskit_aer import AerSimulator


if __name__ == "__main__":
    # --- Simulation Parameters ---
    num_qubits = 8       # Number of qubits (and momentum modes)
    depth = 40         # Trotter steps (tau)
    noise_param = 0.0      # Noiseless case
    num_circuits = 50      # Number of circuits for the Qiskit model
    numshots = 10000        # Shots for Qiskit simulation
    noise_type = "global"    # No noise

    # Define momentum values (ks). For example, use a uniform grid between 0 and pi.
    ks = k_f(num_qubits)

    # --- Run Numeric (Momentum-Space Trotter) Model ---
    # Assumes process_tfim_momentum_trotter(ks, depth, noise_param, num_qubits)
    data_numeric = process_tfim_momentum_trotter(ks, depth, num_qubits, noise_param)
    data_numeric.update({
        "model": "numeric",
        "qubits": num_qubits,
        "depth": depth,
        "noise_param": noise_param
    })

    # --- Run Qiskit Model ---
    # Assumes generate_qiskit_circuits returns a list of circuits and a list of density matrices
    circuits, density_matrices = generate_qiskit_circuits(num_qubits, depth, num_circuits, noise_param, noise_type)
    avg_density_matrix = sum(density_matrices) / num_circuits
    rho_squared = avg_density_matrix @ avg_density_matrix
    purity = np.trace(rho_squared).real

    # Run circuits on the simulator
    simulator = AerSimulator()
    transpiled_circuits = transpile(circuits, simulator, num_processes=-1)
    results = simulator.run(transpiled_circuits, shots=numshots).result().get_counts()

    # Aggregate counts
    counts = {}
    for result in results:
        for key, value in result.items():
            counts[key] = counts.get(key, 0) + value

    # Calculate kink probabilities and moments (assumes calc_kinks_probability and calc_kinks_mean are defined)
    probs = calc_kinks_probability(counts)
    mean = calc_kinks_mean(probs)
    variances = sum((k - mean) ** 2 * v for k, v in probs.items())

    data_qiskit = {
        "density_matrix": avg_density_matrix,
        "kinks_distribution": probs,
        "mean_kinks": mean,
        "var_kinks": variances,
        "purity": purity,
        "model": "qiskit",
        "qubits": num_qubits,
        "depth": depth,
        "noise_param": noise_param
    }

    # --- Plot Comparisons ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Purity Comparison
    purity_numeric = data_numeric.get("purity", np.nan)
    purity_qiskit = data_qiskit.get("purity", np.nan)
    axs[0, 0].bar(['Numeric', 'Qiskit'], [purity_numeric, purity_qiskit], color=['blue', 'orange'])
    axs[0, 0].set_title('Purity Comparison')
    axs[0, 0].set_ylabel('Purity')

    # Plot 2: Mean Kinks Comparison
    mean_numeric = data_numeric.get("mean_kinks", np.nan)
    mean_qiskit = data_qiskit.get("mean_kinks", np.nan)
    axs[0, 1].bar(['Numeric', 'Qiskit'], [mean_numeric, mean_qiskit], color=['blue', 'orange'])
    axs[0, 1].set_title('Mean Kinks Comparison')
    axs[0, 1].set_ylabel('Mean Kinks')

    # Plot 3: Kinks Variance Comparison
    var_numeric = data_numeric.get("var_kinks", np.nan)
    var_qiskit = data_qiskit.get("var_kinks", np.nan)
    axs[1, 0].bar(['Numeric', 'Qiskit'], [var_numeric, var_qiskit], color=['blue', 'orange'])
    axs[1, 0].set_title('Kinks Variance Comparison')
    axs[1, 0].set_ylabel('Variance')

    # Plot 4: Kink Distribution Comparison
    # Extract distributions (assumed to be dictionaries with kink counts as keys and probabilities as values)
    kinks_numeric = data_numeric.get("kinks_distribution", {})
    kinks_qiskit = data_qiskit.get("kinks_distribution", {})

    # Ensure kinks_distribution for numeric model is a dictionary.
    if isinstance(kinks_numeric, np.ndarray):
        kinks_numeric = {i: float(val) for i, val in enumerate(kinks_numeric)}
    if isinstance(kinks_qiskit, np.ndarray):
        kinks_qiskit = {i: float(val) for i, val in enumerate(kinks_qiskit)}

    all_kinks = sorted(set(kinks_numeric.keys()).union(kinks_qiskit.keys()))

    # Get the union of kink counts
    prob_numeric = [kinks_numeric.get(k, 0) for k in all_kinks]
    prob_qiskit = [kinks_qiskit.get(k, 0) for k in all_kinks]

    width = 0.35
    x = np.arange(len(all_kinks))
    axs[1, 1].bar(x - width/2, prob_numeric, width, label='Numeric', color='blue')
    axs[1, 1].bar(x + width/2, prob_qiskit, width, label='Qiskit', color='orange')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(all_kinks)
    axs[1, 1].set_xlabel('Kink Count')
    axs[1, 1].set_ylabel('Probability')
    axs[1, 1].set_title('Kink Distribution')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
