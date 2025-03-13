import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from generate_data import FILE_PATH


def compute_mean_L2_distance(df_ref, df_candidate):
    """
    Given two data frames (with columns 'depth' and 'mean_kinks'),
    compute the L2 norm (over the overlapping depth range) between their mean_kinks.
    The candidate data is interpolated onto the reference (numeric) depths.
    """
    # Determine the overlapping depth range
    depth_min = max(df_ref['depth'].min(), df_candidate['depth'].min())
    depth_max = min(df_ref['depth'].max(), df_candidate['depth'].max())
    if depth_max <= depth_min:
        return np.inf  # no overlap

    # Use only the reference depths that lie within the common range.
    mask = (df_ref['depth'] >= depth_min) & (df_ref['depth'] <= depth_max)
    x_ref = df_ref.loc[mask, 'depth'].values
    if len(x_ref) == 0:
        return np.inf

    # Create an interpolation function for the candidate's mean_kinks
    f_mean = interp1d(df_candidate['depth'].values,
                      df_candidate['mean_kinks'].values,
                      kind='linear', fill_value="extrapolate")

    # Compute the candidate's mean values at the reference depths
    candidate_mean = f_mean(x_ref)
    ref_mean = df_ref.loc[mask, 'mean_kinks'].values

    # Compute and return the L2 norm (Euclidean distance)
    error_mean = np.linalg.norm(ref_mean - candidate_mean)
    return error_mean


def align_models(qubits_number, noise_param_global):
    # Load the data from the CSV
    data = pd.read_csv(FILE_PATH)

    # 1. Select the numeric run using the provided noise parameter and qubits number.
    numeric_data = data[
        (data['model'] == 'independent_ks_numeric') &
        (data['qubits'] == qubits_number) &
        (data['noise_param'] == noise_param_global)
    ]
    if numeric_data.empty:
        print(f"No numeric run found for qubits={qubits_number} and noise={noise_param_global}")
        return

    # Group by depth to average replicates (if any), using only numeric columns.
    numeric_grouped = numeric_data.groupby('depth').mean(numeric_only=True).reset_index()

    # 2. For each of the other two models, choose the noise param that minimizes the L2 error (only for mean_kinks)
    candidate_models = ['qiskit_global_noise', 'qiskit_dephasing']
    chosen_curves = {}  # model name -> (best_noise_param, candidate_grouped_df)

    for model in candidate_models:
        model_data = data[(data['model'] == model) & (data['qubits'] == qubits_number)]
        if model_data.empty:
            print(f"No data for model {model} with qubits={qubits_number}")
            continue

        best_error = np.inf
        best_noise = None
        best_curve = None

        # Iterate over unique noise parameters for this candidate model.
        for candidate_noise in model_data['noise_param'].unique():
            candidate_subset = model_data[model_data['noise_param'] == candidate_noise]
            if candidate_subset.empty:
                continue

            # Group by depth in case there are multiple entries (only numeric columns)
            candidate_grouped = candidate_subset.groupby('depth').mean(numeric_only=True).reset_index()
            error = compute_mean_L2_distance(numeric_grouped, candidate_grouped)
            if error < best_error:
                best_error = error
                best_noise = candidate_noise
                best_curve = candidate_grouped

        if best_noise is not None:
            chosen_curves[model] = (best_noise, best_curve)
            print(f"Chosen noise for {model}: {best_noise:.6f} (mean L2 error = {best_error:.4f})")
        else:
            print(f"No candidate run found for model {model}")

    # 3. Plot the three graphs: mean, variance, and purity.
    fig, (ax_mean, ax_var, ax_purity) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot the numeric run (mean and var normalized by qubits; purity in log₂ scale)
    ax_mean.plot(numeric_grouped['depth'],
                 numeric_grouped['mean_kinks'] / qubits_number,
                 'g-', label=f"Numeric (noise {noise_param_global:.3f})")
    ax_var.plot(numeric_grouped['depth'],
                numeric_grouped['var_kinks'] / qubits_number,
                'g-', label=f"Numeric (noise {noise_param_global:.3f})")
    ax_purity.plot(numeric_grouped['depth'],
                   np.log2(numeric_grouped['purity']),
                   'g-', label=f"Numeric (noise {noise_param_global:.3f})")

    # Define colors for candidate models
    colors = {'qiskit_global_noise': 'b-', 'qiskit_dephasing': 'r-'}

    for model in candidate_models:
        if model not in chosen_curves:
            continue
        best_noise, candidate_curve = chosen_curves[model]
        label = f"{model} (noise {best_noise:.3f})"
        ax_mean.plot(candidate_curve['depth'],
                     candidate_curve['mean_kinks'] / qubits_number,
                     colors[model], label=label)
        ax_var.plot(candidate_curve['depth'],
                    candidate_curve['var_kinks'] / qubits_number,
                    colors[model], label=label)
        ax_purity.plot(candidate_curve['depth'],
                       np.log2(candidate_curve['purity']),
                       colors[model], label=label)

    # Set labels and titles
    ax_mean.set_ylabel("Mean Kinks / Qubits")
    ax_mean.set_title("Mean Kinks")
    ax_var.set_ylabel("Variance of Kinks / Qubits")
    ax_var.set_title("Variance of Kinks")
    ax_purity.set_ylabel("log₂(Purity)")
    ax_purity.set_title("Purity (log₂ scale)")
    ax_purity.set_xlabel("Depth")

    ax_mean.legend()
    ax_var.legend()
    ax_purity.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    for qubits in [4]:
        for noise in [round(i, 6) for i in np.linspace(0, 1.5, 10)]:
            align_models(qubits, noise)