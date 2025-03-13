import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize


def align_models(file, qubits_number):
    # 1. Load the data
    data = pd.read_csv(file)

    # 2. Filter for numeric noise
    numeric_data = data[
        (data['model'] == 'independent_ks_numeric') &
        (data['qubits'] == qubits_number)
    ]

    if numeric_data.empty:
        print("No global noise data found. Exiting.")
        return

    # run a loop over the numeric data noise parameters
    for noise_param in numeric_data['noise_param'].unique():
        numeric_noise_data = numeric_data[numeric_data['noise_param'] == noise_param]

        min_numeric_index = numeric_noise_data['mean_kinks'].idxmin()
        min_numeric_depth = numeric_noise_data.loc[min_numeric_index, 'depth']
        min_numeric_kinks = numeric_noise_data.loc[min_numeric_index, 'mean_kinks']

        global_data = data[
            (data['model'] == 'qiskit_global_noise') &
            (data['qubits'] == qubits_number)
            ]
        dephasing_data = data[
            (data['model'] == 'qiskit_dephasing') &
            (data['qubits'] == qubits_number)
            ]

        if global_data.empty or dephasing_data.empty:
            print("No global or dephasing noise data found. Exiting.")
            return

        # Find the noise parameter for global noise that is closest to the min_numeric_kinks
        global_closest_index = (global_data['mean_kinks'] - min_numeric_kinks).abs().idxmin()
        global_closest_noise_param = global_data.loc[global_closest_index, 'noise_param']

        # Find the noise parameter for dephasing noise that is closest to the min_numeric_kinks
        dephasing_closest_index = (dephasing_data['mean_kinks'] - min_numeric_kinks).abs().idxmin()
        dephasing_closest_noise_param = dephasing_data.loc[dephasing_closest_index, 'noise_param']

        # plot the data
        global_noise_data = global_data[global_data['noise_param'] == global_closest_noise_param]
        dephasing_noise_data = dephasing_data[dephasing_data['noise_param'] == dephasing_closest_noise_param]

        # 7. Create interpolation functions for mean_kinks
        dephasing_sorted = dephasing_noise_data.sort_values('depth')
        numeric_sorted   = numeric_noise_data.sort_values('depth')
        global_sorted    = global_noise_data.sort_values('depth')

        f_dephasing = interp1d(dephasing_sorted['depth'], dephasing_sorted['mean_kinks'],
                               kind='linear', fill_value='extrapolate')
        f_numeric   = interp1d(numeric_sorted['depth'], numeric_sorted['mean_kinks'],
                               kind='linear', fill_value='extrapolate')
        f_global    = interp1d(global_sorted['depth'],  global_sorted['mean_kinks'],
                               kind='linear', fill_value='extrapolate')

        # 8. Domain ranges
        dep_min_depth = dephasing_sorted['depth'].min()
        dep_max_depth = dephasing_sorted['depth'].max()

        num_min_depth = numeric_sorted['depth'].min()
        num_max_depth = numeric_sorted['depth'].max()

        glob_min_depth = global_sorted['depth'].min()
        glob_max_depth = global_sorted['depth'].max()

        # 9. Helper to find a positive scale
        def find_optimal_scale(f_source, f_target, x_source, target_min, target_max):
            """Returns the scale that best aligns f_source(x_source) with f_target(x_source * scale)."""

            def calculate_norm(scale):
                x_mapped = x_source * scale
                mask = (x_mapped >= target_min) & (x_mapped <= target_max)
                if not np.any(mask):
                    return np.inf
                y_source = f_source(x_source[mask])
                y_target = f_target(x_mapped[mask])
                return np.sqrt(np.mean((y_source - y_target)**2))  # L2 norm

            # Use L-BFGS-B to ensure scale>0
            initial_scale = 1.0
            result = minimize(
                calculate_norm,
                x0=[initial_scale],
                method='L-BFGS-B',
                bounds=[(1e-6, None)]
            )
            return result.x[0], result.fun

        # 10. Compute scale for (Dephasing → Numeric)
        x_dep = np.linspace(dep_min_depth, dep_max_depth, 50)
        scale_dep_num, cost_dep_num = find_optimal_scale(
            f_dephasing, f_numeric, x_dep, num_min_depth, num_max_depth
        )

        # 11. Compute scale for (Global → Numeric)
        x_glob = np.linspace(glob_min_depth, glob_max_depth, 50)
        scale_glob_num, cost_glob_num = find_optimal_scale(
            f_global, f_numeric, x_glob, num_min_depth, num_max_depth
        )

        # 12. Take the average of these two scales
        final_scale = (scale_dep_num + scale_glob_num) / 2.0

        # =========== Example: 3 Subplots =============
        fig, (ax_mean, ax_var, ax_purity) = plt.subplots(
            3, 1,
            figsize=(10, 12),
            sharex=True
        )
        ax_mean_2 = ax_mean.twiny()
        ax_var_2 = ax_var.twiny()
        ax_purity_2 = ax_purity.twiny()

        # ------------
        #  Top: MEAN
        # ------------
        # Normalize mean_kinks by qubits_number
        ax_mean.scatter(
            global_noise_data['depth'],
            global_noise_data['mean_kinks'] / qubits_number,
            color='b',
            label='Qiskit Global'
        )
        ax_mean.scatter(
            dephasing_noise_data['depth'],
            dephasing_noise_data['mean_kinks'] / qubits_number,
            color='r',
            label='Qiskit Dephasing'
        )
        ax_mean_2.scatter(
            numeric_noise_data['depth'],
            numeric_noise_data['mean_kinks'] / qubits_number,
            color='g',
            label="Independent K's"
        )

        ax_mean.set_ylabel("Mean Kinks / Qubits")
        ax_mean.set_title("Mean Kinks (Normalized by Qubits)")

        # Keep your x-limit alignment
        ax_mean.set_xlim([0, 50])
        ax_mean_2.set_xlim([0, 50 * final_scale])
        ax_mean_2.set_xlabel("Independent K's Depth", labelpad=10)
        ax_mean_2.xaxis.set_label_position("top")
        ax_mean_2.xaxis.tick_top()

        lines1m, labels1m = ax_mean.get_legend_handles_labels()
        lines2m, labels2m = ax_mean_2.get_legend_handles_labels()
        ax_mean.legend(lines1m + lines2m, labels1m + labels2m, loc='best')
        ax_mean.grid(True, linestyle='--', alpha=0.7)

        # ---------------
        #  Middle: VAR
        # ---------------
        # Normalize var_kinks by qubits_number
        ax_var.scatter(
            global_noise_data['depth'],
            global_noise_data['var_kinks'] / qubits_number,
            color='b',
            label='Qiskit Global'
        )
        ax_var.scatter(
            dephasing_noise_data['depth'],
            dephasing_noise_data['var_kinks'] / qubits_number,
            color='r',
            label='Qiskit Dephasing'
        )
        ax_var_2.scatter(
            numeric_noise_data['depth'],
            numeric_noise_data['var_kinks'] / qubits_number,
            color='g',
            label="Independent K's"
        )

        ax_var.set_ylabel("Variance of Kinks / Qubits")
        ax_var.set_title("Variance of Kinks (Normalized by Qubits)")

        ax_var.set_xlim([0, 50])
        ax_var_2.set_xlim([0, 50 * final_scale])
        ax_var_2.set_xlabel("Independent K's Depth", labelpad=10)
        ax_var_2.xaxis.set_label_position("top")
        ax_var_2.xaxis.tick_top()

        lines1v, labels1v = ax_var.get_legend_handles_labels()
        lines2v, labels2v = ax_var_2.get_legend_handles_labels()
        ax_var.legend(lines1v + lines2v, labels1v + labels2v, loc='best')
        ax_var.grid(True, linestyle='--', alpha=0.7)

        # -----------------
        #  Bottom: PURITY
        # -----------------
        # Plot log2 of purity
        ax_purity.scatter(
            global_noise_data['depth'],
            np.log2(global_noise_data['purity']),
            color='b',
            label='Qiskit Global'
        )
        ax_purity.scatter(
            dephasing_noise_data['depth'],
            np.log2(dephasing_noise_data['purity']),
            color='r',
            label='Qiskit Dephasing'
        )
        ax_purity_2.scatter(
            numeric_noise_data['depth'],
            np.log2(numeric_noise_data['purity']),
            color='g',
            label="Independent K's"
        )

        ax_purity.set_xlabel("Qiskit's Circuit Depth (Global/Dephasing)")
        ax_purity.set_ylabel("log2(Purity)")
        ax_purity.set_title("Purity in log2 Scale")

        ax_purity.set_xlim([0, 50])
        ax_purity_2.set_xlim([0, 50 * final_scale])
        ax_purity_2.set_xlabel("Independent K's Depth", labelpad=10)
        ax_purity_2.xaxis.set_label_position("top")
        ax_purity_2.xaxis.tick_top()

        lines1p, labels1p = ax_purity.get_legend_handles_labels()
        lines2p, labels2p = ax_purity_2.get_legend_handles_labels()
        ax_purity.legend(lines1p + lines2p, labels1p + labels2p, loc='best')
        ax_purity.grid(True, linestyle='--', alpha=0.7)

        # ---------------------------------
        # Overall figure title and show
        # ---------------------------------
        fig.suptitle(
            f"Model Comparison (Qubits: {qubits_number}, "
            f"Global noise: {global_closest_noise_param}, "
            f"Numeric: {noise_param}, "
            f"Dephasing: {dephasing_closest_noise_param})",
            fontsize=14
        )

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


if __name__ == '__main__':
    f = '/Users/rappsh/PycharmProjects/thermal_tfim/data/240225.csv'
    q = 8
    align_models(f, q)


    # for qubits in [8]:
    #     for noise in [round(x, 6) for x in np.linspace(0, 1, 20).tolist()]:
    #         align_models(qubits, noise)



