import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_data_points():
    # Load the data from the CSV file
    file_path = 'data/new_data_from_2025.csv'
    data = pd.read_csv(file_path)

    # Iterate over each model and plot the data
    for model in data['model'].unique():
        model_data = data[data['model'] == model]

        plt.figure(figsize=(10, 6))
        plt.scatter(model_data['depth'], model_data['noise_param'], label=model)

        plt.xlabel('Depth')
        plt.ylabel('Noise Parameter')
        plt.title(f'Depth vs Noise Parameter for Model: {model}')
        plt.legend()
        plt.grid(True)
        plt.show()

def align_models():
    # Load the data from the CSV file
    file_path = 'data/new_data_from_2025.csv'
    data = pd.read_csv(file_path)

    qubits_number = 8
    noise_param_global = 0.578947
    noise_param_numeric = 0.157895

    # Filter the data for the global noise model with the selected parameters
    global_noise_data = data[(data['model'] == 'qiskit_global_noise') &
                             (data['qubits'] == qubits_number) &
                             (data['noise_param'] == noise_param_global)]

    # Find the minimum point in the global noise model plot
    min_global_index = global_noise_data['mean_kinks'].idxmin()
    min_global_depth = global_noise_data.loc[min_global_index, 'depth']
    min_global_kinks = global_noise_data.loc[min_global_index, 'mean_kinks']

    # Exclude the row where depth is 0
    filtered_global_noise_data = global_noise_data[global_noise_data['depth'] != 0]

    # Find the maximum point in the filtered global noise model plot
    max_global_index = filtered_global_noise_data['mean_kinks'].idxmax()
    max_global_depth = filtered_global_noise_data.loc[max_global_index, 'depth']
    max_global_kinks = filtered_global_noise_data.loc[max_global_index, 'mean_kinks']

    # Filter the data for the numeric model with the selected qubits number
    numeric_data = data[(data['model'] == 'independent_ks_numeric') &
                        (data['qubits'] == qubits_number)]

    # Create a list of minimum mean_kinks for every noise parameter
    min_kinks_per_noise_param = numeric_data.groupby('noise_param')['mean_kinks'].min().reset_index()

    # Find the noise parameter with the minimum mean_kinks closest to min_global_kinks
    closest_noise_param_index = (min_kinks_per_noise_param['mean_kinks'] - min_global_kinks).abs().idxmin()
    closest_noise_param = min_kinks_per_noise_param.loc[closest_noise_param_index, 'noise_param']

    numeric_noise_data = data[(data['model'] == 'independent_ks_numeric') &
                              (data['qubits'] == qubits_number) &
                              (data['noise_param'] == closest_noise_param)]

    closest_depth_to_max_global = \
        numeric_noise_data.iloc[(numeric_noise_data['mean_kinks'] - max_global_kinks).abs().argmin()]['depth']
    mean_kinks_at_closest_depth = \
    numeric_noise_data[numeric_noise_data['depth'] == closest_depth_to_max_global]['mean_kinks'].values[0]

    # Find the minimum point in the numeric model plot
    min_numeric_index = numeric_noise_data['mean_kinks'].idxmin()
    min_numeric_depth = numeric_noise_data.loc[min_numeric_index, 'depth']
    min_numeric_kinks = numeric_noise_data.loc[min_numeric_index, 'mean_kinks']

    # Create the figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot global noise model data
    global_line = ax1.plot(global_noise_data['depth'], global_noise_data['mean_kinks'],
                           'b-', label='Global Noise Model')
    ax1.set_xlabel('Global Model Circuit Depth')
    ax1.set_ylabel('Mean Kinks')

    # Create the secondary x-axis
    ax2 = ax1.twiny()

    # Plot numeric model data using the secondary x-axis
    numeric_line = ax2.plot(numeric_noise_data['depth'], numeric_noise_data['mean_kinks'],
                            'g-', label='Numeric Model')
    ax2.set_xlabel('Numeric Model Circuit Depth')

    ax1.set_xlim([0, 60])
    ax2.set_xlim([0, 100])

    # Adjust y-axis limits
    padding = 0.05
    y_min = min(min_global_kinks, min_numeric_kinks)
    y_max = max(max_global_kinks, mean_kinks_at_closest_depth)
    y_range = y_max - y_min
    ax1.set_ylim([y_min - y_range * padding, y_max + y_range * padding])

    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # Add title with parameters
    plt.title(f'Model Comparison (Qubits: {qubits_number}, Global noise: {noise_param_global:.3f}, '
              f'Numeric noise: {closest_noise_param:.3f})')

    # Adjust layout to prevent label clipping
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == '__main__':
    # Load the data from the CSV file
    file_path = 'data/new_data_from_2025.csv'
    data = pd.read_csv(file_path)

    qubits_number = 8
    noise_param_dephasing = 0.368421

    # Filter the data for the global noise model with the selected parameters
    dephasing_noise_data = data[(data['model'] == 'qiskit_dephasing') &
                             (data['qubits'] == qubits_number) &
                             (data['noise_param'] == noise_param_dephasing)]

    # Find the minimum point in the global noise model plot
    min_dephasing_index = dephasing_noise_data['mean_kinks'].idxmin()
    min_dephasing_depth = dephasing_noise_data.loc[min_dephasing_index, 'depth']
    min_dephasing_kinks = dephasing_noise_data.loc[min_dephasing_index, 'mean_kinks']

    # Exclude the row where depth is 0
    filtered_dephasing_noise_data = dephasing_noise_data[dephasing_noise_data['depth'] != 0]

    # Find the maximum point in the filtered global noise model plot
    max_dephasing_index = filtered_dephasing_noise_data['mean_kinks'].idxmax()
    max_dephasing_depth = filtered_dephasing_noise_data.loc[max_dephasing_index, 'depth']
    max_dephasing_kinks = filtered_dephasing_noise_data.loc[max_dephasing_index, 'mean_kinks']

    # Filter the data for the numeric model with the selected qubits number
    numeric_data = data[(data['model'] == 'independent_ks_numeric') &
                        (data['qubits'] == qubits_number)]

    # Create a list of minimum mean_kinks for every noise parameter
    min_kinks_per_noise_param = numeric_data.groupby('noise_param')['mean_kinks'].min().reset_index()

    # Find the noise parameter with the minimum mean_kinks closest to min_global_kinks
    closest_noise_param_index = (min_kinks_per_noise_param['mean_kinks'] - min_dephasing_kinks).abs().idxmin()
    closest_noise_param = min_kinks_per_noise_param.loc[closest_noise_param_index, 'noise_param']

    numeric_noise_data = data[(data['model'] == 'independent_ks_numeric') &
                              (data['qubits'] == qubits_number) &
                              (data['noise_param'] == closest_noise_param)]

    closest_depth_to_max_global = \
        numeric_noise_data.iloc[(numeric_noise_data['mean_kinks'] - max_dephasing_kinks).abs().argmin()]['depth']
    mean_kinks_at_closest_depth = \
    numeric_noise_data[numeric_noise_data['depth'] == closest_depth_to_max_global]['mean_kinks'].values[0]

    # Find the minimum point in the numeric model plot
    min_numeric_index = numeric_noise_data['mean_kinks'].idxmin()
    min_numeric_depth = numeric_noise_data.loc[min_numeric_index, 'depth']
    min_numeric_kinks = numeric_noise_data.loc[min_numeric_index, 'mean_kinks']

    # Create the figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot global noise model data
    global_line = ax1.plot(dephasing_noise_data['depth'], dephasing_noise_data['mean_kinks'],
                           'b-', label='Dephasing Noise Model')
    ax1.set_xlabel('Dephasing Model Circuit Depth')
    ax1.set_ylabel('Mean Kinks')

    # Create the secondary x-axis
    ax2 = ax1.twiny()

    # Plot numeric model data using the secondary x-axis
    numeric_line = ax2.plot(numeric_noise_data['depth'], numeric_noise_data['mean_kinks'],
                            'g-', label='Numeric Model')
    ax2.set_xlabel('Numeric Model Circuit Depth')

    ax1.set_xlim([0, 50])
    ax2.set_xlim([0, 55])

    # Adjust y-axis limits
    padding = 0.05
    y_min = min(min_dephasing_kinks, min_numeric_kinks)
    y_max = max(max_dephasing_kinks, mean_kinks_at_closest_depth)
    y_range = y_max - y_min
    ax1.set_ylim([y_min - y_range * padding, y_max + y_range * padding])

    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # Add title with parameters
    plt.title(f'Model Comparison (Qubits: {qubits_number}, Dephasing noise: {noise_param_dephasing:.3f}, '
              f'Numeric noise: {closest_noise_param:.3f})')

    # Adjust layout to prevent label clipping
    plt.tight_layout()

    # Show the plot
    plt.show()

    
