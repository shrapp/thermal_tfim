import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..discrete_numeric import (k_f, plot_mean_kinks_comparison,
                                process_qiskit_model,
                                process_tfim_momentum_trotter)


def save_data_to_csv(data, filename):
    """Save simulation data to a CSV file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert data to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


def main():
    # Define the qubit numbers and step range
    qubit_numbers = [4, 6, 8, 10]
    step_range = range(0, 31, 3)
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()
    
    # Store all data for saving
    all_data = []
    
    # Plot for each qubit number
    for i, num_qubits in enumerate(qubit_numbers):
        try:
            print(f"Processing {num_qubits} qubits...")
            ks = k_f(num_qubits)
            momentum_means = []
            qiskit_means = []
            
            # Process each step
            for steps in step_range:
                print(f"  Processing step {steps}...")
                
                # Process momentum model
                momentum_results = process_tfim_momentum_trotter(ks, steps, num_qubits, 0.0)
                momentum_means.append(momentum_results["mean_kinks"])
                
                # Process Qiskit model
                qiskit_results = process_qiskit_model(
                    num_qubits=num_qubits,
                    depth=steps,
                    noise_param=0.0,
                    noise_type='global',
                    num_circuits=2,
                    numshots=100000
                )
                qiskit_means.append(qiskit_results["mean_kinks"])
                
                # Store data for this step
                all_data.append({
                    'qubits': num_qubits,
                    'steps': steps,
                    'momentum_mean_kinks': momentum_results["mean_kinks"],
                    'momentum_var_kinks': momentum_results["var_kinks"],
                    'qiskit_mean_kinks': qiskit_results["mean_kinks"],
                    'qiskit_var_kinks': qiskit_results["var_kinks"]
                })
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(list(step_range), [i / num_qubits for i in momentum_means], 'o-',
                     label=f'Momentum (noise=0.0)')
            plt.plot(list(step_range), [i / num_qubits for i in qiskit_means], 'x-',
                     label=f'Qiskit (global, noise=0.0)')
            
            # Get the current figure and axes
            current_fig = plt.gcf()
            current_ax = plt.gca()
            
            # Copy the plot to the subplot
            for line in current_ax.get_lines():
                axs[i].plot(line.get_xdata(), line.get_ydata(), 
                           marker=line.get_marker(), 
                           linestyle=line.get_linestyle(),
                           label=line.get_label())
            
            # Set subplot title and labels
            axs[i].set_title(f'{num_qubits} Qubits')
            axs[i].set_xlabel('Number of Steps')
            axs[i].set_ylabel('Mean Kinks/N')
            axs[i].grid(True)
            axs[i].legend()
            
            # Close the temporary figure
            plt.close(current_fig)
            
        except Exception as e:
            print(f"Error plotting for {num_qubits} qubits: {str(e)}")
            # Plot empty subplot with error message
            axs[i].text(0.5, 0.5, f'Error: {str(e)}', 
                       ha='center', va='center', transform=axs[i].transAxes)
            axs[i].set_title(f'{num_qubits} Qubits (Error)')
    
    # Save all data to CSV
    data_path = os.path.join('..', 'data', 'mean_kinks_comparison_data.csv')
    save_data_to_csv(all_data, data_path)
    
    # Adjust layout and save the figure
    plt.tight_layout()
    output_path = os.path.join('..', 'mean_kinks_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved as '{output_path}'")
    print(f"Data saved to '{data_path}'")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main() 