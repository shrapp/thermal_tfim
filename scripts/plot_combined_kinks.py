import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_combined_kinks():
    # Read the saved data
    data_path = os.path.join('..', 'data', 'mean_kinks_comparison_data.csv')
    df = pd.read_csv(data_path)
    
    # Create a single figure
    plt.figure(figsize=(10, 6))
    
    # Get unique qubit numbers
    qubit_numbers = sorted(df['qubits'].unique())
    
    # Define colors and markers for better distinction
    colors = ['blue', 'green', 'red', 'purple']  # Simple color scheme
    markers = ['o', 's', '^', 'D']  # Different markers for each qubit number
    
    # Plot momentum data
    for i, num_qubits in enumerate(qubit_numbers):
        qubit_data = df[df['qubits'] == num_qubits]
        plt.plot(qubit_data['steps'], 
                qubit_data['momentum_mean_kinks'] / num_qubits,
                f'{markers[i]}-', 
                color=colors[i],
                label=f'Momentum N={num_qubits}',
                alpha=0.7)
    
    # Plot Qiskit data
    for i, num_qubits in enumerate(qubit_numbers):
        qubit_data = df[df['qubits'] == num_qubits]
        plt.plot(qubit_data['steps'], 
                qubit_data['qiskit_mean_kinks'] / num_qubits,
                f'{markers[i]}--', 
                color=colors[i],
                label=f'Qiskit N={num_qubits}',
                alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Number of Steps')
    plt.ylabel('Mean Kinks/N')
    plt.title('Mean Kinks Comparison Across Different System Sizes')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to accommodate the legend
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join('..', 'combined_kinks_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved as '{output_path}'")

if __name__ == '__main__':
    plot_combined_kinks() 