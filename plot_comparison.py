import matplotlib.pyplot as plt

from discrete_numeric import plot_momentum_qiskit_comparison

if __name__ == "__main__":
    # Run comparison with different noise levels
    noise_levels = [0.0, 0.05, 0.1]
    
    plt.figure(figsize=(12, 8))
    
    for noise in noise_levels:
        results = plot_momentum_qiskit_comparison(
            num_qubits=4,
            step_range=range(0, 31, 5),
            noise_param=noise,
            num_circuits=10,
            numshots=100000
        )
        
        # Plot momentum results
        plt.plot(results['steps'], [i/4 for i in results['momentum_means']], 
                'o-', label=f'Momentum (noise={noise})')
        
        # Plot Qiskit results
        plt.plot(results['steps'], [i/4 for i in results['qiskit_means']], 
                'x-', label=f'Qiskit (noise={noise})')
    
    plt.xlabel('Number of Steps')
    plt.ylabel('Mean Kinks/N')
    plt.title('Momentum vs Qiskit Comparison (4 qubits)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig('momentum_qiskit_comparison_multiple_noise.png')
    plt.close() 