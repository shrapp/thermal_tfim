import numpy as np
from matplotlib import pyplot as plt

from discrete_numeric import tfim_momentum_trotter_single_k
from functions import pyplot_settings

if __name__ == '__main__':
    pyplot_settings()
    ks = np.linspace(0, np.pi, 50)
    steps_l = [0, 2, 10]
    ws = [0, 0.3]  # Noiseless and noisy cases
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)  # Side-by-side, shared y-axis

    for i, w in enumerate(ws):
        ax = axs[i]
        ax.set_title('Noiseless' if w == 0 else 'Noisy ($\sigma=0.3$)')
        ax.set_xlabel(f'k mode (normalized by $\pi$)')
        ax.set_ylabel('Probability') if i == 0 else ax.set_ylabel('')  # Label only on left subplot

        for steps in steps_l:
            if steps == 0:
                # Special case for steps=0: no angles, so use exact (noiseless) or approximate
                # Assuming tfim_momentum_trotter_single_k handles steps=0 appropriately
                pks = []
                for k in ks:
                    dm = tfim_momentum_trotter_single_k(k, steps, np.array([]), np.array([]), 0)  # Empty betas/alphas
                    vec = np.array([[np.sin(k / 2)], [np.cos(k / 2)]])
                    pk = np.abs(np.dot(np.array([np.sin(k / 2), np.cos(k / 2)]), np.dot(dm, vec)))[0]
                    pks.append(pk)
            else:
                base_angles = (np.pi / 2) * np.arange(1, steps + 1) / (steps + 1)
                base_angles = base_angles[:, np.newaxis]
                noisy_base = base_angles + w * np.random.randn(steps, 1)
                betas, alphas = -np.sin(noisy_base).flatten(), -np.cos(noisy_base).flatten()
                pks = []
                for k in ks:
                    dm = tfim_momentum_trotter_single_k(k, steps, betas, alphas, 0)
                    vec = np.array([[np.sin(k / 2)], [np.cos(k / 2)]])
                    pk = np.abs(np.dot(np.array([np.sin(k / 2), np.cos(k / 2)]), np.dot(dm, vec)))[0]
                    pks.append(pk)

            ax.plot(ks / np.pi, pks, label=f'steps={steps}')

        ax.legend()
        ax.grid(True, alpha=0.3)  # Optional: light grid for readability

    plt.tight_layout()
    plt.show()
    plt.savefig("thesis_pk_plot.png", dpi=900)