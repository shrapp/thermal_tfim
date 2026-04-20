# thermal_tfim

Research repository for noisy dynamics in the transverse-field Ising model (TFIM), with two main computational viewpoints:

- a continuous-time / analytic-style workflow collected mostly in `functions.py`
- a discrete Trotterized momentum-space workflow collected mostly in `discrete_numeric.py`
- Qiskit circuit generation and simulation utilities used to compare the momentum model against sampled circuit results
- notebooks, cached data, and thesis plotting scripts that document the evolution of the project

This README is intended as a repo map. It explains what each important file or file family is for, how the folders relate to each other, which artifacts are source code versus generated outputs, and how to get started quickly.

## New contributor quick start

If you only want the shortest path into the repo, read files in this order:

1. `discrete_numeric.py` for the current main numerical workflow.
2. `functions.py` for older shared utilities and the continuous-time formulation.
3. `thesis/thesis_graphs_mean_var_fano.py` for the most polished figure-generation workflow.

In one sentence: `discrete_numeric.py` is the current engine, `functions.py` is the older toolbox, notebooks are exploratory, and `data/`, `Plots/`, and most of `thesis/` contain generated results.

## Quick setup

Create or activate a Python environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Some plotting code uses LaTeX-style Matplotlib settings, so figure generation may require a local LaTeX installation depending on which script you run.

## Common workflows

### Run a thesis figure script

Most polished figure-generation scripts live in `thesis/` and can be run directly from the repository root:

```bash
python thesis/thesis_graphs_mean_var_fano.py
python thesis/thesis_graphs_fano_to_noise.py
python thesis/thesis_graphs_models_convergence.py
python thesis/thesis_graphs_kinks_probability.py
python thesis/thesis_graphs_fano_large_system.py
python thesis/thesis_pk_plot.py
```

These scripts usually either:

- compute data and save it as `.pkl`, or
- load previously saved `.pkl` data and regenerate final figures.

### Run the main discrete numerical workflow

`discrete_numeric.py` is primarily a library-style script with reusable functions rather than a single clean CLI entry point. The usual pattern is to import the functions you need from a notebook or short driver script.

Typical examples:

```python
from discrete_numeric import get_data_momentum, plot_qiskit_ratio_and_purity

df = get_data_momentum(
    num_qubits=6,
    steps_list=[0, 2, 4, 8, 16],
    num_circuits=50,
    noise_params_list=[0.0, 0.2],
)
```

```python
from discrete_numeric import compare_qiskit_and_momentum

compare_qiskit_and_momentum(
    num_qubits=6,
    steps=30,
    num_circuits_list=[50, 100],
    noise_params_list=[0.0, 0.2, 0.6],
    num_shots=1000,
)
```

### Run exploratory notebooks

Open notebooks from the repo root with Jupyter:

```bash
jupyter notebook
```

Recommended notebook entry points:

- `notebook.ipynb` for post-processing stored results
- `LZ.ipynb` for Landau-Zener / continuous-time exploration
- `Real_hardware_Dutining.ipynb` for IBM Runtime / hardware work
- `thesis/thesis_graphs.ipynb` for thesis-figure experimentation

### Regenerate cached data and figures

Generated outputs typically appear in these locations:

- `data/` for cached density matrices and observables
- `Plots/` for dated `.svg` exports from comparison scripts
- `thesis/` for thesis-ready `.pdf`, `.png`, `.jpg`, and `.pkl` files

## Repository at a glance

The repository is organized around four layers:

1. `functions.py`: older, broad utility module for TFIM calculations, continuous-time noisy evolution, kink statistics, Qiskit helpers, and plotting helpers.
2. `discrete_numeric.py`: newer, larger implementation focused on a discrete-step momentum model, cached density matrices, observables, and model-comparison plots.
3. notebooks and one-off scripts: exploratory work, hardware experiments, and collaborator-specific comparisons.
4. generated artifacts: `data/`, `Plots/`, and most image/PDF/PKL files under `thesis/`.

## Top-level files and folders

### Core source files

#### `functions.py`

This is the original general-purpose toolbox for the project.

- Defines the core TFIM objects used in the continuous-time formulation: `H0`, `V_func`, `Dissipator`, `psi_dt`, and `rho_dt`.
- Solves noiseless and noisy Landau-Zener style evolution for each momentum mode with `solve_ivp`.
- Converts per-mode excitation probabilities `p_k` into kink-number distributions with Fourier-integral-based functions such as `ln_P_tilda_func`, `P_func`, and `calc_kink_probabilities`.
- Includes thermal reference models (`thermal_prob1`, `thermal_prob2`) and cumulant/statistics helpers.
- Contains an older Qiskit workflow for building TFIM circuits, simulating them with Aer noise models, counting kinks from shot strings, and plotting model comparisons.
- Also provides common styling through `pyplot_settings()`.

In practice, this file mixes physics utilities, data handling, simulation code, and plotting in one place. It is useful as the historical backbone of the repo and still powers some notebooks and thesis scripts.

#### `discrete_numeric.py`

This is the main newer code file for the discrete noisy TFIM workflow.

- Implements the momentum-space Trotter model with `tfim_momentum_trotter_single_k()` and `tfim_momentum_trotter()`.
- Builds full-system observables from independent momentum modes with `process_tfim_momentum_trotter()`.
- Generates corresponding Qiskit circuits with either global angle noise or local/dephasing-style noise.
- Simulates circuits, measures bitstrings, and turns them into kink statistics and purity estimates.
- Caches expensive intermediate results as density matrices and observables in `data/` using `.npz` and `.csv` files.
- Provides many plotting/comparison entry points, including momentum-only plots, Qiskit-vs-momentum comparisons, ratio/Fano-style plots, and helpers for batching many parameter sets.

If you want the current numerical workflow for the project, this is the first file to read.

#### `for_emanuele_3_7_25.py`

Standalone comparison script prepared for a specific discussion/collaboration on 2025-07-03.

- Reimplements only the subset of helpers needed for that task, rather than importing the entire project stack.
- Computes kink statistics from momentum-space density matrices.
- Builds matching Qiskit circuits and compares sampled circuit behavior against the independent-mode momentum model.
- Reads like a frozen experiment snapshot rather than reusable library code.

Use it as an example of a self-contained experiment script, not as the canonical implementation.

#### `__init__.py`

Empty package marker. It allows the repository root to behave like an importable Python package in some tooling setups.

#### `requirements.txt`

Minimal dependency list for the physics and Qiskit workflows.

- numerical stack: `numpy`, `scipy`, `pandas`
- plotting: `matplotlib`, `scienceplots`
- quantum tooling: `qiskit`, `qiskit-aer`, `qiskit_aer`
- parallel / serialization helpers: `joblib`, `dill`

There is some duplication between `qiskit_aer` and `qiskit-aer`, which reflects how the environment evolved over time.


### Notebooks at the repository root

#### `LZ.ipynb`

Notebook for Landau-Zener style derivations and numerical experiments.

- Starts from symbolic/numerical imports and defines the time-dependent TFIM mode Hamiltonian.
- Appears to explore continuous-time dynamics and probability calculations before the later discrete/Trotter workflow became central.

#### `free_fermion_code_new_nsteps_version_27_03_2022.ipynb`

Large legacy notebook for free-fermion calculations.

- Uses `qutip`, `networkx`, FFT-related tooling, and plotting.
- Looks like an older foundation notebook for diagonalization/free-fermion model definitions.
- Likely predates much of the current `functions.py` / `discrete_numeric.py` structure.

This is an archival notebook: important historically, but not the cleanest starting point for new work.

#### `from_scratch050524.ipynb`

Exploratory notebook for noisy probability calculations.

- Calls `calk_noisy_pk` from `functions.py` repeatedly.
- Tests different `tau`, `dt`, and noise choices.
- Looks like a scratchpad used while trying to understand scaling and parameter effects.

#### `notebook.ipynb`

Analysis notebook for stored tabular results.

- Imports everything from `functions.py`.
- Starts by cleaning serialized dataframe content.
- Appears aimed at post-processing previously computed data rather than generating the raw simulations themselves.

### Generated or reference files at the repository root

#### `logger.log`

Main log file written by `functions.py`. This is a generated runtime artifact, not hand-maintained source.

## Main generated-data folders

### `data/`

Large cache directory for numerical outputs. This folder contains more than 13,000 files, so the useful way to understand it is by filename convention rather than file-by-file prose.

Common file families:

- `momentum_rhos_*.npz`: cached density matrices produced by the momentum model.
- `momentum_observables_*.csv`: cached mean/variance observables derived from those density matrices.
- `rhos_N{N}_sigma{...}_step{...}_circuit{...}_k{...}.npy`: per-mode density matrices for a specific system size, noise value, Trotter step, circuit index, and momentum-mode index.
- other `.npz` containers: batched dictionaries keyed by `(steps, noise)` pairs so later runs can reuse earlier expensive computations.

The naming is parameter-centric: system size, noise strength, circuit count, Trotter step, and sometimes momentum mode are encoded directly in the filename so runs can be resumed and aggregated.

Because this folder is machine-generated, it should be thought of as a cache/results store rather than handwritten project structure.

### `Plots/`

Output directory for saved figures, mostly in `.svg` format and grouped by date.

Observed dated subfolders:

- `Plots/20250519`
- `Plots/20250529`
- `Plots/20250609`
- `Plots/20250630`
- `Plots/20250703`
- `Plots/20250717`
- `Plots/20250724`
- `Plots/20250807`

Typical filenames describe exactly what the figure contains, for example:

- `qiskit_vs_momentum_...`: direct comparison plots between sampled circuits and the momentum model.
- `momentum_ratio_noise_...`: ratio/Fano-style observables as a function of noise.
- `qiskit_ratio_and_purity_...`: joint analysis of circuit-level ratio statistics and purity.
- `Mean_kinks_...`: mean-kink curves across steps, circuits, noise values, and shot counts.

This folder is presentation output, not source code.

### `thesis/`

Focused workspace for thesis/paper-quality figure generation and the resulting exported assets.

Important source files inside `thesis/`:

#### `thesis/__init__.py`

Empty package marker for the thesis submodule.

#### `thesis/thesis_graphs.ipynb`

Notebook that explicitly states the plan for five thesis figures.

- mixes data collection, plotting, and figure design work
- acts as a notebook counterpart to the dedicated `thesis_graphs_*.py` scripts

#### `thesis/thesis_pk_plot.py`

Small script that plots momentum-mode excitation probability `p_k` versus normalized momentum for a few step counts, comparing noiseless and noisy cases side by side.

#### `thesis/thesis_graphs_mean_var_fano.py`

Main multi-graph thesis script.

- computes and stores graph data in pickles
- compares momentum predictions, global Qiskit noise, and local Qiskit noise
- produces mean-kink, variance, and Fano-factor plots versus Trotter steps

This is one of the most central thesis-generation scripts.

#### `thesis/thesis_graphs_fano_to_noise.py`

Computes and plots Fano factor as a function of noise strength for momentum, global-circuit, and local-circuit models.

#### `thesis/thesis_graphs_models_convergence.py`

Studies convergence with respect to the number of random circuits.

- compares the momentum-model Fano factor against Qiskit shot-based estimates
- plots the absolute discrepancy versus number of circuits

#### `thesis/thesis_graphs_kinks_probability.py`

Generates kink-number distribution plots for selected scenarios.

- computes full distributions rather than only first two moments
- separates noiseless and noisy cases into different output figures

#### `thesis/thesis_graphs_fano_large_system.py`

Extends the Fano-factor analysis to larger systems, emphasizing momentum-only calculations where full circuit simulation would be too expensive.

#### `thesis/graph1_2_data.pkl`, `thesis/graph3_data.pkl`, `thesis/graph4_data.pkl`, `thesis/graph5_data.pkl`, `thesis/fano_momentum_data.pkl`, `thesis/kink_distributions_data.pkl`, `thesis/ibm_results.pkl`

Serialized intermediate datasets used by the thesis plotting scripts so figures can be regenerated without recomputing every simulation from scratch.

#### Thesis figure exports

Most remaining files in `thesis/` are figure exports in `.jpg`, `.png`, or `.pdf` form. They fall into a few families:

- `mean_kinks_vs_steps_*`: normalized mean-kink figure exports
- `variance_kinks_vs_steps_*`: normalized variance figure exports
- `fano_factor_vs_steps_*`: Fano-factor versus steps plots
- `ratio_vs_noise_*`: Fano/ratio-versus-noise plots
- `diff_vs_circuits_*`: convergence-versus-circuit-count plots
- `kink_distributions*`: probability distribution figures
- `thesis_pk_plot.*`: momentum-mode `p_k` plots
- `fano_vs_steps_momentum_*`: large-system momentum-only Fano plots

For many of these, the same base figure exists in multiple formats (`.jpg`, `.png`, `.pdf`) for slides, manuscripts, and print-quality export.

## How the pieces fit together

Typical workflow in this repo looks like this:

1. choose a system size `N`, step list, noise model, and number of circuit realizations
2. use `discrete_numeric.py` to generate momentum-space density matrices and/or Qiskit circuits
3. cache raw density matrices and derived observables under `data/`
4. save comparison figures under `Plots/`
5. refine selected analyses in `thesis/` into publication-quality graphs and reusable pickle datasets
6. use notebooks for exploration, hardware checks, or ad hoc post-processing
