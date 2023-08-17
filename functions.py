import ast
import os
from multiprocessing import Lock, Pool, Process

import numpy as np
import pandas as pd
from scipy.integrate import quad, solve_ivp
from scipy.special import gammaln, logsumexp

# Parameters
hbar = 1.0   # in natural units
epsilon = 1e-8

def g(t, tau, t_max):
    return (t_max-t)/tau if tau > 0 else t_max

def H(t, tau, t_max, k):
    g_f = g(t, tau, t_max)
    return np.array([[float(g_f) - np.cos(k), np.sin(k)], [np.sin(k), -(g_f - np.cos(k))]])

def psi_dt(t, psi,  tau, t_max, k):
    return -1j * np.dot(H(t, tau, t_max, k), psi)

def rho_dt(t, rho, tau, t_max, k, w):
    h = H(t, tau, t_max, k)
    z = np.array([[1, 0], [0, -1]])
    # convert rho from vector to matrix
    rho = rho.reshape((2,2))
    return (-1j * (np.dot(h, rho) - np.dot(rho, h)) + w**2 * (np.dot(z, np.dot(rho,z)) - rho)).flatten()

def lz_time_evolution_single_k(k, t_max, tau):
    # use scipy.integrate.solve_ivp to solve the ODE H(t) psi(t) = i hbar d/dt psi(t)
    psi0 = np.array([0+0j, 1+0j])
    return  solve_ivp(fun=psi_dt, method='DOP853', t_span=(0, t_max), y0=psi0, args=(tau, t_max, k), rtol = epsilon, atol = epsilon)

def noisy_lz_time_evolution_single_k(k, t_max, tau, w):
    # use scipy.integrate.solve_ivp to solve the ODE H(t) psi(t) = i hbar d/dt psi(t)
    rho0 = np.array([[0+0j, 0+0j], [0+0j, 1+0j]])
    return  solve_ivp(fun=rho_dt, method='DOP853', t_span=(0, t_max), y0=rho0.flatten(), args=(tau, t_max, k, w), rtol = epsilon, atol = epsilon)

def lz_time_evolution(ks, tau):
    t_max = 100*tau
    # use pool to parallelize the calculation for each k
    with Pool() as pool:
        results = pool.starmap(lz_time_evolution_single_k, [(k, t_max, tau) for k in ks])
    return results

def noisy_lz_time_evolution(ks, tau, w):
    t_max = 100*tau
    # use pool to parallelize the calculation for each k
    with Pool() as pool:
        results = pool.starmap(noisy_lz_time_evolution_single_k, [(k, t_max, tau, w) for k in ks])
    return results


def calc_pk(ks, tau):
    # calculate the time evolution for each k
    results = lz_time_evolution(ks, tau)
    # calculate the probabilities for each k
    pks = np.array([np.abs(np.dot(np.array([np.sin(k/2), np.cos(k/2)]), result.y[:,-1]/np.linalg.norm(result.y[:,-1])))**2 for k ,result in zip(ks, results)])
    # remove valuse that are too small
    return  np.where(pks < epsilon/100, 0, pks)

def calk_noisy_pk(ks, tau, w):
    # calculate the time evolution for each k
    results = noisy_lz_time_evolution(ks, tau, w)
    # calculate the probabilities for each k
    pks = np.array([np.abs(np.dot(np.array([np.sin(k/2), np.cos(k/2)]), np.dot(result.y[:,-1].reshape(2,2), np.array([[np.sin(k/2)], [np.cos(k/2)]])))) for k ,result in zip(ks, results)])
    # remove valuse that are too small
    return  np.where(pks < epsilon/100, 0, pks)
     

def k_f(n, N):
    return np.pi * (2 * n + 1) / N

def p_k_analytic(tau, k):
    return np.exp(-2*np.pi*tau*(k**2))*(np.cos(k/2)**2)

def ln_P_tilda_func(theta, pks):
    return np.sum(np.log(1 + pks * (np.exp(1j * theta * 2) - 1)))

def integrand_func(theta, pks, d):
    return np.exp(ln_P_tilda_func(theta, pks) - 1j * theta * d)

def P_func(pks, d):
    integral, _ = quad(lambda theta: np.real(integrand_func(theta, pks, d)), -np.pi, np.pi, limit=10000)
    return np.abs(integral / (2 * np.pi))

def D_func(P_vals):
    N = (len(P_vals) - 1)*2
    ns = np.arange(0,N + 1,2)
    return (1 / N) * np.sum(ns * P_vals)

def beta(d):
    if d == 0:
        return np.inf
    elif d > 0.5:
        return np.nan
    else:
        return -0.5 * (np.log(d) - np.log(1 - d))

def thermal_prob(mean, N):
    beta_val = beta(mean)
    if np.isnan(beta_val):
        return np.nan
    if beta_val == np.inf:
        probabilities = np.zeros(N//2 + 1)
        probabilities[0] = 1
        return probabilities

    # Generate the range of n values
    n_values = np.arange(0, N+1, 2)

    # Compute log-probabilities
    log_unnorm_probs = (-2 * n_values * beta_val) - gammaln(n_values + 1) - gammaln(N - n_values + 1)
    
    # Normalize using logsumexp to avoid overflow/underflow
    log_norm = logsumexp(log_unnorm_probs)
    normalized_logprobs = log_unnorm_probs - log_norm
    
    # Exponentiate to get probabilities
    probabilities = np.exp(normalized_logprobs)

    return probabilities

def calculate_cumulants(probability_mass_function, values):
    # Calculate the moments
    mean = np.sum(probability_mass_function * values)
    second_moment = np.sum(probability_mass_function * values ** 2)
    third_moment = np.sum(probability_mass_function * values ** 3)
    fourth_moment = np.sum(probability_mass_function * values ** 4)

    # Calculate the cumulants using the moments
    variance = second_moment - mean ** 2
    skewness = (third_moment - 3 * mean * second_moment + 2 * mean ** 3)
    kurtosis = (fourth_moment - 4 * mean * third_moment + 6 * mean ** 2 * second_moment - 3 * mean ** 4)

    cumulants = {'mean': mean, 'second_moment': second_moment, 'third_moment': third_moment, 'fourth_moment': fourth_moment, 'variance': variance, 'skewness': skewness, 'kurtosis': kurtosis}
    return cumulants

def calc_kink_probabilities(pks, d_vals):
    with Pool() as pool:
        return np.array(pool.starmap(P_func, [(pks, d) for d in d_vals]))
    
             
def calc_data(Ns, taus, noises):
    processes = []
    lock = Lock()
    for N, tau, noise in [(N, tau, noise) for N in Ns for tau in taus for noise in noises]:
        p = Process(target=calc_data_single, args=(N, tau, lock, noise))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        

def calc_data_single(N, tau, lock, noise):
    os.nice(1) # type: ignore
    tau = round(tau, 6)
    w = round(noise, 6)
    

    with lock:
        # Load data from file if it exists, or create an empty DataFrame
        try:
            df = pd.read_csv('data.csv')
            df = df.sort_values(['N', 'tau', 'noise'])
        except FileNotFoundError:
            df = pd.DataFrame(columns=['N', 'tau', 'type', 'noise','probability', 'mean', 
                                    'second_moment', 'third_moment', 'fourth_moment',
                                    'variance', 'skewness', 'kurtosis'])

    # Check if data[N][tau][w] is in DataFrame
    if (not df.empty) and (not df[(df['N'] == N) & (df['tau'] == tau) & (df['noise'] == w)].empty):
        spesific_df = df[(df['N'] == N) & (df['tau'] == tau) & (df['noise'] == w)]
        ks = k_f(np.arange(0, N/2), N)
        d_vals = np.arange(0,N+1,2)
        if spesific_df[spesific_df['type'] == 'pk'].empty:
            pks_numeric = calk_noisy_pk(ks, tau, w)
            pk_data_df = pd.DataFrame({'probability': [pks_numeric.tolist()], 'N': [N], 'tau': [tau], 'type': ['pk'], 'noise': [w]})
            with lock:
                df = pd.read_csv('data.csv')
                df = pd.concat([df, pk_data_df])
                df = df.sort_values(['N', 'tau', 'noise'])
                df.to_csv('data.csv', index=False)
        else: 
            pks_numeric = np.array(ast.literal_eval(spesific_df[spesific_df['type'] == 'pk']['probability'].iloc[0])).flatten()
        
        if spesific_df[spesific_df['type'] == 'numeric'].empty:
            num_probability_mass_function = calc_kink_probabilities(pks_numeric, d_vals)
            num_probability_mass_function = np.where(num_probability_mass_function < epsilon, 0, num_probability_mass_function)
            num_mask = (np.roll(num_probability_mass_function, 0) == 0) & (np.roll(num_probability_mass_function, -2) == 0)
            num_mask[-1] = False
            num_mask[-2] = False
            num_probability_mass_function[np.roll(num_mask, 1)] = 0
            num_cumulants = calculate_cumulants(num_probability_mass_function, d_vals)
            
            num_data_df = pd.DataFrame({**num_cumulants, 'probability': [num_probability_mass_function.tolist()], 'N': [N], 'tau': [tau], 'type': ['numeric'], 'noise': [w]})
            with lock:
                df = pd.read_csv('data.csv')
                df = pd.concat([df, num_data_df])
                df = df.sort_values(['N', 'tau', 'noise'])
                df.to_csv('data.csv', index=False)
        else:
            num_probability_mass_function = np.array(ast.literal_eval(spesific_df[spesific_df['type'] == 'numeric']['probability'].iloc[0])).flatten()
            
        if spesific_df[spesific_df['type'] == 'thermal'].empty:
            d_num = D_func(num_probability_mass_function)
            therm_probability_mass_function = thermal_prob(d_num, N)
            therm_probability_mass_function = np.where(therm_probability_mass_function < epsilon, 0, therm_probability_mass_function)
            therm_mask = (np.roll(therm_probability_mass_function, 0) == 0) & (np.roll(therm_probability_mass_function, -2) == 0)
            therm_mask[-1] = False
            therm_mask[-2] = False
            therm_probability_mass_function[np.roll(therm_mask, 1)] = 0
            therm_cumulants = calculate_cumulants(therm_probability_mass_function, d_vals)
            
            therm_data_df = pd.DataFrame({**therm_cumulants, 'probability': [therm_probability_mass_function.tolist()], 'N': [N], 'tau': [tau], 'type': ['thermal'], 'noise': [w]})
            with lock:
                df = pd.read_csv('data.csv')
                df = pd.concat([df, therm_data_df])
                df = df.sort_values(['N', 'tau', 'noise'])
                df.to_csv('data.csv', index=False)
                
        if w==0 and spesific_df[spesific_df['type'] == 'analytic'].empty:
            analytic_probability_mass_function = pk_analitic(ks, tau)
            analytic_probability_mass_function = np.where(analytic_probability_mass_function < epsilon, 0, analytic_probability_mass_function)
            analytic_mask = (np.roll(analytic_probability_mass_function, 0) == 0) & (np.roll(analytic_probability_mass_function, -2) == 0)
            analytic_mask[-1] = False
            analytic_mask[-2] = False
            analytic_probability_mass_function[np.roll(analytic_mask, 1)] = 0
            analytic_cumulants = calculate_cumulants(analytic_probability_mass_function, d_vals)
            
            analytic_data_df = pd.DataFrame({**analytic_cumulants, 'probability': [analytic_probability_mass_function.tolist()], 'N': [N], 'tau': [tau], 'type': ['analytic'], 'noise': [w]})
            with lock:
                df = pd.read_csv('data.csv')
                df = pd.concat([df, analytic_data_df])
                df = df.sort_values(['N', 'tau', 'noise'])
                df.to_csv('data.csv', index=False)


def pk_analitic(k_values, tau):
    return np.array([np.exp(-2*np.pi*tau*(k**2))*(np.cos(k/2)**2) for k in k_values])


def get_data_in_range(N, tau_min, tau_max, noise_min=0.0, noise_max=0.0):
    # Load the DataFrame
    df = pd.read_csv('data.csv')
    df = df.sort_values(['N', 'tau', 'noise'])

    # Apply the conditions and get the corresponding data
    return df[(df['N'] == N) & (df['tau'] >= tau_min) & (df['tau'] <= tau_max) & (df['noise'] >= noise_min) & (df['noise'] <= noise_max)]


