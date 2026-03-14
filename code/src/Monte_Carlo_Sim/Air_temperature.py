import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def simulate_thermal_dynamics_mc(mu_obs, sigma_est, k_coeff, n_trials=10000):
    samples = np.random.normal(loc=mu_obs, scale=sigma_est, size=n_trials)
    
    bound_delta = k_coeff * sigma_est
    psi_lower = mu_obs - bound_delta
    psi_upper = mu_obs + bound_delta
    
    return samples, psi_lower, psi_upper

def generate_comparative_analysis(mu, sigma, k, data):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Corrected 'edgecolor' parameter below
    counts, bins, _ = ax.hist(data, bins=50, density=True, alpha=0.5, 
                               color='#5dade2', edgecolor='white', label='Empirical Distribution (MC)')
    
    zeta = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    phi_analytical = (1 / (sigma * np.sqrt(2 * np.pi))) * \
                     np.exp(-0.5 * ((zeta - mu) / sigma)**2)
    
    ax.plot(zeta, phi_analytical, color='#c0392b', lw=2.5, label='Analytical PDF Model')
    
    bounds = [mu - k*sigma, mu + k*sigma]
    for b in bounds:
        ax.axvline(b, color='#2e4053', linestyle='--', lw=1.5)
    
    ax.set_title(r'Stochastic Verification: Empirical vs. Analytical Thermal States', fontsize=12)
    ax.set_xlabel(r'Temperature Deviation $\Delta T$ ($^{\circ}$C)', fontsize=10)
    ax.set_ylabel(r'Probability Density $f(\Delta T)$', fontsize=10)
    ax.legend(frameon=True, loc='upper right')
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    MU_PARAM = 1.2
    SIGMA_PARAM = 0.424
    K_COEFF = 3.0
    
    mc_data, l_bound, u_bound = simulate_thermal_dynamics_mc(MU_PARAM, SIGMA_PARAM, K_COEFF)
    generate_comparative_analysis(MU_PARAM, SIGMA_PARAM, K_COEFF, mc_data)
    
    out_of_regime = np.sum((mc_data < l_bound) | (mc_data > u_bound))
    print(f"Convergence Diagnostics:")
    print(f"Operational Bounds: [{l_bound:.4f}, {u_bound:.4f}]")
    print(f"Out-of-Bound Instances: {out_of_regime}")
    print(f"Integrity Ratio: {(1 - out_of_regime/len(mc_data))*100:.2f}%")