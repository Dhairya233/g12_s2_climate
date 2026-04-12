import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc, norm, gamma

LATENT_HEAT_FUSION = 3.34e5
DENSITY_WATER = 1000
SOLAR_CONSTANT = 342
SENSIBLE_HEAT_COEFF = 15
SECONDS_PER_DAY = 86400

def run_lhs_glacier_model(n_samples: int = 1000) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=5)
    sample_matrix = sampler.random(n=n_samples) 

    temp_final = np.clip(norm.ppf(sample_matrix[:, 0], loc=0, scale=0.433), -1.3, 1.3)
    precip_final = gamma.ppf(sample_matrix[:, 1], a=2.0, scale=1.25)
    albedo_final = np.clip(norm.ppf(sample_matrix[:, 2], loc=0.35, scale=0.075), 0.2, 0.5)
    cloud_final = np.clip(norm.ppf(sample_matrix[:, 3], loc=0.5, scale=0.15), 0, 1)
    threshold_final = np.clip(norm.ppf(sample_matrix[:, 4], loc=1.0, scale=0.5), -0.5, 2.5)

    melt_results = []

    for i in range(n_samples):
        ta, p, alpha, n, ts = temp_final[i], precip_final[i], albedo_final[i], cloud_final[i], threshold_final[i]

        tau = 0.6 * (1 - 0.5 * n)
        q_net = (SOLAR_CONSTANT * tau * (1 - alpha)) + (SENSIBLE_HEAT_COEFF * ta)
        q_melt = max(0, q_net)

        melt_depth = (q_melt * SECONDS_PER_DAY) / (DENSITY_WATER * LATENT_HEAT_FUSION)

        if ta > ts:
            magnitude = melt_depth + (p * 0.001)
        else:
            magnitude = melt_depth * 0.8 
            
        melt_results.append(magnitude)

    return np.array(melt_results)

def main():
    n_samples = 1000
    results = run_lhs_glacier_model(n_samples=n_samples)

    expected_mean = np.mean(results)
    risk_score_95 = np.percentile(results, 95)

    plt.figure(figsize=(12, 6))
    plt.hist(results, bins=60, density=True, color='#34495e', alpha=0.7, edgecolor='white', label='LHS Melt Frequency')
    
    plt.axvline(expected_mean, color='#f1c40f', linestyle='--', linewidth=2, label=f'Expected Mean: {expected_mean:.4f}m')
    plt.axvline(risk_score_95, color='#e74c3c', linestyle='-', linewidth=2, label=f'95th Percentile Risk: {risk_score_95:.4f}m')

    plt.title(f'Probabilistic Melt Distribution Analysis (LHS, n={n_samples})', fontsize=14)
    plt.xlabel('Seasonal Melt Magnitude (meters)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.2)
    plt.show()

    print(f"Analysis Results:")
    print(f"LHS Expected Melt Magnitude: {expected_mean:.4f} meters")
    print(f"95th Percentile Risk Score: {risk_score_95:.4f} meters")

if __name__ == "__main__":
    main()