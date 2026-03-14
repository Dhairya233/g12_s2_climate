import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu = 0.0          
sigma = 0.424     
k = 3             

lower_bound = mu - k * sigma  
upper_bound = mu + k * sigma  

x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
pdf = norm.pdf(x, mu, sigma)
cdf = norm.cdf(x, mu, sigma)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, pdf, 'r-', label='Analytical PDF')
plt.axvline(lower_bound, color='blue', linestyle='--', label=f'Lower Bound ({lower_bound:.2f})')
plt.axvline(upper_bound, color='blue', linestyle='--', label=f'Upper Bound ({upper_bound:.2f})')
plt.fill_between(x, pdf, where=(x >= lower_bound) & (x <= upper_bound), color='red', alpha=0.1)
plt.title("Air Temperature PDF & Chebyshev Bounds")
plt.xlabel("Temperature Error (°C)")
plt.ylabel("Probability Density")
plt.legend()



plt.subplot(1, 2, 2)
plt.plot(x, cdf, 'g-', label='Analytical CDF')
plt.axhline(0.95, color='black', linestyle=':', label='95th Percentile')
plt.title("Cumulative Distribution (CDF)")
plt.xlabel("Temperature Error (°C)")
plt.ylabel("Cumulative Probability")
plt.legend()



plt.tight_layout()
plt.show()

prob_outside = 1 - (norm.cdf(upper_bound, mu, sigma) - norm.cdf(lower_bound, mu, sigma))
print(f"Mathematical Verification:")
print(f"Probability outside 3-sigma: {prob_outside:.4f}")
print(f"Chebyshev Max Allowed (1/k^2): {1/(k**2):.4f}")