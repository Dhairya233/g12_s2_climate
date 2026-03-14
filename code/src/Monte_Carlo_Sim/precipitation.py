import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Gamma distribution parameters
k = 2.5
theta = 3.0

# Monte Carlo parameters
N = 10000

# Chebyshev bound
lower = 0
upper = 21.73

# Monte Carlo sampling
samples = np.random.gamma(k, theta, N)

# Apply bounds
bounded_samples = samples[(samples >= lower) & (samples <= upper)]

# Generate theoretical PDF
x = np.linspace(lower, upper, 1000)
pdf = gamma.pdf(x, a=k, scale=theta)

# Plot Monte Carlo histogram
plt.hist(bounded_samples, bins=50, density=True, alpha=0.6,
         color='skyblue', label='Monte Carlo Simulation')

# Plot theoretical Gamma distribution
plt.plot(x, pdf, 'r', linewidth=2.5, label='Theoretical Gamma PDF')

# Labels
plt.title("Monte Carlo Validation: Precipitation Distribution")
plt.xlabel("Precipitation (mm)")
plt.ylabel("Probability Density")
plt.legend()

plt.show()