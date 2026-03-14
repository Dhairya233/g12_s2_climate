import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu = 1.0
sigma = 0.5
N = 10000

lower = -0.5
upper = 2.5

samples = np.random.normal(mu, sigma, N)

bounded_samples = samples[(samples >= lower) & (samples <= upper)]

x = np.linspace(lower, upper, 1000)
pdf = norm.pdf(x, mu, sigma)

plt.hist(bounded_samples, bins=40, density=True, alpha=0.6, color='skyblue', label='Monte Carlo Simulation')

plt.plot(x, pdf, 'r', linewidth=2.5, label='Theoretical Normal PDF')
plt.title("Monte Carlo Simulation vs Theoretical Distribution\nSnowfall Threshold (Ts)")
plt.xlabel("Snowfall Threshold Temperature (°C)")
plt.ylabel("Probability Density")
plt.legend()

plt.show()