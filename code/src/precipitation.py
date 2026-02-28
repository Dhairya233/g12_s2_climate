import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

shape = 2.0       
scale = 1.25      
mean = shape * scale
std = np.sqrt(shape * (scale**2))
a = 10.0 # Testing value for Markov

x = np.linspace(0, 15, 500)
pdf = gamma.pdf(x, shape, scale=scale)
cdf = gamma.cdf(x, shape, scale=scale)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, pdf, 'b-', label='Analytical PDF (Gamma)')
plt.fill_between(x, pdf, color='blue', alpha=0.1)
plt.title("Precipitation PDF")
plt.xlabel("Precipitation (mm)")
plt.ylabel("Probability Density")
plt.legend()



plt.subplot(1, 2, 2)
plt.plot(x, cdf, 'g-', label='Analytical CDF')
plt.axhline(0.95, color='black', linestyle=':', label='95th Percentile')
plt.title("Precipitation CDF")
plt.xlabel("Precipitation (mm)")
plt.ylabel("Cumulative Probability")
plt.legend()



plt.tight_layout()
plt.show()

actual_prob_above_a = 1 - gamma.cdf(a, shape, scale=scale)
markov_bound = mean / a
print(f"Mathematical Verification:")
print(f"Actual Probability P >= {a}: {actual_prob_above_a:.4f}")
print(f"Markov Max Allowed (E[X]/a): {markov_bound:.4f}")