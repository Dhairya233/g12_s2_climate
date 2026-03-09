import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu = 0.35
sigma = 0.075
x = np.linspace(0.15, 0.55, 1000)

pdf = norm.pdf(x, mu, sigma)
cdf = norm.cdf(x, mu, sigma)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, pdf, color='#d35400', lw=2.5, label='Analytical PDF')
plt.axvline(0.2, color='black', linestyle='--')
plt.axvline(0.5, color='black', linestyle='--')
plt.fill_between(x, pdf, where=((x >= 0.2) & (x <= 0.5)), color='#f1c40f', alpha=0.3)
plt.title('Ice Albedo Probability Density Function')
plt.xlabel('Albedo (α)')
plt.ylabel('Density')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, cdf, color='#27ae60', lw=2.5, label='Analytical CDF')
plt.axvline(0.2, color='black', linestyle='--')
plt.axvline(0.5, color='black', linestyle='--')
plt.axhline(0.95, color='grey', linestyle=':')
plt.title('Ice Albedo Cumulative Distribution Function')
plt.xlabel('Albedo (α)')
plt.ylabel('Cumulative Probability')
plt.legend()

plt.tight_layout()
plt.show()