import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu_n = 0.5
sigma_n = 0.15
x_range = np.linspace(-0.1, 1.1, 1000)

pdf_n = norm.pdf(x_range, mu_n, sigma_n)
cdf_n = norm.cdf(x_range, mu_n, sigma_n)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_range, pdf_n, color='#7f8c8d', lw=2.5, label='Analytical PDF')
plt.axvline(0, color='black', linestyle='--')
plt.axvline(1, color='black', linestyle='--')
plt.fill_between(x_range, pdf_n, where=((x_range >= 0) & (x_range <= 1)), color='#bdc3c7', alpha=0.3)
plt.title('Cloudiness Probability Density Function')
plt.xlabel('Cloud Fraction (n)')
plt.ylabel('Density')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_range, cdf_n, color='#27ae60', lw=2.5, label='Analytical CDF')
plt.axvline(0, color='black', linestyle='--')
plt.axvline(1, color='black', linestyle='--')
plt.axhline(0.95, color='grey', linestyle=':')
plt.title('Cloudiness Cumulative Distribution Function')
plt.xlabel('Cloud Fraction (n)')
plt.ylabel('Cumulative Probability')
plt.legend()

plt.tight_layout()
plt.show()