import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu_ts = 1.0
sigma_ts = 0.5
x_range = np.linspace(-1, 3, 1000)

pdf_ts = norm.pdf(x_range, mu_ts, sigma_ts)
cdf_ts = norm.cdf(x_range, mu_ts, sigma_ts)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_range, pdf_ts, color='#8e44ad', lw=2.5, label='Analytical PDF')
plt.axvline(-0.5, color='black', linestyle='--')
plt.axvline(2.5, color='black', linestyle='--')
plt.fill_between(x_range, pdf_ts, where=((x_range >= -0.5) & (x_range <= 2.5)), color='#9b59b6', alpha=0.3)
plt.title('Snowfall Threshold Probability Density Function')
plt.xlabel('Threshold Temperature (°C)')
plt.ylabel('Density')
plt.legend()



plt.subplot(1, 2, 2)
plt.plot(x_range, cdf_ts, color='#27ae60', lw=2.5, label='Analytical CDF')
plt.axvline(-0.5, color='black', linestyle='--')
plt.axvline(2.5, color='black', linestyle='--')
plt.axhline(0.95, color='grey', linestyle=':')
plt.title('Snowfall Threshold Cumulative Distribution Function')
plt.xlabel('Threshold Temperature (°C)')
plt.ylabel('Cumulative Probability')
plt.legend()



plt.tight_layout()
plt.show()