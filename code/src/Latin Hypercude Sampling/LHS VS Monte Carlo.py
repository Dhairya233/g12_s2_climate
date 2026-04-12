import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, gamma, qmc

# Physics Model
def melt_model(Ta, alpha, n):
    S0 = 342
    Lf = 3.34e5
    rho = 1000
    dt = 86400

    Q = (1 - alpha) * S0 * (1 - n) + 15 * Ta
    melt = (Q * dt) / (Lf * rho)
    return melt

# Monte Carlo
def monte_carlo(N):
    Ta = np.clip(np.random.normal(0, 0.433, N), -1.3, 1.3)
    P = gamma.rvs(a=2.0, scale=1.25, size=N)
    alpha = np.clip(np.random.normal(0.35, 0.075, N), 0.2, 0.5)
    n = np.clip(np.random.normal(0.5, 0.15, N), 0, 1)
    Ts = np.clip(np.random.normal(1.0, 0.5, N), -0.5, 2.5)

    melt = melt_model(Ta, alpha, n)
    return melt, Ta, P, alpha, n, Ts

# LHS
def lhs(N):
    sampler = qmc.LatinHypercube(d=5)
    sample = sampler.random(N)

    Ta = np.clip(norm.ppf(sample[:,0], 0, 0.433), -1.3, 1.3)
    P = gamma.ppf(sample[:,1], a=2.0, scale=1.25)
    alpha = np.clip(norm.ppf(sample[:,2], 0.35, 0.075), 0.2, 0.5)
    n = np.clip(norm.ppf(sample[:,3], 0.5, 0.15), 0, 1)
    Ts = np.clip(norm.ppf(sample[:,4], 1.0, 0.5), -0.5, 2.5)

    melt = melt_model(Ta, alpha, n)
    return melt, Ta, P, alpha, n, Ts

N = 5000

mc_melt, *_ = monte_carlo(N)
lhs_melt, Ta, P, alpha, n, Ts = lhs(N)


# 1) VARIANCE DUEL
N_vals = [500, 1000, 2000, 3000, 5000]
mc_std = []
lhs_std = []

for n_val in N_vals:
    mc, *_ = monte_carlo(n_val)
    lh, *_ = lhs(n_val)

    mc_std.append(np.std(mc)/np.sqrt(n_val))
    lhs_std.append(np.std(lh)/np.sqrt(n_val))

plt.figure()
plt.plot(N_vals, mc_std, marker='o', label="Monte Carlo")
plt.plot(N_vals, lhs_std, marker='o', label="LHS")
plt.title("Variance Duel")
plt.xlabel("Samples")
plt.ylabel("Std Error")
plt.legend()
plt.grid()
plt.show()


# 2) SENSITIVITY ANALYSIS
df = pd.DataFrame({
    "Temp": Ta,
    "Precip": P,
    "Albedo": alpha,
    "Cloud": n,
    "Snow": Ts,
    "Melt": lhs_melt
})

corr = df.corr()["Melt"].drop("Melt")

plt.figure()
corr.plot(kind='bar')
plt.title("Sensitivity Analysis")
plt.ylabel("Correlation")
plt.grid()
plt.show()


# 3) EXTREME CASES
threshold = np.percentile(lhs_melt, 95)
extreme = df[df["Melt"] >= threshold]

plt.figure()
plt.scatter(df["Temp"], df["Albedo"], alpha=0.3, label="All")
plt.scatter(extreme["Temp"], extreme["Albedo"], color='red', label="Top 5%")
plt.xlabel("Temperature")
plt.ylabel("Albedo")
plt.title("Extreme Melt Cases")
plt.legend()
plt.grid()
plt.show()