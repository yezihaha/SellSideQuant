import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# === Black-Scholes Functions ===
def bs_delta_call(S, K, T, r, sigma, t):
    tau = T - t
    if tau <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)

def bs_call_price(S, K, T, r, sigma, t):
    tau = T - t
    if tau <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

# === Stock Path Simulation (GBM or Jump Diffusion) ===
def simulate_stock_path(S0, T, r, sigma, N, mu, lamb, mu_j, sigma_j, include_jumps, seed):
    np.random.seed(seed)
    dt = T / N
    S = np.zeros(N)
    S[0] = S0

    for t in range(1, N):
        Z = np.random.normal()
        jump_multiplier = 1.0
        if include_jumps:
            Nt = np.random.poisson(lamb * dt)
            jump_multiplier = np.random.lognormal(mean=mu_j, sigma=sigma_j, size=Nt).prod() if Nt > 0 else 1.0
        drift = (mu if include_jumps else r) - 0.5 * sigma**2
        S[t] = S[t-1] * np.exp(drift * dt + sigma * np.sqrt(dt) * Z) * jump_multiplier
        
        # Force a large jump at step t == N // 2
        if include_jumps and t == N // 2:
            S[t] *= 2.0  # Sudden 100% upward jump

    return S

# === Delta Hedging Strategy ===
def delta_hedge(S, K, T, r, sigma, t, dt, option_size):
    N = len(S)
    cash_pre = bs_call_price(S[0], K, T, r, sigma, t[0]) - bs_delta_call(S[0], K, T, r, sigma, t[0]) * S[0]
    delta_pre = bs_delta_call(S[0], K, T, r, sigma, t[0])

    portfolio_value = [0.0]
    for i in range(1, N):
        St = S[i]
        time = t[i]
        delta = bs_delta_call(St, K, T, r, sigma, time)
        call_price = bs_call_price(St, K, T, r, sigma, time)
        cash = cash_pre * np.exp(r * dt) - (delta - delta_pre) * St
        
        # 1/ Borrow cash; 2/ Long (buy) underlying; 3/ Write (sell; short) option.
        PV = cash + delta * St + option_size * call_price
        
        portfolio_value.append(PV)
        cash_pre = cash
        delta_pre = delta
    return portfolio_value

# === Parameters ===
S0 = 10
K = 10
T = 1.0
r = 0.05
sigma = 0.4
mu = 0.2 # trend of the stock price path.
lamb = 0.5
mu_j = -0.2
sigma_j = 0.1
include_jumps = True
seed = 42
option_size = -1

# === Hedging Frequencies to Compare ===
frequencies = {
    "Quarterly (N=4)": 4,
    "Monthly (N=12)": 12,
    "Daily (N=252)": 252,
    "High Freq (N=2016)": 8 * 252
}

# === Plot All in One Figure ===
plt.figure(figsize=(12, 8))

for i, (label, N) in enumerate(frequencies.items(), 1):
    t = np.linspace(0, T, N)
    dt = T / N
    S = simulate_stock_path(S0, T, r, sigma, N, mu, lamb, mu_j, sigma_j, include_jumps, seed)
    portfolio_value = delta_hedge(S, K, T, r, sigma, t, dt, option_size)
    option_values = [bs_call_price(S[j], K, T, r, sigma, t[j]) for j in range(N)]

    plt.subplot(2, 2, i)
    plt.plot(t, portfolio_value, label="Portfolio")
    plt.plot(t, option_values, label="Option (BS)", linestyle='--')
    plt.title(f"Delta Hedging: {label}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.suptitle("Delta Hedging Performance at Different Rebalancing Frequencies", fontsize=14, y=1.02)
plt.show()

# === Plot Simulated Stock Price path ===
label = "With Jumps" if include_jumps else "GBM Only"
plt.figure(figsize=(10, 4))
plt.plot(S, label=label)
plt.title("Simulated Stock Price Path")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()