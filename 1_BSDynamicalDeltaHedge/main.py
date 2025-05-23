# Import libraries
import os
print(os.getcwd())

import sys
sys.path.append(r'C:\Users\Kehan\Desktop\kh\2025\GitHub\GitHubPush\SellSideQuant\1_BSDynamicalDeltaHedge\v1')
import numpy as np
import matplotlib.pyplot as plt
from DynamicDeltaHedging import DynamicDeltaHedge

########
# main #
########

# === Parameters ===
params = {
    "S0": 10,
    "K": 10,
    "T": 1.0,
    "r": 0.05,
    "sigma": 0.4,
    "mu": 0.2,
    "lamb": 0.5,
    "mu_j": -0.2,
    "sigma_j": 0.1,
    "include_jumps": False,
    "seed": 42,
    "option_size": 1,
    "gamma_hedge": False,               # <-- Enable gamma hedge
    "gamma_hedge_strike": 10,         # <-- Set strike for gamma hedge option
    "transaction_cost": False,         # <-- Enable transaction cost
    "cost_rate": 0.0008, #0.0008,               # <-- NYSE fee for banks (approx 0.08%)
    "bid_ask_sprd": False,         # <-- Enable bid ask spread
    "bid_ask_sprd_rate": 0.002,               # <-- bid ask spread for SP500 stocks
    "option_type": 'put',
}

# Create an instance of the class
ddh = DynamicDeltaHedge(**params)

# Hedging frequencies to compare
frequencies = {
    "Quarterly (N=4)": 4,
    "Monthly (N=12)": 12,
    "Daily (N=252)": 252,
    "High Freq (N=2016)": 8 * 252
}

plt.figure(figsize=(12, 8))

for i, (label, N) in enumerate(frequencies.items(), 1):
    t = np.linspace(0, params["T"], N)
    dt = params["T"] / N
    S = ddh.simulate_stock_path(N)
    portfolio_value = ddh.delta_hedge(S, t, dt)
    option_values = [ddh.bs_price(S[j], params["K"], params["T"], params["r"], params["sigma"], t[j], option_type=params["option_type"]) for j in range(N)]

    plt.subplot(2, 2, i)
    plt.plot(t, portfolio_value, label="Portfolio (Delta + Gamma Hedge)")
    plt.plot(t, option_values, label="Option (BS)", linestyle='--')
    plt.title(f"Hedging Performance: {label}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.suptitle("Delta + Gamma Hedging Performance at Different Rebalancing Frequencies", fontsize=14, y=1.02)
plt.show()

# Plot Simulated Stock Price path
label = "With Jumps" if params["include_jumps"] else "GBM Only"
plt.figure(figsize=(10, 4))
plt.plot(S, label=label)
plt.title("Simulated Stock Price Path")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()