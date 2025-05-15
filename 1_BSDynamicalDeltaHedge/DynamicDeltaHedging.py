import numpy as np
from scipy.stats import norm

class DynamicDeltaHedge:
    def __init__(self, S0, K, T, r, sigma, mu, lamb, mu_j, sigma_j, include_jumps, seed, option_size):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.mu = mu
        self.lamb = lamb
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.include_jumps = include_jumps
        self.seed = seed
        self.option_size = option_size

    # === Black-Scholes delta for call option ===
    def bs_delta_call(self, S, K, T, r, sigma, t):
        tau = T - t
        if tau <= 0:
            return 1.0 if S > K else 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
        return norm.cdf(d1)

    # === Black-Scholes call price ===
    def bs_call_price(self, S, K, T, r, sigma, t):
        tau = T - t
        if tau <= 0:
            return max(S - K, 0.0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

    # === Simulate stock path with or without jumps ===
    def simulate_stock_path(self, N):
        np.random.seed(self.seed)
        dt = self.T / N
        S = np.zeros(N)
        S[0] = self.S0

        for t in range(1, N):
            Z = np.random.normal()
            jump_multiplier = 1.0
            if self.include_jumps:
                Nt = np.random.poisson(self.lamb * dt)
                jump_multiplier = np.random.lognormal(mean=self.mu_j, sigma=self.sigma_j, size=Nt).prod() if Nt > 0 else 1.0
            drift = (self.mu if self.include_jumps else self.r) - 0.5 * self.sigma ** 2
            S[t] = S[t - 1] * np.exp(drift * dt + self.sigma * np.sqrt(dt) * Z) * jump_multiplier

            if self.include_jumps and t == N // 2:
                S[t] *= 2.0  # large jump at midpoint

        return S

    # === Perform delta hedging ===
    def delta_hedge(self, S, t, dt):
        N = len(S)
        cash_pre = self.bs_call_price(S[0], self.K, self.T, self.r, self.sigma, t[0]) - self.bs_delta_call(S[0], self.K, self.T, self.r, self.sigma, t[0]) * S[0]
        delta_pre = self.bs_delta_call(S[0], self.K, self.T, self.r, self.sigma, t[0])

        portfolio_value = [0.0]
        for i in range(1, N):
            St = S[i]
            time = t[i]
            delta = self.bs_delta_call(St, self.K, self.T, self.r, self.sigma, time)
            call_price = self.bs_call_price(St, self.K, self.T, self.r, self.sigma, time)
            cash = cash_pre * np.exp(self.r * dt) - (delta - delta_pre) * St

            PV = cash + delta * St + self.option_size * call_price

            portfolio_value.append(PV)
            cash_pre = cash
            delta_pre = delta
        return portfolio_value
