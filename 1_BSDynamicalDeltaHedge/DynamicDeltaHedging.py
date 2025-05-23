import numpy as np
from scipy.stats import norm

class DynamicDeltaHedge:
    def __init__(self, S0, K, T, r, sigma, mu, lamb, mu_j, sigma_j, include_jumps, seed,
                 option_size, option_type='call', gamma_hedge=False, gamma_hedge_strike=None,
                 transaction_cost=False, cost_rate=0.001,
                 bid_ask_sprd=False, bid_ask_sprd_rate=0.002):
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
        self.option_type = option_type.lower()  # Ensure consistency
        self.gamma_hedge = gamma_hedge
        self.gamma_hedge_strike = gamma_hedge_strike if gamma_hedge_strike else K
        self.transaction_cost = transaction_cost
        self.cost_rate = cost_rate
        self.bid_ask_sprd = bid_ask_sprd
        self.bid_ask_sprd_rate = bid_ask_sprd_rate

    # === Black-Scholes Greeks ===
    def bs_d1_d2(self, S, K, T, r, sigma, t):
        tau = T - t
        if tau <= 0:
            return float('inf'), float('inf')
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        return d1, d2

    def bs_price(self, S, K, T, r, sigma, t, option_type=None):
        option_type = option_type or self.option_type
        tau = T - t
        if tau <= 0:
            return max(S - K, 0.0) if option_type == 'call' else max(K - S, 0.0)

        d1, d2 = self.bs_d1_d2(S, K, T, r, sigma, t)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        else:
            return K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def bs_delta(self, S, K, T, r, sigma, t, option_type=None):
        option_type = option_type or self.option_type
        tau = T - t
        if tau <= 0:
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        d1, _ = self.bs_d1_d2(S, K, T, r, sigma, t)
        return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

    def bs_gamma(self, S, K, T, r, sigma, t):
        tau = T - t
        if tau <= 0:
            return 0.0
        d1, _ = self.bs_d1_d2(S, K, T, r, sigma, t)
        return norm.pdf(d1) / (S * sigma * np.sqrt(tau))

    # === Stock path simulation ===
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
                jump_multiplier = np.random.lognormal(self.mu_j, self.sigma_j, Nt).prod() if Nt > 0 else 1.0
            drift = (self.mu if self.include_jumps else self.r) - 0.5 * self.sigma**2
            S[t] = S[t-1] * np.exp(drift * dt + self.sigma * np.sqrt(dt) * Z) * jump_multiplier

            if self.include_jumps and t == N // 2:
                S[t] *= 2.0  # artificial jump
        return S

    # === Delta hedging engine ===
    def delta_hedge(self, S, t, dt):
        N = len(S)
        delta_pre = self.bs_delta(S[0], self.K, self.T, self.r, self.sigma, t[0])
        option_price = self.bs_price(S[0], self.K, self.T, self.r, self.sigma, t[0])
        cash_pre = option_price - delta_pre * S[0]
        gamma_hedge_option_qty = 0.0

        portfolio_value = [0.0]

        for i in range(1, N):
            St = S[i]
            time = t[i]
            delta = self.bs_delta(St, self.K, self.T, self.r, self.sigma, time)
            option_price = self.bs_price(St, self.K, self.T, self.r, self.sigma, time)

            gamma_adjustment = 0.0
            if self.gamma_hedge:
                gamma_adjustment, gamma_hedge_option_qty = self.gamma_hedge_adjustment(St, time)

            delta_total = delta + gamma_hedge_option_qty * self.bs_delta(St, self.gamma_hedge_strike, self.T, self.r, self.sigma, time)
            delta_change = delta_total - delta_pre

            cost = 0.0
            if self.transaction_cost:
                cost += abs(delta_change * St * self.cost_rate)
            if self.bid_ask_sprd:
                cost += abs(delta_change * St * self.bid_ask_sprd_rate)

            cash = cash_pre * np.exp(self.r * dt) - delta_change * St - cost

            gamma_hedge_price = self.bs_price(St, self.gamma_hedge_strike, self.T, self.r, self.sigma, time) if self.gamma_hedge else 0.0
            PV = cash + delta_total * St + self.option_size * option_price + gamma_hedge_option_qty * gamma_hedge_price
            portfolio_value.append(PV)

            delta_pre = delta_total
            cash_pre = cash

        return portfolio_value

    # === Gamma hedge engine ===
    def gamma_hedge_adjustment(self, S, t):
        gamma_main = self.bs_gamma(S, self.K, self.T, self.r, self.sigma, t)
        gamma_hedge = self.bs_gamma(S, self.gamma_hedge_strike, self.T, self.r, self.sigma, t)
        hedge_qty = -gamma_main / gamma_hedge if gamma_hedge != 0 else 0.0
        return gamma_main, hedge_qty







