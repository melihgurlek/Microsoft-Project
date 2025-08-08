import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, tickers, data_path, initial_cash=100000, transaction_cost_pct=0.001, max_position_pct=0.25):
        super().__init__()
        self.tickers = tickers
        self.data = pd.read_csv(data_path, header=[0, 1], index_col=0)
        self.dates = self.data.index
        self.current_step = 0
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.portfolio = np.zeros(len(tickers))
        self.transaction_cost_pct = transaction_cost_pct
        self.max_position_pct = max_position_pct  # limit per asset weight

        # Observation space: OHLCV per ticker
        n_features = 5
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(tickers) * n_features,),
            dtype=np.float32
        )
        # Action space: target weights
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(tickers),),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.portfolio = np.zeros(len(self.tickers))
        return self._get_obs(), {}

    def step(self, action):
        action_sum = np.sum(action)
        if action_sum == 0:
            weights = np.ones(len(self.tickers)) / len(self.tickers)
        else:
            weights = action / (action_sum + 1e-8)

        # Enforce max allocation per asset and renormalize
        weights = np.clip(weights, 0, self.max_position_pct)
        weights = weights / np.sum(weights)

        prices = self.data.loc[self.dates[self.current_step],
                               ('Close', slice(None))].values
        portfolio_value = self.cash + np.sum(self.portfolio * prices)

        # Transaction cost based on turnover
        prev_weights = (self.portfolio * prices) / \
            portfolio_value if portfolio_value > 0 else np.zeros(
                len(self.tickers))
        turnover = np.sum(np.abs(weights - prev_weights))
        cost = turnover * portfolio_value * self.transaction_cost_pct
        portfolio_value -= cost

        # Rebalance portfolio
        target_value = portfolio_value * weights
        self.portfolio = target_value / prices
        self.cash = portfolio_value - np.sum(self.portfolio * prices)

        self.current_step += 1
        next_prices = self.data.loc[self.dates[self.current_step],
                                    ('Close', slice(None))].values
        new_value = np.sum(self.portfolio * next_prices) + self.cash

        reward = (new_value - portfolio_value) / max(portfolio_value, 1e-8)
        done = self.current_step >= len(self.dates) - 2
        info = {"portfolio_value": new_value,
                "turnover": turnover, "transaction_cost": cost}
        return self._get_obs(), reward, done, False, info

    def _get_obs(self):
        ohlcv = []
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            values = self.data.loc[self.dates[self.current_step],
                                   (col, slice(None))].values
            ohlcv.extend(values)
        return np.array(ohlcv, dtype=np.float32)

    def render(self):
        pass
