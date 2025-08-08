import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from trading_env import TradingEnv
import torch
import numpy as np
import random

# Reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Environment
TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "SPY"]
env = TradingEnv(tickers=TICKERS, data_path="data/market_data.csv")

check_env(env, warn=True)

# Model (tiny net for local speed)
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=64,
    policy_kwargs=dict(net_arch=[64, 64]),
    verbose=1,
    device="cpu"  # CPU is usually faster for small MLP policies in PPO
)

# Short training run to verify setup
model.learn(total_timesteps=1000)

# Save
model.save("models/local_test_model")
print("Local test model saved.")
