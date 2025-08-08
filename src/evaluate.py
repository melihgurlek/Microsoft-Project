from stable_baselines3 import PPO
from trading_env import TradingEnv
import numpy as np

# Load environment
TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "SPY"]
env = TradingEnv(tickers=TICKERS, data_path="data/market_data.csv")

# Load trained model
model = PPO.load("models/local_test_model", env=env)

obs, _ = env.reset()
total_reward = 0
portfolio_values = []

for step in range(len(env.dates) - 1):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    portfolio_values.append(info["portfolio_value"])
    if done or truncated:
        break

print(f"Evaluation complete. Total Reward: {total_reward:.4f}")
print(f"Final Portfolio Value: {portfolio_values[-1]:.2f}")
