from trading_env import TradingEnv

tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "SPY"]

env = TradingEnv(tickers=tickers, data_path="data/market_data.csv")
obs, _ = env.reset()
total_reward = 0

for step in range(200):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    if step % 50 == 0:
        print(
            f"Step: {step}, Portfolio Value: {info.get('portfolio_value', 'N/A')}")
    if done or truncated:
        obs, _ = env.reset()

print(f"Test run complete. Total Reward: {total_reward:.2f}")
