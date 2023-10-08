import gymnasium as gym
import pandas as pd
import gym_trading_env
from stable_baselines3 import A2C


def preprocess(df: pd.DataFrame):
    # Preprocess
    # df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    # df.set_index("date", inplace=True)

    # Create your features
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()
    df.dropna(inplace=True)
    # print(df)
    return df


env = gym.make(
        "MultiDatasetTradingEnv",
        dataset_dir= 'data/*.pkl',
        preprocess= preprocess,
    verbose=1
    )

# envs = gym.vector.make(
#     "MultiDatasetTradingEnv",
# dataset_dir= 'data/*.pkl',
#         preprocess= preprocess,
#     num_envs=3,
#     trading_fees=0.01 / 100,  # 0.01% per stock buy / sell (Binance fees)
#     borrow_interest_rate=0.0003 / 100,  # 0.0003% per timestep (one timestep = 1h here)
# )

# model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("a2c_cartpole")

# del model # remove to demonstrate saving and loading
#
model = A2C.load("a2c_cartpole")
#
obs = env.reset()[0]

for i in range(10_000):
    action, _states = model.predict(obs)
    obs, reward, done, info, _ = env.step(action)
    env.render()
    if i % 100 == 0:
        env.reset()

env.save_for_render(dir = "render_logs")