from stable_baselines3 import PPO
import gymnasium as gym
import optuna
from gymnasium import register
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from custom_env import TradingEnv
from trading_env_gym_custom.evaluation import evaluate_model
from trading_env_gym_custom.features_eng import get_tvt, simple_reward

print(TradingEnv)


def create_objective(env):
    def objective(trial):
        # Hyperparameters to tune
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
        # Add other hyperparameters you want to tune here

        # Set up the RL model with suggested hyperparameters
        model = PPO('MlpPolicy', env, learning_rate=learning_rate, batch_size=batch_size, verbose=0)

        # Train your model
        model.learn(total_timesteps=16000)

        # Evaluate the trained model (you'll need to define evaluate_model yourself)
        rewards = evaluate_model(model, env)

        return rewards  # Optuna seeks to maximize this value

    return objective


def optimize_hyperparameters(data_path: str, n_trials=10):
    data, _, _ = get_tvt(data_path)
    env = Monitor(gym.make("TradingEnv",
                           name="BTCUSD",
                           df=data,
                           positions=[-1, 0, 1],
                           trading_fees=0,
                           borrow_interest_rate=0,
                           reward_function=simple_reward
                           ))

    objective = create_objective(env)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_hyperparams = study.best_params
    print(f"Optimal hyperparameters: {best_hyperparams}")
