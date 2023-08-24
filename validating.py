import matplotlib.pyplot as plt
import pandas as pd
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import (
    INDICATORS,
)
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.plot import backtest_stats, get_baseline

TRAIN_START_DATE = '2010-01-01'
TRAIN_END_DATE = '2021-10-01'
TRADE_START_DATE = '2021-10-01'
TRADE_END_DATE = '2023-03-01'

processed_full = pd.read_csv("datasets/processed_full.csv")
trade = pd.read_csv("datasets/trade.csv", index_col=0)
train = pd.read_csv("datasets/train.csv", index_col=0)
stock_dimension = len(train.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

# data_risk_indicator = processed_full[(processed_full.date < TRAIN_END_DATE) & (processed_full.date >= TRAIN_START_DATE)]
# insample_risk_indicator = data_risk_indicator.drop_duplicates(subset=['date'])
e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
env_trade, obs_trade = e_trade_gym.get_sb_env()
agent = DRLAgent(env=env_trade)

trained_moedl = agent.get_model("a2c").load("trained_models/trained_a2c")

df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
    model=trained_moedl,
    environment=e_trade_gym)
mvo_df = processed_full.sort_values(['date', 'tic'], ignore_index=True)[['date', 'tic', 'close']]

df_result_a2c = df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
# df_result_ddpg = df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0])
# df_result_td3 = df_account_value_td3.set_index(df_account_value_td3.columns[0])
# df_result_ppo = df_account_value_ppo.set_index(df_account_value_ppo.columns[0])
# df_result_sac = df_account_value_sac.set_index(df_account_value_sac.columns[0])
# df_account_value_a2c.to_csv("df_account_value_a2c.csv")
# baseline stats
print("==============Get Baseline Stats===========")
df_dji_ = get_baseline(
    ticker="^DJI",
    start=TRADE_START_DATE,
    end=TRADE_END_DATE)
stats = backtest_stats(df_dji_, value_col_name='close')
df_dji = pd.DataFrame()
df_dji['date'] = df_account_value_a2c['date']
df_dji['account_value'] = df_dji_['close'] / df_dji_['close'][0] * env_kwargs["initial_amount"]
# df_dji.to_csv("df_dji.csv")
df_dji = df_dji.set_index(df_dji.columns[0])
# df_dji.to_csv("df_dji+.csv")

result = df_result_a2c
# result = pd.merge(result, df_result_td3, left_index=True, right_index=True)
# result = pd.merge(result, df_result_ppo, left_index=True, right_index=True)
# result = pd.merge(result, df_result_sac, left_index=True, right_index=True)
# result = pd.merge(result, MVO_result, left_index=True, right_index=True)
result = pd.merge(result, df_dji, left_index=True, right_index=True)
result.columns = ['a2c', 'dji']

print("result: ", result)
result.to_csv("datasets/result.csv")
