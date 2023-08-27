
import pandas as pd
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.plot import backtest_stats, get_baseline
from create_model import create_env

env = create_env('datasets/full_train.csv')

def validate(trade_path,
             trade_end_date, trade_start_date,
             result_path):
    # FIXME: Refactor this
    env_trade, obs_trade, e_trade_gym, env_kwargs = create_env(trade_path)

    agent = DRLAgent(env=env_trade)
    trained_moedl = agent.get_model("a2c").load("trained_models/trained_a2c")

    df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
        model=trained_moedl,
        environment=e_trade_gym)
    df_result_a2c = df_account_value_a2c.set_index(df_account_value_a2c.columns[0])

    # #baseline stats
    print("==============Get Baseline Stats===========")
    df_dji_ = get_baseline(
        ticker="^DJI",
        start=trade_start_date,
        end=trade_end_date)
    stats = backtest_stats(df_dji_, value_col_name='close')
    df_dji = pd.DataFrame()
    df_dji['date'] = df_account_value_a2c['date']
    df_dji['account_value'] = df_dji_['close'] / df_dji_['close'][0] * env_kwargs["initial_amount"]

    df_dji = df_dji.set_index(df_dji.columns[0])

    result = df_result_a2c
    result = pd.merge(result, df_dji, left_index=True, right_index=True)
    result.columns = ['a2c', 'dji']

    print("result: ", result)
    result.to_csv(result_path)

if __name__ == '__main__':
    validate('datasets/full_trade.csv', '2023-03-01', '2021-10-01', 'results/a2c/results.csv')