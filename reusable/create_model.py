import pandas as pd
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

def create_env(path_in:str):
    data = pd.read_csv(path_in)
    #FIXME ugly patch to work with index
    data = data.drop(columns=['Unnamed: 0'])
    col = data.columns[0]# INDEX COL, day
    data = data.rename(columns={col:'days'})
    data.set_index('days', inplace=True)

    stock_dimension = len(data.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

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
    e_train_gym = StockTradingEnv(df=data, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    return env_train

if __name__ == '__main__':
    env = create_env('datasets/full_train.csv')