from finrl.meta.data_processor import DataProcessor

from crypto_env import CryptoEnv
from train import train
TICKER_LIST = ['BTCUSDT','ETHUSDT','ADAUSDT','BNBUSDT','XRPUSDT',
                'SOLUSDT','DOTUSDT', 'DOGEUSDT','AVAXUSDT','UNIUSDT']
env = CryptoEnv
TRAIN_START_DATE = '2021-09-01'
TRAIN_END_DATE = '2021-09-02'

TEST_START_DATE = '2021-09-21'
TEST_END_DATE = '2021-09-30'

INDICATORS = ['macd', 'rsi', 'cci', 'dx'] #self-defined technical indicator list is NOT supported yet

ERL_PARAMS = {"learning_rate": 2**-15,"batch_size": 2**11,
                "gamma": 0.99, "seed":312,"net_dimension": 2**9,
                "target_step": 5000, "eval_gap": 30, "eval_times": 1}

def get_data(start_date, end_date,data_source, time_interval, ticker_list, **kwargs):
    dp = DataProcessor(data_source)
    df = dp.download_data(ticker_list=ticker_list, start_date=start_date,
                          end_date=end_date, time_interval=time_interval)
    return df
a = get_data(start_date=TRAIN_START_DATE,end_date=TRAIN_END_DATE,data_source='binance', time_interval='5m', ticker_list=TICKER_LIST)
    # return dp.df_to_array()

# train(start_date=TRAIN_START_DATE,
#       end_date=TRAIN_END_DATE,
#       ticker_list=TICKER_LIST,
#       data_source='binance',
#       time_interval='5m',
#       technical_indicator_list=INDICATORS,
#       drl_lib='rllib',
#       env=env,
#       model_name='ppo',
#       current_working_dir='./test_ppo',
#       erl_params=ERL_PARAMS,
#       break_step=5e4,
#       if_vix=False
#       )