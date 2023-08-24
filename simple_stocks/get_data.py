# matplotlib.use('Agg')

import sys

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

sys.path.append("../FinRL")

from finrl import config_tickers
from finrl.main import check_and_make_directories
from finrl.meta.data_processor import DataProcessor
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS
)

check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

TRAIN_START_DATE = '2010-01-01'
TRAIN_END_DATE = '2021-10-01'
TRADE_START_DATE = '2021-10-01'
TRADE_END_DATE = '2023-03-01'

# print(config_tickers.DOW_30_TICKER)
# dp = DataProcessor(data_source='yahoofinance')
# df = dp.download_data(ticker_list=config_tickers.DOW_30_TICKER, start_date=TRAIN_START_DATE,
#                         end_date=TRADE_END_DATE, time_interval='1D')

df = YahooDownloader(start_date=TRAIN_START_DATE,
                     end_date=TRADE_END_DATE,
                     ticker_list=config_tickers.DOW_30_TICKER).fetch_data()
# fe = FeatureEngineer(
#     use_technical_indicator=True,
#     tech_indicator_list=INDICATORS,
#     use_vix=True,
#     use_turbulence=True,
#     user_defined_feature=False)
#
# processed = fe.preprocess_data(df)
#
# processed.to_csv(f'datasets/processed_{TRAIN_START_DATE}_{TRADE_END_DATE}_DOW.csv', index=False)
