import itertools
import pathlib

import pandas as pd
from finrl import config_tickers
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR, INDICATORS
)
from finrl.main import check_and_make_directories
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


def create_features(
        start_date,
        end_date,
        ticker_list,
        indicators,
        save_dir,
        use_vix=True,
        use_turbulence=True,
):
    df = YahooDownloader(start_date=start_date,
                         end_date=end_date,
                         ticker_list=ticker_list).fetch_data()
    df.to_csv("data_example.csv")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=indicators,
        use_vix=use_vix,
        use_turbulence=False,
        user_defined_feature=use_turbulence)
    processed = fe.preprocess_data(df)
    processed.to_csv(f'{save_dir}/processed_{start_date}_{end_date}.csv', index=False)

if __name__ == "__main__":
    create_features(
        start_date='2021-10-01',
        end_date='2021-10-20',
        ticker_list=config_tickers.DOW_30_TICKER,
        indicators=INDICATORS,
        save_dir="data/",
        use_vix=False,
        use_turbulence=False,
    )