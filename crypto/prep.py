import itertools
import pathlib

import pandas as pd
from finrl import config_tickers
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS
)
from finrl.main import check_and_make_directories
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
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
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=indicators,
        use_vix=use_vix,
        use_turbulence=True,
        user_defined_feature=use_turbulence)
    processed = fe.preprocess_data(df)
    processed.to_csv(f'{save_dir}/processed_{TRAIN_START_DATE}_{TRADE_END_DATE}_DOW.csv', index=False)


def clean_data(path: str, output: str):
    processed = pd.read_csv(path)
    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date', 'tic'])

    processed_full = processed_full.fillna(0)

    processed_full.to_csv(f"datasets/{output}.csv")


if __name__ == "__main__":


    root = pathlib.Path(__file__).parent.parent
    #
    dirs = [root/DATA_SAVE_DIR, root/TRAINED_MODEL_DIR, root/TENSORBOARD_LOG_DIR, root/RESULTS_DIR]

    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    #
    TRAIN_START_DATE = '2010-01-01'
    TRAIN_END_DATE = '2021-10-01'
    TRADE_START_DATE = '2021-10-01'
    TRADE_END_DATE = '2023-03-01'
    #
    # create_features(
    #     start_date=TRAIN_START_DATE,
    #     end_date=TRADE_END_DATE,
    #     ticker_list=config_tickers.DOW_30_TICKER,
    #     indicators=INDICATORS,
    #     save_dir=root/DATA_SAVE_DIR,
    #     use_vix=True,
    #     use_turbulence=True
    # )
    clean_data(path="/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/datasets/processed_2010-01-01_2023-03-01_DOW.csv",
                    output="processed_full")