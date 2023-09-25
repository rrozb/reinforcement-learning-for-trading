import itertools
import os
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
       path_in: str,
       path_out: str,
):
    # read all pickles in folder
    data_combined  = []
    for file in os.listdir(path_in):
        if file.endswith(".pkl"):
            pair = file.split("-")[1]
            data = pd.read_pickle(f"{path_in}/{file}")
            data = data.reset_index(drop=True)
            data["tic"] = pair
            data_combined.append(data)

    df = pd.concat(data_combined)
    df["date"] = df["date_close"]
    df = df.drop(columns=["date_close"])
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS)
    processed = fe.preprocess_data(df)

    processed.to_csv(f'{path_out}/processed.csv', index=False)


def clean_data(path: str, output: str):
    processed = pd.read_csv(path)
    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(), processed['date'].max(), freq="1H").astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date', 'tic'])

    processed_full = processed_full.fillna(0)

    processed_full.to_csv(f"datasets/{output}.csv")
#
#
def tvt(path: str, out: str, train_start: str, train_end: str, trade_start: str, trade_end: str):
    processed = pd.read_csv(path)
    train = data_split(processed, train_start, train_end)
    trade = data_split(processed, trade_start, trade_end)
    trade.to_csv(f"datasets/{out}_trade.csv")
    train.to_csv(f"datasets/{out}_train.csv")


if __name__ == "__main__":
    # create_features("data", "features")
    # clean_data("features/processed.csv", "processed_full")
    tvt("datasets/processed_full.csv",
        "full",
        train_start="2022-01-01 01:00:00",
        train_end="2023-08-05 14:00:00",
        trade_start="2023-08-05 15:00:00",
        trade_end="2023-09-14 22:00:00")

    #
    # root = pathlib.Path(__file__).parent.parent
    # #
    # dirs = [root/DATA_SAVE_DIR, root/TRAINED_MODEL_DIR, root/TENSORBOARD_LOG_DIR, root/RESULTS_DIR]
    #
    # check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    # #
    # TRAIN_START_DATE = '2010-01-01'
    # TRAIN_END_DATE = '2021-10-01'
    # TRADE_START_DATE = '2021-10-01'
    # TRADE_END_DATE = '2023-03-01'
    # #
    # create_features(
    #     start_date=TRAIN_START_DATE,
    #     end_date=TRADE_END_DATE,
    #     ticker_list=config_tickers.DOW_30_TICKER,
    #     indicators=INDICATORS,
    #     save_dir=root/DATA_SAVE_DIR,
    #     use_vix=True,
    #     use_turbulence=True
    # )
    # clean_data(path="/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/datasets/processed_2010-01-01_2023-03-01_DOW.csv",
    #                 output="processed_full")
    # tvt(path="/home/rr/Documents/Coding/Work/crypto/reinforcement-learning-for-trading/datasets/processed_full.csv",
    #     out="full",
    #     train_start=TRAIN_START_DATE,
    #     train_end=TRAIN_END_DATE,
    #     trade_start=TRADE_START_DATE,
    #     trade_end=TRADE_END_DATE)
