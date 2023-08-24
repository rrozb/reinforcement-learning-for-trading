from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
import pathlib
from finrl import config_tickers
from finrl.main import check_and_make_directories
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS
)

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

if __name__ == "__main__":


    root = pathlib.Path(__file__).parent.parent

    dirs = [root/DATA_SAVE_DIR, root/TRAINED_MODEL_DIR, root/TENSORBOARD_LOG_DIR, root/RESULTS_DIR]

    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    #
    TRAIN_START_DATE = '2010-01-01'
    TRAIN_END_DATE = '2021-10-01'
    TRADE_START_DATE = '2021-10-01'
    TRADE_END_DATE = '2023-03-01'

    create_features(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=config_tickers.DOW_30_TICKER,
        indicators=INDICATORS,
        save_dir=root/DATA_SAVE_DIR,
        use_vix=True,
        use_turbulence=True
    )