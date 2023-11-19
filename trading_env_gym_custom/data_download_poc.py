import datetime

from gym_trading_env.downloader import download

download(
    exchange_names=["bitfinex2",],
    symbols=["ETH/USD", "BTC/USD"],
    timeframe="1h",
    dir="data",
    since=datetime.datetime(year=2015, month=1, day=1),
    until=datetime.datetime(year=2023, month=11, day=1),
)