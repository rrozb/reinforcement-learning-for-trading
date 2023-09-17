import datetime

from gym_trading_env.downloader import download

download(
    exchange_names=["bitfinex2", "huobi"],
    symbols=["BTC/USDT", "ETH/USDT", "XRP/USDT", "DOGE/USDT",
             "ADA/USDT", "DAI/USDT", "DOT/USDT", "LTC/USDT"],
    timeframe="1h",
    dir="data",
    since= datetime.datetime(year= 2022, month=1, day=1),
    until=datetime.datetime(year=2023, month=9, day=15),
)