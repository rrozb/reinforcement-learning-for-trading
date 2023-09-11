from gym_trading_env.downloader import download
import datetime

download(
    exchange_names = ["binance", "bitfinex2", "huobi"],
    symbols= ["BTC/USDT", "ETH/USDT"],
    timeframe= "15m",
    dir = "data",
    since= datetime.datetime(year= 2023, month= 1, day=1),
    until = datetime.datetime(year= 2023, month= 2, day=1),
)