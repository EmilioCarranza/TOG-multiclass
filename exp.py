# -- Load base packages
import pandas as pd
import data as dt
import pickle
import time
import ccxt

# Get the list of exchanges with the symbol
print(ccxt.exchanges)

# Get public trades from the list of exchanges
exchanges = ['binance']
symbol = 'BTC/USDT'

# Fetch realtime orderbook data until timer is out (60 secs is default)
orderbooks = dt.async_data(symbol=symbol, exchanges=exchanges, output_format='inplace', timestamp_format='timestamp',
                           data_type='orderbooks', file_route='files/orderbooks', stop_criteria=None,
                           elapsed_secs=1800, verbose=2)

# Fetch realtime orderbook data until timer is out (60 secs is default)
start = time.time()
print(time.ctime(start))
publictrades = dt.async_data(symbol=symbol, exchanges=exchanges, output_format='inplace', timestamp_format='timestamp',
                             data_type='publictrades', file_route='files/publictrades', stop_criteria=None,
                             elapsed_secs=1800, verbose=2)

end = time.time()
print(time.ctime(end))
print(end - start)

with open('orderbooks30m.pkl', 'wb') as f:
    pickle.dump(orderbooks, f)

with open('publictrades30m.pkl', 'wb') as f:
    pickle.dump(publictrades, f)

# Basic data check
publictrades['binance'].keys()
orderbooks['binance'].keys()
orderbooks_list = list(orderbooks['binance'])[0]
type(list(orderbooks['binance'])[0])

# create new datasets
orderbook = pd.DataFrame()
y = pd.DataFrame()

for key in orderbooks['binance']:
    print(key)
    y = orderbooks['binance'][key]
    y['timestamp'] = key
    orderbook = orderbook.append(y, ignore_index=True)
orderbook.to_csv('Orderbooks.csv')

public_trades = pd.DataFrame()

for key in publictrades['binance']:
    print(key)
    y = publictrades['binance'][key]
    y['timestamp'] = key
    public_trades = public_trades.append(y, ignore_index=True)
print(public_trades)

public_trades = public_trades.transpose()
public_trades.to_csv('publictrades.csv')

public_trades.to_parquet('pubtrade.parquet')
orderbook.to_parquet('orderbook.parquet')
# order_ohlv= pd.read_csv('Orderbooks.csv',index_col=5,parse_dates=True)
# order_ohlv.index = pd.to_datetime(order_ohlv.index)
# order_ohlv_ask = order_ohlv['ask_price'].resample('1S').ohlc()
# order_ohlv_bid = order_ohlv['bid_price'].resample('1S').sum()
# order_ohlv_askbid = order_ohlv['ask_vol'].resample('1S').sum()
#
#
# order_ohlv_askbid = pd.concat(
#     [order_ohlv_ask, order_ohlv_bid], axis=1,
#     keys=['Ask', 'Bid'])


# import dataset
# orderbooks_file = open('orderbooks30m.pkl', 'rb')
# orderbooks = pickle.load(orderbooks_file)
# print(orderbooks)
# orderbooks_file.close()
#
# publictrades_file = open('publictrades30m.pkl', 'rb')
# publictrades = pickle.load(publictrades_file)
# print(publictrades)
# publictrades_file.close()

# ordbook = pd.read_csv('orderbooks.csv')
# pubtrade = pd.read_csv('publictrades.csv')
#
# ordbook.info()
# pubtrade.info()
#
# ordbook = ordbook.drop(['Unnamed: 0'], axis=1)
# cols = ordbook.columns.tolist()
# cols = cols[-1:] + cols[:-1]
# ordbook = ordbook[cols]
# ordbook = ordbook['timestamp', 'bid_vol', 'bid_price', 'ask_price', 'ask_vol']
# pubtrade.columns = ['timestamp', 'trade_id', 'side', 'price', 'amount']
# pubtrade.drop(['Unnamed: 0'], axis=1)
#
# print(ordbook.shape)
# print(pubtrade.shape)
# count = np.count_nonzero(pubtrade['timestamp'].isna())
# pubtrade['timestamp'].floordiv(1000)
#
# ordbook['timestamp'] = pd.to_datetime(ordbook['timestamp'])
# pubtrade['timestamp'] = pd.to_datetime(pubtrade['timestamp'], errors='coerce')
#
# ordbook.to_parquet('ordbook.parquet')
# pubtrade.to_parquet('pubtrade.parquet')
