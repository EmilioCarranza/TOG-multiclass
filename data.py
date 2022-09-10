# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 20:23:02 2022

@author: Emilio
"""
import pandas as pd
import numpy as np
import time
import json

# -- Cryptocurrency data and trading API
import ccxt

# -- Asynchronous data fetch
import asyncio
import ccxt.async_support as ccxt_async

# --------------------------------------------------------------------------- ASYNCHRONOUS ORDERBOOK DATA -- #
# --------------------------------------------------------------------------------------------------------- #


def async_data(symbol, exchanges, data_type, execution='async', stop_criteria=None, elapsed_secs=60,
               output_format='numpy', timestamp_format='timestamp', verbose=True, file_route='Files/'):
    """
    Asynchronous OrderBook data fetcher. It will asyncronously catch innovations of transactions whenever they
    occur for every exchange is included in the list exchanges, and return the complete orederbook in a in a
    JSON format or DataFrame format with 'ask', 'ask_size', 'bid', 'bid_size'.
    Parameters
    ----------
    symbol: list
        with the names of instruments or markets to fetch the oredebook from.
    exchanges: list
        with the names of exchanges from where the data is intended to be fetched.
    execution: str
        'async': Asyncronous option to fetch several orderbooks in the same call. Depends on
                 asyncio and ccxt.async_support
        'parallel': Run a parallel processing to deploy 1 instance for each symbol at each market. Depends
                 on multiprocessing (pending)
    stop_criteria: dict
        Criteria to stop the execution. Default behavior will be to stop after 1 minute of running.
        'min_count': int
            Stops when all orderbooks have, at least, this number of regisetred timestamps.
        'target_timestamp': datetime
            Stops when its reached a specific timestamp.
        'elapsed_time': (default)
            Stop when the elapsed_secs has passed
    output_format: str {'numpy', 'dataframe'} (default: 'numpy')
        Options for the output format, both is a dictionary with timestamp as key, values are:
        'numpy': numpy array with [0] Bid Volume, [1] Bid Price, [2] Ask Price, [3] Ask Volume
        'dataframe': pd.DataFrame with columns bid_volume, bid_price, ask_price, ask_volume
    verbose: bool
        To print in real time the fetched first ask and bid of every exchange.
    Returns
    -------
    r_data: dict
        A dictionary with the fetched data, with the following structure.
        r_data = {
            instrument: {
                exchange: {
                    timestamp: {'ask': 1.4321, 'ask_size': 0.12,
                                'bid': 1.1234, 'bid_size': 0.21},
                    timestamp: {'ask': 1.4321, 'ask_size': 0.12,
                                'bid': 1.1234, 'bid_size': 0.21}
            }
        }
    References
    ----------
    [1] https://github.com/ccxt/ccxt
    [2] https://docs.python.org/3/library/asyncio.html
    """

    # Coherce to a list type when either exchanges or symbols is only a str
    exchanges = [exchanges] if not isinstance(exchanges, list) else exchanges

    # Store data for every exchange in the list
    r_data = {i_exchange: {} for i_exchange in exchanges}

    async def async_client(exchange, symbol, data_type):

        # Await to be inside exchange limits of calls
        # await asyncio.sleep(exchange.rateLimit / 1000)

        # Initialize client inside the function, later will be closed, since this is runned asyncronuously
        # more than 1 client could be created and later closed.
        client = getattr(ccxt_async, exchange)({'enableRateLimit': True})
        await client.load_markets()

        # Check for symbol support on exchange
        if symbol not in client.symbols:
            raise Exception(exchange + ' does not support symbol ' + symbol)

        # Initial time and counter
        time_1 = time.time()
        time_f = 0
        dct_publictrades = {}
        # Loop until stop criteria is reached
        while time_f < elapsed_secs:

            # Try and await for client response
            try:

                if data_type == 'orderbooks':

                    # Fetch, await and get datetime
                    orderbook = await client.fetch_order_book(symbol)
                    if timestamp_format == 'timestamp':
                        datetime = pd.to_datetime(
                            client.milliseconds()*1000000)
                    elif timestamp_format == 'unix':
                        datetime = client.milliseconds()

                    # Verbosity
                    if verbose == 2:
                        print(datetime, client.id, symbol,
                              'bid_vol: ', orderbook['bids'][1][1], ' bid_price: ', orderbook['bids'][1][0],
                              'ask_price: ', orderbook['asks'][0][0], ' ask_vol: ', orderbook['bids'][0][1])

                    # Unpack values
                    ask_price, ask_vol = np.array(
                        list(zip(*orderbook['asks']))[0:2])
                    bid_price, bid_vol = np.array(
                        list(zip(*orderbook['bids']))[0:2])
                    spread = np.round(ask_price - bid_price, 4)

                    # Final data format for the results
                    r_data[client.id].update({datetime: pd.DataFrame({'bid_vol': bid_vol, 'bid_price': bid_price,
                                                                      'ask_price': ask_price, 'ask_vol': ask_vol})})
                    # End time
                    time_2 = time.time()
                    time_f = round(time_2 - time_1, 4)

                    # Close client
                    await client.close()

                elif data_type == 'publictrades':

                    if verbose == 2:
                        print('Get publictrades in: ', exchange,
                              ' for the symbol: ', symbol)

                    # Fetch, await and get datetime
                    publictrades = await client.fetch_trades(symbol)
                    # publictrades = client.fetch_trades(symbol)
                    if timestamp_format == 'timestamp':
                        datetime = pd.to_datetime(
                            client.milliseconds()*1000000)
                    elif timestamp_format == 'unix':
                        datetime = client.milliseconds()

                    # Get all publictrades in a dataframe
                    # dct_publictrades = {}
                    for i_trade in publictrades:
                        dct_publictrades.update({pd.to_datetime(i_trade['timestamp']*1000000):
                                                 {'trade_id': i_trade['id'], 'side': i_trade['side'],
                                                  'price': i_trade['price'], 'amount': i_trade['amount']}})

                    # Store all dataframes as final result
                    r_data[client.id] = pd.DataFrame(dct_publictrades).T

                    # End time
                    time_2 = time.time()
                    time_f = round(time_2 - time_1, 4)

                    # Close client
                    # await client.close()

            # In case something bad happens with client
            except Exception as e:
                print(type(e).__name__, e.args, str(e))
                pass

        await client.close()

    # ------------------------------------------------------------------------------ MULTIPLE ORDERBOOKS -- #
    async def multi_data(exchanges, symbol):
        # A list of routines (and parameters) to run
        input_coroutines = [async_client(
            exchange, symbol, data_type) for exchange in exchanges]
        # wait for responses
        await asyncio.gather(*input_coroutines, return_exceptions=True)

    # Run event loop in async
    if execution == 'async':
        if verbose == 1 and data_type == 'orderbooks':
            print('Order Books data fetch')

        if verbose == 1 and data_type == 'publictrades':
            print('Public Trades data fetch')

        asyncio.get_event_loop().run_until_complete(multi_data(exchanges, symbol))

    # Run multiple events in parallel
    elif execution == 'parallel':
        raise ValueError('Only supported async')

    # Raise error in case of other value
    else:
        raise ValueError(execution, 'is not supported as a type of execution')

    # ----------------------------------------------------------------------------------- TYPE OF OUTPUT -- #

    # A JSON file writen in directory
    if output_format == 'json':
        # Serializing json
        json_object = pd.DataFrame(r_data).to_json()
        # Label and file creation
        label = str(pd.to_datetime(time.time()*1e9))[:19].replace(' ', 'T')
        with open(file_route + '/orderbooks_' + label + '.json', 'w') as outfile:
            outfile.write(json_object)

    # Just return the DataFrame
    elif output_format == 'inplace':
        return r_data

    # Invalid output
    else:
        raise ValueError('Invalid output value')
