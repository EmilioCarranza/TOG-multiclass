# packages
import pandas as pd
import mplfinance as mpf
from gplearn.genetic import SymbolicRegressor
from sympy import *
from collections import Counter
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
# Call new datasets
ordbook = pd.read_parquet('ordbook.parquet')
pubtrade = pd.read_parquet('pubtrade.parquet')

pubtrade = pubtrade[:-1]

# # Convert timestamp to index.
# ordbook.set_index('timestamp', inplace=True)
# pubtrade.set_index('timestamp', inplace=True)

# Resample to 1s intervals
# ordbook5s = ordbook.resample('1s', ).sum()
# pubtrade5s = pubtrade.resample('1s', ).sum()

# Spread ask price-bid price

bid_spread = ordbook.groupby(ordbook['timestamp'], group_keys=False)['bid_price'].max()
ask_spread = ordbook.groupby(ordbook['timestamp'], group_keys=False)['ask_price'].max()
# ask_volume = ordbook.groupby(ordbook['timestamp'], group_keys=False)['ask_vol'].max()
# bid_volume = ordbook.groupby(ordbook['timestamp'], group_keys=False)['bid_vol'].max()

spread_table = pd.merge(bid_spread, ask_spread, on='timestamp')
spread_table['Spread'] = spread_table['ask_price'] - spread_table['bid_price']
spread_table['basis points'] = spread_table['Spread'] * 10000

# mid-price (promedio entre ask y bid)

spread_table['mid_price'] = (spread_table['ask_price'] + spread_table['bid_price']) * 0.5

# weighted midprice como midprice pero multiplicados por los volumenes

vwap = ordbook.copy()

vwap['volask'] = vwap['ask_price'] * vwap['ask_vol']
vwap['volbid'] = vwap['bid_price'] * vwap['bid_vol']
vwap['total_vol'] = vwap['ask_vol'] + vwap['bid_vol']
vwap['imb_dif'] = vwap['bid_vol'] - vwap['ask_vol']

vwap_grouped = vwap.groupby('timestamp').sum()
vwap_grouped['vwap'] = (vwap_grouped['volask'] + vwap_grouped['volbid']) / vwap_grouped['total_vol']

# Public trade features
trades = (len(pubtrade))
total_volume = (pubtrade['amount']).sum()
print(Counter(pubtrade['side']))

# Describe basic statistics, media, median, variance , sd, sesgo, curtosis,
# quartiles, conteo, min y max.

stats = ordbook.copy()
stats = stats.groupby('timestamp').sum()

# EDA:

# 1.- Describir estadísticas básicas  , media, median ,variance, sd,sesgo,curtosis, quartiles, conteo , min max,
# outliers.
# orderbook_a = (orderbooks['binance'][list(orderbooks['binance'])[2]]['bid_price'])
# orderbook_b = (orderbooks['binance'][list(orderbooks['binance'])[2]])
#
# media = orderbook_a.mean()
# median = orderbook_a.median()
# variance = orderbook_a.var()
# std = orderbook_a.std()
# curt = kurtosis(orderbook_a)
# count = orderbook_a.count()
# max_orderbook = orderbook_a.max()
# min_orderbook = orderbook_a.min()
#
# quantiles = (path_orderbooks['bid_price']).describe()
# # 2.- Atípicos "lado minimo" <= q1-[q3-q1]*1.5todo precio abajo , seria atípico
# lado_minimo = q1 - (q3 - q1) * 1.5
# #     Atípicos "lado maximo" >= q3+[q3-q1]*1.5
# lado_maximo = q3 + (q3 - q1) * 1.5
# 3.- Serie de tiempo(lineas)
series_pubtrade_amount = pubtrade['amount'].astype(float)
series_pubtrade_amount.plot()
pyplot.show()

series_pubtrade_price = pubtrade['price'].astype(float)
series_pubtrade_price.plot()
pyplot.show()

# Plot de precios orderbook
prices_orderbook = []
prices_ordbook = pd.DataFrame(prices_orderbook, index=vwap_grouped.index)
prices_ordbook['ask_price'] = vwap_grouped['ask_price']
prices_ordbook['bid_price'] = vwap_grouped['bid_price']

prices_ordbook.plot()
pyplot.show()

# Decomposition plot for Prices in orderbook
decomposition = seasonal_decompose(prices_ordbook['ask_price'], model='additive', period=300)
decomp = decomposition.plot()
decomp.show()

decomposition2 = seasonal_decompose(prices_ordbook['bid_price'], model='additive', period=300)
decomp2 = decomposition2.plot()
decomp2.show()

# Histograma Frecuencia de Variables

plt.hist(series_pubtrade_price, bins=10)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
# 5.- Boxplot (con media y mediana)
fig = plt.figure(figsize=(10, 7))

# Creating plot
plt.boxplot(series_pubtrade_price)

# show plot
plt.show()

# Crear OHLCV Dataset
# Se genera con el mid price que calculaste utilizando el TOB
# puede ser, 1) la suma del volumen en todos los niveles de todos los libros de órdenes que se tengan durante el
# intervalo de tiempo elegido para el cálculo (este se le llama volumen de órdenes), 2) la suma del volumen de todas
# las public trades ocurridas durante el intervalo de tiempo elegido para el cálculo (a este se le llama volumen de
# operaciones).

ohlcv = spread_table['mid_price'].resample('3S').ohlc()
time = ohlcv.index
ohlcv['volume'] = vwap_grouped['total_vol'].resample('3S').sum()

# Plot OHLC
mpf.plot(ohlcv, type='candle', mav=(10, 20), volume=True)

# Plot OHLC
TICKER = 'BTC/USD'
mpf.plot(ohlcv, figratio=(10, 6), type="candle",
         mav=30, volume=True,
         title=f"Price of {TICKER}",
         tight_layout=True, style="binance")

# Volatility
volatility = []
for i in range(len(ohlcv)):
    vol = ohlcv['high'][i] - ohlcv['low'][i]
    volatility.append(vol)
ohlcv['volatility'] = volatility

# micro trends
high_open = []
for i in range(len(ohlcv)):
    ho = ohlcv['high'][i] - ohlcv['open'][i]
    high_open.append(ho)
ohlcv['high_open'] = high_open

open_low = []
for i in range(len(ohlcv)):
    ol = ohlcv['open'][i] - ohlcv['low'][i]
    open_low.append(ol)
ohlcv['open_low'] = open_low
# Target engineering classification problem sign
ohlc = pd.DataFrame(ohlcv)

high_low = []  # micro volatility
for i in range(len(ohlcv)):
    hl = ohlcv['high'][i] - ohlcv['low'][i]
    high_low.append(hl)
ohlcv['high_low'] = high_low

close_open = []
for i in range(len(ohlcv)):
    co = ohlcv['close'][i] - ohlcv['open'][i]
    close_open.append(co)
ohlcv['close_open'] = close_open

# Target engineering classification problem sign
ohlc = pd.DataFrame(ohlcv)

sign = []

for i in range(len(ohlc)):
    sign1 = ohlc["close"][i] - ohlc["open"][i]
    sign.append(sign1)
for i in range(len(ohlc)):
    if sign[i] <= 0:
        sign[i] = 0
    else:
        sign[i] = 1
ohlc['sign'] = sign
print(ohlc)

# Martingala
ohlc['sign'] = sign
ohlc['sign_t1'] = ohlc['sign'].shift(+1)
ohlc = ohlc.fillna(0)
ohlc2 = pd.DataFrame(ohlc)

accuracy_martingala = accuracy_score(ohlc['sign'], ohlc['sign_t1']) * 100

# Heat map
correlation = ohlc2.corr()
print(correlation)

heat = sns.heatmap(
    correlation,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
heat.set_xticklabels(
    heat.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()

print(ohlc2.describe())
sns.pairplot(ohlc2, hue='sign', height=2.5)

# Second order ops Moving average, std,
ma = ohlc2.rolling(5).mean()
std = ohlc2.std()

# Feature engineering 100 candidate features

# Preprocessing Log, Scale, Standardize (mean, median), Normalize
X = ohlc2.iloc[:, 4:-2]
y = ohlc2['sign']
y_true = y
# Normalizer
transformer = Normalizer().fit(X)  # fit does nothing.
transformed = transformer.transform(X)
X_normalized = transformed
# Scale
X_scale = scale(X)

# Standardize
scaler = StandardScaler()
scaled = scaler.fit_transform(X)
X_standard = scaled

# Heat map
correla = pd.DataFrame(X_scale, index=ohlc2.index)
correla['6'] = ohlc2['sign']
correlation_transformed = correla.corr(method='spearman')
print(correlation_transformed)

heat2 = sns.heatmap(
    correlation_transformed,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
heat2.set_xticklabels(
    heat2.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()

# k fault Symbolic Regressor
#x_test = pd.DataFrame(X_scale,index = X.index, columns= X.columns)
#X_train, X_test, y_train, y_test = train_test_split(x_test, y, test_size=0.30, random_state=False)

# First Test
#function_set = ['add', 'sub', 'mul', 'div', 'cos', 'sin', 'neg', 'inv']
#est_gp = SymbolicRegressor(population_size=5000, function_set=function_set,
                         #  generations=40, stopping_criteria=0.01,
                          # p_crossover=0.7, p_subtree_mutation=0.1,
                           #p_hoist_mutation=0.05, p_point_mutation=0.1,
                          # max_samples=0.9, verbose=1,
                          # parsimony_coefficient=0.01, random_state=0,
                          # feature_names=X_train.columns)
#converter = {
  #  'sub': lambda x, y: x - y,
   # 'div': lambda x, y: x / y,
  #  'mul': lambda x, y: x * y,
  #  'add': lambda x, y: x + y,
  #  'neg': lambda x: -x,
  #  'pow': lambda x, y: x ** y,
  #  'sin': lambda x: sin(x),
  #  'cos': lambda x: cos(x),
  #  'inv': lambda x: 1 / x,
  #  'sqrt': lambda x: x ** 0.5,
  #  'pow3': lambda x: x ** 3
#}
#est_gp.fit(X_train, y_train)
#print('R2:', est_gp.score(X_test, y_test))

#next_e = sympify(est_gp.program, locals=converter)

# Logistic Regression
x_train, x_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.20, random_state=False)
# model fit
logistic_model = LogisticRegression(random_state=None, penalty='elasticnet', solver='saga', l1_ratio=1, max_iter=4000)
logistic_model.fit(x_train, y_train)
# logistic_model.predict_proba((X[:2, :])
y_pred = logistic_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred) * 100
logistic_model.score(X_normalized, y)
confusion_mat = confusion_matrix(y_test, y_pred)

print("Accuracy is", accuracy)
print("Confusion Matrix")
print(confusion_mat)

# Model Evaluation
y_pred2 = pd.DataFrame(y_pred)
# recall tp / (tp + fn)
recall = recall_score(y_test, y_pred, average='micro')
print(recall)
# Precision tp / (tp + fp)
precision = precision_score(y_test, y_pred, average='micro')
print(precision)
# F1 F1 = 2 * (precision * recall) / (precision + recall)
f1_score = 2*(precision*recall)/(precision + recall)
print(f1_score)
# accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print(accuracy)

# Model Explain-ability por como salió y significado de lo que se obtuvo.
