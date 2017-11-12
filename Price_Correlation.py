# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:03:47 2017

@author: buddy
"""
# coding: utf-8

import numpy as np
import pandas as pd
import pickle
import quandl
from datetime import datetime


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

plot=1
plot_heatmap=1

# Pull pricing data for more BTC exchanges
exchanges = ['COINBASE','BITSTAMP','ITBIT']

# Pull pricing data for more altcoins from polonoex 
altcoins = ['ETH','LTC','XRP','ETC','STR','DASH','SC','XMR','XEM','BCH','VTC']

#correlation_heatmap_label = copy.copy(altcoins)
#correlation_heatmap_label.append('BTC')

#parameters for function "get_crypto_data(poloniex_pair)" regarding fetched altcoin timeseries (cos nie wplywa)
base_polo_url = 'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}'
start_date = datetime.strptime('2015-01-01', '%Y-%m-%d') # get data from the start of 2015
end_date = datetime.now() # up until today
pediod = 86400 # pull daily data (86,400 seconds per day)








def get_quandl_data(quandl_id):
    '''Download and cache Quandl dataseries'''
    cache_path = '{}.pkl'.format(quandl_id).replace('/','-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(quandl_id))
    except (OSError, IOError):
        print('Downloading {} from Quandl'.format(quandl_id))
        df = quandl.get(quandl_id, returns="pandas")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df



# Pull Kraken BTC price exchange data
btc_usd_price_kraken = get_quandl_data('BCHARTS/KRAKENUSD')


#check top 5 in list
btc_usd_price_kraken.head()

if plot == 1:
#plot btc price
    plt.figure(0)
    btc_plot_kraken = plt.plot(btc_usd_price_kraken['Weighted Price'], label='BTC price on Kraken')
    plt.legend(loc='best')
    plt.show()



# Pull pricing data for 3 more BTC exchanges
#exchanges = ['COINBASE','BITSTAMP','ITBIT']

exchange_data = {}

exchange_data['KRAKEN'] = btc_usd_price_kraken

for exchange in exchanges:
    exchange_code = 'BCHARTS/{}USD'.format(exchange)
    btc_exchange_df = get_quandl_data(exchange_code)
    exchange_data[exchange] = btc_exchange_df
    
    
def merge_dfs_on_column(dataframes, labels, col):
    '''Merge a single column of each dataframe into a new combined dataframe'''
    series_dict = {}
    for index in range(len(dataframes)):
        series_dict[labels[index]] = dataframes[index][col]
        
    return pd.DataFrame(series_dict)
    
    
    
# Merge the BTC price dataseries' into a single dataframe
btc_usd_datasets = merge_dfs_on_column(list(exchange_data.values()), list(exchange_data.keys()), 'Weighted Price')

#show last observations in avangered price for btc
btc_usd_datasets.tail()

if plot == 1:
#plots values (add function to make a legend for given dataset and labels)
    label_BTC_markets=['BITSTAMP', 'COINBASE','ITBIT','KRAKEN']
    plt.figure(1)
    plot_BTC_markets = plt.plot(btc_usd_datasets)
    plt.legend(plot_BTC_markets, label_BTC_markets, loc=2)
    plt.show()

# Remove "0" values
btc_usd_datasets.replace(0, np.nan, inplace=True)

if plot ==1:
# Plot the revised dataframe without 0 values
    label_BTC_markets=['BITSTAMP', 'COINBASE','ITBIT','KRAKEN']
    plt.figure(1)
    plot_BTC_markets = plt.plot(btc_usd_datasets.index ,btc_usd_datasets)
    plt.legend(plot_BTC_markets, label_BTC_markets, loc=2)
    plt.show()


# Calculate the average BTC price as a new column
btc_usd_datasets['avg_btc_price_usd'] = btc_usd_datasets.mean(axis=1)

if plot == 1:
# Plot the average BTC price
    plt.figure(3)
    label_BTC_markets_average = ['avg btc price usd']
    plot_BTC_markets_average = plt.plot(btc_usd_datasets.index, btc_usd_datasets['avg_btc_price_usd'])
    plt.legend(plot_BTC_markets_average, label_BTC_markets_average, loc=2)
    plt.show()



def get_json_data(json_url, cache_path):
    '''Download and cache JSON data, return as a dataframe.'''
    try:        
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(json_url))
    except (OSError, IOError):
        print('Downloading {}'.format(json_url))
        df = pd.read_json(json_url)
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(json_url, cache_path))
    return df


#importan download info
#base_polo_url = 'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}'
#start_date = datetime.strptime('2015-01-01', '%Y-%m-%d') # get data from the start of 2015
#end_date = datetime.now() # up until today
#pediod = 86400 # pull daily data (86,400 seconds per day)

def get_crypto_data(poloniex_pair):
    '''Retrieve cryptocurrency data from poloniex'''
    json_url = base_polo_url.format(poloniex_pair, start_date.timestamp(), end_date.timestamp(), pediod)
    data_df = get_json_data(json_url, poloniex_pair)
    data_df = data_df.set_index('date')
    print(json_url)
    return data_df


#altcoins = ['ETH','LTC','XRP','ETC','STR','DASH','SC','XMR','XEM']

altcoin_data = {}
for altcoin in altcoins:
    coinpair = 'BTC_{}'.format(altcoin)
    crypto_price_df = get_crypto_data(coinpair)
    altcoin_data[altcoin] = crypto_price_df

#last prices of ETH
altcoin_data['ETH'].tail()


# Calculate USD Price as a new column in each altcoin dataframe
for altcoin in altcoin_data.keys():
    altcoin_data[altcoin]['price_usd'] =  altcoin_data[altcoin]['weightedAverage'] * btc_usd_datasets['avg_btc_price_usd']


# Merge USD price of each altcoin into single dataframe 
combined_df = merge_dfs_on_column(list(altcoin_data.values()), list(altcoin_data.keys()), 'price_usd')


# Add BTC price to the dataframe
combined_df['BTC'] = btc_usd_datasets['avg_btc_price_usd']


# Chart all of the altocoin prices (price in logarithmic scale)
if plot == 1:
    plt.figure('logplot coins prices polonex')
    lable_coins_prices_polonex = altcoins
    logplot_coins_prices_polonex = plt.semilogy(combined_df)
    plt.legend(logplot_coins_prices_polonex, lable_coins_prices_polonex, loc=2)
    plt.title('semilogx')
    plt.grid(True)
    plt.show()



#accesing price of single coin
#plt.figure('log DASH price')
#plt.semilogy(combined_df['DASH'])
#plt.figure('DASH price')
#plt.plot(combined_df.index,combined_df['DASH'])
#plt.scatter(combined_df.index,combined_df['DASH'])

#show label price on mouse click/hover 
#from mpldatacursor import datacursor
#
#for i in range(1, len(combined_df.index)):
#    plt.scatter(combined_df.index,combined_df['BTC'], label='$ID: {}$'.format(i))
#
## Use a DataCursor to interactively display the label for a selected line...
#datacursor(formatter='{label}'.format)
#
#plt.show()

# Calculate the pearson correlation coefficients for cryptocurrencies in 2016
combined_df_2016 = combined_df[combined_df.index.year == 2016]
correlation_price_2016 = combined_df_2016.pct_change().corr(method='pearson')


#correaltion heatmap of prices in 2016
#plt.figure('correaltion heatmap of coin prices in 2016')
#plt.imshow(correlation_price_2016 , cmap='hot', interpolation='nearest')
#plt.show()
#add names instead of number, min max value of heat(normalization on dataset)


# Calculate the pearson correlation coefficients for cryptocurrencies in 2017
combined_df_2017 = combined_df[combined_df.index.year == 2017]
correlation_price_2017 = combined_df_2017.pct_change().corr(method='pearson')


#correaltion heatmap of prices in 2017
#plt.figure('correaltion heatmap of coin prices in 2017')
#plt.imshow(correlation_price_2017 , cmap='hot', interpolation='nearest')
#plt.show()

combined_df_2017_oct = combined_df_2017[combined_df_2017.index.month == 11]
correlation_price_2017_oct = combined_df_2017_oct.pct_change().corr(method='pearson')

#plt.imshow(combined_df_2017_oct , cmap='hot', interpolation='nearest')


correlation_heatmap_label = list(combined_df)
#correlation_heatmap_label.append('BTC')


if plot_heatmap ==1:
#correaltion heatmap of coin prices in 2017
    column_labels = correlation_heatmap_label
    row_labels = correlation_heatmap_label
    data = correlation_price_2017
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=cm.coolwarm, vmin=-1, vmax=1)
        
    
    #TEST Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(heatmap, ticks=[-1, 0, 1])
    
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)
    
    # want a more natural, table-like display
    #ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    
    plt.suptitle('Correaltion heatmap of coin prices in 2017',fontsize=16)    
    
    plt.show()
    




if plot_heatmap ==1:
    #correaltion heatmap of coin prices in November 2017
    column_labels = correlation_heatmap_label
    row_labels = correlation_heatmap_label
    data = correlation_price_2017_oct
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=cm.coolwarm, vmin=-1, vmax=1)
    
    #Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(heatmap, ticks=[-1, 0, 1])
    
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)
    
    # want a more natural, table-like display
    #ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    
    plt.suptitle('Correaltion heatmap of coin prices in November 2017',fontsize=16)     
    
    plt.show()
    
    
    

# Merge USD price of each altcoin into single dataframe 
combined_df_vol = merge_dfs_on_column(list(altcoin_data.values()), list(altcoin_data.keys()), 'volume')

# Add BTC vol to the dataframe in the BTC
combined_df_vol['BTC'] = btc_exchange_df['Volume (BTC)']
#in $
#combined_df_vol['BTC'] = btc_exchange_df['Volume(Currency)']

combined_df_vol.replace(0, np.nan, inplace=True)



combined_df_vol_2017 = combined_df_vol[combined_df_vol.index.year == 2017]
correlation_vol_2017 = combined_df_vol_2017.pct_change().corr(method='pearson')

plt.figure('volumen in time')
plt.semilogy(combined_df_vol)


#jaka jest korelacja ceny i volumenu danej waluty macierz 2x2 vol i cena
#RNN sprawdzic jak przewiduje altcoina cene na podstawie [ceny,vol, jakis inny ciekawy parametr] na podstawie [1 dnia, 7 dni, itp]

a = {}
a['vol'] = combined_df_vol_2017
a['price'] = combined_df_2017

dataframe_a = merge_dfs_on_column(list(a.values()), list(a.keys()), 'BCH')

correlation_vol_price = dataframe_a.pct_change().corr(method='pearson')

vol_names = correlation_heatmap_label
price_names = correlation_heatmap_label

sufix_vol = '_vol'
sufix_price = '_price'

vol_names = [x + sufix_vol for x in vol_names]
price_names = [x + sufix_price for x in price_names]

combined_df_vol_2017_copy = combined_df_vol_2017.copy()
combined_df_2017_copy = combined_df_2017.copy()

combined_df_vol_2017_copy.columns = vol_names
combined_df_2017_copy.columns = price_names

test_dataframe = pd.concat([combined_df_vol_2017_copy, combined_df_2017_copy], axis=1, join='outer')

#frames = [df1, df2, df3]
#result = pd.concat(frames)

correlation_vol_price_all = test_dataframe.pct_change().corr(method='pearson')




btc_usd_datasets.to_csv('avg_BTC_price_USD.csv', sep='\t', encoding='utf-8')

btc_exchange_df.to_csv('BTC_allinfo.csv', sep='\t', encoding='utf-8')



#correlation 'close' to 'price'

combined_df_close = merge_dfs_on_column(list(altcoin_data.values()), list(altcoin_data.keys()), 'close')

# Add BTC vol to the dataframe in the BTC
combined_df_close['BTC'] = btc_exchange_df['Close']


combined_df_close_2017 = combined_df_close[combined_df_close.index.year == 2017]

ab = {}
ab['close'] = combined_df_close_2017
ab['price'] = combined_df_2017

dataframe_ab = merge_dfs_on_column(list(ab.values()), list(ab.keys()), 'BTC')

correlation_close_price = dataframe_ab.pct_change().corr(method='pearson')

#correlation 'close' -1day  to 'price' 

#if i do correlation from price to price moved by x days the correlation is still 1

#dele last obs from pprice and first from close
ac = {}
ac['oneday_move_close'] = combined_df_close_2017.iloc[:-10,:]
#ac['oneday_move_close'] = combined_df_2017.iloc[:-10,:]
ac['oneday_move_price'] = combined_df_2017.iloc[10:,:]

dataframe_ac = merge_dfs_on_column(list(ac.values()), list(ac.keys()), 'BTC')

correlation_close_price_oneday = dataframe_ac.pct_change().corr(method='pearson')

#correlation 'vol' -1day  to 'price' 


ad = {}
ad['oneday_move_vol'] = combined_df_vol_2017.iloc[:-10,:]
ad['oneday_move_price'] = combined_df_2017.iloc[10:,:]

dataframe_ad = merge_dfs_on_column(list(ad.values()), list(ad.keys()), 'BTC')

correlation_vol_price_oneday = dataframe_ad.pct_change().corr(method='pearson')








BTH_kaggle_BTH = pd.read_csv('Data/cryptocurrencypricehistory/bitcoin_cash_price.csv', sep=',')


BTH_kaggle_BTH['avg_price']=(BTH_kaggle_BTH['Open']+BTH_kaggle_BTH['Close'])/2


ass1BTH = []
ass2BTH = []
ass1BTH = BTH_kaggle_BTH['avg_price']
ass2BTH = BTH_kaggle_BTH['Market Cap']
ass1BTH = ass1BTH[:92]
ass2BTH = ass2BTH[:92]
#ass2 = ass2.convert_objects(convert_numeric=True)

ass2BTH = ass2BTH.str.replace(',','').astype(np.float64)

watifBTH = pd.concat([ass1BTH, ass2BTH], axis=1)

correlation_markcap_price_kaggle = watifBTH.pct_change().corr(method='pearson')


BTC_kaggle = pd.read_csv('Data/cryptocurrencypricehistory/bitcoin_price.csv', sep=',')


BTC_kaggle['avg_price']=(BTC_kaggle['Open']+BTC_kaggle['Close'])/2


ass1BTC = []
ass2BTC = []
ass1BTC = BTC_kaggle['avg_price']
ass2BTC = BTC_kaggle['Volume']
ass1BTC = ass1BTC[:1411]
ass2BTC = ass2BTC[:1411]


ass2BTC = ass2BTC.str.replace(',','').astype(np.float64)

watifBTC = pd.concat([ass1BTC, ass2BTC], axis=1)

correlation_vol_price_kaggle = watifBTC.pct_change().corr(method='pearson')



#crypto info BTC_difficulty
BTC_difficulty = pd.read_csv('Data/BTC_difficulty.csv', sep=',')

#crypto info BTC_hash-rate
BTC_hash_rate = pd.read_csv('Data/BTC_hash-rate.csv', sep=',')

#crypto info BTC_miners-revenue
BTC_miners_revenue = pd.read_csv('Data/BTC_miners-revenue.csv', sep=',')

#crypto info market-price
BTC_market_price = pd.read_csv('Data/market-price.csv', sep=',')

ass1BTCd = []
ass2BTCd = []
ass1BTCd = BTC_difficulty['1.0']
ass2BTCd = BTC_market_price['0.0']

hash_rate = []
miners_revenue = []
hash_rate = BTC_hash_rate['0.00000004971026962962963']
miners_revenue = BTC_miners_revenue['0.0']

watifBTC_diff = pd.concat([ass1BTCd, ass2BTCd], axis=1)
watifBTC_hash_rate = pd.concat([hash_rate, ass2BTCd], axis=1)
watifBTC_miners_revenue = pd.concat([miners_revenue, ass2BTCd], axis=1)

correlation_diff_price = watifBTC_diff.pct_change().corr(method='pearson')
correlation_hash_rate_price = watifBTC_hash_rate.pct_change().corr(method='pearson')
correlation_miners_revenue_price = watifBTC_miners_revenue.pct_change().corr(method='pearson')

#SPRAWDZIC CZY DNI SIE ZGADZAJA

#BTC price to google trends
BTC_google_trends_12m = pd.read_csv('Data/google_trends/bitcoin/bitcoin_google_12m.csv', sep=',')

BTC_google_trends_12m = BTC_google_trends_12m.iloc[1:]

BTC_google_trends_12m.index = pd.to_datetime(BTC_google_trends_12m.index)


BTC_google_trends_2017 = BTC_google_trends_12m[BTC_google_trends_12m.index.year == 2017]


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 100))

test = combined_df_2017.ix[:,11:12]
test = test.dropna(axis=0)
combined_df_2017_scale = scaler.fit_transform(test)

test2 = BTC_google_trends_2017.astype(np.float64)

test3 = np.asarray(test2) 


slajz = combined_df_2017_scale[0::7]

plt.plot(test3, color = 'red', label = 'google trends')
plt.plot(slajz, color = 'blue', label = 'btc price')
plt.legend()
plt.show







#LTC price to google trends 
LTC_google_trends_12m = pd.read_csv('Data/google_trends/litecoin/litecoin_google_12m.csv', sep=',')

LTC_google_trends_12m = LTC_google_trends_12m.iloc[1:]

LTC_google_trends_12m.index = pd.to_datetime(LTC_google_trends_12m.index)


LTC_google_trends_2017 = LTC_google_trends_12m[LTC_google_trends_12m.index.year == 2017]


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 100))

test = combined_df_2017.ix[:,5:6]
test = test.dropna(axis=0)
combined_df_2017_scale = scaler.fit_transform(test)

test2 = LTC_google_trends_2017.astype(np.float64)

test3 = np.asarray(test2) 


slajz = combined_df_2017_scale[0::7]

plt.plot(test3, color = 'red', label = 'google trends')
plt.plot(slajz, color = 'blue', label = 'ltc price')
plt.legend()
plt.show


