import requests
import pandas as pd
from datetime import datetime
import arrow

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
import statsmodels.api as sm
import numpy as np

import talib


def FeaturesTargets(stock):
    #Create targets in this case a pct_change for x days in the future given interval is 2
    stock['5dFut'] = stock['close'].shift(-5)
    stock['5dFutPct'] = stock['5dFut'].pct_change(5)
    stock['5dClosePct'] = stock['close'].pct_change(5)

    featureNames = ['5dClosePct']

    DailyReturns = stock['close'].pct_change()

    #Create moving averages and RSI's with fiven ranges, this is to find something that reflects
    #a correlation with the target being the 5dFutPct
    for i in [5, 15, 50, 150]:
        stock['MovAv' + str(i)] = talib.SMA(stock['close'].values,timeperiod = i)
        stock['RSI' + str(i)] = talib.RSI(stock['close'].values, timeperiod = i)
        stock['Ewma' + str(i)] = DailyReturns.ewm(span = i).mean()

        featureNames = featureNames + ['MovAv' + str(i), 'RSI' + str(i), 'Ewma' + str(i)]

    #remove Nan values
    stock = stock.dropna()

    pd.set_option('mode.chained_assignment', None)

    #more features
    newFeatures = ['Volume1dChange', 'Volume1dChangeSMA']
    featureNames.extend(newFeatures)
    stock['Volume1dChange'] = stock['volume'].pct_change()
    stock['Volume1dChangeSMA'] = talib.SMA(stock['Volume1dChange'].values, timeperiod = 5)

    #remove NaN values again as it volume features cause bugs to linear modelling (Unknown error)
    stock = stock.dropna()

    features = stock[featureNames]
    targets = stock['5dFutPct']

    #Creates data frame for just target and features
    FeatureTargetCols = ['5dFutPct'] + featureNames
    FeatureTargetDf = stock[FeatureTargetCols]

    #Adds constant to features data frame
    linearFeatures = sm.add_constant(features)

    #85 percent of data set use to train machine learning
    trainSize = int(0.85 * features.shape[0])

    trainFeatures = linearFeatures[:trainSize]
    trainTargets = targets[:trainSize]

    testFeatures = linearFeatures[trainSize:]
    testTargets = targets[trainSize:]

    return linearFeatures[:trainSize], targets[:trainSize], linearFeatures[trainSize:], targets[trainSize:], FeatureTargetDf
    
    

#This function retrives and stores stock data in data frame
def GatherData(Symbol, Range, Interval):
    Request = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/{Symbol}?range={Range}&interval={Interval}'.format(**locals()))
    data = Request.json()
    body = data['chart']['result'][0]
    Time = pd.Series(body['timestamp'])
    #To convert to AEST 15 hours (54000 seconds) difference with EST
    Constant = pd.Series([54000 for x in range(len(Time.index))])
    Time = Time.add(Constant)
    Date = pd.Series(map(lambda x: arrow.get(x).to('EST').datetime.replace(tzinfo=None), Time), name='Datetime(AEST)')
    Frame = pd.DataFrame(body['indicators']['quote'][0], index=Date)

    #Only retrieve these values of data
    return Frame.loc[:, ('open', 'high', 'low', 'close', 'volume')]

def GatherClose(Symbol, Range, Interval):
    Request = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/{Symbol}?range={Range}&interval={Interval}'.format(**locals()))
    data = Request.json()
    body = data['chart']['result'][0]
    Time = pd.Series(body['timestamp'])
    #To convert to AEST 15 hours (54000 seconds) difference with EST
    Constant = pd.Series([54000 for x in range(len(Time.index))])
    Time = Time.add(Constant)
    Date = pd.Series(map(lambda x: arrow.get(x).to('EST').datetime.replace(tzinfo=None), Time), name='Datetime(AEST)')
    Frame = pd.DataFrame(body['indicators']['quote'][0], index=Date)

    #Only retrieve these values of data
    return Frame.loc[:, ('close')]

#This function produces a heatmap given a correlation dataframe
def Heatmap(Corr):
    sns.heatmap(Corr, annot = True, annot_kws = {"size" : 10})
    #Plot heatmap
    plt.yticks(rotation=0, size =14); plt.xticks(rotation=90, size = 14)
    #fits heatmap tightly
    plt.tight_layout()
    plt.show()





    
