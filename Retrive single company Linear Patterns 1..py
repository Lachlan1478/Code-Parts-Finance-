import requests
import pandas as pd
from datetime import datetime
import arrow
import matplotlib.pyplot as plt
import talib
import seaborn as sns
import statsmodels.api as sm
import numpy as np

Int = '60m'
Rnge = '200d'
Ticker = 'APT.AX'

#This function retrives and stores stock data in data frame
def GatherData(Symbol, Range, Interval):
    Request = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/{Symbol}?range={Range}&interval={Interval}'.format(**locals()))
    data = Request.json()
    body = data['chart']['result'][0]
    Time = pd.Series(body['timestamp'])
    Date = pd.Series(map(lambda x: arrow.get(x).to('EST').datetime.replace(tzinfo=None), body['timestamp']), name='Datetime(EST)')
    Frame = pd.DataFrame(body['indicators']['quote'][0], index=Date)

    #Only retrieve these values of data
    return Frame.loc[:, ('open', 'high', 'low', 'close', 'volume')]

#This function produces a heatmap given a correlation dataframe
def Heatmap(Corr):
    sns.heatmap(Corr, annot = True, annot_kws = {"size" : 10})
    #Plot heatmap
    plt.yticks(rotation=0, size =14); plt.xticks(rotation=90, size = 14)
    #fits heatmap tightly
    plt.tight_layout()
    plt.show()

#####
##############
#####
    
Stock = GatherData(Ticker, Rnge, Int);

length = len(Stock.index)-1

'close of the day before'
Past = Stock.iloc[length - 6]['close']
Present = Stock.iloc[length]['close']

'Calculate Percentage Change 1day'
PercentageChange = int((Present - Past)/Past *100)

print("Percentage Change is: " + str(PercentageChange))
print(Stock.head())


Stock['5dFut'] = Stock['close'].shift(-5)
Stock['5dFutPct'] = Stock['5dFut'].pct_change(5)
Stock['5dClosePct'] = Stock['close'].pct_change(5)

featureNames = ['5dClosePct']

#Create moving averages and RSI's with fiven ranges, this is to find something that reflects
#a correlation with the target being the 5dFutPct
for i in [14, 30, 50, 200]:
    Stock['MovAv' + str(i)] = talib.SMA(Stock['close'].values,timeperiod = i) /Stock['close']
    Stock['RSI' + str(i)] = talib.RSI(Stock['close'].values, timeperiod = i)

    featureNames = featureNames + ['MovAv' + str(i), 'RSI' + str(i)]
    

#remove NaN values
Stock = Stock.dropna()

features = Stock[featureNames]

print(features)

targets = Stock['5dFutPct']

FeatureTargetCols = ['5dFutPct'] + featureNames
FeatureTargetDf = Stock[FeatureTargetCols]

#Pearson correlation between variables(-1 negatively correlated, 1 positively correlated)
correlation = FeatureTargetDf.corr()

#Adds constant to data frame
linearFeatures = sm.add_constant(features)


#85 percent of data set use to train machine learning
trainSize = int(0.85 * features.shape[0])

trainFeatures = linearFeatures[:trainSize]
trainTargets = targets[:trainSize]

testFeatures = linearFeatures[trainSize:]
testTargets = targets[trainSize:]

print('step 1')

#
model = sm.OLS(trainTargets, trainFeatures)

results = model.fit()
print('summary')
print(results.summary())
print('lol')

#pvalues are the percentage chance that the coefficient of the feature does not differ from 0
#Lower value such as less then 0.05 means a lot different from zero
print(results.pvalues)

trainPredictions = results.predict(trainFeatures)
testPredictions = results.predict(testFeatures)

plt.scatter(trainPredictions,trainTargets,alpha =0.2,color ='b',label ='train')
plt.scatter(testPredictions, testTargets,alpha=0.2, color = 'r', label='test')

xmin, xmax = plt.xlim()
plt.plot(np.arange(xmin,xmax,0.01), np.arange(xmin,xmax,0.01), c='k')

plt.xlabel('predictions')
plt.ylabel('actual')
plt.legend()
plt.show()












    












                                        
    

