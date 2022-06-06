import requests
import pandas as pd
from datetime import datetime
import arrow
import matplotlib.pyplot as plt
import talib
import seaborn as sns

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

Stock = GatherData(Ticker, Rnge, Int);

length = len(Stock.index)-1

'close of the day before'
Past = Stock.iloc[length - 6]['close']
Present = Stock.iloc[length]['close']

'Calculate Percentage Change 1day'
PercentageChange = (Present - Past)/Past *100

print(PercentageChange)
print(Stock.head())

#remove NaN values
Stock = Stock.dropna()
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
    

features = Stock[featureNames]
targets = Stock['5dFutPct']

FeatureTargetCols = ['5dFutPct'] + featureNames
FeatureTargetDf = Stock[FeatureTargetCols]

#Find Pearson correlation between variables(-1 negatively correlated, 1 positively correlated)
correlation = FeatureTargetDf.corr()

#Heatmap to view correlations
sns.heatmap(correlation, annot = True, annot_kws = {"size" : 10})

#Plot heatmap
plt.yticks(rotation=0, size =14); plt.xticks(rotation=90, size = 14)
#fits heatmap tightly
plt.tight_layout()
plt.show()











                                        
    

