import Machsetup as ms
import Machlearn as ml

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

Tickers = ['Z1P.AX', 'BHP.AX', 'ORG.AX']
Rnge = '300d'
Int = '1d'

Stock1 = ms.GatherClose(Tickers[0], Rnge, Int)
Stock2 = ms.GatherClose(Tickers[1], Rnge, Int)
Stock3 = ms.GatherClose(Tickers[2], Rnge, Int)

JointDf = pd.concat([Stock1, Stock2, Stock3], axis=1).dropna()

#Sort into monthly data values only
monthlyDf = JointDf.resample('BMS').first()


DailyReturns = JointDf.pct_change()

MonthlyReturns = monthlyDf.pct_change().dropna()

print(MonthlyReturns.tail())

#convariances for each month
covariances = {}

rtdIdx = DailyReturns.index

for i in MonthlyReturns.index:
    #Mask daily returns for each month and year
    mask = (rtdIdx.month == i.month) & (rtdIdx.year == i.year)
    covariances[i] = DailyReturns[mask].cov()

    print(covariances[i])

PortReturns, PortVolatility, PortWeights = {}, {}, {}

for date in sorted(covariances.keys()):
    cov = covariances[date]
    #Generate random weights for each stock to find the best one for each month
    #Change range to bigger numbers for more effective use
    for portfolio in range(5000):
        #Generate three random number between 0 and 1
        weights = np.random.random(3)
        #Divide each weight by sum to normalize values (Sum to one)
        weights /= np.sum(weights)
        returns = np.dot(weights, MonthlyReturns.loc[date])
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        
        PortReturns.setdefault(date, []).append(returns)
        PortVolatility.setdefault(date, []).append(volatility)
        PortWeights.setdefault(date, []).append(weights)

print(PortWeights[date][0])

date = sorted(covariances.keys())[-1]

#plt.scatter(x = PortVolatility[date], y = PortReturns[date], alpha = 0.1)
#plt.xlabel('Volatility')
#plt.ylabel('Returns')
#plt.show()

SharpeRatio, maxSharpeIdxs = {}, {}
for date in PortReturns.keys():
    for i, ret in enumerate(PortReturns[date]):
        SharpeRatio.setdefault(date, []).append(ret / PortVolatility[date][i])

    maxSharpeIdxs[date] = np.argmax(SharpeRatio[date])

print(PortReturns[date][maxSharpeIdxs[date]])

#Create Features (exponentiall-weighted moving average)
ewmaDaily = DailyReturns.ewm(span=30).mean()
ewmaMonthly = ewmaDaily.resample('BMS').first()
ewmaMonthly = ewmaMonthly.shift(1).dropna()

print(ewmaMonthly.iloc[-1])

targets, features = [], []

for date, ewma in ewmaMonthly.iterrows():
    bestIdx = maxSharpeIdxs[date]
    targets.append(PortWeights[date][bestIdx])
    features.append(ewma)

targets = np.array(targets)
features = np.array(features)
print(targets[-5:])

date = sorted(covariances.keys())[-1]
CurrentReturns = PortReturns[date]
CurrentVolatility = PortVolatility[date]

plt.scatter(x = CurrentVolatility, y = CurrentReturns, alpha=0.1, color = 'blue')
bestIdx = maxSharpeIdxs[date]

plt.scatter(x = CurrentVolatility[bestIdx], y = CurrentReturns[bestIdx], marker = 'x', color = 'orange')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.show()

trainSize = int(0.85 * features.shape[0])
trainFeat = features[:trainSize]
testFeat = features[trainSize:]
trainTarg = targets[:trainSize]
testTarg = targets[trainSize:]

RFRScore, RFRTrain = ml.FindBestGridRFR(trainFeat, trainTarg, testFeat, testTarg)
print("Random forest regressor score: ")
print(RFRTrain)
print(RFRScore)


      

        



    
