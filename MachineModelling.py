from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import pandas as pd

#Personal libraries
import Machsetup as ms
import Machlearn as ml


Ticker = 'WOW.AX'
Rnge = '500d'
Int = '1d'

Stock = ms.GatherData(Ticker, Rnge, Int)
print(Stock.head())

#Create features and Targets
trainFeat, trainTarg, testFeat, testTarg, FeatTargdf = ms.FeaturesTargets(Stock)



#correlation for possible heatmap
correlation = FeatTargdf.corr()
print(correlation)

#Top features
x = correlation['5dFutPct']
x = x.abs()
x.sort_values(ascending=False, inplace=True)

#Top five features
#names of top 5 featuress
y = list(x.index[1:6])
print("Top five features: ", y)
#Only use top 5 features
#trainFeat = trainFeat[y]
#testFeat = testFeat[y]

#Create linear model
linearScore = ml.LinearModel(trainFeat, trainTarg, testFeat, testTarg)
print("Linear Model Score: ")
print(linearScore)

DecisionScore, DecisionTrain = ml.DecisionTree(trainFeat, trainTarg, testFeat, testTarg)
print("Decision Tree Score: ")
print(DecisionTrain)
print(DecisionScore)

#RFRScore, RFRTrain = ml.FindBestGridRFR(trainFeat, trainTarg, testFeat, testTarg)
#print("Random forest regressor score: ")
#print(RFRTrain)
#print(RFRScore)

GBRScore, GBRTrain, GBRPredict = ml.FindBestGridGBR(trainFeat, trainTarg, testFeat, testTarg)
print("Gradient Boosting regressor score: ")
print(GBRTrain)
print(GBRScore)

size = int(0.85 * Stock.shape[0])

plt.plot(Stock['close'][:size], color = 'green')
plt.plot(Stock['close'][size:], color = 'r')

plt.show()

scaledTrainFeat = scale(trainFeat)
scaledTestFeat = scale(testFeat)

#ma.Neighbours(scaledTrainFeat, trainTarg, scaledTestFeat, testTarg)

#ma.KerasModels(scaledTrainFeat, trainTarg, scaledTestFeat, testTarg)






