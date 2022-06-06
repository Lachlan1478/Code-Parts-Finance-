import Machsetup as ms
import Machlearn as ml

import pandas as pd


Tickers = []

TopScores = []
TopTickers = []

Rnge = '300d'
Int = '1d'

Path = "ASX200_Tickers.txt"

Read = open(Path, "r")
for i in Read:
    Tickers = [line.strip('"'+'\n') for line in open(Path)]

Length = len(Tickers)
for i in range(0, Length):
    Tickers[i] = Tickers[i]+'.AX'

print(Tickers)
print(Length)
for i in range(0, Length):
    #Removal of any stocks that cause issues
    #Update index value considering deleted values
    j = i - (Length - len(Tickers))
    print(j)

    try:
        Stock = ms.GatherData(Tickers[j], Rnge, Int)
    except:
        print(Tickers[j])
        del Tickers[j]

    #Create features and Targets
    trainFeat, trainTarg, testFeat, testTarg, FeatTargdf = ms.FeaturesTargets(Stock)

    #correlation for possible heatmap
    correlation = FeatTargdf.corr()


    #Top features
    x = correlation['5dFutPct']
    x = x.abs()
    x.sort_values(ascending=False, inplace=True)

    #Top five features
    #names of top 5 featuress
    y = list(x.index[1:6])

    GBRScore, GBRTrain, GBRPredict = ml.FindBestGridGBR(trainFeat, trainTarg, testFeat, testTarg)

    print(Tickers[j])
    print(f"Full features: {GBRScore}")
    if(GBRScore > 0.5):
        TopScores.append(GBRScore)
        TopTickers.append(Tickers[j])
        print("not edited")

    trainFeat = trainFeat[y]
    testFeat = testFeat[y]
        
    GBRScore, GBRTrain, GBRPredict = ml.FindBestGridGBR(trainFeat, trainTarg, testFeat, testTarg)

    
    print(f"Shortened Features: {GBRScore}")
    
          
    if(GBRScore > 0.5):
        TopScores.append(GBRScore)
        TopTickers.append(Tickers[j])
        print("edited")

print(TopTickers)
print(TopScores)
        
