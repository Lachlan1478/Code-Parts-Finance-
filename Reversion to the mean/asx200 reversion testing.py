#This program reads in the ticker value of the all ordinaries or the asx 200
#From a text file, retrieves data given a certain interval and range, then
#Prints any significant movements in prices

import requests
import pandas as pd
from datetime import datetime
import arrow

import time

import Machsetup as ms

Balance = 5000


Int = '1d'
Rnge = '504d'

Path = "ALLORDS_Tickers.txt"

Changes = 0
sucChanges = 0
alrightChanges = 0
badChanges = 0

PercentDiff = 0

Read = open(Path, "r")
for i in Read:
    Tickers = [line.strip('"'+'\n') for line in open(Path)]

Length = len(Tickers)
for i in range(0, Length):
    Tickers[i] = Tickers[i]+'.AX'

print(Length)


for i in range(0, Length):

    #Removal of any stocks that cause issues
    #Update index value considering deleted values
    j = i - (Length - len(Tickers))
    print(j)
    try:
        Stock = ms.GatherData(Tickers[j], Rnge, Int)
        
        IndexList = Stock.index.tolist()
        
        
        for x in range(100, 358):
            #close of day before
            Past = Stock.iloc[-x - 1]['close']
            Present = Stock.iloc[-x]['close']

            #Calculate Percentage Change
            PercentageChange = (Present - Past)/Past * 100

            if (PercentageChange < -20):
                Changes += 1

                Fut = Stock.iloc[-x + 5]['close']

                FutPercentageChange = (Fut - Present)/Present * 100

                Balance = Balance + 5000 * FutPercentageChange/100 - 40

                PercentDiff += FutPercentageChange

                print(IndexList[-x])
                print(FutPercentageChange)                

                if (FutPercentageChange > 5):
                    sucChanges += 1
                    print("good")                    
                    
                if(FutPercentageChange > 0 and FutPercentageChange < 5):
                    alrightChanges += 1
                    print("alright")
                    
                if(FutPercentageChange < -5):
                    badChanges += 1
                    print("bad")

                
        
    except:
        print(Tickers[j])
        del Tickers[j]


print(f"Number of changes: {Changes}")
print(f"Number of successful changes: {sucChanges}")
print(f"Number of alright changes: {alrightChanges}")
print(f"Number of bad changes: {badChanges}")

print(f"Percentage: {sucChanges/Changes * 100}")

print(Balance)
print(PercentDiff)


    














    
