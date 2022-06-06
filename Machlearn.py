import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid



#This function fits a linear model to a training data set to make predictions
#For a test data set
def LinearModel(TrainFeat, TrainTarg, TestFeat, TestTarg):
    model = sm.OLS(TrainTarg, TrainFeat)

    results = model.fit()
    print('summary')
    print(results.summary())

    #pvalues are the percentage chance that the coefficient of the feature does not differ from 0
    #Lower value such as less then 0.05 means a lot different from zero
    print(results.pvalues)

    #trainPredictions = results.predict(TrainFeat)
    testPredictions = results.predict(TestFeat)

    TestScore = r2_score(TestTarg, testPredictions)

    return TestScore

#This function fits features into leaf nodes using the decision tree regressor module
def DecisionTree(trainFeat, trainTarg, testFeat, testTarg):
    Scores = []
    Trains = []
    #Loop through max depths to optimize model
    for i in range(2, 8):
        decisionTree = DecisionTreeRegressor(max_depth = i)#max number of trees grouping features together
        decisionTree.fit(trainFeat, trainTarg)

        trainPredict = decisionTree.predict(trainFeat)
        testPredict = decisionTree.predict(testFeat)

        TestScore = r2_score(testTarg, testPredict)
        TrainScore = r2_score(trainTarg, trainPredict)
        
        Scores.append(TestScore)
        Trains.append(TrainScore)

    BestScoreIndex = np.argmax(Scores)
    print("Best Depth: ")
    print(BestScoreIndex+2)
    
    return Scores[BestScoreIndex], Trains[BestScoreIndex]

#This function creates a dictionary of hyperparameters for a RandomForestRegressor and finds
#The best ones
def FindBestGridRFR(trainFeat, trainTarg, testFeat, testTarg):
    RFR = RandomForestRegressor()
    #Creates dictionary for hyper parameters
    #SET Possible parameters here to test
    #nestimators is number of trees in forest
    grid = {'n_estimators': [50,100,150,200,250], 'max_depth': [3,4,5,6,7,8,9,10], 'max_features': [4,6,8,10], 'random_state': [0,42]}
    testScores = []
    Trains = []
    
    for g in ParameterGrid(grid):
        RFR.set_params(**g) # ** is unpacking the grid
        RFR.fit(trainFeat, trainTarg)

        trainPredict = RFR.predict(trainFeat)
        testPredict = RFR.predict(testFeat)
    
        testScores.append(r2_score(testTarg, testPredict))
        Trains.append(r2_score(trainTarg, trainPredict))

    bestScoreIndex = np.argmax(testScores)
    print(ParameterGrid(grid)[bestScoreIndex])

    return testScores[bestScoreIndex], Trains[bestScoreIndex]

#This function creates a dictionary of hyperparameters for a GradientBoostingRegressor and finds
#The best ones
def FindBestGridGBR(trainFeat, trainTarg, testFeat, testTarg):
    GBR = GradientBoostingRegressor()
    #Creates dictionary for hyper parameters
    #SET Possible parameters here to test
    grid = {'n_estimators': [50,100,150,200,250], 'max_depth': [2,3,4,5,6,7,8,9,10], 'random_state': [0,42], 'learning_rate':[0.005,0.01, 0.1], 'subsample':[0.6,1]}
    testScores = []
    Trains = []

    for g in ParameterGrid(grid):
        GBR.set_params(**g) # ** is unpacking the grid
        GBR.fit(trainFeat, trainTarg)

        trainPredict = GBR.predict(trainFeat)
        testPredict = GBR.predict(testFeat)
    
        testScores.append(r2_score(testTarg, testPredict))
        Trains.append(r2_score(trainTarg, trainPredict))

    bestScoreIndex = np.argmax(testScores)
    #print(ParameterGrid(grid)[bestScoreIndex])

    GBR.set_params(**ParameterGrid(grid)[bestScoreIndex])
    GBR.fit(trainFeat, trainTarg)

    testPredict = GBR.predict(testFeat)
    #print(testPredict[-1])    

    return testScores[bestScoreIndex], Trains[bestScoreIndex], testPredict








    
