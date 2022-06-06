from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from keras.models import Sequential
from keras.layers import Dense
import keras.losses
import tensorflow as tf
from keras.layers import Dropout
import numpy as np

#S stands for scaled 
def Neighbours(STrainFeat, trainTarg, STestFeat, testTarg):
    #Visually inspect command window to find which number of neighbors is the best
    for i in range(2,13):
        KNN = KNeighborsRegressor(n_neighbors = i)

        KNN.fit(STrainFeat, trainTarg)

        print("number of neighbors = ", i)
        print('train, test scores')
        print(KNN.score(STrainFeat, trainTarg))
        print(KNN.score(STestFeat, testTarg))
        

def KerasModels(STrainFeat, trainTarg, STestFeat, testTarg):
    model_1 = Sequential()
    model_1.add(Dense(100, input_dim=STrainFeat.shape[1], activation='relu'))
    model_1.add(Dense(20, activation = 'relu'))
    model_1.add(Dense(1, activation = 'linear'))

    model_1.compile(optimizer ='adam', loss = 'mse')

    trainPreds1 = model_1.predict(STrainFeat)
    testPreds1 = model_1.predict(STestFeat)

    print("model 1")
    print(r2_score(trainTarg, trainPreds1))
    print(r2_score(testTarg, testPreds1))

    # Create loss function (punishes values that are in top left or bottom right of graph)
    def sign_penalty(y_true, y_pred):
        #Change penalty here
        penalty = 100.
        loss = tf.where(tf.less(y_true * y_pred, 0), \
                         penalty * tf.square(y_true - y_pred), \
                         tf.square(y_true - y_pred))

        return tf.reduce_mean(loss, axis=-1)

    keras.losses.sign_penalty = sign_penalty  # enable use of loss with keras
    print(keras.losses.sign_penalty)
          
    #create model 2 using custom loss
    model_2 = Sequential()
    model_2.add(Dense(100, input_dim=STrainFeat.shape[1], activation = 'relu'))
    model_2.add(Dense(20, activation = 'relu'))
    model_2.add(Dense(1, activation = 'linear'))

    #Fit model with custom loss function
    model_2.compile(optimizer='adam', loss=sign_penalty)

    print("model 2")
    trainPreds2 = model_2.predict(STrainFeat)
    testPreds2 = model_2.predict(STestFeat)

    print(r2_score(trainTarg, trainPreds2))
    print(r2_score(testTarg, testPreds2))

    #create model 3 using dropout to counter overfitting of models
    model_3 = Sequential()
    model_3.add(Dense(100, input_dim = STrainFeat.shape[1], activation = 'relu'))
    model_3.add(Dropout(0.2))
    model_3.add(Dense(20, activation = 'relu'))
    model_3.add(Dense(1, activation = 'linear'))

    model_3.compile(optimizer='adam', loss = 'mse')

    #Check loss function has flattened out (Cant be fucked)

    trainPreds3 = model_3.predict(STrainFeat)
    testPreds3 = model_3.predict(STestFeat)

    print("Model 3")
    print(r2_score(trainTarg, trainPreds3))
    print(r2_score(testTarg, testPreds3))

    trainPreds = np.mean(np.hstack((trainPreds1, trainPreds2, trainPreds3)), axis=1)
    testPreds = np.mean(np.hstack((testPreds1, testPreds2, testPreds3)), axis=1)

    print("Combined model")
    print(r2_score(trainTarg, trainPreds))
    print(r2_score(testTarg, testPreds))

