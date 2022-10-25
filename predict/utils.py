from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from keras.layers import Dense,LSTM , GRU, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import ElasticNet
import os
import copy
import datetime
import joblib
import tensorflow as tf

# setting a seed for reproducibility
numpy.random.seed(10)
# read all stock files in directory indivisual_stocks_5yr
def read_all_stock_files(folder_path):
    allFiles = []
    for (_, _, files) in os.walk(folder_path):
        allFiles.extend(files)
        break

    dataframe_dict = {}
    for stock_file in allFiles:
        df = pd.read_csv(folder_path + "/" +stock_file)
        dataframe_dict[(stock_file.split('_'))[0]] = df

    return dataframe_dict
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)
# create dataset from the dataframe
# def create_preprocessed_Dataset(df,look_back=1):
#     df.drop(df.columns.difference(['date', 'open']), 1, inplace=True)
#     df = df['open']
#     dataset = df.values
#     dataset = dataset.reshape(-1, 1)
#     dataset = dataset.astype('float32')

#     # split into train and test sets
#     train_size = len(dataset) - 2
#     train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

#     # reshape into X=t and Y=t+1
#     trainX, trainY = create_dataset(train, look_back)
#     testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    # trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    # testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    return trainX, trainY, testX, testY
# extract input dates and opening price value of stocks
def getData(df):
    # Create the lists / X and Y data sets
    dates = []
    prices = []

    # Get the number of rows and columns in the data set
    # df.shape

    # Get the last row of data (this will be the data that we test on)
    last_row = df.tail(1)

    # Get all of the data except for the last row
    df = df.head(len(df) - 1)
  

    # df

    # The new shape of the data
    # df.shape

    # Get all of the rows from the Date Column
    df_dates = df.loc[:, 'date']
    # Get all of the rows from the Open Column
    df_open = df.loc[:, 'open']

    # Create the independent data set X
    for date in df_dates:
        dates.append([int(date.split('-')[2])])

    # Create the dependent data se 'y'
    for open_price in df_open:
        prices.append(float(open_price))

    # See what days were recorded
    last_date = int(((list(last_row['date']))[0]).split('-')[2])
    last_price = float((list(last_row['open']))[0])
    return dates, prices, last_date, last_price


# def SVR_linear(dates, prices, test_date, df):
#     svr_lin = SVR(kernel='linear', C=1e3)
#     trainX, trainY, testX, testY = create_preprocessed_Dataset(df)
#     X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size = 0.33, random_state = 42)
#     svr_lin.fit(X_train, y_train)
#     decision_boundary = svr_lin.predict(trainX)
#     y_pred = svr_lin.predict(X_test)
#     test_score = mean_squared_error(y_test, y_pred)
#     prediction = svr_lin.predict(testX)[0]

#     return (decision_boundary, prediction, test_score)

# def SVR_rbf(dates, prices, test_date, df):
#     svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#     trainX, trainY, testX, testY = create_preprocessed_Dataset(df)
#     # trainX = [item for sublist in trainX for item in sublist]
#     # testX = [item for sublist in testX for item in sublist]
#     X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
#     svr_rbf.fit(trainX, trainY)
#     decision_boundary = svr_rbf.predict(trainX)
#     y_pred = svr_rbf.predict(X_test)
#     test_score = mean_squared_error(y_test, y_pred)
#     prediction = svr_rbf.predict(testX)[0]
#     return (decision_boundary, prediction, test_score)

# def linear_regression(dates, prices, test_date, df):
#     lin_reg = LinearRegression()
#     trainX, trainY, testX, testY = create_preprocessed_Dataset(df)
#     # trainX = [item for sublist in trainX for item in sublist]
#     # testX = [item for sublist in testX for item in sublist]
#     X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
#     lin_reg.fit(trainX, trainY)
#     decision_boundary = lin_reg.predict(trainX)
#     y_pred = lin_reg.predict(X_test)
#     test_score = mean_squared_error(y_test, y_pred)
#     prediction = lin_reg.predict(testX)[0]
#     return (decision_boundary, prediction, test_score)

# def random_forests(dates, prices, test_date, df):
#     rand_forst = RandomForestRegressor(n_estimators=10, random_state=0)
#     trainX, trainY, testX, testY = create_preprocessed_Dataset(df)
#     # trainX = [item for sublist in trainX for item in sublist]
#     # testX = [item for sublist in testX for item in sublist]
#     X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
#     rand_forst.fit(trainX, trainY)
#     decision_boundary = rand_forst.predict(trainX)
#     y_pred = rand_forst.predict(X_test)
#     test_score = mean_squared_error(y_test, y_pred)
#     prediction = rand_forst.predict(testX)[0]

#     return (decision_boundary, prediction, test_score)

# def KNN(dates, prices, test_date, df):
#     knn = KNeighborsRegressor(n_neighbors=2)
#     trainX, trainY, testX, testY = create_preprocessed_Dataset(df)
#     # trainX = [item for sublist in trainX for item in sublist]
#     # testX = [item for sublist in testX for item in sublist]
#     X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
#     knn.fit(trainX, trainY)
#     decision_boundary = knn.predict(trainX)
#     y_pred = knn.predict(X_test)
#     test_score = mean_squared_error(y_test, y_pred)
#     prediction = knn.predict(testX)[0]

#     return (decision_boundary, prediction, test_score)

# def DT(dates, prices, test_date, df):
#     decision_trees = tree.DecisionTreeRegressor()
#     trainX, trainY, testX, testY = create_preprocessed_Dataset(df)
#     # trainX = [item for sublist in trainX for item in sublist]
#     # testX = [item for sublist in testX for item in sublist]
#     X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
#     decision_trees.fit(trainX, trainY)
#     decision_boundary = decision_trees.predict(trainX)
#     y_pred = decision_trees.predict(X_test)
#     test_score = mean_squared_error(y_test, y_pred)
#     prediction = decision_trees.predict(testX)[0]
#     return (decision_boundary, prediction, test_score)

# def elastic_net(dates, prices, test_date, df):
#     regr = ElasticNet(random_state=0)
#     trainX, trainY, testX, testY = create_preprocessed_Dataset(df)
#     # trainX = [item for sublist in trainX for item in sublist]
#     # testX = [item for sublist in testX for item in sublist]
#     X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.33, random_state=42)
#     regr.fit(trainX, trainY)
#     decision_boundary = regr.predict(trainX)
#     y_pred = regr.predict(X_test)
#     test_score = mean_squared_error(y_test, y_pred)
#     prediction = regr.predict(testX)[0]

#     return (decision_boundary, prediction, test_score)

def LSTM_model(dates, prices, test_date, df,look_back = 1,epochs=50, batch_size=8,d=0.2):
    df.drop(df.columns.difference(['date', 'open']), 1, inplace=True)
    df = df['open']
    dataset = df.values
    dataset = dataset.reshape(-1, 1)
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = len(dataset) - look_back -1
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.10, random_state=42)
    # print(f'X_train.shape:{X_train.shape}')
    # print(f'X_test.shape:{X_test.shape}')
    # print(f'y_train.shape:{y_train.shape}')
    # print(f'y_test.shape:{y_test.shape}')

    train_rate=0.80
    train_size=int(trainX.shape[0]*train_rate)
    X_train = trainX[:train_size]
    X_test =  trainX[train_size:]
    y_train = trainY[:train_size]
    y_test = trainY[train_size:]

    # print(f'XX_train.shape:{X_train.shape}')
    # print(f'XX_test.shape:{X_test.shape}')
    # print(f'yy_train.shape:{y_train.shape}')
    # print(f'yy_test.shape:{y_test.shape}')

    # reshape input to be [samples, time steps, features]
    X_train = numpy.reshape(X_train, (X_train.shape[0], look_back, 1))
    
    X_test = numpy.reshape(X_test, (X_test.shape[0], look_back, 1))
    testX = numpy.reshape(testX, (testX.shape[0], look_back,1))
    # create and fit the LSTM network
    model = Sequential()
    # bug修正：look_back和feature number 放反
    model.add(LSTM(64,return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(d))
    model.add(LSTM(64,return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(32))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())

    # tarin model
    start_time=datetime.datetime.now()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    end_time=datetime.datetime.now()
    time_delta=(end_time-start_time).seconds

    # make predictions
    trainPredict = model.predict(X_train)
    mainTestPredict = model.predict(X_test)
    #testPredict = model.predict(testX)

    # future prediction
    predict_period=10
    futureX=copy.deepcopy(testX)
    predictList=[]
    for _ in range(predict_period):
        testPredict = model.predict(futureX)
        predictList.append(testPredict)
        futureX=np.delete(np.concatenate((futureX,testPredict.reshape(1,1,1)),axis=1),0,axis=1)
        
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    y_train = scaler.inverse_transform([y_train])

    #testPredict = scaler.inverse_transform(testPredict)
    predictList=[scaler.inverse_transform(x)[0][0] for x in predictList ]
    testY = scaler.inverse_transform([testY])

    mainTestPredict = scaler.inverse_transform(mainTestPredict)
    mainTestPredict = [item for sublist in mainTestPredict for item in sublist]
    y_test = scaler.inverse_transform([y_test])
    test_score=[]
    test_score.append(mean_squared_error(y_test[0], mainTestPredict))
    test_score.append(r2_score(y_test[0], mainTestPredict))
    test_score.append(time_delta)
    #test_score = mean_squared_error(y_test[0], mainTestPredict)
    # calculate root mean squared error
    trainPredict = [item for sublist in trainPredict for item in sublist]
    #print(trainPredict, predictList)

    #for drawing piture


    #return (trainPredict, (testPredict[0]), test_score)
    return (trainPredict+mainTestPredict, predictList, test_score)

def GRU_model(dates, prices, test_date, df,look_back = 1,epochs=50, batch_size=8, d=0.2):
    df.drop(df.columns.difference(['date', 'open']), 1, inplace=True)
    df = df['open']
    dataset = df.values
    dataset = dataset.reshape(-1, 1)
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = len(dataset) - look_back -1
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    #X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.20, random_state=42)
    train_rate=0.80
    train_size=int(trainX.shape[0]*train_rate)
    X_train = trainX[:train_size]
    X_test =  trainX[train_size:]
    y_train = trainY[:train_size]
    y_test = trainY[train_size:]

    # print(f'XX_train.shape:{X_train.shape}')
    # print(f'XX_test.shape:{X_test.shape}')
    # print(f'yy_train.shape:{y_train.shape}')
    # print(f'yy_test.shape:{y_test.shape}')







    # reshape input to be [samples, time steps, features]
    X_train = numpy.reshape(X_train, (X_train.shape[0], look_back, 1))
    X_test = numpy.reshape(X_test, (X_test.shape[0], look_back, 1))
    testX = numpy.reshape(testX, (testX.shape[0], look_back,1))
    # create and fit the LSTM network
    model = Sequential()
    # bug修正：look_back和feature number 放反
    model.add(GRU(64,return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(d))
    model.add(GRU(64,return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(32))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())

    # train model
    start_time=datetime.datetime.now()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    end_time=datetime.datetime.now()
    time_delta=(end_time-start_time).seconds

    # make predictions
    trainPredict = model.predict(X_train)
    mainTestPredict = model.predict(X_test)
    #testPredict = model.predict(testX)

    # future prediction
    predict_period=10
    futureX=copy.deepcopy(testX)
    predictList=[]
    for _ in range(predict_period):
        testPredict = model.predict(futureX)
        predictList.append(testPredict)
        futureX=np.delete(np.concatenate((futureX,testPredict.reshape(1,1,1)),axis=1),0,axis=1)
        


    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    y_train = scaler.inverse_transform([y_train])

    #testPredict = scaler.inverse_transform(testPredict)
    predictList=[scaler.inverse_transform(x)[0][0] for x in predictList ]
    testY = scaler.inverse_transform([testY])

    mainTestPredict = scaler.inverse_transform(mainTestPredict)
    mainTestPredict = [item for sublist in mainTestPredict for item in sublist]
    y_test = scaler.inverse_transform([y_test])
    test_score=[]
    test_score.append(mean_squared_error(y_test[0], mainTestPredict))
    test_score.append(r2_score(y_test[0], mainTestPredict))
    test_score.append(time_delta)
    #test_score = mean_squared_error(y_test[0], mainTestPredict)
    
    # calculate root mean squared error
    trainPredict = [item for sublist in trainPredict for item in sublist]
    #print(trainPredict, predictList)

    #return (trainPredict, (testPredict[0]), test_score)
    return (trainPredict+mainTestPredict, predictList, test_score)

def LSTM_cci30_model(dates, prices, test_date, df,look_back = 10,epochs=50, batch_size=8, d=0.2):
    look_back = 10
    df.drop(df.columns.difference(['date', 'open']), 1, inplace=True)
    df = df['open']
    dataset = df.values
    dataset = dataset.reshape(-1, 1)
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = joblib.load('predict/scaler_cci30_5.save')
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = len(dataset) - look_back -1
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    #X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.20, random_state=42)
    train_rate=0.80
    train_size=int(trainX.shape[0]*train_rate)
    X_train = trainX[:train_size]
    X_test =  trainX[train_size:]
    y_train = trainY[:train_size]
    y_test = trainY[train_size:]

    # print(f'XX_train.shape:{X_train.shape}')
    # print(f'XX_test.shape:{X_test.shape}')
    # print(f'yy_train.shape:{y_train.shape}')
    # print(f'yy_test.shape:{y_test.shape}')


    # reshape input to be [samples, time steps, features]
    X_train = numpy.reshape(X_train, (X_train.shape[0], look_back, 1))
    X_test = numpy.reshape(X_test, (X_test.shape[0], look_back, 1))
    testX = numpy.reshape(testX, (testX.shape[0], look_back,1))
    # create and fit the LSTM network
    model = tf.keras.models.load_model('predict/bdse27_lstm_model_cci30.h5')

    # train model
    start_time=datetime.datetime.now()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    end_time=datetime.datetime.now()
    time_delta=(end_time-start_time).seconds

    # make predictions
    trainPredict = model.predict(X_train)
    mainTestPredict = model.predict(X_test)
    #testPredict = model.predict(testX)

    # future prediction
    predict_period=10
    futureX=copy.deepcopy(testX)
    predictList=[]
    for _ in range(predict_period):
        testPredict = model.predict(futureX)
        predictList.append(testPredict)
        futureX=np.delete(np.concatenate((futureX,testPredict.reshape(1,1,1)),axis=1),0,axis=1)
        


    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    y_train = scaler.inverse_transform([y_train])

    #testPredict = scaler.inverse_transform(testPredict)
    predictList=[scaler.inverse_transform(x)[0][0] for x in predictList ]
    testY = scaler.inverse_transform([testY])

    mainTestPredict = scaler.inverse_transform(mainTestPredict)
    mainTestPredict = [item for sublist in mainTestPredict for item in sublist]
    y_test = scaler.inverse_transform([y_test])
    test_score=[]
    test_score.append(mean_squared_error(y_test[0], mainTestPredict))
    test_score.append(r2_score(y_test[0], mainTestPredict))
    test_score.append(time_delta)
    #test_score = mean_squared_error(y_test[0], mainTestPredict)
    
    # calculate root mean squared error
    trainPredict = [item for sublist in trainPredict for item in sublist]
    #print(trainPredict, predictList)

    #return (trainPredict, (testPredict[0]), test_score)
    return (trainPredict+mainTestPredict, predictList, test_score)

def GRU_cci30_model(dates, prices, test_date, df,look_back = 10,epochs=50, batch_size=8, d=0.2):
    look_back = 10
    df.drop(df.columns.difference(['date', 'open']), 1, inplace=True)
    df = df['open']
    dataset = df.values
    dataset = dataset.reshape(-1, 1)
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = joblib.load('predict/scaler_cci30_5.save')
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = len(dataset) - look_back -1
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    #X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.20, random_state=42)
    train_rate=0.80
    train_size=int(trainX.shape[0]*train_rate)
    X_train = trainX[:train_size]
    X_test =  trainX[train_size:]
    y_train = trainY[:train_size]
    y_test = trainY[train_size:]

    # print(f'XX_train.shape:{X_train.shape}')
    # print(f'XX_test.shape:{X_test.shape}')
    # print(f'yy_train.shape:{y_train.shape}')
    # print(f'yy_test.shape:{y_test.shape}')


    # reshape input to be [samples, time steps, features]
    X_train = numpy.reshape(X_train, (X_train.shape[0], look_back, 1))
    X_test = numpy.reshape(X_test, (X_test.shape[0], look_back, 1))
    testX = numpy.reshape(testX, (testX.shape[0], look_back,1))
    # create and fit the LSTM network
    model = tf.keras.models.load_model('predict/bdse27_GRU_model_cci30.h5')

    # train model
    start_time=datetime.datetime.now()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    end_time=datetime.datetime.now()
    time_delta=(end_time-start_time).seconds

    # make predictions
    trainPredict = model.predict(X_train)
    mainTestPredict = model.predict(X_test)
    #testPredict = model.predict(testX)

    # future prediction
    predict_period=10
    futureX=copy.deepcopy(testX)
    predictList=[]
    for _ in range(predict_period):
        testPredict = model.predict(futureX)
        predictList.append(testPredict)
        futureX=np.delete(np.concatenate((futureX,testPredict.reshape(1,1,1)),axis=1),0,axis=1)
        


    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    y_train = scaler.inverse_transform([y_train])

    #testPredict = scaler.inverse_transform(testPredict)
    predictList=[scaler.inverse_transform(x)[0][0] for x in predictList ]
    testY = scaler.inverse_transform([testY])

    mainTestPredict = scaler.inverse_transform(mainTestPredict)
    mainTestPredict = [item for sublist in mainTestPredict for item in sublist]
    y_test = scaler.inverse_transform([y_test])
    test_score=[]
    test_score.append(mean_squared_error(y_test[0], mainTestPredict))
    test_score.append(r2_score(y_test[0], mainTestPredict))
    test_score.append(time_delta)
    #test_score = mean_squared_error(y_test[0], mainTestPredict)
    
    # calculate root mean squared error
    trainPredict = [item for sublist in trainPredict for item in sublist]
    #print(trainPredict, predictList)

    #return (trainPredict, (testPredict[0]), test_score)
    return (trainPredict+mainTestPredict, predictList, test_score) 