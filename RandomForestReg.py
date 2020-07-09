import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV as gsc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import matplotlib.dates as mdates
import warnings
import itertools
import dateutil
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn import metrics
from sklearn.linear_model import Ridge,Lasso
from sklearn.ensemble  import RandomForestRegressor
from sklearn.model_selection import GridSearchCV as GSC
import time

def main ():
    data = pd.read_csv('Original_with_dummies.csv')
    y = data.AQI
    data = data.drop('AQI', axis=1)
    normalize(data)
    data['AQI'] = y
    #
    # AQI(data)
    # AQI_Feature_importance(data)
    # AQI_Domain_Knowledge(data)
    # AQI_without_Domain_Knowledge(data)

    ## Predincting for nxt day
    y = pd.read_csv('/home/gaurav/Desktop/IIITD/ML/project_k/untitled folder/AQI_prediction_add.csv')
    AQI_Future(data, y.AQI_predicted)
    AQI_Feature_importance_Future(data, y.AQI_predicted)
    # AQI_Domain_Knowledge_Future(data, y.AQI_predicted)
    # AQI_without_Domain_Knowledge_Future(data, y.AQI_predicted)
    # AQI_predicted

    # data['date'] = pd.to_datetime(data['date'],format='%Y-%m-%d')
    # for i in range(len(data)):
    #     data.year[i] = str(data.date[i])[:4]
    # print(data.year)
    # df = data[['AQI','month']].groupby(["month"]).median().reset_index().sort_values(by='month',ascending=False)
    # f,ax=plt.subplots(figsize=(15,10))
    # sns.pointplot(x='month', y='AQI', data=df)
    # plt.show()
    # df=data[['AQI','date']]
    # df["date"] = pd.to_datetime(df['date'])
    # print(df.tail(20))
    # df=data.set_index('date')
    # df.sort_values(by='date',ascending=False)
    # df.plot(figsize=(15, 6))
    # plt.show()
    # AQI()
    # temp()
    # pressure()


def normalize(data):
    for c in data.columns:
        mean = data[c].mean()
        max = data[c].max()
        min = data[c].min()
        data[c] = (data[c] - min) / (max - min)
    return data






#Predicting AQI using all features
def AQI(data):
    y=data.AQI
    data=data.drop('AQI',axis=1)

    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    graph_testing(y_pred,y_val)


#Predicting AQI using features from features importance graph
def AQI_Feature_importance(data):
    tree_clf = ExtraTreesRegressor()
    y=data['AQI']
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    data=data.drop('AQI',axis=1)
    tree_clf.fit(data, y)
    importances = tree_clf.feature_importances_
    feature_names = data.columns
    imp_features=dict(zip(feature_names,importances))
    features_up = sorted(imp_features.items(), key=lambda x: x[1], reverse=True)
    features_down = sorted(imp_features.items(), key=lambda x: x[1], reverse=False)

    #best features
    x = [i[0] for i in features_up]
    print(x)
    x=data[x]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    graph_testing(y_pred,y_val)

#Predicting AQI using all features
def AQI_Domain_Knowledge(data):
    y=data.AQI
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    # df[['Name', 'Qualification']]
    x=data[[' _tempm',' _wdird',' _wspdm','year','Type_Industrial Area','month_10','month_11',]]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    graph_testing(y_pred,y_val)

def AQI_without_Domain_Knowledge(data):
    y=data.AQI
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    data=data.drop('AQI',axis=1)
    data=data.drop('NO2',axis=1)
    data=data.drop('SO2',axis=1)
    data=data.drop('SPM',axis=1)
    data=data.drop('RSPM',axis=1)
    data=data.drop('ni',axis=1)
    data=data.drop('si',axis=1)
    data=data.drop('rpi',axis=1)
    data=data.drop('spi',axis=1)

    data=data.drop(' _tempm',axis=1)
    data=data.drop(' _wdird',axis=1)
    data=data.drop(' _wspdm',axis=1)
    data=data.drop('year',axis=1)
    data=data.drop('Type_Industrial Area',axis=1)
    data=data.drop('month_10',axis=1)
    data=data.drop('month_11',axis=1)

    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr =RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    graph_testing(y_pred,y_val)



def AQI_Future(data,y):
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    graph_testing(y_pred,y_val)


#Predicting AQI using features from features importance graph
def AQI_Feature_importance_Future(data,y):
    tree_clf = ExtraTreesRegressor()
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    tree_clf.fit(data, y)
    importances = tree_clf.feature_importances_
    feature_names = data.columns
    imp_features=dict(zip(feature_names,importances))
    features_up = sorted(imp_features.items(), key=lambda x: x[1], reverse=True)
    features_down = sorted(imp_features.items(), key=lambda x: x[1], reverse=False)

    #best features
    x = [i[0] for i in features_up]
    print(x)
    x=data[x]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    graph_testing(y_pred,y_val)

#Predicting AQI using all features
def AQI_Domain_Knowledge_Future(data,y):
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    x=data[[' _tempm',' _wdird',' _wspdm','year','Type_Industrial Area','month_10','month_11',]]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    graph_testing(y_pred,y_val)

def AQI_without_Domain_Knowledge_Future(data,y):
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    data=data.drop('AQI',axis=1)
    data=data.drop('NO2',axis=1)
    data=data.drop('SO2',axis=1)
    data=data.drop('SPM',axis=1)
    data=data.drop('RSPM',axis=1)
    data=data.drop('ni',axis=1)
    data=data.drop('si',axis=1)
    data=data.drop('rpi',axis=1)
    data=data.drop('spi',axis=1)

    data=data.drop(' _tempm',axis=1)
    data=data.drop(' _wdird',axis=1)
    data=data.drop(' _wspdm',axis=1)
    data=data.drop('year',axis=1)
    data=data.drop('Type_Industrial Area',axis=1)
    data=data.drop('month_10',axis=1)
    data=data.drop('month_11',axis=1)

    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    graph_testing(y_pred,y_val)



def AQI_Future(data,y):
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    graph_testing(y_pred,y_val)


#Predicting AQI using features from features importance graph
def AQI_Feature_importance_Future(data,y):
    tree_clf = ExtraTreesRegressor()
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    tree_clf.fit(data, y)
    importances = tree_clf.feature_importances_
    feature_names = data.columns
    imp_features=dict(zip(feature_names,importances))
    features_up = sorted(imp_features.items(), key=lambda x: x[1], reverse=True)
    features_down = sorted(imp_features.items(), key=lambda x: x[1], reverse=False)

    #best features
    x = [i[0] for i in features_up]
    print(x)
    x=data[x]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    graph_testing(y_pred,y_val)

#Predicting AQI using all features
def AQI_Domain_Knowledge_Future(data,y):
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    x=data[[' _tempm',' _wdird',' _wspdm','year','Type_Industrial Area','month_10','month_11',]]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    graph_testing(y_pred,y_val)

def AQI_without_Domain_Knowledge_Future(data,y):
    # data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    data=data.drop('AQI',axis=1)
    data=data.drop('NO2',axis=1)
    data=data.drop('SO2',axis=1)
    data=data.drop('SPM',axis=1)
    data=data.drop('RSPM',axis=1)
    data=data.drop('ni',axis=1)
    data=data.drop('si',axis=1)
    data=data.drop('rpi',axis=1)
    data=data.drop('spi',axis=1)

    data=data.drop(' _tempm',axis=1)
    data=data.drop(' _wdird',axis=1)
    data=data.drop(' _wspdm',axis=1)
    data=data.drop('year',axis=1)
    data=data.drop('Type_Industrial Area',axis=1)
    data=data.drop('month_10',axis=1)
    data=data.drop('month_11',axis=1)

    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = RandomForestRegressor()
    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
    graph_training(y_pred,y_train)
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
    graph_testing(y_pred,y_val)
def graph_training(y_pred,y_train):
    all_samples = [i for i in range(0, 250)]
    y_pred=y_pred[0:250]
    y_train=y_train[0:250]
    plt.plot(all_samples, y_pred,label='Predicted')
    plt.plot(all_samples , y_train,label='Expected')
    plt.xlabel("No of Samples")
    plt.ylabel("AQI")
    plt.title("Training")
    plt.legend()
    plt.show()


def graph_testing(y_pred,y_val):
    all_samples = [i for i in range(0, 250)]
    y_pred=y_pred[0:250]
    y_val=y_val[0:250]
    plt.plot(all_samples, y_pred,label='Predicted')
    plt.plot(all_samples , y_val,label='Expected')
    plt.xlabel("No of Samples")
    plt.ylabel("AQI")
    plt.title("Validation")
    plt.legend()
    plt.show()



def temp():
    data=pd.read_csv('AQI_NAN_Mean.csv')
    y=data.temp
    data=data.drop('temp',axis=1)
    data=data.drop('datetime_utc',axis=1)
    data=data.drop('date',axis=1)
    regr = RandomForestRegressor()
    regr.fit(data, y)
    print("xxxx")
    print(regr.score(data, y)*100)
    y_pred = regr.predict(data)
    # print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))
    print ("With SO2 ,NO2,etc")
    print('Mean Absolute Error:', metrics.mean_absolute_error(y,y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y,y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y,y_pred)))

def pressure():
    data=pd.read_csv('AQI_NAN_Mean.csv')
    y=data.AQI
    data=data.drop('AQI',axis=1)
    data=data.drop('datetime_utc',axis=1)
    data=data.drop('date',axis=1)
    data=data.drop('SO2',axis=1)
    data=data.drop('NO2',axis=1)
    data=data.drop('RSPM',axis=1)
    data=data.drop('SPM',axis=1)
    data=data.drop('si',axis=1)
    data=data.drop('ni',axis=1)
    data=data.drop('rpi',axis=1)
    data=data.drop('spi',axis=1)
    # data=data.drop('AQI',axis=1)



    regr =RandomForestRegressor()
    regr.fit(data, y)
    print("xxxx")
    print(regr.score(data, y)*100)
    y_pred = regr.predict(data)
    # print('Mean Squared Error:', metrics.mean_squared_error(y, y_pred))

    print('Mean Absolute Error:', metrics.mean_absolute_error(y,y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y,y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y,y_pred)))

main()
