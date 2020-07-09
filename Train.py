import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn import metrics
import time

# Method to train and test mlp regressor
def mlp_regressor(data):

    y = data.AQI
    data = data.drop('AQI', axis=1)
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    regr = MLPRegressor(hidden_layer_sizes=(10),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=1000,
                                       learning_rate_init=0.01,
                                       alpha=0.01)

    regr.fit(X_train, y_train)
    print("xxxx")
    y_pred = regr.predict(X_train)
    print('Training Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
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
    y_pred = regr.predict(X_val)
    print('Testing Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))
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






def SVM(X_train, X_val, y_train, y_val):
    model=SVR(gamma='scale')
    print("SVM Fitting.....")
    start_time=time.time()
    model=model.fit(X_train,y_train)
    end_time=time.time()
    y_pred=model.predict(X_val)
    Score=model.accuracy_score(y_val,y_pred)
    print("SVR Score:",Score)
    print("SVR fitting time:",end_time-start_time)



data = pd.read_csv('/home/gaurav/Desktop/IIITD/ML/project_k/untitled folder/Original_with_dummies.csv')
# mlp_regressor(data)
y = data.AQI
data = data.drop('AQI', axis=1)
X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
SVM(X_train, X_val, y_train, y_val)