import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV as gsc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix,precision_score,recall_score
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import matplotlib.dates as mdates
import warnings
import itertools
import dateutil
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scikitplot as skplt
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier


def main ():
    data = pd.read_csv('/home/gaurav/Desktop/IIITD/ML/project_k/untitled folder/Final_Classification_with_dummies.csv')
    y = data.extreme_weather
    data = data.drop('extreme_weather', axis=1)
    normalize(data)
    ExtremeWeatherConditions_bagging(data, y)
    ExtremeWeatherConditions_boostiong(data,y)






def normalize(data):
    for c in data.columns:
        mean = data[c].mean()
        max = data[c].max()
        min = data[c].min()
        data[c] = (data[c] - min) / (max - min)
    return data

#
#
#
def ExtremeWeatherConditions_boostiong(data,y):
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    model= AdaBoostClassifier()
    model=model.fit(X_train,y_train)

    print("Training", model.score(X_train, y_train))
    print("Testing", model.score(X_val, y_val))

    prediction = model.predict(X_val)
    confusionMetrics(y_val, prediction)
    print("Precision Score:", precision_score(y_val, prediction))
    print("Recall Score:", recall_score(y_val, prediction))
    ROC(y_train, model.predict_proba(X_train))
    ROC(y_val, model.predict_proba(X_val))

    # print(confusion_matrix())
    a=y_val.to_numpy()
    b = X_val.to_numpy()
    misclassified = np.where(a!= model.predict(b))
    print(misclassified)
    # confusionMetrics(y_val, model.predict(X_val))

def ExtremeWeatherConditions_bagging(data,y):
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    model=BaggingClassifier()
    model=model.fit(X_train,y_train)

    print("Training", model.score(X_train, y_train))
    print("Testing", model.score(X_val, y_val))

    prediction = model.predict(X_val)
    confusionMetrics(y_val, prediction)
    print("Precision Score:", precision_score(y_val, prediction))
    print("Recall Score:", recall_score(y_val, prediction))
    ROC(y_train, model.predict_proba(X_train))
    ROC(y_val, model.predict_proba(X_val))

    # print(confusion_matrix())
    a=y_val.to_numpy()
    b = X_val.to_numpy()
    misclassified = np.where(a!= model.predict(b))
    print(misclassified)


def confusionMetrics(a, b):
    print(confusion_matrix(a, b))





def ROC(t,p):
    skplt.metrics.plot_roc(t, p, title="ROC Curve For Svm model")
    plt.show()



main()











