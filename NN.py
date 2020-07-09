import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
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
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix,precision_score,recall_score



def main ():
    data=pd.read_csv('Final_Classification_with_dummies.csv')
    y=data.extreme_weather
    data=data.drop('extreme_weather',axis=1)
    data=normalize(data)
    NN_ExtremeWeatherConditions_Feature_importance(data, y)
    ##Logistic Regression
    print ("Logistic")
    # logistic_ExtremeWeatherConditions(data,y)


    ##Decision treeprint ("Decision Tree")
    print ("Decision Tree")
    # Decision_Tree_Extreme_weather(data,y)
    # Decision_Tree_Extreme_weather_Feature_importance(data,y)





def normalize(data):
    for c in data.columns:
        mean = data[c].mean()
        max = data[c].max()
        min = data[c].min()
        data[c] = (data[c] - min) / (max - min)
    return data




## Logistic Regression
def logistic_ExtremeWeatherConditions(data,y):
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    model=LogisticRegression()
    model=model.fit(X_train,y_train)
    print("Training",model.score(X_train,y_train))
    print("Testing",model.score(X_val,y_val))

    ROC(y_train,model.predict_proba(X_train))
    prediction = model.predict(X_val)
    confusionMetrics(y_val, prediction)
    print(precision_score(y_val, prediction))
    print(recall_score(y_val, prediction))

    ROC(y_val,model.predict_proba(X_val))


def NN_ExtremeWeatherConditions_Feature_importance(data,y):

    tree_clf = ExtraTreesRegressor()
    tree_clf.fit(data, y)
    importances = tree_clf.feature_importances_
    feature_names = data.columns
    imp_features=dict(zip(feature_names,importances))
    features_up = sorted(imp_features.items(), key=lambda x: x[1], reverse=True)
    features_down = sorted(imp_features.items(), key=lambda x: x[1], reverse=False)
    x = [i[0] for i in features_up]
    x=data[x]
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    model=MLPClassifier(solver='adam', alpha=0.01,hidden_layer_sizes=(100,100), random_state=1, batch_size=500)
    model=model.fit(X_train,y_train)
    print("Training",model.score(X_train,y_train))
    print("Testing",model.score(X_val,y_val))

    prediction=model.predict(X_val)
    confusionMetrics(y_val, prediction)
    print("Precision Score:",precision_score(y_val,prediction))
    print("Recall Score:",recall_score(y_val, prediction))
    ROC(y_train, model.predict_proba(X_train))
    ROC(y_val, model.predict_proba(X_val))


def confusionMetrics(a, b):
    print(confusion_matrix(a, b))


def ROC(t,p):
    skplt.metrics.plot_roc(t, p, title="ROC Curve For Svm model")
    plt.show()



main()