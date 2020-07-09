import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, RandomForestClassifier
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
from statistics import mode


from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,precision_score,recall_score
from statsmodels.compat import scipy


def main ():
    data=pd.read_csv('Final_Classification_with_dummies.csv')
    y=data.extreme_weather
    data=data.drop('extreme_weather',axis=1)
    data=normalize(data)

    #Logistic Regression
    print ("Logistic")

    # logistic_ExtremeWeatherConditions(data,y)
    # logistic_ExtremeWeatherConditions_Feature_importance(data,y)
    #
    # ##Decision treeprint ("Decision Tree")
    # print ("Decision Tree")
    # Decision_Tree_Extreme_weather(data,y)
    # Decision_Tree_Extreme_weather_Feature_importance(data,y)
    #
    # ##Naive naive_bayes
    # print ("Naive Bayes")
    # naive_bayes_Extreme_weather(data,y)
    # naive_bayes_Extreme_weather_Feature_importance(data,y)

    ## KNN
    # print ("KNN")
    # KNN_Extreme_weather(data,y)
    # KNN_Extreme_weather_Feature_importance(data,y)

    ##Neural Network
    # print ("NN")
    # NN_Extreme_weather(data,y)
    # NN_Extreme_weather_Feature_importance(data,y)

    bagging(data,y)




def normalize(data):
    for c in data.columns:
        mean = data[c].mean()
        max = data[c].max()
        min = data[c].min()
        data[c] = (data[c] - min) / (max - min)
    return data

def ROC(t,p):
    skplt.metrics.plot_roc(t, p, title="ROC Curve For Svm model")
    plt.show()




## Logistic Regression
def logistic_ExtremeWeatherConditions(data,y):
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    model=LogisticRegression()
    model=model.fit(X_train,y_train)
    print("Training Accuracy",model.score(X_train,y_train))
    prediction=model.predict(X_train)
    print("Precision Score:",precision_score(y_train,prediction))
    print("Recall Score:",recall_score(y_train, prediction))
    print("Confusion MATRIX",confusion_matrix(y_train,prediction))
    # ROC(y_train,model.predict_proba(X_train))
    print("Testing Accuracy",model.score(X_val,y_val))
    prediction=model.predict(X_val)
    print("Precision Score:",precision_score(y_val,prediction))
    print("Recall Score:",recall_score(y_val, prediction))
    print("Confusion MATRIX",confusion_matrix(y_val,prediction))
    # ROC(y_val,model.predict_proba(X_val))







def logistic_ExtremeWeatherConditions_Feature_importance(data,y):
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
    model=LogisticRegression()
    model=model.fit(X_train,y_train)
    print("Training",model.score(X_train,y_train))
    print("Testing",model.score(X_val,y_val))
    ROC(y_train,model.predict_proba(X_train))
    ROC(y_val,model.predict_proba(X_val))
    return model






#Predicting AQI using all features
def Decision_Tree_Extreme_weather(data,y):
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    print("Training",clf.score(X_train,y_train))
    print("Testing",clf.score(X_val,y_val))
    ROC(y_train,clf.predict_proba(X_train))
    ROC(y_val,clf.predict_proba(X_val))



#Predicting AQI using features from features importance graph
def Decision_Tree_Extreme_weather_Feature_importance(data,y):
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
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    print("Training",clf.score(X_train,y_train))
    print("Testing",clf.score(X_val,y_val))
    ROC(y_train,clf.predict_proba(X_train))
    ROC(y_val,clf.predict_proba(X_val))
    return clf

def naive_bayes_Extreme_weather(data,y):
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    model = GaussianNB()
    model = model.fit(X_train,y_train)
    print("Training",model.score(X_train,y_train))
    print("Testing",model.score(X_val,y_val))
    ROC(y_train,model.predict_proba(X_train))
    ROC(y_val,model.predict_proba(X_val))



#Predicting AQI using features from features importance graph
def naive_bayes_Extreme_weather_Feature_importance(data,y):
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
    model = GaussianNB()
    model = model.fit(X_train,y_train)
    print("Training",model.score(X_train,y_train))
    print("Testing",model.score(X_val,y_val))
    ROC(y_train,model.predict_proba(X_train))
    ROC(y_val,model.predict_proba(X_val))
    return model


def KNN_Extreme_weather(data,y):
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    model = KNeighborsClassifier(n_neighbors=3)
    model = model.fit(X_train,y_train)
    print("Training",model.score(X_train,y_train))
    print("Testing",model.score(X_val,y_val))
    ROC(y_train,model.predict_proba(X_train))
    ROC(y_val,model.predict_proba(X_val))



#Predicting AQI using features from features importance graph
def KNN_Extreme_weather_Feature_importance(data,y):
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
    model = KNeighborsClassifier(n_neighbors=3)
    model = model.fit(X_train,y_train)
    print("Training",model.score(X_train,y_train))
    print("Testing",model.score(X_val,y_val))
    ROC(y_train,model.predict_proba(X_train))
    ROC(y_val,model.predict_proba(X_val))
    return model

def RandomForest_Extreme_weather_Feature_importance(data,y):
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
    model = RandomForestClassifier(max_depth=None, random_state=0, n_estimators=100)
    model = model.fit(X_train,y_train)
    print("Training",model.score(X_train,y_train))
    print("Testing",model.score(X_val,y_val))
    ROC(y_train,model.predict_proba(X_train))
    ROC(y_val,model.predict_proba(X_val))
    return model





def bagging(data,y):
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    models = []

    # models.append(logistic_ExtremeWeatherConditions_Feature_importance(X_train, y_train))
    #
    # ##Decision treeprint ("Decision Tree")
    # print ("Decision Tree")

    models.append(Decision_Tree_Extreme_weather_Feature_importance(X_train, y_train))
    # #
    # # ##Naive naive_bayes
    # # print ("Naive Bayes")
    #
    models.append(RandomForest_Extreme_weather_Feature_importance(X_train, y_train))
    #
    # ## KNN
    # # print ("KNN")
    models.append(KNN_Extreme_weather_Feature_importance(X_train, y_train))

    predictions=[]
    print("Bagging......")
    X_df = pd.DataFrame(X_val)

    for i in X_df.index:
        x = X_df.loc[i,:].values
        x=x.reshape(1, -1)
        pred = []
        for i in range (len(models)):
            pred.append(models[i].predict(x))
        # print(np.sum(pred))
        if(np.sum(pred)>=2):

            predictions.append(1)
        else:
            predictions.append(0)

            # print(predictions)

    print("bagging test Accuracy",metrics.accuracy_score(predictions,y_val))








main()
