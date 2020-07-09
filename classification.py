
from sklearn.ensemble import RandomForestClassifier
import pickle
import scikitplot as skplt
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,precision_score,recall_score
from sklearn.ensemble import ExtraTreesRegressor
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from sklearn.neural_network import MLPClassifier



def main ():
    data=pd.read_csv('Final_Classification_with_dummies.csv')
    y=data.extreme_weather
    data=data.drop('extreme_weather',axis=1)
    data=normalize(data)

    #Logistic Regression
    # print ("Logistic")
    # logistic_ExtremeWeatherConditions(data,y)
    # logistic_ExtremeWeatherConditions_Feature_importance(data,y)
    # #
    # ##Decision tree
    # print ("Decision Tree")
    # Decision_Tree_Extreme_weather(data,y)
    # Decision_Tree_Extreme_weather_Feature_importance(data,y)
    #
    # ##Naive naive_bayes
    # print ("Naive Bayes")
    naive_bayes_Extreme_weather(data,y)
    naive_bayes_Extreme_weather_Feature_importance(data,y)

    ## KNN
    # print ("KNN")
    KNN_Extreme_weather(data,y)
    KNN_Extreme_weather_Feature_importance(data,y)

    ##Neural Network
    # print ("NN")
    NN_Extreme_weather(data,y)
    NN_Extreme_weather_Feature_importance(data,y)

    ##Random forest
    RandomForest_Extreme_weather(data,y)
    RandomForest_Extreme_weather_Feature_importance(data,y)

##Analysing extreme weather condition in all three seasons
    count_weather_classes()

    ##misclassification
    misclassification()

    ##Bagging
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
    ROC(y_train,model.predict_proba(X_train))
    print("Testing Accuracy",model.score(X_val,y_val))
    prediction=model.predict(X_val)
    print("Precision Score:",precision_score(y_val,prediction))
    print("Recall Score:",recall_score(y_val, prediction))
    print("Confusion MATRIX",confusion_matrix(y_val,prediction))
    ROC(y_val,model.predict_proba(X_val))






def logistic_ExtremeWeatherConditions_Feature_importance(data,y):
    tree_clf = ExtraTreesRegressor()
    tree_clf.fit(data, y)
    importances = tree_clf.feature_importances_
    feature_names = data.columns
    imp_features=dict(zip(feature_names,importances))
    features_up = sorted(imp_features.items(), key=lambda x: x[1], reverse=True)
    features_down = sorted(imp_features.items(), key=lambda x: x[1], reverse=False)
        #best features
    print(features_up[:10])

    import time
    start=time.time()
    plt.bar(range(len(features_up)), [imp[1] for imp in features_up], align='center')
    plt.xticks(np.arange(len(features_up)),[x[0] for x in features_up], rotation='vertical', label='Features')
    plt.yticks(label='Feature Importance')
    # plt.title('features')
    plt.show()
    print(time.time()-start)

    import time
    start=time.time()
    plt.bar(range(len(features_up[4:])), [imp[1] for imp in features_up[4:]], align='center')
    plt.title('features')
    plt.show()
    print(time.time()-start)

    # x = [i[0] for i in features_up]
    # x=data[x]
    # X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
    # model=LogisticRegression()
    # model=model.fit(X_train,y_train)
    # print("Training Accuracy",model.score(X_train,y_train))
    # prediction=model.predict(X_train)
    # print("Precision Score:",precision_score(y_train,prediction))
    # print("Recall Score:",recall_score(y_train, prediction))
    # print("Confusion MATRIX",confusion_matrix(y_train,prediction))
    # ROC(y_train,model.predict_proba(X_train))
    # print("Testing Accuracy",model.score(X_val,y_val))
    # prediction=model.predict(X_val)
    # print("Precision Score:",precision_score(y_val,prediction))
    # print("Recall Score:",recall_score(y_val, prediction))
    # print("Confusion MATRIX",confusion_matrix(y_val,prediction))
    # ROC(y_val,model.predict_proba(X_val))






#Predicting AQI using all features
def Decision_Tree_Extreme_weather(data,y):
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    model = tree.DecisionTreeClassifier()
    model = model.fit(X_train,y_train)
    print("Training Accuracy",model.score(X_train,y_train))
    prediction=model.predict(X_train)
    print("Precision Score:",precision_score(y_train,prediction))
    print("Recall Score:",recall_score(y_train, prediction))
    print("Confusion MATRIX",confusion_matrix(y_train,prediction))
    ROC(y_train,model.predict_proba(X_train))
    print("Testing Accuracy",model.score(X_val,y_val))
    prediction=model.predict(X_val)
    print("Precision Score:",precision_score(y_val,prediction))
    print("Recall Score:",recall_score(y_val, prediction))
    print("Confusion MATRIX",confusion_matrix(y_val,prediction))
    ROC(y_val,model.predict_proba(X_val))


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
    model = tree.DecisionTreeClassifier()
    model = model.fit(X_train,y_train)
    print("Training Accuracy",model.score(X_train,y_train))
    prediction=model.predict(X_train)
    print("Precision Score:",precision_score(y_train,prediction))
    print("Recall Score:",recall_score(y_train, prediction))
    print("Confusion MATRIX",confusion_matrix(y_train,prediction))
    ROC(y_train,model.predict_proba(X_train))
    print("Testing Accuracy",model.score(X_val,y_val))
    prediction=model.predict(X_val)
    print("Precision Score:",precision_score(y_val,prediction))
    print("Recall Score:",recall_score(y_val, prediction))
    print("Confusion MATRIX",confusion_matrix(y_val,prediction))
    ROC(y_val,model.predict_proba(X_val))
    return model

def naive_bayes_Extreme_weather(data,y):
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    model = GaussianNB()
    model = model.fit(X_train,y_train)
    print("Training Accuracy",model.score(X_train,y_train))
    prediction=model.predict(X_train)
    print("Precision Score:",precision_score(y_train,prediction))
    print("Recall Score:",recall_score(y_train, prediction))
    print("Confusion MATRIX",confusion_matrix(y_train,prediction))
    ROC(y_train,model.predict_proba(X_train))
    print("Testing Accuracy",model.score(X_val,y_val))
    prediction=model.predict(X_val)
    print("Precision Score:",precision_score(y_val,prediction))
    print("Recall Score:",recall_score(y_val, prediction))
    print("Confusion MATRIX",confusion_matrix(y_val,prediction))
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
    print("Training Accuracy",model.score(X_train,y_train))
    prediction=model.predict(X_train)
    print("Precision Score:",precision_score(y_train,prediction))
    print("Recall Score:",recall_score(y_train, prediction))
    print("Confusion MATRIX",confusion_matrix(y_train,prediction))
    ROC(y_train,model.predict_proba(X_train))
    print("Testing Accuracy",model.score(X_val,y_val))
    prediction=model.predict(X_val)
    print("Precision Score:",precision_score(y_val,prediction))
    print("Recall Score:",recall_score(y_val, prediction))
    print("Confusion MATRIX",confusion_matrix(y_val,prediction))
    ROC(y_val,model.predict_proba(X_val))


def KNN_Extreme_weather(data,y):
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    model = KNeighborsClassifier(n_neighbors=3)
    model = model.fit(X_train,y_train)
    print("Training Accuracy",model.score(X_train,y_train))
    prediction=model.predict(X_train)
    print("Precision Score:",precision_score(y_train,prediction))
    print("Recall Score:",recall_score(y_train, prediction))
    print("Confusion MATRIX",confusion_matrix(y_train,prediction))
    ROC(y_train,model.predict_proba(X_train))
    print("Testing Accuracy",model.score(X_val,y_val))
    prediction=model.predict(X_val)
    print("Precision Score:",precision_score(y_val,prediction))
    print("Recall Score:",recall_score(y_val, prediction))
    print("Confusion MATRIX",confusion_matrix(y_val,prediction))
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
    print("Training Accuracy",model.score(X_train,y_train))
    prediction=model.predict(X_train)
    print("Precision Score:",precision_score(y_train,prediction))
    print("Recall Score:",recall_score(y_train, prediction))
    print("Confusion MATRIX",confusion_matrix(y_train,prediction))
    ROC(y_train,model.predict_proba(X_train))
    print("Testing Accuracy",model.score(X_val,y_val))
    prediction=model.predict(X_val)
    print("Precision Score:",precision_score(y_val,prediction))
    print("Recall Score:",recall_score(y_val, prediction))
    print("Confusion MATRIX",confusion_matrix(y_val,prediction))
    ROC(y_val,model.predict_proba(X_val))
    return model

def NN_Extreme_weather(data,y):
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    model=MLPClassifier()
    model=model.fit(X_train,y_train)
    print("Training Accuracy",model.score(X_train,y_train))
    prediction=model.predict(X_train)
    print("Precision Score:",precision_score(y_train,prediction))
    print("Recall Score:",recall_score(y_train, prediction))
    print("Confusion MATRIX",confusion_matrix(y_train,prediction))
    ROC(y_train,model.predict_proba(X_train))
    print("Testing Accuracy",model.score(X_val,y_val))
    prediction=model.predict(X_val)
    print("Precision Score:",precision_score(y_val,prediction))
    print("Recall Score:",recall_score(y_val, prediction))
    print("Confusion MATRIX",confusion_matrix(y_val,prediction))
    ROC(y_val,model.predict_proba(X_val))


#Predicting AQI using features from features importance graph
def NN_Extreme_weather_Feature_importance(data,y, layer):
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
    model=MLPClassifier(solver='adam', alpha='0.01', hidden_layer_sizes=layer,batch_size=500,random_state=1)
    model=model.fit(X_train,y_train)
    print("Training Accuracy",model.score(X_train,y_train))
    prediction=model.predict(X_train)
    print("Precision Score:",precision_score(y_train,prediction))
    print("Recall Score:",recall_score(y_train, prediction))
    print("Confusion MATRIX",confusion_matrix(y_train,prediction))
    ROC(y_train,model.predict_proba(X_train))
    print("Testing Accuracy",model.score(X_val,y_val))
    prediction=model.predict(X_val)
    print("Precision Score:",precision_score(y_val,prediction))
    print("Recall Score:",recall_score(y_val, prediction))
    print("Confusion MATRIX",confusion_matrix(y_val,prediction))
    ROC(y_val,model.predict_proba(X_val))


def RandomForest_Extreme_weather_Feature_importance(data,y):
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    model=RandomForestClassifier(max_depth=None, random_state=0,n_estimators=100)
    model=model.fit(X_train,y_train)

    print("Training", model.score(X_train, y_train))
    print("Testing", model.score(X_val, y_val))

    prediction = model.predict(X_val)
    confusionMetrics(y_val, prediction)
    print("Precision Score:", precision_score(y_val, prediction))
    print("Recall Score:", recall_score(y_val, prediction))
    ROC(y_train, model.predict_proba(X_train))
    ROC(y_val, model.predict_proba(X_val))
    return model



def RandomForest_Extreme_weather(data,y):
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
    model=RandomForestClassifier(max_depth=None, random_state=0,n_estimators=100)
    model=model.fit(X_train,y_train)
    print("Training", model.score(X_train, y_train))
    print("Testing", model.score(X_val, y_val))
    prediction = model.predict(X_val)
    confusionMetrics(y_val, prediction)
    print("Precision Score:", precision_score(y_val, prediction))
    print("Recall Score:", recall_score(y_val, prediction))
    ROC(y_train, model.predict_proba(X_train))
    ROC(y_val, model.predict_proba(X_val))
def confusionMetrics(a, b):
    print(confusion_matrix(a, b))


def misclassification():
    with open("misclassified.txt", "rb") as fp:   # Unpickling
        misclassified = pickle.load(fp)[0]

    data = pd.read_csv('Final_Classification_with_dummies.csv')
    mc = [1 if i in misclassified else 0 for i in data.index]
    data['ind'] = [i for i in data.index]
    data['misclassified'] = mc
    # data.plot.scatter(x='ind', y='AQI', c='misclassified',colormap='viridis')
    # sns.heatmap(pd.crosstab(data.extreme_weather, data.year,values= data['AQI'], aggfunc="mean"),
    #                 cmap="coolwarm", annot=True, cbar=True)
    # plt.title("Average Temprature 1996-2016")
    # plt.plot()
    d = data.loc[11500:, :]
    sns.countplot(x='extreme_weather',data=d)
    plt.show()
    sns.countplot(x='misclassified',data=d)
    plt.show()


def bagging(data,y):
    X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=1)
    models = []
    models.append(Decision_Tree_Extreme_weather_Feature_importance(X_train, y_train))
    models.append(RandomForest_Extreme_weather_Feature_importance(X_train, y_train))
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

    print("bagging test Accuracy",metrics.accuracy_score(predictions,y_val))

def count_weather_classes():

    data1 = pd.read_csv('Classification_autumn_data.csv')
    y = data1.extreme_weather
    a = y.to_numpy()
    count_0 = np.where(a == 0)
    count_0_autom=len(count_0[0])
    count_1_autom=len(y)-count_0_autom
    data1 = pd.read_csv('Classification_summer_data.csv')
    y = data1.extreme_weather
    a = y.to_numpy()
    count_0 = np.where(a == 0)
    print(len(count_0[0]))

    count_0_summer=len(count_0[0])
    count_1_summer=len(y)-count_0_autom
    data1 = pd.read_csv('Classification_winter_data.csv')
    y = data1.extreme_weather
    a = y.to_numpy()
    count_0 = np.where(a == 0)


    count_0_winter=len(count_0[0])
    count_1_winter=len(y)-count_0_autom
    print(count_1_autom,count_1_summer,count_1_winter)
    print(count_0_autom, count_0_summer, count_0_winter)
    tips=[count_0_autom,count_1_autom,count_0_summer,count_1_summer,count_0_winter,count_1_winter]
    LABELS= ['count_0_autom', 'count_1_autom', 'count_0_summer', 'count_1_summer', 'count_0_winter', 'count_1_winter']
    # sns.boxplot(data=t,palette='coolwarm',orient='h')
    plt.plot(tips)
    d=[0,1,2,3,4,5]
    plt.xticks(d, LABELS)
    # tips=pd.DataFrame(tips)
    # sns.boxplot(data = tips, color='' dodge = False)
    # # ax.show()
    plt.show()


main()
