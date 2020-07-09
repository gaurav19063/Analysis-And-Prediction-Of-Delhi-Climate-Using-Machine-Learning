import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor


def main():
    merging_gaussian()
    calculate_aqi()
    one_hot_encode()
    df = pd.read_csv('Original_without_dummies.csv')
    prepare_future_value(df)
    seasons ()
    heatmap(df)
    temp(df)
    cat(df)
    relplot(df)
    arima(df)



def merging_gaussian():
    weather,pollution=data_extraction()
    date = [x[:8] for x in weather.datetime_utc]
    weather['date'] = date
    merge(weather, pollution)



def calculate_aqi():
    data = pd.read_csv('FINAL_weather_Variance.csv')
    data['SO2']=data.SO2.mask(data.SO2==0,data['SO2'].mean(skipna=True))
    data['NO2']=data.NO2.mask(data.NO2==0,data['NO2'].mean(skipna=True))
    data['RSPM']=data.RSPM.mask(data.RSPM==0,data['RSPM'].mean(skipna=True))
    data['SPM']=data.SPM.mask(data.SPM==0,data['SPM'].mean(skipna=True))
    data['si']=data['SO2'].apply(calculate_si)
    data['ni']=data['NO2'].apply(calculate_ni)
    data['rpi']=data['RSPM'].apply(calculate_rpi)
    data['spi']=data['SPM'].apply(calculate_spi)

    data['si']=data['SO2']
    data['ni']=data['NO2']
    data['rpi']=data['RSPM']
    data['spi']=data['SPM']

    print (count_null(data))
    # data=data.fillna(0)
    # print(data[' _pressurem'].describe())
    # print()
    data[' _pressurem'] = data[' _pressurem'].replace(-9999,data[' _pressurem'].median() )
    data=data.fillna(data.mean())
    # data=data.drop(' _conds',axis=1)
    # data=data.drop(' _wdire',axis=1)
    data['AQI']=data.apply(lambda data:calculate_aqi(data['si'],data['ni'],data['spi'],data['rpi']),axis=1)
    data.to_csv("Original_without_dummies.csv")

def one_hot_encode():
    data=pd.read_csv("Original_without_dummies.csv")
    data=pd.get_dummies(data, columns=[' _wdire'], prefix = [' _wdire'])
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    data=pd.get_dummies(data, columns=['Type'], prefix = ['Type'])
    data.to_csv("Original_with_dummies.csv")

def prepare_future_value(data):

    n = 0
    AQI_predicted = []
    for i in range(len(data) - 12):

        aqi_mean = 0
        for j in range(i, i + 9):
            aqi_mean += data.at[j, 'AQI']
        aqi_mean /= 10
        AQI_predicted.append(aqi_mean)
        print(i, aqi_mean)
        n = i

    while n < len(data):
        aqi_mean = 0
        counter = 0
        for j in range(len(data) - 1, n, -1):
            aqi_mean += data.at[j, 'AQI']
            print(j, len(data)-1, n)
            counter += 1
        if counter != 0:
            aqi_mean /= counter
            AQI_predicted.append(aqi_mean)
        print(len(AQI_predicted))
        n += 1

    data['AQI_predicted'] = AQI_predicted
    data.to_csv("AQI_prediction_add.csv")


def seasons ():
    winter = [11, 12, 1, 2]
    summer = [3, 4, 5, 6]
    autumn = [7, 8, 9, 10]
    data=pd.read_csv("Original_with_dummies.csv")
    winter_df = data[data['month'].isin(winter)]
    summer_df = data[data['month'].isin(summer)]
    autumn_df = data[data['month'].isin(autumn)]
    winter_df.to_csv("winter_data.csv")
    summer_df.to_csv("summer_data.csv")
    autumn_df.to_csv("autumn_data.csv")


def time_series_plot(data, col):
    data.set_index('datetime_utc', inplace= True)
    data = data[[col]]
    data[col].fillna(data[col].mean(), inplace=True)
    plt.plot(data)
    plt.title('Time Series')
    plt.xlabel('Date')
    plt.ylabel('temperature')
    plt.show()


def heatmap(df):

    plt.figure(figsize=(90, 90))
    sns.set(font_scale=0.3, context='poster')
    p = sns.heatmap(df.corr(), annot=True, linewidths=0.8)
    p.set_xticklabels(p.get_xticklabels(), fontsize=8)
    p.set_yticklabels(p.get_yticklabels(), rotation=0, fontsize=8)
    plt.show()


def temp(df):
    plt.figure(figsize=(15, 10))
    temp = df['date'].dt.year.unique()
    sns.heatmap(pd.crosstab(temp, df['date'].dt.month,values= df['AQI'], aggfunc="mean"),
                cmap="coolwarm", annot=True, cbar=True)
    plt.title("Average Temprature 1996-2016")
    plt.plot()
    plt.show()


def cat(df):
    cmap = plt.cm.Set1
    by_issued_amount = df.groupby(['year', ' _conds']).temp.mean()
    by_issued_amount.unstack().plot(stacked=False, colormap=cmap, grid=False, legend=True, figsize=(15, 6))
    plt.show()


def relplot(df):
    sns.relplot(x="year", y="AQI", kind="line", sort=False, data=df)
    plt.show()
def feautures_importance(data):
    tree_clf = ExtraTreesRegressor()
    data=data.drop('datetime_utc',axis=1)
    y=data['AQI']
    data=data.drop('AQI',axis=1)
    data=pd.get_dummies(data, columns=['month'], prefix = ['month'])
    # data=data.drop('year',axis=1)

    tree_clf.fit(data, y)

    importances = tree_clf.feature_importances_
    feature_names = data.columns
    print(feature_names)

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

def data_extraction():
    path1="delhi_weather .csv"
    path2="new_pollution.csv"
    weather = pd.read_csv(path1,sep=',')
    pollution = pd.read_csv(path2,sep=',')
    return weather,pollution

def arima(data_en):
    print(len(data_en))
    data_en['date'] = pd.to_datetime(data_en['datetime_utc'])

    data_en['date']=data_en['date'].dt.date
    data_en['date'] = pd.to_datetime(data_en['date'], format='%Y-%m-%d')
    df_aqi = data_en[["date", "AQI"]]
    df_aqi = df_aqi.set_index("date")
    df_aqi = df_aqi.dropna()
    df_AQI_resample = df_aqi.resample(rule="M").mean().ffill()
    df_AQI_resample["AQI_first_diff"] = df_AQI_resample["AQI"] - df_AQI_resample["AQI"].shift(7)
    model = sm.tsa.statespace.SARIMAX(df_AQI_resample["AQI"], order=(0, 1, 0), seasonal_order=(1, 1, 1, 48))
    results = model.fit()
    print(results.summary())
    df_AQI_resample['forecast'] = results.predict(start=110, end=155, dynamic=True)
    df_AQI_resample[['AQI', 'forecast']].plot(figsize=(20, 10))
    plt.show()

def merge(weather, pollution):
    c = 0
    weather["Type"] = ""
    weather['SO2'] = ''
    weather['NO2'] = ''
    weather['RSPM'] = ''
    weather['SPM'] = ''
    all_columns = list(pollution.columns.values) + list(weather.columns.values)
    # print(all_columns)
    df = pd.DataFrame(columns=all_columns)
    # weather['date'] = ''
    for i in range(len(pollution)):
        # print(type(weather['date']), type(pollution.date[i]))

        z = np.where(weather['date'] == str(pollution.date[i]))[0]

        print(i, z, pollution.date[i], c)
        if z.size != 0:
            SO2=0
            NO2=0
            RSPM=0
            SPM=0
            count_SO2=0
            count_NO2 = 0
            count_RSPM = 0
            count_SPM = 0

            for k in range (20):
                if i>20:


                    # print("hello")
                    if (np.math.isnan(pollution.NO2[i-k]))==True:
                        pollution.NO2[i - k]=0
                        count_NO2=count_NO2+1
                    if (np.math.isnan(pollution.SO2[i - k])) == True:
                        pollution.SO2[i - k] = 0
                        count_SO2=count_SO2+1
                    if (np.math.isnan(pollution.RSPM[i-k]))==True:
                        pollution.RSPM[i - k]=0
                        count_RSPM=count_RSPM+1
                    if (np.math.isnan(pollution.SPM[i-k]))==True:
                        pollution.SPM[i - k]=0
                        count_SPM=count_SPM+1



                    # np.math.isnan(pollution.NO2[i])
                    print (pollution.NO2[i]-pollution.NO2[i-k])
                    print (np.power ((pollution.NO2[i]-pollution.NO2[i-k]),2))
                    NO2=NO2+np.power ((pollution.NO2[i]-pollution.NO2[i-k]),2)
                    SO2=SO2+np.power ((pollution.SO2[i]-pollution.SO2[i-k]),2)
                    RSPM=RSPM+np.power ((pollution.RSPM[i]-pollution.RSPM[i-k]),2)
                    SPM=SPM+np.power ((pollution.SPM[i]-pollution.SPM[i-k]),2)
            NO2=np.sqrt(NO2)/(20-count_NO2)
            SO2=np.sqrt(SO2)/(20-count_SO2)
            SPM=np.sqrt(SPM)/(20-count_SPM)
            RSPM=np.sqrt(RSPM)/(20-count_RSPM)
            print ("SO2,NO2,SPM,RSPM",SO2,NO2,SPM,RSPM)

            for x in z:
                weather.Type[x] = pollution.Type[i]
                weather.SO2[x] = np.random.normal(pollution.SO2[i], SO2)
                weather.NO2[x] = np.random.normal(pollution.NO2[i], NO2)
                print ("SO2",weather.SO2[x],pollution.SO2[i])
                weather.RSPM[x] = np.random.normal(pollution.RSPM[i], RSPM)
                weather.SPM[x] = np.random.normal(pollution.SPM[i], SPM)
                # print(type(weather.loc[x, :]), type(pollution.loc[i, :]))
                k = pollution.loc[i, :].tolist() + weather.loc[x, :].tolist()
                df.loc[c] = k
                # print(df.loc[c, :])
                c += 1
                print(i, c, x)


    df.to_csv("FINAL_weather_Variance.csv")


##Calculating AQI SI,NI,...
def calculate_si(so2):
    si=0
    if (so2<=40):
     si= so2*(50/40)
    if (so2>40 and so2<=80):
     si= 50+(so2-40)*(50/40)
    if (so2>80 and so2<=380):
     si= 100+(so2-80)*(100/300)
    if (so2>380 and so2<=800):
     si= 200+(so2-380)*(100/800)
    if (so2>800 and so2<=1600):
     si= 300+(so2-800)*(100/800)
    if (so2>1600):
     si= 400+(so2-1600)*(100/800)
    return si

# df.head()


#Function to calculate no2 individual pollutant index(ni)
def calculate_ni(no2):
    ni=0
    if(no2<=40):
     ni= no2*50/40
    elif(no2>40 and no2<=80):
     ni= 50+(no2-14)*(50/40)
    elif(no2>80 and no2<=180):
     ni= 100+(no2-80)*(100/100)
    elif(no2>180 and no2<=280):
     ni= 200+(no2-180)*(100/100)
    elif(no2>280 and no2<=400):
     ni= 300+(no2-280)*(100/120)
    else:
     ni= 400+(no2-400)*(100/120)
    return ni

# df.head()

#Function to calculate no2 individual pollutant index(rpi)
def calculate_rpi(rspm):
    rpi=0
    if(rpi<=30):
     rpi=rpi*50/30
    elif(rpi>30 and rpi<=60):
     rpi=50+(rpi-30)*50/30
    elif(rpi>60 and rpi<=90):
     rpi=100+(rpi-60)*100/30
    elif(rpi>90 and rpi<=120):
     rpi=200+(rpi-90)*100/30
    elif(rpi>120 and rpi<=250):
     rpi=300+(rpi-120)*(100/130)
    else:
     rpi=400+(rpi-250)*(100/130)
    return rpi

# df.tail()


#Function to calculate no2 individual pollutant index(spi)
def calculate_spi(spm):
    spi=0
    if(spm<=50):
     spi=spm
    if(spm<50 and spm<=100):
     spi=spm
    elif(spm>100 and spm<=250):
     spi= 100+(spm-100)*(100/150)
    elif(spm>250 and spm<=350):
     spi=200+(spm-250)
    elif(spm>350 and spm<=450):
     spi=300+(spm-350)*(100/80)
    else:
     spi=400+(spm-430)*(100/80)
    return spi

# df.tail()


#function to calculate the air quality index (AQI) of every data value
#its is calculated as per indian govt standards
def calculate_aqi(si,ni,spi,rpi):
    aqi=0
    if(si>ni and si>spi and si>rpi):
     aqi=si
     print (si , aqi)
    elif(spi>si and spi>ni and spi>rpi):
     aqi=spi
     print (spi , aqi)
    elif(ni>si and ni>spi and ni>rpi):
     aqi=ni
     print (ni , aqi)
    elif(rpi>si and rpi>spi and rpi>ni):
     aqi=rpi
     print (rpi , aqi)
    print(aqi,si,ni,spi,rpi)
    return aqi

# df.head()


# Function to count the number of null values for each column
def count_null(data):
    print(len(data.index) - data.count())


# Function to remove unwanted columns
def remove_columns(df):
    df.drop(df.columns[[0, 6, 8, 18, 19]], axis = 1, inplace=True)
    return df