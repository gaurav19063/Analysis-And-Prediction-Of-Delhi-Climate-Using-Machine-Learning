import numpy as np
import pandas as pd
import random
import math

def main():
    weather,pollution=data_extraction()
    date = [x[:8] for x in weather.datetime_utc]
    print(date)
    weather['date'] = date
    print(pollution)
    merge(weather, pollution)


def data_extraction():
    path1="/home/gaurav/Desktop/IIITD/ML/project_k/untitled folder/delhi_weather .csv"
    path2="/home/gaurav/Desktop/IIITD/ML/project_k/untitled folder/new_pollution.csv"
    weather = pd.read_csv(path1,sep=',')
    pollution = pd.read_csv(path2,sep=',')
    return weather,pollution


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


main()
