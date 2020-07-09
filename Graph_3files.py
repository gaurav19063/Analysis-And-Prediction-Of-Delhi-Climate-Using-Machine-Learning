import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



def main ():

    count_weather_classes()




def count_weather_classes():

    data1 = pd.read_csv('/home/gaurav/Desktop/IIITD/ML/project_k/untitled folder/Classification_autumn_data.csv')
    y = data1.extreme_weather
    a = y.to_numpy()
    count_0 = np.where(a == 0)
    count_0_autom=len(count_0[0])
    count_1_autom=len(y)-count_0_autom
    data1 = pd.read_csv('/home/gaurav/Desktop/IIITD/ML/project_k/untitled folder/Classification_summer_data.csv')
    y = data1.extreme_weather
    a = y.to_numpy()
    count_0 = np.where(a == 0)
    print(len(count_0[0]))

    count_0_summer=len(count_0[0])
    count_1_summer=len(y)-count_0_autom
    data1 = pd.read_csv('/home/gaurav/Desktop/IIITD/ML/project_k/untitled folder/Classification_winter_data.csv')
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











