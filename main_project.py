import pandas as pd
import numpy as np

def main ():
    data=data_extraction()
    # print (data.info)
    print(data.info)
    data['si']=data['SO2'].apply(calculate_si)
    data['ni']=data['NO2'].apply(calculate_ni)
    data['rpi']=data['RSPM'].apply(calculate_si)
    data['spi']=data['SPM'].apply(calculate_spi)
    # data['si']=data['SO2'].fillna(0)
    # data['ni']=data['NO2'].fillna(0)
    # data['rpi']=data['RSPM'].fillna(0)
    # data['spi']=data['SPM'].fillna(0)

    print (count_null(data))
    # data=data.fillna(0)
    # print(data[' _pressurem'].describe())
    # print()
    data[' _pressurem'] = data[' _pressurem'].replace(-9999,data[' _pressurem'].median() )
    data=data.fillna(data.mean())
    data=data.drop(' _conds',axis=1)
    data=data.drop(' _wdire',axis=1)
    data['AQI']=data.apply(lambda data:calculate_aqi(data['si'],data['ni'],data['spi'],data['rpi']),axis=1)

    data.to_csv("AQI_NAN_Mean_Without_dummies.csv")
    # print (count_null(data))
    # print("here",data.groupby('_precipm').count())
    # print(data['AQI'].median())
    # print(data['AQI'].mean())

    # count_null(data)
    # data = remove_columns(data)
    # data.to_csv("FINAL_delhi_weather.csv")
    # print(data)
    # count_null(data)



def data_extraction():
    data = pd.read_csv('FINAL_weather_Variance.csv')
    print(data.info)
    # print(count_null(data))
    # data=pd.get_dummies(data, columns=[' _conds'], prefix = [' _conds'])
    # data=pd.get_dummies(data, columns=[' _wdire'], prefix = [' _wdire'])
    # data=pd.get_dummies(data, columns=['Type'], prefix = ['Type'])
    return data

#Function to calculate so2 individual pollutant index(si)
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
def calculate_(rspm):
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



main()
