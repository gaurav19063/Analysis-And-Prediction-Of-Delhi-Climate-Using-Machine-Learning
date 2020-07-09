import  pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm



def main ():
    data_w_en,data_en = data_extraction()
    arima(data_en)












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























def data_extraction():
    data1 = pd.read_csv('/home/gaurav/Desktop/IIITD/ML/project_k/untitled folder/Original_without_dummies.csv')
    data2 = pd.read_csv('/home/gaurav/Desktop/IIITD/ML/project_k/untitled folder/Original_with_dummies _AREMA.csv')
    print(data1.info,data2.info)
    # print(count_null(data))
    # data=pd.get_dummies(data, columns=[' _conds'], prefix = [' _conds'])
    # data=pd.get_dummies(data, columns=[' _wdire'], prefix = [' _wdire'])
    # data=pd.get_dummies(data, columns=['Type'], prefix = ['Type'])
    return data1,data2


























main()

