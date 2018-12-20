import xlrd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import pandas as pd

ARIMA_build_data = xlrd.open_workbook(r'Data/ARIMA build data.xls')
ARIMA_build_data_tourist_number = ARIMA_build_data.sheet_by_name('Sheet1').col_values(1)[1:]
ARIMA_build_data_time = ARIMA_build_data.sheet_by_name('Sheet1').col_values(0)[1:]
#因共线性剔除了部分变量
北京 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(2)[1:]
上海 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(3)[1:]
广东 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(4)[1:]
河南 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(6)[1:]
四川 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(7)[1:]
重庆 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(8)[1:]
江苏 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(9)[1:]
湖北 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(10)[1:]
浙江 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(11)[1:]
福建 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(12)[1:]
山东 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(14)[1:]
广西 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(17)[1:]
贵州 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(18)[1:]
海南 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(19)[1:]
河北 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(20)[1:]
江西 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(23)[1:]
辽宁 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(24)[1:]
山西 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(28)[1:]
陕西 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(29)[1:]
新疆 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(33)[1:]
安徽 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(35)[1:]

y = ARIMA_build_data_tourist_number 
x = np.transpose([北京,上海,广东,河南,四川,重庆,江苏,湖北,浙江,福建,山东,广西,贵州,海南,河北,江西,辽宁,山西,陕西,新疆,安徽])

ARIMA = sm.tsa.statespace.SARIMAX(endog=ARIMA_build_data_tourist_number ,exog=x, order=(0,1,0))
res = ARIMA.fit(disp=False)
#print(res.summary())
def predict():
    北京 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(2)[1:]
    上海 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(3)[1:]
    广东 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(4)[1:]
    河南 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(6)[1:]
    四川 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(7)[1:]
    重庆 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(8)[1:]
    江苏 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(9)[1:]
    湖北 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(10)[1:]
    浙江 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(11)[1:]
    福建 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(12)[1:]
    山东 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(14)[1:]
    广西 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(17)[1:]
    贵州 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(18)[1:]
    海南 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(19)[1:]
    河北 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(20)[1:]
    江西 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(23)[1:]
    辽宁 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(24)[1:]
    山西 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(28)[1:]
    陕西 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(29)[1:]
    新疆 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(33)[1:]
    安徽 = ARIMA_build_data.sheet_by_name('Sheet2').col_values(35)[1:]
    x = np.transpose([北京,上海,广东,河南,四川,重庆,江苏,湖北,浙江,福建,山东,广西,贵州,海南,河北,江西,辽宁,山西,陕西,新疆,安徽])
    ARIMA_Predict = res.predict(exog = x,end = ARIMA.nobs + 3)
    #print(ARIMA_Predict)
    print(ARIMA.nobs )
    plt.plot(ARIMA_build_data_tourist_number,color = 'red',linewidth = '10')
    plt.plot(ARIMA_Predict,color = 'green')
    plt.xlabel(ARIMA_build_data_time)
    plt.xticks(rotation=30)
    plt.show()


if __name__ == '__main__':
    predict()



#y = pd.Series(ARIMA_build_data_tourist_number, index=ARIMA_build_data_time )
#arma_mod = sm.tsa.ARMA(y, order=(0,0))
#arma_res = arma_mod.fit( disp=-1)
#print(arma_res.summary())


#model = sm.OLS(ARIMA_build_data_tourist_number , 上海,北京,广东)
#results = model.fit()
#print(results.summary())
