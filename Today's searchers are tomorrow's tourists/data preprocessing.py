import xlrd
import xlwt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#get data from ARIMA build data.xls
ARIMA_build_data = xlrd.open_workbook(r'Data/ARIMA build data.xls')
ARIMA_build_data_time = ARIMA_build_data.sheet_by_name('Sheet1').col_values(0)[1:]
ARIMA_build_data_tourist_number = ARIMA_build_data.sheet_by_name('Sheet1').col_values(1)[1:]
#print(ARIMA_build_data_time)
ARIMA_build_data_tourist_number_diff = np.diff(ARIMA_build_data_tourist_number)#对数据一阶差分

#tourist number trend
def tourist_number_trend():
    plt.plot(ARIMA_build_data_time, ARIMA_build_data_tourist_number)
    plt.title('tourist number trend')
    plt.xlabel('Time')
    plt.ylabel('Tourist number')
    plt.xticks(rotation=30)  #标签旋转
    plt.show()
    
#对数据一阶差分，验证数据的平稳性
def tourist_number_trend_diff():
    ARIMA_build_data_tourist_number_diff = np.diff(ARIMA_build_data_tourist_number)
    plt.plot(ARIMA_build_data_time[1:],ARIMA_build_data_tourist_number_diff)
    plt.title('tourist number trend diff')
    plt.xticks(rotation=30)
    plt.show()

#自相关和偏自相关图，需要保存后再查看
def ACF_PACF():
    acf = plot_acf(ARIMA_build_data_tourist_number).savefig(r'Data/ACF-ARIMA_build_data_tourist_number.png')
    pacf = plot_pacf(ARIMA_build_data_tourist_number).savefig(r'Data/PACF-ARIMA_build_data_tourist_number.png')

if __name__ == '__main__':
    ACF_PACF()