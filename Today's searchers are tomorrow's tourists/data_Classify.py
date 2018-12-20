import xlrd
import tsfresh

ARIMA_build_data = xlrd.open_workbook(r'Data/ARIMA build data.xls')
北京 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(2)[1:]
上海 = ARIMA_build_data.sheet_by_name('Sheet1').col_values(3)[1:]



print(tsfresh.feature_extraction.feature_calculators.agg_linear_trend(北京,上海))