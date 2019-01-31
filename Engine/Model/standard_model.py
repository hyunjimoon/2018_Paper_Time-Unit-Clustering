import numpy as np
import pandas as pd

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from statsmodels.tsa.api import SimpleExpSmoothing, Holt

"""
class NextoptETS(object):
    def __init__(self):
        self.model = None
    def fit(self, train, optimize=True, smoothing_level=None):
        if optimize:
            self.model = SimpleExpSmoothing(np.array(train['y']).fit(optimize=optimize))
        else:
            self.model = SimpleExpSmoothing(np.array(train['y']).fit(smoothing_level=smoothing_level))
    def predict(self, test):
        return self.model.forecast(len(test))
"""


class StandardModel(object):
    def __init__(self, train, test, frequency):

        # Data
        self.train = train
        self.test = test

        # Frequency
        self.frequency = frequency

        # DataFrame
        self.fit_df = None
        self.forecast_df = None

        # 1. load objects and packages
        StandardModel.check_forecast_package()

    # 1. check forecast package
    @staticmethod
    def check_forecast_package():
        try:
            importr('forecast')
        except:
            print("There is no forecast package on your computer.\n")
            print("Start installing forecast package.")
            utils = importr('utils')
            utils.install_packages('forecast')
            print("Forecast package installed.")

    def make_output_format(self):
        self.fit_df.set_index(self.train.index, inplace=True)
        self.forecast_df.set_index(self.test.index, inplace=True)

    def change_column_name(self):
        self.fit_df.columns = ['yhat']
        self.forecast_df.columns = ['yhat',
                                    'yhat_lower_80', 'yhat_upper_80',
                                    'yhat_lower_95', 'yhat_upper_95']


class ExponentialMovingAverage(StandardModel):
    def __init__(self, train, test, frequency):
        super(ExponentialMovingAverage, self).__init__(train, test, frequency)

    def fit_and_forecast(self, horizon):
        r_string = """
            function(data, frequency, horizon){
                library(forecast)
                ts_data <- ts(data, frequency=frequency)
                fit <- HoltWinters(ts_data, beta=FALSE, gamma=FALSE)
                fitted_df <- data.frame(fit$fitted)
                forecast <- forecast(fit, h = horizon)
                forecast_df <- data.frame(forecast)
                output <- list(fitted_df, forecast_df)
                return(output)
            }
        """

        r_func = robjects.r(r_string)

        # Run R
        pandas2ri.activate()
        output_list = r_func(self.train, self.frequency, horizon)
        self.fit_df = pandas2ri.ri2py(output_list[0])
        self.forecast_df = pandas2ri.ri2py(output_list[1])
        pandas2ri.deactivate()

        self.make_output_format()
        self.change_column_name()

    def make_output_format(self):
        self.fit_df = self.fit_df[['xhat']]
        self.fit_df.loc[-1] = None
        self.fit_df.sort_index(ascending=True, inplace=True)
        self.fit_df.set_index(self.train.index, inplace=True)
        self.forecast_df.set_index(self.test.index, inplace=True)


class STL(StandardModel):
    def __init__(self, train, test, frequency):
        super(STL, self).__init__(train, test, frequency)

    def fit_and_forecast(self, horizon):
        r_string = """
            function(data, frequency, horizon){
                library(forecast)
                ts_data <- ts(data, frequency=frequency)
                fit <- stl(ts_data, s.window="periodic")
                fitted_df <- data.frame(fit$time.series)
                forecast <- forecast(fit, h = horizon)
                forecast_df <- data.frame(forecast)
                output <- list(fitted_df, forecast_df)
                return(output)
            }
        """

        r_func = robjects.r(r_string)

        # Run R
        pandas2ri.activate()
        output_list = r_func(self.train['y'], self.frequency, horizon)
        self.fit_df = pandas2ri.ri2py(output_list[0])
        self.forecast_df = pandas2ri.ri2py(output_list[1])
        pandas2ri.deactivate()

        self.make_output_format()
        self.change_column_name()

    def make_output_format(self):
        self.fit_df['yhat'] = self.fit_df['seasonal'] + self.fit_df['trend']
        self.fit_df.set_index(self.train.index, inplace=True)
        self.forecast_df.set_index(self.test.index, inplace=True)

    def change_column_name(self):
        self.forecast_df.columns = ['yhat',
                                    'yhat_lower_80', 'yhat_upper_80',
                                    'yhat_lower_95', 'yhat_upper_95']


class HoltWinters(StandardModel):
    def __init__(self, train, test, frequency):
        super(HoltWinters, self).__init__(train, test, frequency)

    def fit_and_forecast(self, horizon):
        r_string = """
            function(data, frequency, horizon){
                library(forecast)
                ts_data <- ts(data, frequency=frequency)
                fit <- HoltWinters(ts_data)
                fitted_df <- data.frame(fit$fitted)
                forecast <- forecast(fit, h = horizon)
                forecast_df <- data.frame(forecast)
                output <- list(fitted_df, forecast_df)
                return(output)
            }
        """

        r_func = robjects.r(r_string)

        # Run R
        pandas2ri.activate()
        output_list = r_func(self.train, self.frequency, horizon)
        self.fit_df = pandas2ri.ri2py(output_list[0])
        self.forecast_df = pandas2ri.ri2py(output_list[1])
        pandas2ri.deactivate()

        self.make_output_format()
        self.change_column_name()

    def make_output_format(self):
        for i in range(self.frequency):
            index = (i+1) * (-1)
            self.fit_df.loc[index] = None

        self.fit_df.sort_index(ascending=True, inplace=True)
        self.fit_df.set_index(self.train.index, inplace=True)
        self.forecast_df.set_index(self.test.index, inplace=True)

    def change_column_name(self):
        self.fit_df.columns = ['yhat', 'level', 'trend', 'season']
        self.forecast_df.columns = ['yhat',
                                    'yhat_lower_80', 'yhat_upper_80',
                                    'yhat_lower_95', 'yhat_upper_95']


class AutoArima(StandardModel):
    def __init__(self, train, test, frequency, seasonal):
        super(AutoArima, self).__init__(train, test, frequency)
        self.seasonal = seasonal

    def fit_and_forecast(self, horizon):
        r_string = """
            function(data, frequency, seasonal, horizon){
                library(forecast)
                ts_data <- ts(data, frequency=frequency)
                fit <- auto.arima(ts_data, seasonal=seasonal)
                fitted_df <- data.frame(fit$fitted)
                forecast <- forecast(fit, h = horizon)
                forecast_df <- data.frame(forecast)
                output <- list(fitted_df, forecast_df)
                return(output)
            }
        """

        r_func = robjects.r(r_string)

        pandas2ri.activate()
        output_list = r_func(self.train, self.frequency, self.seasonal, horizon)
        self.fit_df = pandas2ri.ri2py(output_list[0])
        self.forecast_df = pandas2ri.ri2py(output_list[1])
        pandas2ri.deactivate()

        self.make_output_format()
        self.change_column_name()

    def make_output_format(self):
        self.fit_df.set_index(self.train.index, inplace=True)
        self.forecast_df.set_index(self.test.index, inplace=True)


class TBATS(StandardModel):
    def __init__(self, train, test, frequency):
        super(TBATS, self).__init__(train, test, frequency)

    def fit_and_forecast(self, horizon):
        r_string = """
            function(data, frequency, horizon){
                library(forecast)
                
                if(length(frequency) == 1){
                    ts_data <- ts(data, frequency=frequency)
                }else{
                    ts_data <- msts(data, seasonal.periods=frequency)
                }
 
                fit <- tbats(ts_data)
                fitted_df <- data.frame(fit$fitted.values)
                forecast <- forecast(fit, h = horizon)
                forecast_df <- data.frame(forecast)
                output <- list(fitted_df, forecast_df)
                return(output)
            }
        """

        r_func = robjects.r(r_string)

        pandas2ri.activate()
        output_list = r_func(self.train, robjects.IntVector(self.frequency), horizon)
        self.fit_df = pandas2ri.ri2py(output_list[0])
        self.forecast_df = pandas2ri.ri2py(output_list[1])
        pandas2ri.deactivate()

        self.make_output_format()
        self.change_column_name()


class AutoRegressiveNN(StandardModel):
    def __init__(self, train, test, frequency):
        super(AutoRegressiveNN, self).__init__(train, test, frequency)

    def fit_and_forecast(self, horizon):
        r_string = """
            function(data, frequency, horizon){
                library(forecast)
                
                if(length(frequency) == 1){
                    ts_data <- ts(data, frequency=frequency)
                }else{
                    ts_data <- msts(data, seasonal.periods=frequency)
                }
                
                fit <- nnetar(ts_data)
                fitted_df <- data.frame(fit$fitted)
                forecast <- forecast(fit, h = horizon)
                forecast_df <- data.frame(forecast$mean)
                output <- list(fitted_df, forecast_df)
                return(output)
            }
        """

        r_func = robjects.r(r_string)

        pandas2ri.activate()
        output_list = r_func(self.train, robjects.IntVector(self.frequency), horizon)
        self.fit_df = pandas2ri.ri2py(output_list[0])
        self.forecast_df = pandas2ri.ri2py(output_list[1])
        pandas2ri.deactivate()

        self.make_output_format()
        self.change_column_name()

    def make_output_format(self):
        self.fit_df.set_index(self.train.index, inplace=True)
        self.forecast_df.set_index(self.test.index, inplace=True)

    def change_column_name(self):
        self.fit_df.columns = ['yhat']
        self.forecast_df.columns = ['yhat']