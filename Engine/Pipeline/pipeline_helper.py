import copy
import numpy as np
import pandas as pd


class Evaluator(object):
    def __init__(self, y_df, yhat_df, use_log):
        self._yhat_df = yhat_df
        self._y_df = y_df
        self._result = None

        if use_log is True:
            self._yhat = np.array(np.exp(yhat_df['yhat']))
            self._y = np.array(np.exp(y_df['y']))
            self._yhat_df['yhat'] = np.exp(self._yhat_df['yhat'])
            self._y_df['y'] = np.exp(self._y_df['y'])
        else:
            self._yhat = np.array(yhat_df['yhat'])
            self._y = np.array(y_df['y'])

    @property
    def y_df(self):
        return self._y_df

    @property
    def yhat_df(self):
        return self._yhat_df

    @property
    def y(self):
        return self._y

    @property
    def yhat(self):
        return self._yhat

    @property
    def rmse(self):
        return np.sqrt(((self._y - self._yhat)**2).mean())

    @property
    def mape(self):
        # avoid zero division error
        return np.mean(np.abs((self._y - self._yhat) / (self._y + 0.0001))) * 100

    @property
    def smape(self):
        return (
                    np.sum(np.abs(self._y - self._yhat)) / np.sum(self._y + self._yhat)
                ) * 2 * 100

    @property
    def y_sum(self):
        return np.sum(self._y)

    @property
    def yhat_sum(self):
        return np.sum(self._yhat)

    @property
    def total_diff(self):
        return self.y_sum - self.yhat_sum

    @property
    def total_daily_diff(self):
        return np.sum(np.abs(self._y - self._yhat))

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, result):
        self._result = result

    def get_result(self, i):
        df = pd.DataFrame(
            columns=['Fold', 'From', 'To', 'Real', 'Predict',
                     'Error', 'Error(%)', 'Total Error', 'Total Error(%)',
                     'RMSE', 'MAPE', 'sMAPE'], index=[i]
        )

        df['Fold'] = i
        df['From'] = self.y_df['ds'].min()
        df['To'] = self.y_df['ds'].max()
        df['Real'] = self.y_sum
        df['Predict'] = self.yhat_sum
        df['Error'] = self.total_diff
        df['Error(%)'] = (self.total_diff / self.y_sum) * 100
        df['Total Error'] = self.total_daily_diff
        df['Total Error(%)'] = (self.total_daily_diff / self.y_sum) * 100
        df['RMSE'] = self.rmse
        df['MAPE'] = self.mape
        df['sMAPE'] = self.smape

        self._result = df
        return df


class PostProcessor(object):
    def __init__(self, forecast):
        self.forecast = forecast

    def convert_minus_zero(self):
        self.forecast['yhat'] = self.forecast['yhat'].apply(lambda x: x - x if x < 0 else x)
        print("POSTPROCESS YHAT = 0 IF YHAT < 0")

    def convert_sunday_zero(self):
        self.forecast.loc[self.forecast.ds.dt.weekday == 6, 'yhat'] = 0
        print("POSTPROCESS YHAT = 0 IF DAYTIME IS SUNDAY")

    def holiday_correction(self, ratio, df):
        self.forecast.loc[self.forecast.ds.isin(df.ds),'yhat'] = \
            self.forecast.loc[self.forecast.ds.isin(df.ds),'yhat'] * ratio
        print("POSTPROCESS Holiday")

    def postprocess(self, minus_zero, sunday_zero, holiday_correction_dict):
        if minus_zero:
            self.convert_minus_zero()
        if sunday_zero:
            self.convert_sunday_zero()
        if holiday_correction_dict is not None:
            for ratio, df in holiday_correction_dict.items():
                self.holiday_correction(ratio, df)

        return self.forecast


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg


@pd.api.extensions.register_dataframe_accessor("train_test_split")
class TrainTestSplit(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._attribute = list()
        self._y = 'y'

        # horizon default
        self._unit = 'd'
        self._horizon = 30
        self._end_date = self._obj.index.max()

    @property
    def attribute(self):
        return self._attribute

    @attribute.setter
    def attribute(self, attribute):
        self._attribute = attribute

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = y

    @property
    def xy(self):
        xy= copy.deepcopy(self._attribute)
        xy.append(self._y)
        return xy

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, horizon):
        self._horizon = horizon

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, unit):
        self._unit = unit

    @property
    def end_date(self):
        return self._end_date

    @end_date.setter
    def end_date(self, end_date):
        self._end_date = end_date

    @property
    def train(self):
        start = self._obj.index.min()
        end = self.end_date - TrainTestSplit.date(self.horizon, unit=self.unit)

        return self._obj[start:end]

    @property
    def trainX(self):
        return self.train[self._attribute]

    @property
    def trainY(self):
        return self.train[self._y]

    @property
    def test(self):
        start = self.end_date - \
                TrainTestSplit.date(self._horizon, unit=self.unit) + \
                TrainTestSplit.date(1, unit='D')
        end = self.end_date

        return self._obj[start:end]

    @property
    def testX(self):
        return self.test[self._attribute]

    @property
    def testY(self):
        return self.test[self._y]

    @staticmethod
    def date(horizon, unit):
        if unit == 'D' or unit == 'd' or unit == 'day':
            return pd.DateOffset(days=horizon)
        elif unit == 'W' or unit == 'w' or unit == 'week':
            return pd.DateOffset(weeks=horizon)
        elif unit == 'M' or unit == 'mon' or unit == 'month':
            return pd.DateOffset(months=horizon)
        elif unit == "Q" or unit == 'q' or unit == 'quarter':
            return pd.DateOffset(quarters=horizon)
        elif unit == 'Y' or unit == 'y' or unit == 'year':
            return pd.DateOffset(years=horizon)
        elif unit == 'H' or unit == 'h' or unit == 'hour':
            return pd.DateOffset(hours=horizon)
        elif unit == 'm' or unit == 'min' or unit == 'minute':
            return pd.DateOffset(minutes=horizon)
        elif unit == 's' or unit == 'sec' or unit =='second':
            return pd.DateOffset(seconds=horizon)





