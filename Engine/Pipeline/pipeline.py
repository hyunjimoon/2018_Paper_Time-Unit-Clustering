from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from .pipeline_helper import Evaluator, PostProcessor, TrainTestSplit

from ..Data.transformer import Transformer
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


# 1.Base
# 1.1 Base - Pipeline
class NextoptBasePipeline(object):
    def __init__(self):
        self._raw_data = None
        self._fold = 0
        self._horizon = 30

        self._unit = 'd'

        self._train = {}
        self._test = {}
        self._future = False

        self._model = {}
        self._regressor = []

        self._forecast = {}
        self._forecast_train = {}
        self._forecast_test = {}

        self._postprocessor = {}
        self._postprocessed_test = {}
        self._postprocessed_train = {}

        self._evaluator = {}
        self._fitted_value = None
        self._forecast_value = None

        self._result_summary = None
        self._use_log = False

    # 1. Property
    @property
    def raw_data(self):
        return self._raw_data

    @raw_data.setter
    def raw_data(self, raw_data):
        self._raw_data = raw_data

        if self.raw_data.index.name != 'ds':
            try:
                self.raw_data.set_index('ds', inplace=True)
                print("SET INDEX AS DS(DateTimeIndex)")
            except KeyError:
                raise KeyError("DATAFRAME FOR MODEL MUST HAVE A 'DS'(DATETIMEINDEX) COLUMN")

    @property
    def fold(self):
        return self._fold

    @fold.setter
    def fold(self, fold):
        self._fold = fold

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
    def train(self):
        return self._train

    @train.setter
    def train(self, train):
        self._train = train

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, test):
        self._test = test

    @property
    def future(self):
        return self._future

    @future.setter
    def future(self, future):
        self._future = future

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        for i in range(self._fold):
            self._model[i] = deepcopy(model)

        # for future
        self._model[self.fold] = deepcopy(model)

    @property
    def forecast(self):
        return self._forecast

    @forecast.setter
    def forecast(self, forecast):
        self._forecast = forecast

    @property
    def forecast_train(self):
        return self._forecast_train

    @forecast_train.setter
    def forecast_train(self, forecast):
        self._forecast_train = forecast

    @property
    def forecast_test(self):
        return self._forecast_test

    @forecast_test.setter
    def forecast_test(self, forecast):
        self._forecast_test = forecast

    @property
    def evaluator(self):
        return self._evaluator

    @evaluator.setter
    def evaluator(self, evaluator):
        self._evaluator = evaluator

    @property
    def postprocessor(self):
        return self._postprocessor

    @postprocessor.setter
    def postprocessor(self, postprocessor):
        self._postprocessor = postprocessor

    @property
    def fitted_value(self):
        return self._fitted_value

    @property
    def forecast_value(self):
        return self._forecast_value

    @property
    def postprocessed_test(self):
        return self._postprocessed_test

    @postprocessed_test.setter
    def postprocessed_test(self, test):
        self._postprocessed_test = test

    @property
    def postprocessed_train(self):
        return self._postprocessed_train

    @postprocessed_train.setter
    def postprocessed_train(self, train):
        self._postprocessed_train = train

    @property
    def result_summary(self):
        return self._result_summary

    @property
    def use_log(self):
        return self._use_log

    @use_log.setter
    def use_log(self, use_log):
        self._use_log = use_log

    # 2. Function
    def cross_validation_split(self, fold=3, end_date=None):
        self._fold = fold

        if not end_date:
            end_date = self.raw_data.index.max()
        else:
            end_date = pd.to_datetime(end_date)

        for i in range(self._fold):
            self.train[i], self.test[i] = self.train_test_split(end_date)
            end_date = end_date - TrainTestSplit.date(self.horizon, self.unit)
        print("CROSS VALIDATION SPLIT IS DONE.")

    def train_test_split(self, end_date):
        if end_date:
            self.raw_data.train_test_split.end_date = end_date

        self.raw_data.train_test_split.horizon = self.horizon
        self.raw_data.train_test_split.unit = self.unit

        df_train, df_val = self.raw_data.train_test_split.train, self.raw_data.train_test_split.test
        return [df_train, df_val]

    def convert_raw_test(self, end_date=None):
        if not end_date:
            end_date = self.raw_data.index.max() + TrainTestSplit.date(self.horizon, self.unit)
        else:
            end_date = pd.to_datetime(end_date)

        future = pd.DataFrame(index=pd.date_range(self.raw_data.index.max() + pd.DateOffset(days=1),
                                                  end_date, freq=self.unit),
                              columns=self.raw_data.columns)
        future.index.name = 'ds'
        self.raw_data = pd.concat([self.raw_data, future])
        self.future = True

    def reset_index(self, index):
        self.train[index].reset_index(drop=False, inplace=True)
        self.test[index].reset_index(drop=False, inplace=True)

    def fit(self):
        for i in range(self._fold):
            self.model[i].fit(self.train[i])
            print("FOLD ", i + 1, " FIT DONE")

    def predict(self):
        for i in range(self._fold):
            forecast = pd.DataFrame({
                'ds': pd.concat(
                    [self.train[i]['ds'], self.test[i]['ds']], axis=0)
            })

            self.forecast[i] = self.model[i].predict(forecast)
            self.forecast_train[i] = self.extract_forecast_train(i)
            self.forecast_test[i] = self.extract_forecast_test(i)
            print("FOLD ", i + 1, " FORECAST DONE")

    def fit_and_predict(self):
        self.fit()
        self.predict()

    def extract_forecast_train(self, index):
        return self.forecast[index].loc[
            self.forecast[index].ds.isin(self.train[index].ds)
        ][['ds', 'yhat']].reset_index(drop=True)

    def extract_forecast_test(self, index):
        return self.forecast[index].loc[
            self.forecast[index].ds.isin(self.test[index].ds)
        ][['ds', 'yhat']].reset_index(drop=True)

    def postprocess(self, convert_minus_zero=True, convert_sunday_zero=False, holiday_correction_dict=None):
        for i in range(self._fold):
            if self.forecast_train[i].index.name == 'ds':
                self.reset_forecast_train_index(i)
                self.reset_forecast_test_index(i)

            if self.train[i].index.name == 'ds':
                self.reset_index(i)

            self.postprocessor[i] = PostProcessor(deepcopy(self.forecast_test[i]))
            self.postprocessed_test[i] = self.postprocessor[i].postprocess(convert_minus_zero,
                                                                           convert_sunday_zero,
                                                                           holiday_correction_dict)
            self.postprocessor[i] = PostProcessor(deepcopy(self.forecast_train[i]))
            self.postprocessed_train[i] = self.postprocessor[i].postprocess(convert_minus_zero,
                                                                            convert_sunday_zero,
                                                                            holiday_correction_dict)

    def reset_forecast_train_index(self, index):
        self.forecast_train[index].reset_index(drop=False, inplace=True)

    def reset_forecast_test_index(self, index):
        self.forecast_test[index].reset_index(drop=False, inplace=True)

    def evaluate(self, type='test', length=30):
        if type == 'test':
            for i in range(self._fold):
                self.evaluator[i] = Evaluator(self.test[i],
                                              self.postprocessed_test[i],
                                              self.use_log)
        if type == 'train':
            for i in range(self._fold):
                self.evaluator[i] = Evaluator(self.train[i][-length:],
                                              self.postprocessed_train[i][-length:],
                                              self.use_log)

    def print_forecast_value(self):
        for i in reversed(range(self._fold)): # 1 0
            result = pd.DataFrame(
                columns=['fold', 'ds', 'Real', 'Prediction', 'Error'],
                index=[j for j in range(30 * (self._fold - (i + 1)), 30 * (self._fold - i))]
            )

            result['fold'] = self._fold - i
            result['ds'] = self.test[i]['ds'].values
            result['Real'] = self.test[i]['y'].values
            result['Prediction'] = self.postprocessed_test[i]['yhat'].values
            result['Error'] = result['Real'] - result['Prediction']

            self._forecast_value = pd.concat([self._forecast_value, result])
        print(self.forecast_value.round(2))

    def save_forecast_value(self, path=None, format='csv'):
        if format == 'csv' or format == ".csv":
            self.forecast_value.round(2).to_csv(path)
        elif format == 'pickle' or format == 'pkl' or format == ".pkl":
            self.forecast_value.round(2).to_pickle(path)
        elif format == 'excel' or format == 'xlsx' or format == ".xlsx":
            self.forecast_value.round(2).to_excel(path)
        print("Saved forecast.")

    def print_summary(self):
        for i in reversed(range(self.fold)):
            if i == self._fold - 1:
                self._result_summary = self.evaluator[i].get_result(i + 1)
                continue

            summary = self.evaluator[i].get_result(i + 1)
            self._result_summary = pd.concat([self.result_summary, summary])
        print(self.result_summary.round(2))

    def save_summary(self, path=None, format='csv'):
        if format == 'csv' or format == ".csv":
            self.result_summary.round(2).to_csv(path)
        elif format == 'pickle' or format == 'pkl' or format == ".pkl":
            self.result_summary.round(2).to_pickle(path)
        elif format == 'excel' or format == 'xlsx' or format == ".xlsx":
            self.result_summary.round(2).to_excel(path)
        print("Saved summary.")

    def plot_fit(self, length=30, save=False, path=None):
        for i in reversed(range(self._fold)):
            print("Fold ", i + 1, "\t",
                  self.postprocessed_train[i][-length:].ds.min(),
                  " - ",
                  self.postprocessed_train[i][-length:].ds.max())

        for i in reversed(range(self._fold)):
            result = pd.concat([self.train[i][-length:]['y'],
                                self.postprocessed_train[i][-length:]['yhat']],
                                axis=1)
            result.index = self.train[i][-length:]['ds']
            ax = result.plot()
            plt.title("Fold " + str(i+1))

            if save is True:
                fig = ax.get_figure()
                fig.savefig(path + "fold_" + str(i+1) + '.png')

    def plot_forecast(self, save=False, path=None):
        for i in reversed(range(self._fold)):
            print("Fold ", i+1, "\t", self.test[i].ds.min(), " - ", self.test[i].ds.max())

        for i in reversed(range(self._fold)):
            result = pd.concat([self.test[i]['y'], self.postprocessed_test[i]['yhat']], axis=1)
            result.index = self.test[i]['ds']
            ax = result.plot()
            plt.title("Fold " + str(i+1))

            if save is True:
                fig = ax.get_figure()
                fig.savefig(path + "fold_" + str(i+1) + '.png')


# 1.2 Base - Weekday Divide and Merge Pipeline
class NextoptWeekdayDMPipeline(NextoptBasePipeline):
    def __init__(self):
        super(NextoptWeekdayDMPipeline, self).__init__()

        self._division_condition = {}
        self._divided_train = {}
        self._divided_forecast = {}

    @property
    def division_condition(self):
        return self._division_condition

    @division_condition.setter
    def division_condition(self, condition_lst):
        for i, condition in enumerate(condition_lst):
            tmp_list = []

            if isinstance(condition, int):
                condition = str(condition)

            for char in condition:
                tmp_list.append(int(char))

            self._division_condition[i] = tmp_list

        self.divide_train()
        self.initialize_divided_forecast()
        print("DIVIDE EACH FOLD BY WEEKDAY")

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        for i in range(self._fold):
            for j in self._division_condition:
                self._model[i] = {j: {}}

        for i in range(self._fold):
            for j in self._division_condition.keys():
                self._model[i][j] = deepcopy(model)

    @property
    def divided_forecast(self):
        return self._divided_forecast

    def divide_train(self):
        for i in range(self._fold):
            for j in self._division_condition:
                self._divided_train[i] = {j: {}}

        for i in range(self._fold):
            for j in self._division_condition.keys():
                self._divided_train[i][j] = \
                    self.train[i].transformer.get(weekday=self._division_condition[j])

        print("DIVIDE TRAIN BY WEEKDAY")

    def initialize_divided_forecast(self):
        for i in range(self._fold):
            for j in self._division_condition:
                self._divided_forecast[i] = {j: {}}

    def fit(self):
        for i in range(self._fold):
            for j in self._division_condition.keys():
                self.model[i][j].fit(self._divided_train[i][j])

    def predict(self):
        for i in range(self._fold):
            for j in self._division_condition.keys():
                forecast = pd.DataFrame({
                    'ds': pd.concat(
                        [self._divided_train[i][j]['ds'], self.test[i]['ds']], axis=0)
                })
                self.divided_forecast[i][j] = self.model[i][j].predict(forecast)
                print("FOLD ", i + 1, " DIVISION ", j + 1, " FORECAST DONE")

    def merge_forecast(self):
        for i in range(self._fold):
            self.forecast[i] = pd.DataFrame()

            for j in self._division_condition.keys():
                self.set_divide_forecast_index(i, j)
                self.forecast[i] = pd.concat(
                    [
                        self.forecast[i],
                        self.divided_forecast[i][j].transformer.get(
                            weekday=self.division_condition[j]
                        )
                    ], axis=0
                )

            self.reset_forecast_index(i)
            self.forecast[i].sort_values('ds', inplace=True)
            self.forecast[i].reset_index(drop=True, inplace=True)

            self.forecast_train[i] = self.extract_forecast_train(i)
            self.forecast_test[i] = self.extract_forecast_test(i)

        print("MERGE DIVIDED FORECAST INTO FORECAST_TRAIN & FORECAST_TEST")

    def reset_divide_train_index(self, i, j):
        self._divided_train[i][j].reset_index(drop=False, inplace=True)

    def set_divide_forecast_index(self, i, j):
        self.divided_forecast[i][j].set_index('ds', inplace=True)

    def reset_forecast_index(self, i):
        self.forecast[i].reset_index(drop=False, inplace=True)


# 2. Model Pipeline Model
# 2-1 Prophet - Pipeline
class NextoptProphetPipeline(NextoptBasePipeline):
    def __init__(self):
        super(NextoptProphetPipeline, self).__init__()

    def add_seasonality(self, name, period, fourier_order):
        for i in range(self.fold):
            self.model[i].add_seasonality(name=name, period=period, fourier_order=fourier_order)

    def add_regressor(self, column):
        for i in range(self._fold):
            self._model[i].add_regressor(column)
        self._regressor.append(column)

    def fit(self):
        for i in range(self._fold):
            print("FOLD ", i + 1, " RESET TRAIN, TEST INDEX FOR PROPHET MODEL")
            self.reset_index(i)
            self.model[i].fit(self.train[i])
            print("FOLD ", i + 1, " FIT DONE")

    def predict(self):
        for i in range(self._fold):
            forecast = pd.DataFrame({
                'ds': pd.concat(
                    [self.train[i]['ds'], self.test[i]['ds']], axis=0)
            })
            if bool(self._regressor):
                for reg_col in self._regressor:
                    forecast[reg_col] = pd.concat([self.train[i][reg_col], self.test[i][reg_col]], axis=0)

            self.forecast[i] = self.model[i].predict(forecast)
            self.forecast_train[i] = self.extract_forecast_train(i)
            self.forecast_test[i] = self.extract_forecast_test(i)
            print("FOLD ", i + 1, " FORECAST DONE")

    def plot(self):
        for i in reversed(range(self._fold)):
            print("Fold ", i, "\t", self.test[i].ds.min(), " - ", self.test[i].ds.max())
            self.model[i].plot(self.forecast[i])

    def plot_components(self):
        for i in reversed(range(self._fold)):
            print("Fold ", i, "\t", self.test[i].ds.min(), " - ", self.test[i].ds.max())
            self.model[i].plot_components(self.forecast[i])


# 2-2 Prophet - Weekday Divide and Merge Pipeline
class NextoptWeekdayDMProphetPipeline(NextoptWeekdayDMPipeline):
    def __init__(self):
        super(NextoptWeekdayDMProphetPipeline, self).__init__()
        self._cancel_weekly_seasonality_for_one_day = False

    @property
    def cancel_weekly_seasonality_for_one_day(self):
        return self._cancel_weekly_seasonality_for_one_day

    @cancel_weekly_seasonality_for_one_day.setter
    def cancel_weekly_seasonality_for_one_day(self, value):
        self._cancel_weekly_seasonality_for_one_day = value

    def add_seasonality(self, name, period, fourier_order):
        for i in range(self.fold):
            for j in self._division_condition.keys():
                self.model[i][j].add_seasonality(name=name, period=period, fourier_order=fourier_order)

    def add_regressor(self, column):
        for i in range(self._fold):
            for j in self._division_condition.keys():
                self._model[i][j].add_regressor(column)
        self._regressor.append(column)

    def fit(self):
        for i in range(self._fold):
            for j in self._division_condition.keys():
                print("FOLD ", i + 1, " DIVISION ", j + 1, " RESET TRAIN, TEST INDEX FOR PROPHET MODEL")

                if self._divided_train[i][j].index.name == 'ds':
                    self.reset_divide_train_index(i, j)

                if self.train[i].index.name == 'ds':
                    self.reset_index(i)

                if self.cancel_weekly_seasonality_for_one_day:
                    if len(self.division_condition[j]) == 1:
                        self.model[i][j].weekly_seasonality = False

                self.model[i][j].fit(self._divided_train[i][j])

    def predict(self):
        for i in range(self._fold):
            for j in self._division_condition.keys():
                forecast = pd.DataFrame({
                    'ds': pd.concat(
                        [self._divided_train[i][j]['ds'], self.test[i]['ds']], axis=0)
                })

                if bool(self._regressor):
                    for reg_col in self._regressor:
                        forecast[reg_col] = pd.concat([self._divided_train[i][j][reg_col],
                                                       self.test[i][reg_col]], axis=0)

                self.divided_forecast[i][j] = self.model[i][j].predict(forecast)
                print("FOLD ", i + 1, " DIVISION ", j + 1, " FIT AND FORECAST DONE")

    def plot(self):
        pass

    def plot_components(self):
        for i in reversed(range(self._fold)):
            for j in self._division_condition.keys():
                print("Fold ", i, "\t", self.test[i].ds.min(), " - ", self.test[i].ds.max())
                self.model[i][j].plot_components(self.divided_forecast[i][j])


# 3. LSTM - Pipeline
class NextoptLSTMPipeline(NextoptBasePipeline):
    def __init__(self):
        super(NextoptLSTMPipeline, self).__init__()

        self._lags = None
        self._preprocessed_train_X = {}
        self._preprocessed_train_y = {}

        self._preprocessed_test_X = {}
        self._preprocessed_test_y = {}

    @property
    def lags(self):
        return self._lags

    @lags.setter
    def lags(self, lags):
        self._lags = lags

    @property
    def preprocessed_train_X(self):
        return self._preprocessed_train_X

    @property
    def preprocessed_train_y(self):
        return self._preprocessed_train_y

    @property
    def preprocessed_test_X(self):
        return self._preprocessed_test_X

    @property
    def preprocessed_test_y(self):
        return self._preprocessed_test_y

    def preprocess(self, lags=7):
        self.lags = lags

        self.preprocess_train()
        self.preprocess_test()

    def preprocess_train(self):
        for i in range(self.fold):
            X_train, y_train = [], []

            for idx in range(len(self.train[i]) - self.lags):
                x_train = self.train[i]['y'][idx:(idx + self.lags)].values
                X_train.append(x_train)

                y_train.append(self.train[i]['y'][idx + self.lags])

            self._preprocessed_train_X[i] = self.reshape(np.array(X_train))
            self._preprocessed_train_y[i] = np.array(y_train).reshape(-1, 1)
            print("FOLD ", i + 1, " TRAIN PREPROCESS DONE")

    def preprocess_test(self):
        for i in range(self.fold):
            X_test, y_test = [], []
            cnt = 0

            for idx in range(len(self.train[i]) - self.lags, len(self.train[i])):
                x_test_front = self.train[i]['y'][idx:len(self.train[i])].values.tolist()
                x_test_end = self.test[i]['y'][0:cnt].tolist()
                x_test = x_test_front + x_test_end
                cnt = cnt + 1

                X_test.append(x_test)

            for idx in range(len(self.test[i]) - self.lags):
                x_test = self.train[i]['y'][idx:(idx + self.lags)].values
                X_test.append(x_test)

            for idx in range(len(self.test[i])):
                y_test.append(self.test[i]['y'][idx])

            self._preprocessed_test_X[i] = self.reshape(np.array(X_test))
            self._preprocessed_test_y[i] = np.array(y_test).reshape(-1, 1)
            print("FOLD ", i + 1, " TEST PREPROCESS DONE")

    def reshape(self, x):
        return np.reshape(x, (x.shape[0], 1, x.shape[1]))

    def fit(self, epochs=30, batch_size=1, verbose=2):
        for i in range(self._fold):
            self.model[i].fit(X_train=self.preprocessed_train_X[i],
                              y_train=self.preprocessed_train_y[i],
                              X_val=self.preprocessed_test_X[i],
                              y_val=self.preprocessed_test_y[i],
                              lags=self.lags,
                              epochs=epochs,
                              batch_size=batch_size,
                              verbose=verbose)
            print("FOLD ", i + 1, " FIT DONE\n")

    def predict(self):
        for i in range(self._fold):
            fit_value_end, forecast_value = self.model[i].predict(self.preprocessed_train_X[i], self.preprocessed_test_X[i])
            fit_value_front = np.array([self.test[i]['y'][j] for j in range(self.lags)])
            fit_value = np.append(fit_value_front, fit_value_end)

            self.forecast_train[i] = pd.DataFrame(fit_value, columns=['yhat'], index=self.train[i].index)
            self.forecast_test[i] = pd.DataFrame(forecast_value, columns=['yhat'], index=self.test[i].index)
            self.forecast[i] = pd.concat([self.forecast_train[i], self.forecast_test[i]])

            print("FOLD ", i + 1, " FORECAST DONE")

    def fit_and_predict(self, epochs=30, batch_size=1, verbose=2):
        self.fit(epochs=epochs, batch_size=batch_size, verbose=verbose)
        self.predict()

    def plot_history(self):
        for i in range(self.fold):
            self.model[i].plot_history()


# 4. R Model - Exponential Smoothing, STL, Autoarima, Holtwinters, TBATS, Autoregressive NN
# Needs to be ported into python later
class NextoptRPipeline(NextoptBasePipeline):
    def __init__(self):
        super(NextoptRPipeline, self).__init__()
        NextoptRPipeline.check_forecast_package()
        self._frequency = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        for i in range(self._fold):
            self._model[i] = deepcopy(model)
            self._frequency = self._model[i].frequency

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, frequency):
        self._frequency = frequency

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

    def fit(self):
        print("USE FIT AND PREDICT")

    def predict(self):
        print("USE FIT AND PREDICT")

    def fit_and_predict(self):
        for i in range(self.fold):
            self.forecast_train[i], self.forecast_test[i] = \
                self.model[i].fit_and_predict(self.train[i]['y'], self.horizon)
            self.make_output_format(i)
            self.change_column_name(i)
            self.reset_forecast_train_index(i)
            self.reset_forecast_test_index(i)
            print("FOLD ", i + 1, " FORECAST DONE")

    def make_output_format(self, idx):
        self.forecast_train[idx].set_index(self.train[idx].index, inplace=True)
        self.forecast_test[idx].set_index(self.test[idx].index, inplace=True)

    def change_column_name(self, idx):
        self.forecast_train[idx].columns = ['yhat']
        self.forecast_test[idx].columns = ['yhat',
                                           'yhat_lower_80', 'yhat_upper_80',
                                           'yhat_lower_95', 'yhat_upper_95']

"""ETS는 쉽게 python porting 가능하다. """
class NextoptETSPipeline(NextoptRPipeline):
    def __init__(self):
        super(NextoptETSPipeline, self).__init__()

    def make_output_format(self, idx):
        self.forecast_train[idx] = self.forecast_train[idx][['xhat']]
        self.forecast_train[idx].loc[-1] = self.train[idx].iloc[0]['y']
        self.forecast_train[idx].sort_index(ascending=True, inplace=True)
        self.forecast_train[idx].set_index(self.train[idx].index, inplace=True)
        self.forecast_test[idx].set_index(self.test[idx].index, inplace=True)


class NextoptSTLPipeline(NextoptRPipeline):
    def __init__(self):
        super(NextoptSTLPipeline, self).__init__()

    def make_output_format(self, idx):
        self.forecast_train[idx]['yhat'] = self.forecast_train[idx]['seasonal'] + self.forecast_train[idx]['trend']
        self.forecast_train[idx].set_index(self.train[idx].index, inplace=True)
        self.forecast_test[idx].set_index(self.test[idx].index, inplace=True)

    def change_column_name(self, idx):
        self.forecast_test[idx].columns = ['yhat',
                                           'yhat_lower_80', 'yhat_upper_80',
                                           'yhat_lower_95', 'yhat_upper_95']


"""Holtwinters는 쉽게 python porting 가능하다. """
class NextoptHWPipeline(NextoptRPipeline):
    def __init__(self):
        super(NextoptHWPipeline, self).__init__()

    def make_output_format(self, idx):
        for i in range(self.frequency):
            index = (i + 1) * (-1)
            self.forecast_train[idx].loc[index] = self.train[idx].iloc[-index-1]['y']

        self.forecast_train[idx].sort_index(ascending=True, inplace=True)
        self.forecast_train[idx].set_index(self.train[idx].index, inplace=True)
        self.forecast_test[idx].set_index(self.test[idx].index, inplace=True)

    def change_column_name(self, idx):
        self.forecast_train[idx].columns = ['yhat', 'level', 'trend', 'season']
        self.forecast_test[idx].columns = ['yhat',
                                    'yhat_lower_80', 'yhat_upper_80',
                                    'yhat_lower_95', 'yhat_upper_95']


class NextoptAutoRegressiveNNPipeline(NextoptRPipeline):
    def __init__(self):
        super(NextoptAutoRegressiveNNPipeline, self).__init__()

    def fit(self):
        print("USE FIT AND PREDICT")

    def predict(self):
        print("USE FIT AND PREDICT")

    def fit_and_predict(self):
        for i in range(self.fold):
            self.forecast_train[i], self.forecast_test[i] = \
                self.model[i].fit_and_predict(self.train[i]['y'], self.horizon)
            self.make_output_format(i)
            self.change_column_name(i)
            self.reset_forecast_train_index(i)
            self.reset_forecast_test_index(i)
            print("FOLD ", i + 1, " FORECAST DONE")

    def change_column_name(self, idx):
        self.forecast_train[idx].columns = ['yhat']
        self.forecast_test[idx].columns = ['yhat']


"""AutoArima는 쉽게 python porting이 가능하다."""
class NextoptAutoArimaPipeline(NextoptRPipeline):
    def __init__(self):
        super(NextoptAutoArimaPipeline, self).__init__()


class NextoptTBATSPipeline(NextoptRPipeline):
    def __init__(self):
        super(NextoptTBATSPipeline, self).__init__()