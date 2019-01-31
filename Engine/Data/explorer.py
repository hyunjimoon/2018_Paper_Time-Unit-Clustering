from copy import deepcopy

import pandas as pd
from matplotlib import pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
import ruptures as rpt
from luminol.anomaly_detector import AnomalyDetector
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot


# from pandas.plotting import autocorrelation_plot

@pd.api.extensions.register_dataframe_accessor("explorer")
class Explorer(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

        if not type(self._obj.index) == pd.core.indexes.datetimes.DatetimeIndex:
            raise TypeError("YOU NEED TO SET INDEX AS DATETIME FORMAT")

        self.weekday_obj = self.group_by_weekday()  # weekday
        self._figsize = (8, 8)

        self.outliers = None

    # 1.set figsize
    @property
    def figsize(self):
        return self._figsize

    @figsize.setter
    def figsize(self, value):
        self._figsize = value

    @property
    def obj(self):
        return self._obj

    @property
    def outliers(self):
        return self._outliers

    @outliers.setter
    def outliers(self, outliers):
        self._outliers = outliers

    # 2. group by
    def group_by_weekday(self):
        return self._obj.groupby(self._obj.index.weekday)

    # 3. statistics
    def stats_by_feature(self):
        obj = self._obj.describe().T
        sum_obj = pd.DataFrame(self._obj.sum())
        sum_obj.columns = ['sum']
        return pd.concat([obj, sum_obj], axis=1)

    def stats_by_freq(self, freq='Y'):
        obj = self._obj.groupby(pd.Grouper(freq=freq)).describe()
        obj.columns = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

        sum_obj = pd.DataFrame(self._obj.groupby(pd.Grouper(freq=freq)).sum())
        sum_obj.columns = ['sum']
        return pd.concat([obj, sum_obj], axis=1)

    def stats_by_weekday(self):
        obj = self.weekday_obj.describe()
        obj.columns = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

        sum_obj = pd.DataFrame(self.weekday_obj.sum())
        sum_obj.columns = ['sum']
        return pd.concat([obj, sum_obj], axis=1)

    # 4. plot
    # 4-1. line and bar plot
    def plot_by_feature(self, kind='bar', how='sum', style=None, stacked=None):
        if how == 'sum':
            self._obj.sum().plot(kind=kind, style=style, stacked=stacked,
                                 figsize=self._figsize,
                                 ylim=[0, self._obj.sum().max() * 1.2])
        elif how == 'mean':
            self._obj.mean().plot(kind=kind, style=style, stacked=stacked,
                                  figsize=self._figsize,
                                  ylim=[0, self._obj.mean().max() * 1.2])

    def plot_by_freq(self, freq='Y', kind='line', how='sum', style=None, stacked=None):
        freq_obj = self._obj.groupby(pd.Grouper(freq=freq))
        if how == 'sum':
            freq_obj.sum().plot(
                kind=kind, style=style, stacked=stacked,
                figsize=self._figsize, ylim=[0, freq_obj.sum().max().max() * 1.2])
        elif how == 'mean':
            freq_obj.mean().plot(
                kind=kind, style=style, stacked=stacked,
                figsize=self._figsize, ylim=[0, freq_obj.mean().max().max() * 1.2])

    def plot_by_weekday(self, kind='line', how='sum', style=None, stacked=None):
        if how == 'sum':
            self.weekday_obj.sum().plot(kind=kind, style=style, stacked=stacked,
                                        figsize=self._figsize,
                                        ylim=[0, self.weekday_obj.sum().max().max() * 1.2])
        elif how == 'mean':
            self.weekday_obj.mean().plot(kind=kind, style=style, stacked=stacked,
                                         figsize=self._figsize,
                                         ylim=[0, self.weekday_obj.mean().max().max() * 1.2])

    # 4-2. histogram and density
    def hist(self, bins=15):
        self._obj.hist(figsize=self._figsize, bins=bins)

    def density(self):
        self._obj.plot(kind='kde', figsize=self._figsize)

    # 4-3. boxplot
    def boxplot(self, freq='Y'):
        groups = pd.DataFrame()
        code = self._obj.columns[0]

        for name, group in self._obj.groupby(pd.Grouper(freq=freq)):
            if freq == 'Y':
                groups[name.year] = pd.Series(group[code].values)
            elif freq == 'Q':
                groups[name.year + (name.quarter - 1) * 1 / 4] = pd.Series(group[code].values)
            elif freq == 'M':
                groups[name.year + (name.month - 1) * 1 / 12] = pd.Series(group[code].values)
        groups.boxplot(figsize=self._figsize)

    # 4-4. scatter
    def scatter(self, color=None, label=None):
        plt.scatter(self._obj.index,
                    self._obj[self._obj.columns[0]],
                    c=color, label=label, alpha=0.8)

    # 4-5. trend, seasonal
    def decompose_yearly(self):
        return_obj = seasonal_decompose(self._obj, freq=365)
        return return_obj

    def decompose_monthly(self):
        return_obj = seasonal_decompose(self._obj, freq=30)
        return return_obj

    def decompose_weekly(self):
        return_obj = seasonal_decompose(self._obj, freq=7)
        return return_obj

    # 4-6. trend changepoint
    def prophet_change_point(self):
        obj = self._obj.reset_index()
        obj.columns = ['ds', 'y']

        model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
        model.fit(obj)

        # 1. plot1
        plt.figure(figsize=self.figsize)
        plt.plot(self._obj)
        for i, changepoint in enumerate(model.changepoints):
            print(i, changepoint)
            plt.axvline(changepoint, ls='--', lw=1, c='r')

        # 2. plot2
        deltas = model.params['delta'].mean(0)
        fig = plt.figure(facecolor='w', figsize=self.figsize)
        ax = fig.add_subplot(111)
        ax.bar(range(len(deltas)), deltas)
        ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
        ax.set_ylabel('Rate change')
        ax.set_xlabel('Potential changepoint')
        fig.tight_layout()

    # 4-7. Outlier
    # get_outlier
    def detect_outlier(self):
        ts_dict = {}
        values = self._obj.values

        for i in range(len(self._obj)):
            ts_dict[i] = int(values[i])

        # 2. detect anomaly
        detector = AnomalyDetector(ts_dict)
        score = detector.get_all_scores()

        # 3. outlier dictionary
        outlier_dict = {}
        for timestamp, value in score.iteritems():
            outlier_dict[timestamp] = value

        self.outliers = sorted(outlier_dict.items(), key=lambda kv: kv[1], reverse=True)

    def show_outlier(self, count=15):
        outlier_timestamp = []

        # 1. print
        for i in range(count):
            outlier = self._obj.iloc[self.outliers[i][0]]
            outlier_timestamp.append(outlier.name)

            print("time: ", outlier.name, "\tvalue: ", outlier[0], "\tscore: ", self.outliers[i][1])

        # 2. detect
        obj = deepcopy(self._obj)
        obj['outlier'] = False
        obj.loc[outlier_timestamp, 'outlier'] = True

        plt.figure(figsize=self.figsize)
        plt.scatter(obj.index, obj[obj.columns[0]], c=obj.outlier, label=obj.outlier)

    def delete_outlier(self, count=15):
        outlier_index_list = [self.outliers[:count][i][0] for i in range(count)]

        obj = deepcopy(self._obj)
        obj.loc[self._obj.index[outlier_index_list]] = None
        return obj

    # 4-8. holiday
    def plot_holiday(self, holiday=None):
        try:
            obj = pd.merge(self.obj.reset_index(), holiday, on='ds', how='left')
        except:
            raise TypeError("holiday should be a dataframe with 'ds', and 'holiday'")

        obj = obj.fillna(0)
        obj['factor'], _ = pd.factorize(obj['holiday'])

        plt.figure(figsize=self.figsize)
        plt.scatter(obj.ds.values, obj[obj.columns[1]].values,
                    c=obj['factor'].values)

    def split_holiday(self, holiday=None):
        try:
            obj = pd.merge(self.obj.reset_index(), holiday, on='ds', how='left')
        except:
            raise TypeError("holiday should be a dataframe with 'ds', and 'holiday'")

        normal_day = obj.loc[obj['holiday'].isnull() == True]
        holiday = obj.loc[obj['holiday'].isnull() == False]

        return normal_day[normal_day.columns[:2]].set_index('ds'), holiday[holiday.columns[:2]].set_index('ds')


    """
    def ruptures_change_point(self, model='l2', algorithm='BottomUp', breakpoint=2):
        # check trend change point
        if algorithm == 'BottomUp':
            algo = rpt.BottomUp(model=model).fit(self._obj.values)
        result_list = algo.predict(n_bkps=breakpoint)

        # 1. print
        for i, result in enumerate(result_list):
            if i == len(result_list) - 1:
                break

            print(self._obj.iloc[result-1])
            print(self._obj.iloc[result])

        # 2. plot
        rpt.display(self._obj.values, result_list, figsize=self.figsize)
    """

    """
    # 4-6. autocorreation
    def plot_autocorrelation(self):
        plt.figure(figsize=self._figsize)
        autocorrelation_plot(self._obj[self._obj.columns[0]])
    
    # 4-7. comparison plot
    def plot_comparison(self, code, time_grouper, how):

    # 5-1. trend change point
    
    # 5-2. outlier
    def plot_outlier(self):
        pass
    """
