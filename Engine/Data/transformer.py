import pandas as pd


@pd.api.extensions.register_dataframe_accessor("transformer")
class Transformer(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._dt_format = '%Y%m%d'  # Datetime Default

    # 1. Property
    @property
    def dt_format(self):
        return self._dt_format

    @dt_format.setter
    def dt_format(self, dt_format):
        self._dt_format = dt_format

    @property
    def obj(self):
        return self._obj

    # 2. aggregate by feature
    def aggregate(self, index_col='일자', feature_col='유형코드', value='수량', add_total=True, add_weekday=True):
        """
        Aggregate a pandas DataFrame into a new DataFrame.

        New Panda DataFrame Format

        Index: index_col in pd.datetime format
        Column: feature_col
        Value: value

        :param index_col: pandas datetime format, defualt '일자'
        :param feature_col: string, defualt '유형코드'
        :param value: string, default '수량'
        :param add_total: bool, default True
        :param add_weekday: bool, default True
        :return: new pandas DataFrame which is aggregated.
        """
        # transform by index and feature
        return_obj = self.transform(index_col, feature_col, value)
        #return_obj = return_obj.fillna(0)

        # add_total and weekday column
        if add_total:
            return_obj['total'] = return_obj.sum(axis=1)
        if add_weekday:
            return_obj['weekday'] = return_obj.index.weekday

        return return_obj

    # 2-1. transform format
    def transform(self, index_col, feature_col, value_col):
        return_obj = self._obj.groupby(
            [self._obj[index_col], self._obj[feature_col]]
        )[value_col].sum().unstack(feature_col)

        # change columns
        return_obj.columns = list(map(str, return_obj.columns.tolist()))
        return_obj.index = pd.to_datetime(return_obj.index, format=self._dt_format)

        if self.dt_format == '%Y%m%d':
            freq = 'D'

        return_obj = return_obj.reindex(
            pd.date_range(
                min(return_obj.index), max(return_obj.index), freq=freq,
            )
        )
        return_obj.index.name = 'ds'
        return return_obj

    # 3. get
    def get(self, start=None, end=None, code='all', weekday='all'):
        """
        Select rows which meet the condition

        :param start: pandas datetime, default no change
        :param end: pandas datatime, default no change
        :param code: string, default 'all'
        :param weekday: string, default 'all'
        :return:
        """

        if code == 'all':
            code = self._obj.columns
        if weekday == 'all':
            weekday = [0, 1, 2, 3, 4, 5, 6]

        if start is None and end is None:
            return_obj = self._obj
        elif start is None:
            return_obj = self._obj[:end]
        elif end is None:
            return_obj = self._obj[start:]
        else:
            return_obj = self._obj[start:end]

        return_obj = return_obj[return_obj.index.weekday.isin(weekday)]
        return_obj = return_obj[code]

        return return_obj





