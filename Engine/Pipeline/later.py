# 향후 사용 가능 함수 추가

"""
class DFM(object):
    def __init__(self, splitted_data_dict, horizon, forecast_model, **kwargs):
        self.splitted_data_dict = copy.deepcopy(splitted_data_dict)
        self.splitted_key = list(self.splitted_data_dict.keys())
        self.horizons = dict()

        self.split_each_data_into_train_test(horizon)
        self.set_horizons_for_each_test_data()

        self.models = dict()
        self.fit_df = list()
        self.forecast_df = list()
        for key in self.splitted_key:
            self.splitted_data_dict[key].update(kwargs)
            self.validate_option_dict_by_keys(self.splitted_data_dict[key], forecast_model)
            self.models[key] = forecast_model(**self.splitted_data_dict[key])


    def set_horizons_for_each_test_data(self):
        for key in self.splitted_key:
            self.horizons[key] = self.splitted_data_dict[key]['test'].shape[0]

    def split_each_data_into_train_test(self, horizon):
        enddate_list = list()
        for key in self.splitted_key:
            enddate_list.append(self.splitted_data_dict[key].index.max())
            enddate = max(enddate_list)

        for key in self.splitted_key:
            self.splitted_data_dict[key].train_test_split.enddate = enddate
            df_train, df_test = self.splitted_data_dict[key].preprocessor.train_test_split(horizon)
            self.splitted_data_dict[key] = {
                'train': df_train,
                'test': df_test
            }

    def validate_option_dict_by_keys(self, option_dict, forecast_model):
        sig = inspect.signature(forecast_model)
        param_list = list(option_dict.keys())
        error_flag = False
        error_message = str()

        # 모델에 필요한 파라미터들을 체크합니다. default 값이 없는 파라미터가
        # 주어진 input 값에 없다면 주어진 에러 메세지를 추가합니다.
        for key in sig.parameters.keys():
            if (sig.parameters[key].default == inspect._empty) \
                    and (key not in param_list):
                error_flag = True
                error_message += "Model *{}* needs parameter *{}*, but input has not.\n".format(
                    forecast_model.__name__, key)
            elif (key in param_list):
                param_list.remove(key)
            else:
                pass

        # 파라미터들을 체크한 후 input 값에 남은 파라미터가 없는지 체크합니다.
        # 남은 파라미터가 있다면 해당 모델에서 받아들이지 못하므로 에러 메세지를 띄웁니다.
        if param_list:
            error_flag = True
            for unexpected_param in param_list:
                error_message += "Unexpected parameter *{}* in input.\n".format(
                    unexpected_param)

        if error_flag:
            raise KeyError(error_message)

    def fit_and_forecast(self, concaternate_results= True, axis=0):
        for key, model in self.models.items():
            model.fit_and_forecast(horizon=self.horizons[key])
            self.fit_df.append(model.fit_df)
            self.forecast_df.append(model.forecast_df)

        if concaternate_results:
            self.fit_df = pd.concat(self.fit_df, axis=axis).sort_index()
            self.forecast_df = pd.concat(self.forecast_df, axis=axis).sort_index()





@pd.api.extensions.register_dataframe_accessor("data_handler")
class DataHandlerExtension(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def divide_df_by_value_condition(self, dict_of_condition, column_for_condition):
        result_dict= dict()
        for condition_key, condition in dict_of_condition.items():
            result_dict[condition_key] = copy.deepcopy(
                self._obj[
                        self._obj[column_for_condition].isin(condition)
                    ]
                )
        return result_dict

    def unique_condition(self, series):
        unique_values_list= list(series.unique())
        return {'{}_in_{}'.format(str(i), series.name): (i, )
                for i in unique_values_list}

    def divide_df_by_dt_cycle(self, cycle, start_num= 0,
                              dict_of_cycle_condition=None):
        len_of_rows= self._obj.shape[0]
        name_of_cycle_column= 'cycle{}'.format(cycle)
        cycle_series= pd.Series(
            np.remainder(np.arange(len_of_rows)+start_num, cycle),
            index= self._obj.index,
            name= name_of_cycle_column
            )
        self._obj= self._obj.join(cycle_series)
        if not dict_of_cycle_condition:
            dict_of_cycle_condition= self._obj.data_handling.unique_condition(cycle_series)

        result_dict= self._obj.data_handling.divide_df_by_value_condition(
            dict_of_cycle_condition, name_of_cycle_column)

        # delete cycle columns
        del self._obj[name_of_cycle_column]
        for divided_df in result_dict.values():
            del divided_df[name_of_cycle_column]

        return result_dict

    def divide_df_by_dt_period(self, period, start_num=0,
                               dict_of_period_condition=None):
        len_of_rows= self._obj.shape[0]
        name_of_period_column= 'period{}'.format(period)
        period_series= pd.Series(
            (np.arange(len_of_rows)+start_num+1)//period,
            index= self._obj.index,
            name= name_of_period_column
            )
        self._obj= self._obj.join(period_series)
        if not dict_of_period_condition:
            dict_of_period_condition= self._obj.data_handling.unique_condition(period_series)

        result_dict = self._obj.data_handling.divide_df_by_value_condition(
            dict_of_period_condition, name_of_period_column)

        # delete period columns
        del self._obj[name_of_period_column]
        for divided_df in result_dict.values():
            del divided_df[name_of_period_column]

        return result_dict
"""