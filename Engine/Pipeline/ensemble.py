import numpy as np
import pandas as pd
from copy import deepcopy

from .pipeline import NextoptBasePipeline
from .pipeline_helper import InputError


class NextoptPipelines(NextoptBasePipeline):
    def __init__(self, *args):
        super(NextoptPipelines, self).__init__()
        self._pipelines = args
        self.check_all_folds_same()

    def check_all_folds_same(self):
        self._fold = self._pipelines[0].fold

        for pipeline in self._pipelines:
            if self._fold != pipeline.fold:
                raise InputError("InputError", "EACH PIPELINE MUST HAVE THE SAME FOLD.")

    def check_dates(self):
        pass

    def sum(self):
        self._train = deepcopy(self._pipelines[0].train)
        self._test = deepcopy(self._pipelines[0].test)
        self._forecast = deepcopy(self._pipelines[0].forecast)
        self._forecast_train = deepcopy(self._pipelines[0].forecast_train)
        self._forecast_test = deepcopy(self._pipelines[0].forecast_test)
        self._postprocessed_train = deepcopy(self._pipelines[0].postprocessed_train)
        self._postprocessed_test = deepcopy(self._pipelines[0].postprocessed_test)

        # add operation
        for fold in range(self._fold):
            for pipeline in self._pipelines[1:]:
                self.train[fold]['y'] = self.train[fold]['y'].add(pipeline.train[fold]['y'])
                self.test[fold]['y'] = self.test[fold]['y'].add(pipeline.test[fold]['y'])
                self.forecast_train[fold]['yhat'] = self.forecast_train[fold]['yhat'].add(pipeline.forecast_train[fold]['yhat'])
                self.forecast_test[fold]['yhat'] = self.forecast_test[fold]['yhat'].add(pipeline.forecast_test[fold]['yhat'])
                self.postprocessed_train[fold]['yhat'] = self.postprocessed_train[fold]['yhat'].add(pipeline.postprocessed_train[fold]['yhat'])
                self.postprocessed_test[fold]['yhat'] = self.postprocessed_test[fold]['yhat'].add(pipeline.postprocessed_test[fold]['yhat'])

    def average(self):
        self.sum()

        for fold in range(self._fold):
            self.train[fold].loc[:, 'y'] = self.train[fold].loc[:, 'y'] / len(self._pipelines)
            self.test[fold].loc[:, 'y'] = self.test[fold].loc[:, 'y'] / len(self._pipelines)
            self.forecast_train[fold].loc[:, 'yhat'] = self.forecast_train[fold].loc[:, 'yhat'] / len(self._pipelines)
            self.forecast_test[fold].loc[:, 'yhat'] = self.forecast_test[fold].loc[:, 'yhat'] / len(self._pipelines)
            self.postprocessed_train[fold].loc[:, 'yhat'] = self.postprocessed_train[fold].loc[:, 'yhat'] / len(self._pipelines)
            self.postprocessed_test[fold].loc[:, 'yhat'] = self.postprocessed_test[fold].loc[:, 'yhat' ] / len(self._pipelines)


# 2. Ensemble each pipeline
class NextoptEnsemblePipelines(NextoptPipelines):
    def __init__(self, *args):
        super(NextoptEnsemblePipelines, self).__init__(*args)
        self._main_pipeline = self._pipelines[0]
        self._sub_pipeline = self._pipelines[1:]
        self._evaluator = []
        self._forecast_values = []
        self._result_summaries = []

    @property
    def main_pipeline(self):
        return self._main_pipeline

    @main_pipeline.setter
    def main_pipeline(self, pipeline):
        self._main_pipeline = pipeline

    @property
    def pipeline(self):
        return self._main_pipeline

    @property
    def evaluator(self):
        return self._evaluator

    @evaluator.setter
    def evaluator(self, evaluator_list):
        self._evaluator = evaluator_list


    def evaluate(self):
        for pipeline in self._pipelines:
            pipeline.evaluate()
            self.evaluator.append(pipeline.evaluator)
            self._forecast_values.append(pipeline.forecast_value)
            self._result_summaries.append(pipeline.result_summary)


    def print_forecast_value(self):
        for pipeline in self._pipelines:
            print(pipeline.__class__.__name__)
            pipeline.print_forecast_value()
            print('\n\n')

    def save_forecast_value(self, path=None, format='csv', by='best'):
        if by == 'best':
            self._main_pipeline.save_forecast_value(path, format)

    def print_summary(self):
        for pipeline in self._pipelines:
            print(pipeline.__class__.__name__)
            pipeline.print_summary()
            print('\n\n')

    def save_summary(self, path=None, format='csv', by='best'):
        if by == 'best':
            self._main_pipeline.save_summary(path, format)

    def compare_summary(self, column='Error', by='mean'):
        if by == 'mean':
            score_by_mean = [int(summary[column].mean())
                         for summary in self._result_summaries]
            min_index = score_by_mean.index(
            max(score_by_mean)
            )
        self._main_pipeline = self._pipelines[min_index]
        print("COMPARING BY '"+ column + "' IS DONE: PIPELINE WITH BEST RESULT WOULD BE '" +
              self._pipelines[min_index].__class__.__name__ + "'.")

