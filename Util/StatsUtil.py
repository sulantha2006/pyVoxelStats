import pandas, re
import statsmodels.formula.api as smf
from Util.Params import StatsModelsParams


class DataMatrix:
    def __init__(self):
        self._X = None
        self._Y = None


class Dataset:
    def __init__(self, file_name, file_type='csv', delimiter=',', filter_string=None, string_model_obj=None):
        self._file_name = file_name
        self._file_type = file_type
        self._delimiter = delimiter
        self._filter_string = filter_string
        self._data_table_full = self.__load_data()
        self.string_model_obj = string_model_obj
        self._data_table = self.__get_data_table()

    def __load_data_file(self):
        data_set = None
        if self._file_type == 'csv':
            data_set = pandas.read_csv(self._file_name, delimiter=self._delimiter)
        return data_set

    def __filter_data_file(self, data_set):
        return data_set.query(self._filter_string)

    def __load_data(self):
        data_set = self.__load_data_file()
        if self._filter_string:
            filtered_data = self.__filter_data_file(data_set)
        else:
            filtered_data = data_set
        return filtered_data

    def __get_data_table(self, used_vars=None):
        if not used_vars:
            used_vars = self.string_model_obj._used_vars
        return self._data_table_full[used_vars]


class StatsModel():
    def __init__(self, type):
        self._type = type

    def fit(self, data_frame):
        print('Not yet implemented')
        return None

    def filter_result(self, result):
        print('Not yet implemented')
        return None


class LM(StatsModel):
    def __init__(self, string_model):
        StatsModel.__init__(self, 'lm')
        self.string_model = string_model
        self.model_wise_results_names = StatsModelsParams.ResultsModelWiseResults['lm']
        self.var_wise_results_names = StatsModelsParams.ResultsModelVariableWiseResults['lm']

    def fit(self, data_frame):
        mod = smf.ols(formula=self.string_model._string_model_str, data=data_frame)
        res = mod.fit()
        return self.filter_result(res)

    def filter_result(self, result):
        result_f = {}
        for vard in self.model_wise_results_names:
            result_f[vard] = getattr(result, vard)
        variable_names_in_model_op = result.model.exog_names
        for vard in self.var_wise_results_names:
            result_f[vard] = {name: getattr(result, vard)[name] for name in variable_names_in_model_op}
        result_f['variable_names_in_model_op'] = variable_names_in_model_op
        return result_f


class GLM(StatsModel):
    def __init__(self, string_model):
        StatsModel.__init__(self, 'glm')
        self.string_model = string_model


class LME(StatsModel):
    def __init__(self, string_model):
        StatsModel.__init__(self, 'lme')
        self.string_model = string_model


class GLME(StatsModel):
    def __init__(self, string_model):
        StatsModel.__init__(self, 'glme')
        self.string_model = string_model


class StringModel:
    def __init__(self, string_model_str, voxel_vars, multi_var_ops=None):
        self._string_model_str = string_model_str
        self._voxel_vars = voxel_vars
        self._used_vars = self.__get_used_vars(self._string_model_str)
        self.multi_var_ops = multi_var_ops

    def __get_used_vars(self, string_model):
        all_strings = re.findall(r"[.C\(\w\)\w']+", string_model)
        all_strings = [re.sub(r'C\(([\w]+)\)', r'\1', st) for st in all_strings]
        unique_vars = set(all_strings)
        return list(unique_vars)
