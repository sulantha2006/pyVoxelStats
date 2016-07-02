import pandas, re
import statsmodels.formula.api as smf
import statsmodels.tools.sm_exceptions as sme
from pyVoxelStats import pyVoxelStats


class DataMatrix(pyVoxelStats):
    def __init__(self):
        pyVoxelStats.__init__(self)
        self._X = None
        self._Y = None


class Dataset(pyVoxelStats):
    def __init__(self, file_name, file_type='csv', delimiter=',', filter_string=None, string_model_obj=None):
        pyVoxelStats.__init__(self)
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


class StatsModel(pyVoxelStats):
    def __init__(self, type):
        pyVoxelStats.__init__(self)
        self._type = type
        self.model_wise_results_names = [re.sub('\\[|\\]', '', s.strip().replace("'", '')) for s in
                                         self.config['ResultsModelWiseResults'][self._type].split(',')]
        self.var_wise_results_names = [re.sub('\\[|\\]', '', s.strip().replace("'", '')) for s in
                                       self.config['ResultsModelVariableWiseResults'][self._type].split(',')]

    def fit(self, data_frame):
        print('Not yet implemented')
        return None

    def filter_result(self, result, model):
        result_f = {}
        for vard in self.model_wise_results_names:
            if result:
                result_f[vard] = getattr(result, vard)
            else:
                result_f[vard] = 0
        variable_names_in_model_op = model.exog_names
        for vard in self.var_wise_results_names:
            if result:
                result_f[vard] = {name: getattr(result, vard)[name] for name in variable_names_in_model_op}
            else:
                result_f[vard] = {name: 0 for name in variable_names_in_model_op}
        result_f['variable_names_in_model_op'] = variable_names_in_model_op
        return result_f


class LM(StatsModel):
    def __init__(self, string_model):
        StatsModel.__init__(self, 'lm')
        self.string_model = string_model

    def fit(self, data_frame):
        mod = smf.ols(formula=self.string_model._string_model_str, data=data_frame)
        try:
            res = mod.fit()
        except (sme.PerfectSeparationError, sme.MissingDataError) as e:
            res = None
            # print('Statistics exception; result for the voxel may be set to 0 : ' + str(e))
        return self.filter_result(res, mod)


class GLM(StatsModel):
    def __init__(self, string_model, family_obj):
        StatsModel.__init__(self, 'glm')
        self.string_model = string_model
        self.family_obj = family_obj

    def fit(self, data_frame):
        mod = smf.glm(formula=self.string_model._string_model_str, data=data_frame, family=self.family_obj)
        try:
            res = mod.fit()
        except (sme.PerfectSeparationError, sme.MissingDataError) as e:
            res = None
            # print('Statistics exception; result for the voxel may be set to 0 : ' + str(e))
        return self.filter_result(res, mod)


class LME(StatsModel):
    def __init__(self, string_model):
        StatsModel.__init__(self, 'lme')
        self.string_model = string_model


class GEE(StatsModel):
    def __init__(self, string_model, family_obj, covariance_obj, time):
        StatsModel.__init__(self, 'gee')
        self.string_model = string_model
        self.family_obj = family_obj
        self.covariance_obj = covariance_obj
        self.time=time

    def fit(self, data_frame):
        mod = smf.gee(formula=self.string_model._string_model_str, data=data_frame, family=self.family_obj,
                      cov_struct=self.covariance_obj, time=self.time)
        try:
            res = mod.fit()
        except (sme.PerfectSeparationError, sme.MissingDataError) as e:
            res = None
            # print('Statistics exception; result for the voxel may be set to 0 : ' + str(e))
        return self.filter_result(res, mod)


class StringModel(pyVoxelStats):
    def __init__(self, string_model_str, voxel_vars, multi_var_ops=None):
        pyVoxelStats.__init__(self)
        self._string_model_str = string_model_str
        self._voxel_vars = voxel_vars
        self._used_vars = self.__get_used_vars(self._string_model_str)
        self.multi_var_ops = multi_var_ops

    def __get_used_vars(self, string_model):
        all_strings = re.findall(r"[.C\(\w\)\w']+", string_model)
        all_strings = [re.sub(r'C\(([\w]+)\)', r'\1', st) for st in all_strings]
        unique_vars = set(all_strings)
        return list(unique_vars)
