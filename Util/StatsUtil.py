import pandas, re

class DataMatrix:
    def __init__(self):
        self._X = None
        self._Y = None

class Dataset:
    def __init__(self, file_name, file_type='csv', delimiter=',', filter_string=None):
        self._file_name = file_name
        self._file_type = file_type
        self._delimiter = delimiter
        self._filter_string = filter_string
        self._data_table_full = self.__load_data()

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

    def get_data_table(self, used_vars):
        return self._data_table_full[used_vars]

class StatsModel():
    def __init__(self, type):
        self._type = type

    def fit(self, X, y):
        pass

class LM(StatsModel):
    def __init__(self):
        StatsModel.__init__(self, 'lm')

class GLM(StatsModel):
    def __init__(self):
        StatsModel.__init__(self, 'glm')

class LME(StatsModel):
    def __init__(self):
        StatsModel.__init__(self, 'lme')

class GLME(StatsModel):
    def __init__(self):
        StatsModel.__init__(self, 'glme')

class StringModel:
    def __init__(self, string_model, voxel_vars):
        self._string_model = string_model
        self._voxel_vars = voxel_vars
        self._used_vars = self.__get_used_vars(self._string_model)

    def __get_used_vars(self, string_model):
        all_strings = re.findall(r"[.\w']+",  string_model)
        unique_vars = set(all_strings)
        return list(unique_vars)
