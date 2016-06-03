import pandas, re

class DataMatrix:
    def __init__(self):
        self.X = None
        self.Y = None

class Dataset:
    def __init__(self, file_name, file_type='csv', delimiter=',', filter_string=None):
        self._file_name = file_name
        self._file_type = file_type
        self._delimiter = delimiter
        self.filter_string = filter_string
        self.data_table_full = self.load_data()

    def load_data_file(self):
        data_set = None
        if self._file_type == 'csv':
            data_set = pandas.read_csv(self._file_name, delimiter=self._delimiter)
        return data_set

    def filter_data_file(self, data_set):
        return data_set.query(self.filter_string)

    def load_data(self):
        data_set = self.load_data_file()
        if self.filter_string:
            filtered_data = self.filter_data_file(data_set)
        else:
            filtered_data = data_set
        return filtered_data

    def get_data_table(self, used_vars):
        return self.data_table_full[used_vars]

class StatsModel():
    def __init__(self, type):
        self.type = type

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
        self.string_model = string_model
        self.voxel_vars = voxel_vars
        self.used_vars = self.get_used_vars(self.string_model)

    def get_used_vars(self, string_model):
        all_strings = re.findall(r"[.\w']+",  string_model)
        unique_vars = set(all_strings)
        return list(unique_vars)
