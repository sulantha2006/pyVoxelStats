import pandas

class DataMatrix:
    def __init__(self):
        self.X = None
        self.Y = None

class Dataset:
    def __init__(self, file_name, file_type='csv', delimiter=','):
        self._file_name = file_name
        self._file_type = file_type
        self._delimiter = delimiter
        self.data_table = self.load_data_file()

    def load_data_file(self):
        data_set = None
        if self._file_type == 'csv':
            data_set = pandas.read_csv(self._file_name, delimiter=self._delimiter)
        return data_set

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