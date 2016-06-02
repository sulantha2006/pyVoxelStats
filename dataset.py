import pandas

class dataset:
    def __init__(self, file_name, file_type='csv', delimiter=','):
        self._file_name = file_name
        self._file_type = file_type
        self._delimiter = delimiter
        self.data_table = self.load_data_file()

    def load_data_file(self):
        if self._file_type == 'csv':
            data_set = pandas.read_csv(self._file_name, delimiter=self._delimiter)
        return data_set
