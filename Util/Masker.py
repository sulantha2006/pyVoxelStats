from Util.FileUtil import FileReaderWriter
import numpy

class Masker:
    def __init__(self, file_type, mask_file):
        self._file_type = file_type
        self._mask_file = mask_file
        self._fileReaderWriter = FileReaderWriter(self._file_type)
        self._mask_array = self.__read_mask()

    def __read_mask(self):
        mask_array = self._fileReaderWriter.read_file(self._mask_file)
        return mask_array > 0.9

    def __mask_data(self, data):
        return data[self._mask_array]

    def __mask_image(self, image_file):
        image_data = self._fileReaderWriter.read_file(image_file)
        return self.__mask_data(image_data)

    def __rebuild_image_data(self, data):
        new_image_data = numpy.zeros(self._mask_array.shape, dtype=data.dtype)
        new_image_data[self._mask_array] = data
        return new_image_data

    def save_image_from_data(self, data, file_name):
        new_data = self.__rebuild_image_data(data)
        self._fileReaderWriter.write_file(new_data, file_name, self._mask_file)

    def get_data_from_image(self, image_file_name):
        return self.__mask_image(image_file_name)

