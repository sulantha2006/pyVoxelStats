from Util.FileUtil import FileReaderWriter
import numpy
from pyVoxelStats.pyVoxelStats import pyVoxelStats


class Masker(pyVoxelStats):
    def __init__(self, file_type, mask_file):
        pyVoxelStats.__init__(self)
        self._file_type = file_type
        self._mask_file = mask_file
        self._fileReaderWriter = FileReaderWriter(self._file_type)
        self._mask_array = self.__read_mask()
        self._mask_ndim = self._mask_array.ndim
        self._mask_shape = self._mask_array.shape

    def __read_mask(self):
        mask_array = self._fileReaderWriter.read_file(self._mask_file)
        return mask_array > 0.9

    def __mask_data(self, data):
        dim_dif = data.ndim - self._mask_ndim
        if dim_dif > 0:
            time_rep_array = [1] * data.ndim
            for dim in range(dim_dif):
                time_rep_array[dim] = data.shape[dim]
            mask = numpy.tile(self._mask_array, tuple(time_rep_array))

        elif dim_dif == 0:
            mask = self._mask_array
        else:
            print('Mask dimensions and data dimensions mismatch. Mask dimensions are higher than data dimension')
            return None
        return data[mask]

    def __mask_image(self, image_file):
        image_data = self._fileReaderWriter.read_file(image_file)
        return self.__mask_data(image_data)

    def mask_image(self, image_file):
        image_data = self._fileReaderWriter.read_file(image_file)
        return self.__mask_data(image_data)

    def __rebuild_image_data(self, data):
        # TODO Add rebuild when image dims are higher than mask dims - eg: 4D
        new_image_data = numpy.zeros(self._mask_array.shape, dtype=data.dtype)
        new_image_data[self._mask_array] = data
        return new_image_data

    def save_image_from_data(self, data, file_name):
        new_data = self.__rebuild_image_data(data)
        self._fileReaderWriter.write_file(new_data, file_name, self._mask_file)

    def get_data_from_image(self, image_file_name):
        return self.__mask_image(image_file_name)
