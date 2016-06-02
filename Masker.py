from Util.FileUtil import FileReaderWriter
import numpy

class Masker:
    def __init__(self, file_type, mask_file):
        self.file_type = file_type
        self.mask_file = mask_file
        self.fileReaderWriter = FileReaderWriter(self.file_type)
        self.mask_array = self.read_mask()

    def read_mask(self):
        mask_array = self.fileReaderWriter.read_file(self.mask_file)
        return mask_array > 0.9

    def mask_data(self, data):
        return data[self.mask_array]

    def mask_image(self, image_file):
        image_data = self.fileReaderWriter.read_file(image_file)
        return self.mask_data(image_data)

    def rebuild_image_data(self, data):
        new_image_data = numpy.zeros(self.mask_array.shape, dtype=data.dtype)
        new_image_data[self.mask_array] = data
        return new_image_data

    def save_image_from_data(self, data, file_name):
        new_data = self.rebuild_image_data(data)
        self.fileReaderWriter.write_file(new_data, file_name, self.mask_file)

