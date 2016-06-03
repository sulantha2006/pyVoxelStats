import nibabel
import pyminc.volumes.factory as fc

class FileReaderWriter:
    def __init__(self, file_type):
        self.file_type = file_type
        self.file_readers = {'minc':MincUtil, 'nifti':NiftiUtil}
        self.file_writers = {'minc':MincUtil, 'nifti':NiftiUtil}

    def read_file(self, image_file):
        return self.file_readers[self.file_type]().load(image_file)

    def write_file(self, data, file_name, ref_file):
        return self.file_writers[self.file_type]().save(data, file_name, ref_file)

class FileUtil:
    def load(self, file_name):
        pass

    def save(self, data, file_name, ref_file):
        pass

class MincUtil(FileUtil):
    def __init__(self):
        pass

    def load(self, file_name):
        vol_h = fc.volumeFromFile(file_name)
        data = vol_h.getdata()
        vol_h.closeVolume()
        return data

    def save(self, data, file_name, ref_file):
        out_h = fc.volumeLikeFile(ref_file, file_name)
        out_h.data = data
        out_h.writeFile()
        out_h.closeVolume()

class NiftiUtil(FileUtil):
    def __init__(self):
        print('Reading and writing Nifti files are still experimental. Not recommended for use')

    def load(self, file_name):
        img = nibabel.load(file_name)
        print('Reading and writing Nifti files are still experimental. Not recommended for use')
        return img.get_data()

    def save(self, data, file_name, ref_file):
        ref_img = nibabel.load(ref_file)
        affine = ref_img.affine
        print('Reading and writing Nifti files are still experimental. Not recommended for use')
        nibabel.save(nibabel.Nifti1Image(data, affine), file_name)