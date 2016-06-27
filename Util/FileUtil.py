import nibabel
import pyminc.volumes.factory as fc


class FileReaderWriter:
    def __init__(self, file_type):
        self._file_type = file_type
        self._file_readers = {'minc': MincUtil, 'nifti': NiftiUtil}
        self._file_writers = {'minc': MincUtil, 'nifti': NiftiUtil}

    def read_file(self, image_file):
        return self._file_readers[self._file_type]()._load(image_file)

    def write_file(self, data, file_name, ref_file):
        return self._file_writers[self._file_type]()._save(data, file_name, ref_file)


class FileUtil:
    def _load(self, file_name):
        pass

    def _save(self, data, file_name, ref_file):
        pass


class MincUtil(FileUtil):
    def __init__(self):
        pass

    def _load(self, file_name):
        vol_h = fc.volumeFromFile(file_name)
        data = vol_h.getdata()
        vol_h.closeVolume()
        return data

    def _save(self, data, file_name, ref_file):
        out_h = fc.volumeLikeFile(ref_file, file_name, dtype='short', volumeType='ushort')
        out_h.data = data
        out_h.writeFile()
        out_h.closeVolume()


class NiftiUtil(FileUtil):
    def __init__(self):
        pass

    def _load(self, file_name):
        img = nibabel.load(file_name)
        return img.get_data()

    def _save(self, data, file_name, ref_file):
        ref_img = nibabel.load(ref_file)
        affine = ref_img.affine
        nibabel.save(nibabel.Nifti1Image(data, affine), file_name)
