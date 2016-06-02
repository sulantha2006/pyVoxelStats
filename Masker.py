import nibabel, numpy

class Masker:
    def __init__(self, file_type, mask_file):
        self.file_type = file_type
        self.mask_file = mask_file
        self.mask, self.mask_array = self.read_mask()

    def read_mask(self):
        img = nibabel.load(self.mask_file)
        mask_data = img.get_data()
        return img, mask_data > 0.9

    def mask_image(self, image_file):
        img = nibabel.load(image_file)
        image_data = img.get_data()
        return self.mask_data(image_data)

    def mask_data(self, data):
        return data[self.mask_array]

    def rebuild_image(self, data):
        new_image_data = numpy.empty(self.mask_array.shape, dtype=data.dtype)
        new_image_data[self.mask_array] = data
        affine = self.mask.affine
        if self.file_type == 'minc':
            new_image = nibabel.Minc2Image(new_image_data, affine)
        elif self.file_type == 'nifti':
            new_image = nibabel.Nifti1Image(new_image_data, affine)
        return new_image

