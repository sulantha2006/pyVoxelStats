import nibabel

class Masker:
    def __init__(self, file_type, mask_file):
        self.file_type = file_type
        self.mask_file = mask_file
        self.mask = self.read_mask()

    def read_mask(self):
        img = nibabel.load(self.mask_file)
        mask_data = img.get_data()
        return mask_data > 0.9

    def mask_image(self, image_file):
        img = nibabel.load(image_file)
        image_data = img.get_data()
        return self.mask_data(image_data)

    def mask_data(self, data):
        return data[self.mask]

    def rebuild_image(self, data):
        new_image = data.reshape(self.mask.shape)
        return new_image

