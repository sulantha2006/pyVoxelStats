from Masker import Masker
import nibabel

masker = Masker('nifti', 'VoxelStatsTestData/Masks/FullBrain.nii')
image_masked = masker.mask_image('VoxelStatsTestData/NIFTY/I300779.nii')
masker.save_image_from_data(image_masked, 'Test/MincWrite.nii')
