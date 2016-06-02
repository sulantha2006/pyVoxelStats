from Masker import Masker
import nibabel

masker = Masker('minc', 'VoxelStatsTestData/Masks/FullBrain.mnc')
image_masked = masker.mask_image('VoxelStatsTestData/MINC/I300779.mnc')
rebuilt = masker.rebuild_image(image_masked)
rebuilt.to_filename('Test/MaskerRebuilt.mnc')
