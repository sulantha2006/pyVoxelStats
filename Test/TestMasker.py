from pyVoxelStats.Util.Masker import Masker

masker = Masker('minc', '/home/sulantha/PycharmProjects/pyVoxelStats/VoxelStatsTestData/Masks/FullBrain.mnc')
image_masked = masker.mask_image('/home/sulantha/PycharmProjects/pyVoxelStats/VoxelStatsTestData/MINC/I300779.mnc')
masker.save_image_from_data(image_masked, '/home/sulantha/Desktop/MincWrite3.mnc')
