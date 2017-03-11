import sys
sys.path.append('/home/sulantha/PycharmProjects/pyVoxelStats')
from Util.Masker import Masker

masker = Masker('minc', '/home/sulantha/PycharmProjects/pyVoxelStats/VoxelStatsTestData/Rat4D/Mask.mnc')
image_masked = masker.mask_image('/home/sulantha/PycharmProjects/pyVoxelStats/VoxelStatsTestData/Rat4D/A1795dynres_nlinatlas.mnc')
masker.save_image_from_data(image_masked, '/home/sulantha/Desktop/MincWrite3.mnc')
