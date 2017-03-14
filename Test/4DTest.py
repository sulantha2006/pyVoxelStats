'''Running a Voxel-wise linear model with 4d image files'''

from pyVoxelStats.pyVoxelStats.pyVoxelStatsLM import pyVoxelStatsLM



model_string = 'NAV_4d ~ PBR_3d + D + C(G)'
csv_file = '/home/sulantha/PycharmProjects/pyVoxelStats/VoxelStatsTestData/CSV/4D_Rat_data.csv'
mask_file = '/home/sulantha/PycharmProjects/pyVoxelStats/VoxelStatsTestData/Rat4D/Mask.mnc'
voxel_variables = ['NAV_4d', 'PBR_3d']
#subset_string = 'b > 1'
#multi_variable_operations = ['A*(-1)']
file_type = 'minc'

lm = pyVoxelStatsLM(file_type, model_string, csv_file, mask_file, voxel_variables)
lm.set_up_cluster(profile_name='sgeov', no_start=True)
#lm.set_up_cluster(workers=4)
results = lm.evaluate()

lm.save('/home/sulantha/Desktop/PBR_RAT_4d.mnc', 'tvalues', 'PBR_3d')