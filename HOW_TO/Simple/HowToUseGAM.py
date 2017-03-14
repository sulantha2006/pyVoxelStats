'''Running a Voxel-wise linear model with multiple volumetric variables'''

from pyVoxelStats.pyVoxelStats.pyVoxelStatsGAM import pyVoxelStatsGAM


model_string = 'A ~ s(C_d)'
csv_file = 'VoxelStatsTestData/CSV/Data.csv'
mask_file = 'VoxelStatsTestData/Masks/FullBrain.mnc'
voxel_variables = ['A', 'C_d']
subset_string = 'b > 1'
multi_variable_operations = ['A*(-1)']
file_type = 'minc'

lm = pyVoxelStatsGAM(file_type, model_string, csv_file, mask_file, voxel_variables, subset_string=subset_string,
                    multi_variable_operations=multi_variable_operations)
#lm.set_up_cluster(profile_name='sgeov', no_start=True)
lm.set_up_cluster(workers=56)
lm.evaluate()
#lm.set_up_cluster(workers=4)
results = lm.res
