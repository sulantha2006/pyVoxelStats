'''Running a Voxel-wise linear model with multiple volumetric variables'''

from pyVoxelStatsLM import pyVoxelStatsLM

model_string = 'A ~ e + C_d + C(f)'
csv_file = 'VoxelStatsTestData/CSV/Data.csv'
mask_file = 'VoxelStatsTestData/Masks/FullBrain.mnc'
voxel_variables = ['A', 'C_d']
subset_string = 'b > 1'
multi_variable_operations = ['A*(-1)']
file_type = 'minc'

lm = pyVoxelStatsLM(file_type, model_string, csv_file, mask_file, voxel_variables, subset_string,
                    multi_variable_operations)
lm.set_up_cluster(profile_name='sgeov')
results = lm.evaluate()
#lm.save('Output.mnc', 'tvalues', 'C_d')
