'''Running a Voxel-wise linear model with multiple volumetric variables'''

from pyVS.pyVoxelStats.pyVoxelStatsLM import pyVoxelStatsLM



model_string = 'A ~ e + C_d + C(f)'
csv_file = 'VoxelStatsTestData/CSV/Data.csv'
mask_file = 'VoxelStatsTestData/Masks/cerebellum_full_mask2.mnc'
voxel_variables = ['A', 'C_d']
subset_string = 'b > 1'
multi_variable_operations = ['A*(-1)']
file_type = 'minc'

lm = pyVoxelStatsLM(file_type, model_string, csv_file, mask_file, voxel_variables, subset_string,
                    multi_variable_operations)
lm.set_no_parallel(False)
lm.enable_save_model()
lm.set_up_cluster()
lm.evaluate()
#lm.set_up_cluster(workers=4)
results = lm.res