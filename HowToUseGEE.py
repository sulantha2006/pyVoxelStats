'''Running a Voxel-wise linear model with multiple volumetric variables'''

import statsmodels.api

from pyVoxelStats.pyVoxelStatsGEE import pyVoxelStatsGEE

model_string = 'h ~ e + C_d + D'
csv_file = 'VoxelStatsTestData/CSV/Data.csv'
mask_file = 'VoxelStatsTestData/Masks/cerebellum_full_mask2.mnc'
voxel_variables = ['D', 'C_d']
subset_string = 'b > 1'
multi_variable_operations = ['A*(-1)']
file_type = 'minc'
family = 'binomial' ## This can either be a string or a statsmodels.api.families object.

cov = statsmodels.api.cov_struct.Exchangeable()

glm = pyVoxelStatsGEE(file_type, model_string, csv_file, mask_file, voxel_variables, family, groups='ID', cov_struct=cov,
                      subset_string=subset_string, multi_variable_operations=multi_variable_operations)
glm.set_up_cluster(no_start=False)
#glm.set_up_cluster(workers=4)
results = glm.evaluate()

