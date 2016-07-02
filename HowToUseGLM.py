'''Running a Voxel-wise linear model with multiple volumetric variables'''

from pyVoxelStatsGLM import pyVoxelStatsGLM
import yappi

# yappi.start()
model_string = 'h ~ e + C_d + A'
csv_file = 'VoxelStatsTestData/CSV/Data.csv'
mask_file = 'VoxelStatsTestData/Masks/cerebellum_full_mask2.mnc'
voxel_variables = ['A', 'C_d']
subset_string = 'b > 1'
multi_variable_operations = ['A*(-1)']
file_type = 'minc'
family = 'binomial' ## This can either be a string or a statsmodels.api.families object.

glm = pyVoxelStatsGLM(file_type, model_string, csv_file, mask_file, voxel_variables, family, subset_string,
                    multi_variable_operations)
# glm.set_up_cluster(profile_name='sgeov', no_start=True)
glm.set_up_cluster(workers=4)
results = glm.evaluate()

#glm.save('Output.mnc', 'tvalues', 'C_d')
# stats = yappi.get_func_stats()
# stats.save('pstats.stats', type='pstat')
# with open('stats.stats', 'w') as f:
#     import pstats
#
#     ps = pstats.Stats('pstats.stats', stream=f)
#     ps.sort_stats('cumtime')
#     ps.print_stats()
