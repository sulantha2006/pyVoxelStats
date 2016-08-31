'''Running a Voxel-wise linear model with multiple volumetric variables'''

from pyVoxelStats.pyVoxelStatsGAM import pyVoxelStatsGAM


# yappi.start()
model_string = 'A ~ s(C_d)'
csv_file = 'VoxelStatsTestData/CSV/Data.csv'
mask_file = 'VoxelStatsTestData/Masks/FullBrain.mnc'
voxel_variables = ['A', 'C_d']
subset_string = 'b > 1'
multi_variable_operations = ['A*(-1)']
file_type = 'minc'

lm = pyVoxelStatsGAM(file_type, model_string, csv_file, mask_file, voxel_variables, subset_string=subset_string,
                    multi_variable_operations=multi_variable_operations)
lm.set_up_cluster(profile_name='sgeov', no_start=True)
#lm.set_up_cluster(workers=4)
results = lm.evaluate()

#lm.save('Output.mnc', 'tvalues', 'C_d')
# stats = yappi.get_func_stats()
# stats.save('pstats.stats', type='pstat')
# with open('stats.stats', 'w') as f:
#     import pstats
#
#     ps = pstats.Stats('pstats.stats', stream=f)
#     ps.sort_stats('cumtime')
#     ps.print_stats()
