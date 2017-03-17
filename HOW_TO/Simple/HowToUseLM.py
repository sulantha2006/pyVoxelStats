'''Running a Voxel-wise linear model with multiple volumetric variables'''
from pyVS.pyVoxelStats.pyVoxelStatsLM import pyVoxelStatsLM

model_string = 'A_minc ~ e_val + C_d_minc + C(f)'
csv_file = 'VoxelStatsTestData/CSV/Data_MT.csv'
mask_file = 'VoxelStatsTestData/Masks/cerebellum_full_mask2.mnc'
voxel_variables = ['A_minc', 'C_d_minc']
subset_string = 'b_val > 1'
multi_variable_operations = ['A_minc*(-1)']
file_type = 'minc'

lm = pyVoxelStatsLM(file_type, model_string, csv_file, mask_file, voxel_variables, subset_string,
                    multi_variable_operations)
lm.set_no_parallel(False)
lm.enable_save_model()
lm.set_up_cluster()
lm.evaluate()
#lm.set_up_cluster(workers=4)
#results = lm.res

## Pickle analysis if needed to write results later
import pickle
## Saving
pickle.dump(lm, 'VoxelStatsTestData/Pickles/LM.pkl')
## Loading Back
lm2 = pickle.load('VoxelStatsTestData/Pickles/LM.pkl')