'''Running a Voxel-wise Partial Correlation '''
from pyVS.pyVoxelStats.pyVoxelStatsPCorr import pyVoxelStatsPCorr

model_string = 'A_minc ~ e_val + C_d_minc + C(f)'
csv_file = 'VoxelStatsTestData/CSV/Data_MT.csv'
mask_file = 'VoxelStatsTestData/Masks/cerebellum_full_mask2.mnc'
voxel_variables = ['A_minc', 'C_d_minc']
#subset_string = 'b_val > 1'
#multi_variable_operations = ['A_minc*(-1)']
file_type = 'minc'
var_x = 'e_val'
var_y = 'A_minc'

pcorr = pyVoxelStatsPCorr(file_type, model_string, csv_file, mask_file, voxel_variables, var_x=var_x, var_y=var_y)

#lm.enable_save_model()
pcorr.set_up_cluster(clus_json='/home/sulantha/ipyparallel_json/ipcontroller-client.json', no_start=True)
#pcorr.set_no_parallel(True)
pcorr.evaluate()
results = pcorr.res
print(pcorr.res['df'])
pcorr.save('/home/sulantha/Desktop/holishit.mnc', 'r_prime')
pcorr.save('/home/sulantha/Desktop/holishitt.mnc', 't_')
pcorr.save('/home/sulantha/Desktop/holishitp.mnc', 'p_')

#lm.set_up_cluster(workers=4)
#results = lm.res
