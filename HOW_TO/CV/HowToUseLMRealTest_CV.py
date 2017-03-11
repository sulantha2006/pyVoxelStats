from pyVoxelStats.pyVoxelStatsLM import pyVoxelStatsLM
from sklearn.model_selection import KFold

model_string = 'Flubet_scan ~ MMSE + Age + C(Gender_code)'
csv_file = '/data/data03/sulantha/VoxelStatsPaper/CSVs/DataCSV.csv'
mask_file = 'VoxelStatsTestData/Masks/cerebellum_mask_rsl_8mm_blur_075_mask2_8mm_test_mask.mnc'
voxel_variables = ['Flubet_scan']
#subset_string = ''
file_type = 'minc'

lm = pyVoxelStatsLM(file_type, model_string, csv_file, mask_file, voxel_variables)
lm.enable_save_model()
lm.set_no_parallel(True)

lm.set_up_cluster()

#lm.set_up_cluster(profile_name='default')
kf = KFold(n_splits=3, shuffle=True)
lm.cv_evaluate(kf)
pred_mean = lm.preds.get_rmse()
print(pred_mean)
lm.save_image_data('/home/sulantha/Desktop/Holymm.mnc', pred_mean)
