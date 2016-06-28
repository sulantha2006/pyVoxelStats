from pyVoxelStatsLM import pyVoxelStatsLM

model_string = 'fdg ~ AV45_pathbl + AV45_bl_norm + VBM + Age + PTGENDER + APOE + PTEDUCAT'
csv_file = '/data/data03/tharick/Paper_3/All_baseline_fdg_av45.csv'
mask_file = '/data/data03/sulantha/quarantine/mni_icbm152_t1_tal_nlin_sym_09a_mask.mnc'
voxel_variables = ['AV45_pathbl', 'fdg', 'VBM']
subset_string = 'diag_3bl == 2 & Baseline == 1'
file_type = 'minc'

lm = pyVoxelStatsLM(file_type, model_string, csv_file, mask_file, voxel_variables, subset_string)
lm.set_up_cluster(profile_name='sgeov', workers=215)
results = lm.evaluate()

lm.save('/home/sulantha/Desktop/Av45_bathbl_fromThraick_pyvs_speedtest_2.mnc', 'tvalues', 'AV45_pathbl')

