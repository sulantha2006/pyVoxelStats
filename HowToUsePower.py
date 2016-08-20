'''Running a Voxel-wise power analysis with multiple volumetric variables'''

from pyVoxelStatsPower import pyVoxelStatsPower

model_string = '(0.842 + 1.96)**2 + 2*std**2 / (0.25*mean)**2'
csv_file = '/home/sulantha/Downloads/table_ADNI_CSF_CN_Sulantha.csv'
mask_file = '/data/data03/quarantine/mni_icbm152_t1_tal_nlin_asym_09a_mask.mnc'
voxel_variables = ['perdiff_fdg_unnorm']
file_type = 'minc'
filter = 'Both == 1'

pw = pyVoxelStatsPower(file_type, model_string, csv_file, mask_file, voxel_variables, subset_string=filter)
pw.set_up_cluster(profile_name='sgeov', no_start=True)
results = pw.evaluate()
pw.save('Output.mnc', 'ss')