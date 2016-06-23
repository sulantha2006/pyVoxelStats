from Util.StatsUtil import Dataset, StringModel, LM
from Util.Masker import Masker
from Util.VoxelOperation import VoxelOperation


string_model = 'A ~ b + C(C)'
data_file = 'VoxelStatsTestData/CSV/Data.csv'
mask_file = 'VoxelStatsTestData/Masks/FullBrain.mnc'
voxel_vars = ['A', 'C']
filter_string = 'b > 1'

string_model_obj = StringModel(string_model, voxel_vars)
data_set = Dataset(data_file, filter_string=filter_string, string_model_obj=string_model_obj)
masker = Masker('minc', mask_file)
stats_model = LM()

voxel_op = VoxelOperation(string_model_obj, data_set, masker, stats_model)
voxel_op.set_up()

