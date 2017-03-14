from Util.Masker import Masker
from Util.VoxelOperation import VoxelOperation

from pyVoxelStats.Util.StatsUtil import Dataset, StringModel
from pyVoxelStats.Util.StatsUtil import GAM
from pyVoxelStats.pyVoxelStats import pyVoxelStats


class pyVoxelStatsGAM(pyVoxelStats):
    def __init__(self, file_type, model_string, csv_file, mask_file, voxel_variables, family_str=None, method_str=None,
                 subset_string=None, multi_variable_operations=None):
        pyVoxelStats.__init__(self, file_type, model_string, csv_file, mask_file, voxel_variables, subset_string,
                              multi_variable_operations)
        self.family_str = family_str
        self.method_str = method_str

    def evaluate(self):
        self.string_model_obj = StringModel(self.string_model, self.voxel_vars, self.multi_var_operations)
        self.data_set = Dataset(self.data_file, filter_string=self.filter_string,
                                string_model_obj=self.string_model_obj)
        self.masker = Masker(self.file_type, self.mask_file)
        self.stats_model = GAM(self.string_model_obj, self.family_str, self.method_str)

        voxel_op = VoxelOperation(self.string_model_obj, self.data_set, self.masker, self.stats_model)
        voxel_op.set_up_cluster(profile_name=self.cluster_profile, workers=self.clus_workers, no_start=self.clus_no_start)
        voxel_op.set_up()
        voxel_op.execute()
        self.res = voxel_op.results.get_results()
        self.models = voxel_op.results.get_models()
        return self.res