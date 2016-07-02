from Util.StatsUtil import Dataset, StringModel, GLM
from Util.Masker import Masker
from Util.VoxelOperation import VoxelOperation
from pyVoxelStats import pyVoxelStats
import statsmodels.api as smapi


class pyVoxelStatsGLM(pyVoxelStats):
    def __init__(self, file_type, model_string, csv_file, mask_file, voxel_variables, family_obj, subset_string=None,
                 multi_variable_operations=None):
        pyVoxelStats.__init__(self)
        self.file_type = file_type
        self.string_model = model_string
        self.data_file = csv_file
        self.mask_file = mask_file
        self.voxel_vars = voxel_variables
        self.filter_string = subset_string
        self.multi_var_operations = multi_variable_operations

        self.family_dict = dict(binomial=smapi.families.Binomial, gamma=smapi.families.Gamma, gaussian=smapi.families.Gaussian,
                                poisson=smapi.families.Poisson, inversegaussian=smapi.families.InverseGaussian,
                                negativebinomial=smapi.families.NegativeBinomial)
        self.family_obj = self.get_family(family_obj)

        self.string_model_obj = None
        self.data_set = None
        self.masker = None
        self.stats_model = None

        self.cluster_profile = 'default'
        self.clus_workers = None
        self.clus_no_start = False

        self.res = None

    def get_family(self, family_obj):
        if type(family_obj) is str:
            return self.family_dict[family_obj]()
        else:
            return family_obj

    def set_up_cluster(self, profile_name='default', workers=None, no_start=False):
        self.cluster_profile = profile_name
        self.clus_workers = workers
        self.clus_no_start = no_start

    def evaluate(self):
        self.string_model_obj = StringModel(self.string_model, self.voxel_vars, self.multi_var_operations)
        self.data_set = Dataset(self.data_file, filter_string=self.filter_string,
                                string_model_obj=self.string_model_obj)
        self.masker = Masker(self.file_type, self.mask_file)
        self.stats_model = GLM(self.string_model_obj, self.family_obj)

        voxel_op = VoxelOperation(self.string_model_obj, self.data_set, self.masker, self.stats_model)
        voxel_op.set_up_cluster(profile_name=self.cluster_profile, workers=self.clus_workers, no_start=self.clus_no_start)
        voxel_op.set_up()
        voxel_op.execute()
        self.res = voxel_op.results.get_results()
        return self.res

    def save(self, file_name, result_field, var_name=None):
        if var_name:
            print('Saving - {0} - {1} -> {2} ....'.format(result_field, var_name, file_name))
            self.masker.save_image_from_data(self.res[result_field][var_name], file_name)
        else:
            print('Saving - {0} -> {1} ....'.format(result_field, file_name))
            self.masker.save_image_from_data(self.res[result_field], file_name)
        print('Saving done.')
