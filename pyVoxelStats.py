import os, configparser
import Util.Params.pyVSParams
from Util.StatsUtil import Dataset, StringModel
from Util.Masker import Masker


class pyVoxelStats:
    def __init__(self, file_type, model_string, csv_file, mask_file, voxel_variables, subset_string=None,
                 multi_variable_operations=None):
        self.config = self.check_config()
        self.string_model = model_string
        self.file_type = file_type
        self.data_file = csv_file
        self.mask_file = mask_file
        self.voxel_vars = voxel_variables
        self.filter_string = subset_string
        self.multi_var_operations = multi_variable_operations

        self.string_model_obj = StringModel(self.string_model, self.voxel_vars, self.multi_var_operations)
        self.data_set = Dataset(self.data_file, filter_string=self.filter_string,
                                string_model_obj=self.string_model_obj)
        self.masker = Masker(self.file_type, self.mask_file)

        self.string_model_obj = None
        self.data_set = None
        self.masker = None
        self.stats_model = None
        self.res = None

        self.cluster_profile = 'default'
        self.clus_workers = None
        self.clus_no_start = False

    def set_new_config(self, config_path):
        config_dict = Util.Params.pyVSParams.config_dict
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        cf = configparser.ConfigParser()
        for k in config_dict.keys():
            cf[k] = config_dict[k]
        with open(config_path, 'w') as configfile:
            cf.write(configfile)
        return cf

    def check_config(self, config_path='{0}/.pyVS/pyVSParams.ini'.format(os.path.expanduser('~'))):
        if os.path.exists(config_path):
            cf = configparser.ConfigParser()
            cf.read(config_path)
            config_dict = Util.Params.pyVSParams.config_dict
            if 'version' not in cf['VSVoxelOPS'] or (
                config_dict['VSVoxelOPS']['version'] > float(cf['VSVoxelOPS']['version'])):
                cf = self.set_new_config(config_path=config_path)
            return cf
        else:
            return self.set_new_config(config_path=config_path)

    def save(self, file_name, result_field, var_name=None):
        if var_name:
            print('Saving - {0} - {1} -> {2} ....'.format(result_field, var_name, file_name))
            self.masker.save_image_from_data(self.res[result_field][var_name], file_name)
        else:
            print('Saving - {0} -> {1} ....'.format(result_field, file_name))
            self.masker.save_image_from_data(self.res[result_field], file_name)
        print('Saving done.')

    def set_up_cluster(self, profile_name='default', workers=None, no_start=False):
        self.cluster_profile = profile_name
        self.clus_workers = workers
        self.clus_no_start = no_start

    def evaluate(self):
        raise Exception('Not yet implemented')
