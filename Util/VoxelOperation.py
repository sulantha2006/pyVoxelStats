class VoxelOperation:
    def __init__(self, string_model_obj, dataset_obj, masker_obj, stats_obj):
        self.string_model_obj = string_model_obj
        self.dataset_obj = dataset_obj
        self.masker_obj = masker_obj
        self.stats_obj = stats_obj
        self.voxel_var_data_map = {}
        self.operation_dataset = {}
        self.total_voxel_ops = None

    def set_up(self):
        self.read_voxel_vars()
        self.set_up_data_for_op()


    def read_voxel_vars(self):
        for var in self.string_model_obj._voxel_vars:
            list_of_files = self.dataset_obj._data_table[var]
            var_data_list = []
            for file in list_of_files:
                var_data_list.append(self.masker_obj.get_data_from_image(file))
            self.voxel_var_data_map[var] = var_data_list

    def set_up_data_for_op(self):
        if not self.voxel_var_data_map:
            print('Voxel Data not read. Cannot continue.')
            return None
        else:
            for var in self.string_model_obj._used_vars:
                if var not in self.string_model_obj._voxel_vars:
                    self.operation_dataset[var] = dict(data=VarDataAccessPointer(self.dataset_obj._data_table[var]),
                                                       shape=self.dataset_obj._data_table[var].shape)
                else:
                    self.operation_dataset[var] = dict(data=VarDataAccessPointer(self.voxel_var_data_map[var]),
                                                       shape=self.voxel_var_data_map[var][0].shape)
        self.total_voxel_ops = max(self.operation_dataset[k]['shape'][1] for k in self.operation_dataset)

    def execute(self):
        pass

class VarDataAccessPointer:
    def __init__(self, var_data):
        self.var_data = var_data

    def get_data(self, pointer_loc):
        return self.var_data[:, pointer_loc % self.var_data.shape[1]]


