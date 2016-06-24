import numpy

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
            self.voxel_var_data_map[var] = numpy.vstack(var_data_list)

    def set_up_data_for_op(self):
        if not self.voxel_var_data_map:
            print('Voxel Data not read. Cannot continue.')
            return None
        else:
            for var in self.string_model_obj._used_vars:
                if var not in self.string_model_obj._voxel_vars:
                    self.operation_dataset[var] = dict(data=VarDataAccessPointer(self.dataset_obj._data_table[var], 1),
                                                       shape=self.dataset_obj._data_table[var].shape)
                else:
                    self.operation_dataset[var] = dict(data=VarDataAccessPointer(self.voxel_var_data_map[var], self.voxel_var_data_map[var].shape[1]),
                                                       shape=self.voxel_var_data_map[var].shape)
        self.total_voxel_ops = max(self.operation_dataset[k]['shape'][1] if len(self.operation_dataset[k]['shape']) > 1 else 1 for k in self.operation_dataset)

    def __get_block_from_var_dict(self, block_variable_dict):
        pass

    def __get_data_block(self, blockSize, block_number):
        data_block = []
        finished = False
        if (blockSize*block_number) > self.total_voxel_ops:
            finished = True
            return data_block, finished
        elif (blockSize*(block_number+1) > self.total_voxel_ops) & (blockSize*block_number < self.total_voxel_ops):
            block_var_dict = {}
            for var in self.operation_dataset:
                block_var_dict[var] = self.operation_dataset[var]['data'].get_data_block(int(blockSize*block_number),
                                                                                 int(self.total_voxel_ops))
        else:
            block_var_dict = {}
            for var in self.operation_dataset:
                block_var_dict[var] = self.operation_dataset[var]['data'].get_data_block(int(blockSize*block_number),
                                                                                         int((blockSize*(block_number + 1)) - 1))
        return (self.__get_block_from_var_dict(block_var_dict), finished)


    def execute(self):
        slice_count = 200
        blockSize = numpy.ceil(self.total_voxel_ops/slice_count)
        for art_slice in range(slice_count):
            data_block, finished = self.__get_data_block(blockSize, art_slice)


class VarDataAccessPointer:
    def __init__(self, var_data, outer_shape):
        self.var_data = var_data
        self.outer_shape = outer_shape

    def get_data(self, pointer_loc):
        return self.var_data[:, pointer_loc % self.outer_shape]

    def get_data_block(self, loc_1, loc_2):
        return self.var_data[:, loc_1 % self.outer_shape: loc_2 % self.outer_shape]


