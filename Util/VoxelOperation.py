import numpy, pandas, copy, os, datetime
import ipyparallel as ipp

class VoxelOperation:
    def __init__(self, string_model_obj, dataset_obj, masker_obj, stats_obj):
        self.string_model_obj = string_model_obj
        self.dataset_obj = dataset_obj
        self.masker_obj = masker_obj
        self.stats_obj = stats_obj
        self.voxel_var_data_map = {}
        self.operation_dataset = {}
        self.total_voxel_ops = None
        self.results = None
        self.par_view = ipp.Client()[:]
        self.number_of_engines = len(self.par_view)

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
        self.results = VoxelOpResultsWrapper(self.total_voxel_ops, self.stats_obj)

    def __get_block_from_var_dict(self, block_variable_dict, start_loc):
        data_block = []
        vars = list(block_variable_dict.keys())
        blockSize = block_variable_dict[vars[0]].shape[1]
        for i in range(blockSize):
            try:
                data = {var:block_variable_dict[var][:,i] for var in vars}
            except:
                print('Hi')
            data_block.append(dict(data_block=pandas.DataFrame.from_dict(data), location=start_loc+i,
                              stats_obj=copy.copy(self.stats_obj)))
        return data_block



    def __get_data_block(self, blockSize, block_number):
        finished = False
        if (blockSize*block_number) > self.total_voxel_ops:
            finished = True
            return ([], finished)
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
        return (self.__get_block_from_var_dict(block_var_dict, int(blockSize*block_number)), finished)


    def execute(self):
        slice_count = 200
        blockSize = numpy.ceil(self.total_voxel_ops/slice_count)
        for art_slice in range(slice_count):
            print('Slice - {0}'.format(art_slice), end="")
            sl_st_time = datetime.datetime.now().replace(microsecond=0)
            data_block, finished = self.__get_data_block(blockSize, art_slice)
            if finished:
                print('Analysis complete')
                return 0
            else:
                self.par_view.map(os.chdir, [os.getcwd()]*self.number_of_engines)
                par_results = self.par_view.map_sync(run_par, data_block)
                for i in par_results:
                    self.results.modify_temp_result(i.res, i.loc)
            sl_end_time = datetime.datetime.now().replace(microsecond=0)
            print(' - {0}'.format(sl_end_time - sl_st_time))


def run_par(data_block):
    loc = data_block['location']
    res = data_block['stats_obj'].fit(data_block['data_block'])
    return ParRes(loc, res)

class ParRes:
    def __init__(self, loc, res):
        self.loc = loc
        self.res = res

class VarDataAccessPointer:
    def __init__(self, var_data, outer_shape):
        self.var_data = var_data
        self.outer_shape = outer_shape

    def get_data(self, pointer_loc):
        if self.outer_shape == 1:
            return numpy.tile(self.var_data.as_matrix(), (1, 1)).transpose()
        return self.var_data[:, pointer_loc % self.outer_shape]

    def get_data_block(self, loc_1, loc_2):
        if self.outer_shape == 1:
            blockSize = loc_2-loc_1
            return numpy.tile(self.var_data.as_matrix(), (blockSize, 1)).transpose()
        return self.var_data[:, loc_1 % self.outer_shape: loc_2 % self.outer_shape]

class VoxelOpResultsWrapper:
    def __init__(self, total_voxel_operations, stats_model):
        self.total_ops = total_voxel_operations
        self.stats_model = stats_model
        self.temp_results = [0]*self.total_ops
        self.results = None

    def modify_temp_result(self, value, loc):
        self.temp_results.insert(loc, value)

    def get_results(self):
        if not self.results:
            self.__get_final_voxel_op_result()
        return self.results

    def __get_final_voxel_op_result(self):
        print('Building final results.')
        res = {}
        for var in self.stats_model.model_wise_results_names:
            res[var] = numpy.zeros(self.total_ops)
        model_var_names = self.temp_results[0]['variable_names_in_model_op']
        for var in self.stats_model.var_wise_results_names:
            res[var]={name:numpy.zeros(self.total_ops) for name in model_var_names}

        for i in range(self.total_ops):
            print(i)
            obj = self.temp_results[i]
            for var in self.stats_model.model_wise_results_names:
                print(var)
                res[var][i] = obj[var]
            for var in self.stats_model.var_wise_results_names:
                for name in model_var_names:
                    res[var][name][i] = obj[var][name]
        print('Final results building finished. ')
        self.results = res



