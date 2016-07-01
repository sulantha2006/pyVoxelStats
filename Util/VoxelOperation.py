import numpy, pandas, os, datetime, subprocess, time, numexpr
import ipyparallel as ipp
from pyVoxelStats import pyVoxelStats
from multiprocessing import Pool
from ShareObj import ShareObj
cimport numpy


class VoxelOperation(pyVoxelStats):
    def __init__(self, string_model_obj, dataset_obj, masker_obj, stats_obj):
        pyVoxelStats.__init__(self)
        self.string_model_obj = string_model_obj
        self.dataset_obj = dataset_obj
        self.masker_obj = masker_obj
        self.stats_obj = stats_obj
        self.voxel_var_data_map = {}
        self.operation_dataset = {}
        self.total_voxel_ops = None
        self.results = None
        self.rc = None
        self.par_view = None
        self.number_of_engines = 0

        self.temp_package = None

        self.debug = False


    def set_up_cluster(self, profile_name='default', workers=None, no_start=False):
        print('Setting up cluster .....')
        if not no_start:
            p0 = subprocess.Popen(['ipcluster stop --profile={0}'.format(profile_name)], shell=True)
            time.sleep(5)
            if workers:
                str_args = ['ipcluster start --profile={1} -n {0}'.format(workers, profile_name)]
            else:
                str_args = ['ipcluster start --profile={0}'.format(profile_name)]
            p = subprocess.Popen(str_args, shell=True)
            if profile_name == 'default':
                time.sleep(10)
            else:
                time.sleep(140)
        self.rc = ipp.Client(profile=profile_name)
        self.par_view = self.rc.direct_view(targets='all')
        self.number_of_engines = len(self.par_view)
        print('Connected to {0} workers. '.format(self.number_of_engines))

    def set_up(self):
        print('Setting up voxel operations ... ')
        self.read_voxel_vars()
        self.set_up_data_for_op()

    def read_voxel_vars(self):
        print('File reading ...')
        total_files = 0
        pool = Pool(processes=24)
        for var in self.string_model_obj._voxel_vars:
            list_of_files = self.dataset_obj._data_table[var]
            var_data_list = pool.map(self.masker_obj.get_data_from_image, list_of_files)
            total_files += len(var_data_list)
            self.voxel_var_data_map[var] = numpy.vstack(var_data_list)
        pool.close()
        pool.join()
        print('Done. Total files read : {0}'.format(total_files))

    def __apply_voxel_ops(self, var_name, var_data):
        if not self.string_model_obj.multi_var_ops:
            return var_data
        data_dict = {var_name : var_data}
        for op in self.string_model_obj.multi_var_ops:
            try:
                ret_data = numexpr.evaluate(op, local_dict=data_dict)
                return ret_data
            except:
                pass
        return var_data

    def set_up_data_for_op(self):
        if not self.voxel_var_data_map:
            print('Voxel Data not read. Cannot continue.')
            return None
        else:
            for var in self.string_model_obj._used_vars:
                if var not in self.string_model_obj._voxel_vars:
                    self.operation_dataset[var] = dict(data=VarDataAccessPointer(self.__apply_voxel_ops(var, self.dataset_obj._data_table[var]), 1),
                                                       shape=self.dataset_obj._data_table[var].shape)
                else:
                    self.operation_dataset[var] = dict(
                        data=VarDataAccessPointer(self.__apply_voxel_ops(var, self.voxel_var_data_map[var]), self.voxel_var_data_map[var].shape[1]),
                        shape=self.voxel_var_data_map[var].shape)
        self.total_voxel_ops = max(
            self.operation_dataset[k]['shape'][1] if len(self.operation_dataset[k]['shape']) > 1 else 1 for k in
            self.operation_dataset)
        self.results = VoxelOpResultsWrapper(self.total_voxel_ops, self.stats_obj)


    def __get_block_from_var_dict(self, block_variable_dict, start_loc):
        vars = list(block_variable_dict.keys())
        cdef int blockSize = block_variable_dict[vars[0]].shape[1]
        cdef int n_subs = block_variable_dict[vars[0]].shape[0]
        #formats = ['{0}f8'.format(blockSize)] * len(vars)
        #dtype = dict(names=vars, formats=formats)
        n_formats = ['f8'] * len(vars)
        n_dtype = dict(names=vars, formats=n_formats)
        #data = numpy.zeros(n_subs, dtype=dtype)
        cdef numpy.ndarray n_data = numpy.zeros(n_subs, dtype=n_dtype)

        def f(int k):
            for var in vars:
                n_data[var][:] = block_variable_dict[var][:, [k]].T
            return dict(data_block=pandas.DataFrame(n_data), location=start_loc + k, stats_obj=self.stats_obj)

        data_block = [f(x) for x in range(blockSize)]
        return data_block

    def __get_data_block(self, blockSize, block_number):
        finished = False
        if (blockSize * block_number) > self.total_voxel_ops:
            finished = True
            return ([], finished)
        elif (blockSize * (block_number + 1) > self.total_voxel_ops) & (
                blockSize * block_number < self.total_voxel_ops):
            block_var_dict = {}
            for var in self.operation_dataset:
                block_var_dict[var] = self.operation_dataset[var]['data'].get_data_block(int(blockSize * block_number),
                                                                                         int(self.total_voxel_ops - 1))
        else:
            block_var_dict = {}
            for var in self.operation_dataset:
                block_var_dict[var] = self.operation_dataset[var]['data'].get_data_block(int(blockSize * block_number),
                                                                                         int((blockSize * (
                                                                                         block_number + 1)) - 1))
        return (self.__get_block_from_var_dict(block_var_dict, int(blockSize * block_number)), finished)

    def execute(self):
        print('Execution started ... ')
        ex_st_time = datetime.datetime.now()
        slice_count = int(self.config['VSVoxelOPS']['slice_count'])
        print('Slices - {0}'.format(slice_count))
        blockSize = numpy.ceil(self.total_voxel_ops / slice_count)
        all_results = []
        for art_slice in range(slice_count):
            print('Slice - {0}/{1}'.format(art_slice + 1, slice_count))
            sl_st_time = datetime.datetime.now()
            data_block, finished = self.__get_data_block(blockSize, art_slice)
            d_end_time = datetime.datetime.now()
            if self.debug: print('Block creation time - {0}'.format(d_end_time-sl_st_time))
            if finished:
                print('Analysis complete')
            else:
                self.par_view.map(os.chdir, [os.getcwd()] * self.number_of_engines)
                pr_st_time = datetime.datetime.now()
                par_results = self.par_view.map_sync(run_par, data_block)
                pr_end_time = datetime.datetime.now()
                all_results.extend(par_results)
                ext_end_time = datetime.datetime.now()
                if self.debug: print('Parallel time - {0}'.format(pr_end_time - pr_st_time))
                if self.debug: print('Extend time - {0}'.format(ext_end_time - pr_end_time))
            sl_end_time = datetime.datetime.now()
            print(' - Time: {0} - Remaining time : {1}'.format((sl_end_time - sl_st_time), (sl_end_time - sl_st_time) * (slice_count - art_slice + 1)))
        self.results.temp_results = all_results
        full_end_time = datetime.datetime.now()
        print('Total execution time {0}'.format((full_end_time - ex_st_time)))

    def execute_OLD(self):
        print('Execution started ... ')
        sl_st_time = datetime.datetime.now()
        self.par_view.map(os.chdir, [os.getcwd()] * self.number_of_engines)
        data_block, finished = self.__get_data_block(self.total_voxel_ops, 0)
        print('Parallel execution...')
        par_results = self.par_view.map_sync(run_par, data_block)
        for i in par_results:
            self.results.modify_temp_result(i.res, i.loc)
        sl_end_time = datetime.datetime.now()
        print('Finished : Total execution time {0}'.format((sl_end_time - sl_st_time)))

def ParGetBlock(i):
    data = {var: ShareObj.block_dict['block_variable_dict'][var][:, i] for var in ShareObj.block_dict['var_names']}
    return dict(data_block=pandas.DataFrame.from_dict(data), location=ShareObj.block_dict['start_loc'] + i,
                stats_obj=ShareObj.block_dict['stats_obj'])

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
            return self.var_data.as_matrix()
        return self.var_data[:, pointer_loc % self.outer_shape]

    def get_data_block(self, loc_1, loc_2):
        data = [self.get_data(loc) for loc in range(loc_1, loc_2 + 1)]
        return numpy.vstack(data).T


class VoxelOpResultsWrapper:
    def __init__(self, total_voxel_operations, stats_model):
        self.total_ops = total_voxel_operations
        self.stats_model = stats_model
        self.temp_results = None
        self.results = None

    def modify_temp_result_parallel(self, result):
        self.modify_temp_result(result.res, result.loc)

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
        model_var_names = self.temp_results[0].res['variable_names_in_model_op']
        for var in self.stats_model.var_wise_results_names:
            res[var] = {name: numpy.zeros(self.total_ops) for name in model_var_names}

        for obj in self.temp_results:
            i = obj.loc
            result = obj.res
            for var in self.stats_model.model_wise_results_names:
                res[var][i] = result[var]
            for var in self.stats_model.var_wise_results_names:
                for name in model_var_names:
                    res[var][name][i] = result[var][name]
        print('Final results building finished. ')
        self.temp_results = None
        self.results = res
