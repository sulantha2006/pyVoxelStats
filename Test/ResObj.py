import numpy
class ResObj:
    def __init__(self, index, result):
        self.loc = index ### This is the location, where the values should go in the final result dictionary
        self.res = result ### This is a dictionary that has values for this location.

        self.loc = 2
        self.res = {'value1':5.4, 'value2':2.3, 'valuen':{'sub_value1':4.5, 'sub_value2':3.4, 'sub_value3':7.6}}


def make_final_result(list_of_results):
    no_sub_result_variables = ['value1', 'value2']
    sub_result_variables = ['valuen']
    sub_value_variables = ['sub_value1', 'sub_value3', 'sub_value3']

    final_result = {}
    num_of_results = len(list_of_results)
    for var in no_sub_result_variables:
        final_result[var] = numpy.zeros(num_of_results)
    for var in sub_result_variables:
        final_result[var] = {sub_var:numpy.zeros(num_of_results) for sub_var in sub_value_variables}

    for obj in list_of_results:
        i = obj.loc
        result = obj.res
        for var in no_sub_result_variables:
            final_result[var][i] = result[var]
            for var in sub_result_variables:
                for name in sub_value_variables:
                    try:
                        final_result[var][name][i] = result[var][name]
                    except KeyError as e:
                        ##TODO Add a debug check
                        pass


def value_to_record(value):
    model_wise_results_names = ['value1', 'value2']
    var_wise_results_names = ['valuen']
    model_var_names = ['sub_value1', 'sub_value3', 'sub_value3']
    tup = ()
    if model_wise_results_names:
        tup = tup + tuple(value.res[var] for var in model_wise_results_names)
    if var_wise_results_names:
        tup = tup + tuple(
            tuple(value.res[var][name] for name in model_var_names) for var in var_wise_results_names)
    return tup


def make_result(temp_results):
    model_wise_results_names = ['value1', 'value2']
    var_wise_results_names = ['valuen']
    model_var_names = ['sub_value1', 'sub_value2', 'sub_value3']
    temp_results.sort(key=lambda x: x.loc, reverse=False)
    dtype = []
    if model_wise_results_names:
        for var in model_wise_results_names:
            dtype.append(("{0}".format(var), "f8"))
    if var_wise_results_names:
        for var in var_wise_results_names:
            dtype.append(("{0}".format(var), [("{0}".format(name), "f8") for name in model_var_names]))

    arr = numpy.fromiter(map(value_to_record, temp_results), dtype=dtype, count=len(temp_results))
    return arr

ress = [ResObj(i, {'value1':5.4, 'value2':2.3, 'valuen':{'sub_value1':4.5, 'sub_value2':3.4, 'sub_value3':7.6}}) for i in range(20000000)]
a = make_result(ress)
print(a[0]['value1'])