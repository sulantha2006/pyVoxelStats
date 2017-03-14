import argparse
import sys

sys.path.append('/home/sulantha/PycharmProjects/pyVoxelStats')
sys.path.append('/home/sulantha/PycharmProjects/pyVoxelStats/Util')
from pyVoxelStats.pyVoxelStats.pyVoxelStatsPower import pyVoxelStatsPower

parser = argparse.ArgumentParser(description='pyVoxelStats Voxelwise Power analysis.')
parser.add_argument('--model', nargs=1, help='Model to evaluate', required=True)
parser.add_argument('--csv', nargs=1, help='CSV file with data', required=True)
parser.add_argument('--mask', nargs=1, help='Mask file', required=True)
parser.add_argument('--vol_var', nargs=1, help='Volumetric variable', required=True)
parser.add_argument('--filter_string', nargs=1, help='Filter string', required=False)
parser.add_argument('--multi_var_operations', nargs='*', help='Multivalue operations list', required=False)
parser.add_argument('--output', nargs=1, help='Output file name', required=True)
parser.add_argument('--cl_profile', nargs=1, help='Cluster profile', required=False, default='default')
parser.add_argument('--cl_workers', nargs=1, help='Cluster profile', required=False, default=None, type=int)
parser.add_argument('--cl_nostart', help='Do not restart cluster', required=False, action='store_true', default=False)
args = parser.parse_args()

model_string = args.model[0]
csv_file = args.csv[0]
mask_file = args.mask[0]
voxel_variables = args.vol_var
subset_string = args.filter_string[0] if args.filter_string else None
multi_variable_operations = args.multi_var_operations
output=args.output[0]

cl_profile = args.cl_profile
cl_workers = args.cl_workers[0] if args.cl_workers else None
cl_nostart = args.cl_nostart

file_type = 'minc'

pw = pyVoxelStatsPower(file_type, model_string, csv_file, mask_file, voxel_variables, subset_string)
if cl_workers:
    pw.set_up_cluster(profile_name=cl_profile, workers=cl_workers, no_start=cl_nostart)
else:
    pw.set_up_cluster(profile_name=cl_profile, no_start=cl_nostart)
results = pw.evaluate()
pw.save(output, 'ss')


