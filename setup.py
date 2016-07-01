from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(
    ext_modules = cythonize(["pyVoxelStats.pyx", "pyVoxelStatsLM.pyx", "Util/StatsUtil.pyx", "Util/VoxelOperation.pyx"]), include_dirs=[numpy.get_include()]
)
