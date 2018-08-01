from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
import os

include_dirs = [
    np.get_include(),
    "extern",
    "cpp/box",
    "cpp/bond",
    "cpp/util",
    "cpp/locality",
    "cpp/cluster",
    "cpp/density",
    "cpp/voronoi",
    "cpp/kspace",
    "cpp/order",
    "cpp/environment",
    "cpp/interface",
    "cpp/pmft",
    "cpp/parallel",
    "cpp/registration",
]

libraries = ["tbb"]

compile_args = link_args = ["-std=c++11"]

extensions = [
    # Compile cluster first so that Cluster.cc has been compiled and is available for the order module.
    Extension("freud.order",
        sources=["freud/order.pyx", "cpp/util/HOOMDMatrix.cc", "cpp/order/wigner3j.cc", "cpp/cluster/Cluster.cc"],
	language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        libraries=libraries,
        include_dirs=include_dirs),
    Extension("freud.*",
        sources=["freud/*.pyx", "cpp/util/HOOMDMatrix.cc"],
	language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        libraries=libraries,
        include_dirs=include_dirs),
]

# Gets the version
version = '0.8.2'

# Read README for PyPI, fallback if it fails.
desc = 'Perform various analyses of particle simulations.'
try:
    readme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'README.md')
    with open(readme_file) as f:
        readme = f.read()
except ImportError:
    readme = desc

setup(name = 'freud',
      version=version,
      description=desc,
      long_description=readme,
      long_description_content_type='text/markdown',
      url='http://bitbucket.org/glotzer/freud',
      packages = ['freud'],
      python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*',
      ext_modules = cythonize(extensions),
)
