from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
import io
import sys
import contextlib
import tempfile
import os
import platform

import logging

logger = logging.getLogger(__name__)


def find_tbb(argv):
    """Function to find paths to TBB.

    For finding TBB, the order of precedence is the
    following:
        1. The -TBB_INCLUDE/-TBB_LINK passed to setup.py (must specify both).
        2. The -TBB_ROOT passed to setup.py.
        3. The TBB_INCLUDE/TBB_LINK environment variables (must specify both).
        4. The TBB_ROOT environment variable.

    Args:
        argv (str): The value of sys.argv (arguments to the file).

    Returns:
        tuple:
            The tbb include and lib directories passed as args. Returns None
            if nothing was provided.
    """
    valid_tbb_opts = set(['-TBB_ROOT', '-TBB_INCLUDE', '-TBB_LINK'])
    provided_opts = valid_tbb_opts.intersection(sys.argv)
    err_str = ("You must provide either '-TBB_ROOT' or BOTH '-TBB_INCLUDE' "
               "and '-TBB_LINK' as command line arguments. These may also be "
               "specified as environment variables "
               " (e.g. TBB_ROOT=/usr/local python setup.py install).")

    tbb_include = tbb_link = None
    if len(provided_opts) == 3:
        logger.warning("-TBB_ROOT is ignored if both -TBB_INCLUDE and "
                       "-TBB_LINK are specified.")
        tbb_include = sys.argv[sys.argv.index('-TBB_INCLUDE') + 1]
        tbb_link = sys.argv[sys.argv.index('-TBB_LINK') + 1]
    elif len(provided_opts) == 2:
        if '-TBB_ROOT' in provided_opts:
            logger.warning("Using -TBB_ROOT and ignoring {}".format(
                           provided_opts.difference(set(["-TBB_ROOT"]))))
            root = sys.argv[sys.argv.index('-TBB_ROOT') + 1]
            tbb_include = os.path.join(root, 'include')
            tbb_link = os.path.join(root, 'lib')
        else:
            tbb_include = sys.argv[sys.argv.index('-TBB_INCLUDE') + 1]
            tbb_link = sys.argv[sys.argv.index('-TBB_LINK') + 1]
    elif '-TBB_ROOT' in provided_opts:
        root = sys.argv[sys.argv.index('-TBB_ROOT') + 1]
        tbb_include = os.path.join(root, 'include')
        tbb_link = os.path.join(root, 'lib')
    elif '-TBB_LINK' in provided_opts:
        tbb_link = sys.argv[sys.argv.index('-TBB_LINK') + 1]
    elif '-TBB_INCLUDE' in provided_opts:
        tbb_include = sys.argv[sys.argv.index('-TBB_INCLUDE') + 1]
    elif provided_opts:
            raise RuntimeError(err_str)
    else:
        include = os.getenv("TBB_INCLUDE")
        link = os.getenv("TBB_LINK")
        root = os.getenv("TBB_ROOT")
        if link and include:
            if root:
                logger.warning("TBB_ROOT is ignored if both TBB_INCLUDE and "
                               "TBB_LINK are defined.")
            tbb_include = include
            tbb_link = link
        elif root:
            if link or include:
                logger.warning("Using environment variable TBB_ROOT and "
                               "ignoring {}".format("TBB_LINK" if link
                                                    else "TBB_INCLUDE"))
        else:
            tbb_include = include
            tbb_link = link

    return tbb_include, tbb_link


# Ensure that builds on Mac use correct stdlib.
if platform.system() == 'Darwin':
        os.environ["MACOSX_DEPLOYMENT_TARGET"]= "10.9"

tbb_include, tbb_link = find_tbb(sys.argv)

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

if tbb_include:
    include_dirs.append(tbb_include)

libraries = ["tbb"]
library_dirs = [tbb_link] if tbb_link else []

compile_args = link_args = ["-std=c++11"]

extensions = [
    # Compile cluster first so that Cluster.cc has been compiled and is
    # available for the order module.
    Extension("freud.order",
              sources=["freud/order.pyx",
                       "cpp/util/HOOMDMatrix.cc",
                       "cpp/order/wigner3j.cc",
                       "cpp/cluster/Cluster.cc"],
              language="c++",
              extra_compile_args=compile_args,
              extra_link_args=link_args,
              libraries=libraries,
              library_dirs=library_dirs,
              include_dirs=include_dirs),
    Extension("freud.*",
              sources=["freud/*.pyx", "cpp/util/HOOMDMatrix.cc"],
              language="c++",
              extra_compile_args=compile_args,
              extra_link_args=link_args,
              libraries=libraries,
              library_dirs=library_dirs,
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


@contextlib.contextmanager
def stderr_manager(f):
    """Context manager for capturing C-level standard error in a file.

    Adapted from
    http://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python

    Args:
        f (file-like object): File to which to write output.
    """

    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)

    def _redirect_stderr(to_fd, original_stderr_fd):
        """Redirect stderr to the given file descriptor."""
        sys.stderr.close()
        os.dup2(to_fd, original_stderr_fd)
        sys.stderr = io.TextIOWrapper(os.fdopen(original_stderr_fd, 'wb'))

    try:
        _redirect_stderr(f.fileno(), stderr_fd)
        yield
    finally:
        _redirect_stderr(saved_stderr_fd, stderr_fd)
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)


tfile = tempfile.TemporaryFile(mode='w+b')
try:
    with stderr_manager(tfile):
        setup(name='freud',
              version=version,
              description=desc,
              long_description=readme,
              long_description_content_type='text/markdown',
              url='http://bitbucket.org/glotzer/freud',
              packages=['freud'],
              python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*',
              ext_modules=cythonize(extensions))
except SystemExit:
    err_str = "tbb/tbb.h"
    if err_str in tfile.read().decode():
        raise RuntimeError("Unable to find tbb. If you have TBB on your "
                           "system, try specifying the location using the "
                           "-TBB_ROOT or the -TBB_INCLUDE/-TBB_LINK "
                           "arguments to setup.py.")
    else:
        raise
except: # noqa
    sys.stderr.write(tfile.read().decode())
    raise
else:
    sys.stderr.write(tfile.read().decode())
