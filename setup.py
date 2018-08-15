from setuptools import setup
from distutils.extension import Extension
import numpy as np
import io
import contextlib
import tempfile
import os
import sys
import platform
import glob

import logging

logger = logging.getLogger(__name__)

######################################
# Define helper functions for setup.py
######################################


def find_tbb(argv):
    """Function to find paths to TBB.

    For finding TBB, the order of precedence is the
    following:
        1. The --TBB-INCLUDE/--TBB-LINK passed to setup.py (must specify both).
        2. The --TBB-ROOT passed to setup.py.
        3. The TBB_INCLUDE/TBB_LINK environment variables (must specify both).
        4. The TBB_ROOT environment variable.

    Args:
        argv (str): The value of sys.argv (arguments to the file).

    Returns:
        tuple:
            The tbb include and lib directories passed as args. Returns None
            if nothing was provided.
    """
    valid_tbb_opts = set(['--TBB-ROOT', '--TBB-INCLUDE', '--TBB-LINK'])
    provided_opts = valid_tbb_opts.intersection(sys.argv)
    err_str = ("You must provide either '--TBB-ROOT' or BOTH '--TBB-INCLUDE' "
               "and '--TBB-LINK' as command line arguments. These may also be "
               "specified as environment variables "
               " (e.g. TBB_ROOT=/usr/local python setup.py install).")

    tbb_include = tbb_link = None
    if len(provided_opts) == 3:
        logger.warning("--TBB-ROOT is ignored if both --TBB-INCLUDE and "
                       "--TBB-LINK are specified.")
        tbb_include = sys.argv[sys.argv.index('--TBB-INCLUDE') + 1]
        tbb_link = sys.argv[sys.argv.index('--TBB-LINK') + 1]
    elif len(provided_opts) == 2:
        if '--TBB-ROOT' in provided_opts:
            logger.warning("Using --TBB-ROOT and ignoring {}".format(
                           provided_opts.difference(set(["--TBB-ROOT"]))))
            root = sys.argv[sys.argv.index('--TBB-ROOT') + 1]
            tbb_include = os.path.join(root, 'include')
            tbb_link = os.path.join(root, 'lib')
        else:
            tbb_include = sys.argv[sys.argv.index('--TBB-INCLUDE') + 1]
            tbb_link = sys.argv[sys.argv.index('--TBB-LINK') + 1]
    elif '--TBB-ROOT' in provided_opts:
        root = sys.argv[sys.argv.index('--TBB-ROOT') + 1]
        tbb_include = os.path.join(root, 'include')
        tbb_link = os.path.join(root, 'lib')
    elif '--TBB-LINK' in provided_opts:
        tbb_link = sys.argv[sys.argv.index('--TBB-LINK') + 1]
    elif '--TBB-INCLUDE' in provided_opts:
        tbb_include = sys.argv[sys.argv.index('--TBB-INCLUDE') + 1]
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

    # Delete the options and their values.
    for arg in provided_opts:
        i = sys.argv.index(arg)
        sys.argv.remove(arg)
        del sys.argv[i]

    return tbb_include, tbb_link


@contextlib.contextmanager
def stderr_manager(f):
    """Context manager for capturing C-level standard error in a file.

    Capturing C++ level output cannot be done by simply repointing sys.stdout,
    we need to repoint the underlying file descriptors.

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
        if sys.version_info > (3, 0):
            sys.stderr = io.TextIOWrapper(os.fdopen(original_stderr_fd, 'wb'))
        else:
            sys.stderr = os.fdopen(original_stderr_fd, 'wb')

    try:
        _redirect_stderr(f.fileno(), stderr_fd)
        yield
    finally:
        _redirect_stderr(saved_stderr_fd, stderr_fd)
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)


############
# Parse args
############

warnings_str = "--PRINT-WARNINGS"
coverage_str = "--COVERAGE"
cython_str = "--ENABLE-CYTHON"

if warnings_str in sys.argv:
    sys.argv.remove(warnings_str)
    print_warnings = True
else:
    print_warnings = False

if coverage_str in sys.argv:
    sys.argv.remove(coverage_str)
    directives = {'linetrace': True}
    macros = [('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')]
else:
    directives = {}
    macros = []

if cython_str in sys.argv:
    sys.argv.remove(cython_str)
    use_cython = True
    ext = '.pyx'
else:
    use_cython = False
    ext = '.cpp'

#########################
# Set extension arguments
#########################

tbb_include, tbb_link = find_tbb(sys.argv)

include_dirs = [
    np.get_include(),
    "extern",
]
include_dirs.extend(glob.glob(os.path.join('cpp', '*')))

if tbb_include:
    include_dirs.append(tbb_include)

# Add sys.prefix to include path for finding conda tbb
include_dirs.append(os.path.join(sys.prefix, 'include'))

libraries = ["tbb"]
library_dirs = [tbb_link] if tbb_link else []

compile_args = link_args = ["-std=c++11"]

ext_args = dict(
    language="c++",
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    libraries=libraries,
    library_dirs=library_dirs,
    include_dirs=include_dirs,
    define_macros=macros
)

###################
# Set up extensions
###################


# Need to find files manually; cythonize accepts glob syntax, but basic
# extension modules with C++ do not
files = glob.glob(os.path.join('freud', '*') + ext)
files.remove(os.path.join('freud', 'order' + ext))  # Is compiled separately
modules = [f.replace(ext, '') for f in files]
modules = [m.replace(os.path.sep, '.') for m in modules]

# Compile order separately since it requires that Cluster.cc and a few
# other things be compiled in addition to the main source.
s = [os.path.join("freud", "order" + ext),
     os.path.join("cpp", "util", "HOOMDMatrix.cc"),
     os.path.join("cpp", "cluster", "Cluster.cc")]
# Ensure changes in CC files are captured
s.extend(glob.glob(os.path.join("cpp", "order", "*.cc")))
extensions = [Extension("freud.order", sources=s, **ext_args)]

for f, m in zip(files, modules):
    s = [f, os.path.join("cpp", "util", "HOOMDMatrix.cc")]
    # Ensure changes in CC files are captured
    s.extend(glob.glob(os.path.join("cpp", m.replace('freud.', ''), "*.cc")))
    extensions.append(Extension(m, sources=s, **ext_args))

if use_cython:
    from Cython.Build import cythonize
    extensions = cythonize(extensions, compiler_directives=directives)

####################################
# Perform set up with error handling
####################################

# Ensure that builds on Mac use correct stdlib.
if platform.system() == 'Darwin':
    os.environ["MACOSX_DEPLOYMENT_TARGET"]= "10.9"

version = '0.9.0'

# Read README for PyPI, fallback to short description if it fails.
desc = 'Perform various analyses of particle simulations.'
try:
    readme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'README.md')
    with open(readme_file) as f:
        readme = f.read()
except ImportError:
    readme = desc

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
              ext_modules=extensions)
except SystemExit:
    # For now, the only error we're explicitly checking for is whether or not
    # TBB is missing
    err_str = "tbb/tbb.h"
    err_out = tfile.read().decode()
    if err_str in err_out:
        sys.stderr.write("Unable to find tbb. If you have TBB on your "
                         "system, try specifying the location using the "
                         "--TBB-ROOT or the --TBB-INCLUDE/--TBB-LINK "
                         "arguments to setup.py.\n")
    else:
        sys.stderr.write(err_out)
        raise
except: # noqa
    sys.stderr.write(tfile.read().decode())
    raise
else:
    if print_warnings:
        sys.stdout.write("Printing warnings: ")
        sys.stderr.write(tfile.read().decode())
    else:
        out = tfile.read()
        if out:
            sys.stdout.write("Some warnings were emitted during compilations."
                             "Call setup.py with the {} argument "
                             "to see these warnings.\n".format(warnings_str))
