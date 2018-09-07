import setuptools
from distutils.extension import Extension
import numpy as np
import io
import contextlib
import tempfile
import os
import sys
import platform
import glob
import multiprocessing.pool
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
parallel_str = "-j"
thread_str = "--NTHREAD"

if warnings_str in sys.argv:
    sys.argv.remove(warnings_str)
    print_warnings = True
else:
    print_warnings = False

if coverage_str in sys.argv:
    sys.argv.remove(coverage_str)
    directives = {'embedsignature': True, 'binding': True, 'linetrace': True}
    macros = [('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')]
else:
    directives = {'embedsignature': True, 'binding': True}
    macros = []

if cython_str in sys.argv:
    sys.argv.remove(cython_str)
    use_cython = True
    ext = '.pyx'
else:
    use_cython = False
    ext = '.cpp'

if parallel_str in sys.argv:
    # Delete both the option and the associated value from argv
    i = sys.argv.index(parallel_str)
    nthreads = int(sys.argv[i+1])
else:
    nthreads = 1

if thread_str in sys.argv:
    i = sys.argv.index(thread_str)
    sys.argv.remove(thread_str)
    nthreads_ext = int(sys.argv[i])
    del sys.argv[i]

    # Hack for increasing parallelism during builds.
    def parallelCCompile(self, sources, output_dir=None, macros=None,
                         include_dirs=None, debug=0, extra_preargs=None,
                         extra_postargs=None, depends=None):
        # source: https://stackoverflow.com/questions/11013851/speeding-up-build-process-with-distutils  # noqa
        # monkey-patch for parallel compilation
        macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
            output_dir, macros, include_dirs, sources, depends, extra_postargs)
        cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

        # The number of parallel threads to attempt
        N = nthreads_ext

        def _single_compile(obj):
            try:
                src, ext = build[obj]
            except KeyError:
                return
            self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        # convert to list, imap is evaluated on-demand
        list(multiprocessing.pool.ThreadPool(N).imap(_single_compile, objects))
        return objects
    setuptools.distutils.ccompiler.CCompiler.compile=parallelCCompile


#######################
# Configure ReadTheDocs
#######################

on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    use_cython = True
    ext = '.pyx'


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
files.extend(glob.glob(os.path.join('freud', 'util', '*') + ext))
modules = [f.replace(ext, '') for f in files]
modules = [m.replace(os.path.sep, '.') for m in modules]

# Source files required for all modules.
sources_in_all = [
    os.path.join("cpp", "util", "HOOMDMatrix.cc"),
    os.path.join("cpp", "locality", "LinkCell.cc"),
    os.path.join("cpp", "locality", "NearestNeighbors.cc"),
    os.path.join("cpp", "locality", "NeighborList.cc"),
    os.path.join("cpp", "box", "Box.cc")
]

# Any source files required only for specific modules.
# Dict keys should be specified as the module name without
# "freud.", i.e. not the fully qualified name.
extra_module_sources = dict(
    order=[os.path.join("cpp", "cluster", "Cluster.cc")]
)

extensions = []
for f, m in zip(files, modules):
    m_name = m.replace('freud.', '')
    # Use set to avoid doubling up on things in sources_in_all
    sources = set(sources_in_all + [f])
    sources.update(extra_module_sources.get(m_name, []))
    sources.update(glob.glob(os.path.join('cpp', m_name, '*.cc')))

    extensions.append(Extension(m, sources=list(sources), **ext_args))

if use_cython:
    from Cython.Build import cythonize
    extensions = cythonize(extensions,
                           compiler_directives=directives,
                           nthreads=nthreads)


####################################
# Perform setup with error handling
####################################

# Ensure that builds on Mac use correct stdlib.
if platform.system() == 'Darwin':
    os.environ["MACOSX_DEPLOYMENT_TARGET"]= "10.9"

version = '0.10.0'

# Read README for PyPI, fallback to short description if it fails.
desc = 'Perform various analyses of particle simulations.'
try:
    readme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'README.md')
    with open(readme_file) as f:
        readme = f.read()
except ImportError:
    readme = desc

# Using a temporary file as a buffer to hold stderr output allows us
# to parse error messages from the underlying compiler and parse them
# for known errors.
tfile = tempfile.TemporaryFile(mode='w+b')
try:
    with stderr_manager(tfile):
        setuptools.setup(name='freud',
                         version=version,
                         description=desc,
                         long_description=readme,
                         long_description_content_type='text/markdown',
                         url='http://bitbucket.org/glotzer/freud',
                         packages=['freud'],
                         python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*',
                         ext_modules=extensions)
except SystemExit:
    # The errors we're explicitly checking for are whether or not
    # TBB is missing, and whether a parallel compile resulted in a
    # distutils-caused race condition.
    parallel_err = "file not recognized: file truncated"
    tbb_err = "'tbb/tbb.h' file not found"

    err_out = tfile.read().decode()
    sys.stderr.write(err_out)
    if tbb_err in err_out:
        sys.stderr.write("\n\033[1mUnable to find tbb. If you have TBB on "
                         "your system, try specifying the location using the "
                         "--TBB-ROOT or the --TBB-INCLUDE/--TBB-LINK "
                         "arguments to setup.py.\033[0m\n")
    elif parallel_err in err_out and nthreads > 1:
        sys.stderr.write("\n\033[1mYou attempted parallel compilation on a "
                         "Python version where this leads to a race "
                         "in distutils. Please recompile without the -j "
                         "option and try again.\033[0m\n")
    else:
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
