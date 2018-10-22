import io
import contextlib
import tempfile
import os
import sys
import platform
import glob
import multiprocessing.pool
import logging
import argparse
import numpy as np
try:
    from setuptools import Extension, setup, distutils
except ImportError:
    # Compatibility with distutils
    import distutils
    from distutils import Extension, setup

logger = logging.getLogger(__name__)


######################################
# Define helper functions for setup.py
######################################

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
tbb_root_str = "--TBB-ROOT"
tbb_include_str = "--TBB-INCLUDE"
tbb_link_str = "--TBB-LINK"

parser = argparse.ArgumentParser(
    description="These are the additional arguments provided by freud "
                "specific build steps. Any arguments not listed in this "
                "usage will be passed on to setuptools.setup.",
    add_help=False)
parser.add_argument(
    '-h',
    '--help',
    action='store_true',
    help='show this help message'
)
parser.add_argument(
    warnings_str,
    action="store_true",
    dest="print_warnings",
    help="Print out all warnings issued during compilation."
)
parser.add_argument(
    coverage_str,
    action="store_true",
    dest="use_coverage",
    help="Compile Cython with coverage"
)
parser.add_argument(
    cython_str,
    action="store_true",
    dest="use_cython",
    help="Compile with Cython instead of using precompiled C++ files"
)
parser.add_argument(
    parallel_str,
    type=int,
    dest="nthreads",
    default=1,
    help="The number of modules to simultaneously compile. Affects both "
         "cythonization and actual compilation of C++ source."
)
parser.add_argument(
    thread_str,
    type=int,
    dest="nthreads_ext",
    default=1,
    help="The number of threads to use to simultaneously compile a single "
         "module. Helpful when constantly recompiling a single module with "
         "many source files, for example during development."
)
parser.add_argument(
    tbb_root_str,
    dest="tbb_root",
    help="The root directory where TBB is installed."
)
parser.add_argument(
    tbb_include_str,
    dest="tbb_include",
    help="The includes directory where the TBB headers are found."
)
parser.add_argument(
    tbb_link_str,
    dest="tbb_link",
    help="The lib directory where TBB shared libraries are found."
)

# Parse known args then rewrite sys.argv for setuptools.setup to use
args, extras = parser.parse_known_args()
if args.nthreads > 1:
    # Make sure number of threads to use gets passed through to setup.
    extras.extend(["-j", str(args.nthreads)])

# Override argparse default helping so that setup can proceed.
if args.help:
    parser.print_help()
    print("\n\nThe subsequent help is for standard setup.py usage.\n\n")
    extras.append('-h')

sys.argv = ['setup.py'] + extras


#######################
# Configure ReadTheDocs
#######################

on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    args.use_cython = True


################################
# Modifications to setup process
################################

# Decide whether or not to use Cython
if args.use_cython:
    try:
        from Cython.Build import cythonize
    except ImportError:
        raise RuntimeError("Could not find cython so cannot build with "
                           "cython. Try again without the --ENABLE-CYTHON "
                           "option.")
    ext = '.pyx'
else:
    ext = '.cpp'

# Decide whether or not to compile with coverage support
if args.use_coverage:
    directives = {'embedsignature': True, 'binding': True, 'linetrace': True}
    macros = [('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')]
else:
    directives = {'embedsignature': True, 'binding': True}
    macros = []


# Enable build parallel compile within modules.
def parallelCCompile(self, sources, output_dir=None, macros=None,
                     include_dirs=None, debug=0, extra_preargs=None,
                     extra_postargs=None, depends=None):
    # source: https://stackoverflow.com/questions/11013851/speeding-up-build-process-with-distutils  # noqa
    # monkey-patch for parallel compilation
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    # The number of parallel threads to attempt
    N = args.nthreads_ext

    def _single_compile(obj):
        try:
            src, ext = build[obj]
        except KeyError:
            return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    # convert to list, imap is evaluated on-demand
    list(multiprocessing.pool.ThreadPool(N).imap(_single_compile, objects))
    return objects


distutils.ccompiler.CCompiler.compile=parallelCCompile


#########################
# Set extension arguments
#########################

def find_tbb(tbb_root=None, tbb_include=None, tbb_link=None):
    """Function to find paths to TBB.

    For finding TBB, the order of precedence is:
        1. The --TBB-INCLUDE/--TBB-LINK passed to setup.py (must specify both).
        2. The --TBB-ROOT passed to setup.py.
        3. The TBB_INCLUDE/TBB_LINK environment variables (must specify both).
        4. The TBB_ROOT environment variable.

    Args:
        tbb_root (str): The location where TBB is installed.
        tbb_include (str): The directory where the TBB headers are found.
        tbb_root (str): The directory where TBB shared libraries are found.

    Returns:
        tuple:
            The tbb include and lib directories passed as args. Returns None
            if nothing was provided.
    """
    err_str = ("You must provide either {} or BOTH {} and {} as command line "
               "arguments. These may also be specified as environment "
               "variables (e.g. {}=/usr/local python setup.py install).")
    err_str = err_str.format(
        tbb_root_str, tbb_include_str, tbb_link_str, tbb_root_str)

    if tbb_root and tbb_include and tbb_link:
        logger.warning("{} is ignored if both {} and {} are specified.".format(
                       tbb_root_str, tbb_include_str, tbb_link_str))
    elif tbb_include and tbb_link:
        pass
    elif tbb_root:
        if tbb_include or tbb_link:
            logger.warning("Using {} and ignoring {}".format(tbb_root_str,
                           tbb_include_str if tbb_include else tbb_link_str))
        tbb_include = os.path.join(tbb_root, 'include')
        tbb_link = os.path.join(tbb_root, 'lib')
    elif tbb_include or tbb_link:
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
            tbb_include = os.path.join(root, 'include')
            tbb_link = os.path.join(root, 'lib')
        elif include or link:
            raise RuntimeError(err_str)

    return tbb_include, tbb_link


tbb_include, tbb_link = find_tbb(args.tbb_root, args.tbb_include,
                                 args.tbb_link)

include_dirs = [
    "extern",
    np.get_include()] + glob.glob(os.path.join('cpp', '*'))

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

if args.use_cython:
    extensions = cythonize(extensions,
                           compiler_directives=directives,
                           nthreads=args.nthreads)


####################################
# Perform setup with error handling
####################################

# Ensure that builds on Mac use correct stdlib.
if platform.system() == 'Darwin':
    os.environ["MACOSX_DEPLOYMENT_TARGET"]= "10.12"

version = '0.11.3'

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
        setup(name='freud-analysis',
              version=version,
              description=desc,
              long_description=readme,
              long_description_content_type='text/markdown',
              url='http://bitbucket.org/glotzer/freud',
              packages=['freud'],
              python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*',
              install_requires=['numpy>=1.10'],
              ext_modules=extensions)
except SystemExit:
    # The errors we're explicitly checking for are whether or not
    # TBB is missing, and whether a parallel compile resulted in a
    # distutils-caused race condition.
    parallel_err = "file not recognized: file truncated"
    tbb_err = "'tbb/tbb.h' file not found"

    err_out = tfile.read().decode('utf-8')
    sys.stderr.write(err_out.encode('utf-8'))
    if tbb_err in err_out:
        sys.stderr.write("\n\033[1mUnable to find tbb. If you have TBB on "
                         "your system, try specifying the location using the "
                         "--TBB-ROOT or the --TBB-INCLUDE/--TBB-LINK "
                         "arguments to setup.py.\033[0m\n")
    elif parallel_err in err_out and args.nthreads > 1:
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
    if args.print_warnings:
        sys.stdout.write("Printing warnings: ")
        sys.stderr.write(tfile.read().decode())
    else:
        out = tfile.read()
        if out:
            sys.stdout.write("Some warnings were emitted during compilations."
                             "Call setup.py with the {} argument "
                             "to see these warnings.\n".format(warnings_str))
