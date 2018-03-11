from __future__ import print_function
import sys
from subprocess import call, Popen
import os
import shutil

def help_text():
    print('This script must be called with `python setup.py install` or '
          '`python setup.py install --force`.')
    sys.exit(1)

if len(sys.argv) == 1:
    help_text()
if sys.argv[1] != 'install':
    help_text()
if len(sys.argv) == 3 and sys.argv[2] != '--force':
    help_text()
if len(sys.argv) > 3:
    help_text()

# Check if running on ReadTheDocs
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    print('Detected ReadTheDocs build environment.')

# Remove the build directory if on ReadTheDocs
if on_rtd and os.path.isdir('build'):
    shutil.rmtree('build')

# Create and enter the build directory
call(['mkdir', '-p', 'build'])
os.chdir('build')

cmake_command = ['cmake', '../']

try:
    import cython
except ModuleNotFoundError:
    print('Cython not found. Using existing Cython cpp files.')
else:
    print('Detected Cython. Rebuilding Cython cpp files.')
    cmake_command.append('-DENABLE_CYTHON=ON')

# Run CMake, make install
if on_rtd:
    # ReadTheDocs needs clang since it uses less memory than gcc
    clang_env = os.environ.copy()
    clang_env['CC'] = shutil.which('clang')
    clang_env['CXX'] = shutil.which('clang++')
    print('Calling', ' '.join(cmake_command))
    cmake_process = Popen(cmake_command, env=clang_env)
    if cmake_process.wait() != 0:
        print('Errors occurred during CMake.')
        exit(1)
    exit_code = call(['make', 'install', '-j1'])
    exit(exit_code)
else:
    print('Calling', ' '.join(cmake_command))
    exit_code = call(cmake_command)
    if exit_code != 0:
        exit(exit_code)
    exit_code = call(['make', 'install', '-j4'])
    exit(exit_code)
