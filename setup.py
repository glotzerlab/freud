from __future__ import print_function
import sys
from subprocess import call, Popen
import os
import shutil

def help_text():
    print('This script must be called with `python setup.py install` or `python setup.py install --force`.')
    sys.exit(1)

if len(sys.argv) == 1:
    help_text()
if sys.argv[1] != 'install':
    help_text()
if len(sys.argv) == 3 and sys.argv[2] != '--force':
    help_text()
if len(sys.argv) > 3:
    help_text()

#Check if running on ReadTheDocs
on_rtd = os.environ.get('READTHEDOCS') == 'True'

# Remove the build directory if on ReadTheDocs
if on_rtd and os.path.isdir('build'):
    shutil.rmtree('build')

# Create and enter the build directory
call(['mkdir', '-p', 'build'])
os.chdir('build');

# Run CMake, make install
if on_rtd:
    # ReadTheDocs needs clang since it uses less memory than gcc
    clang_env = os.environ.copy()
    clang_env['CC'] = shutil.which('clang')
    clang_env['CXX'] = shutil.which('clang++')
    try:
        import cython
        cmake_process = Popen(['cmake', '../', '-DENABLE_CYTHON=ON'], env=clang_env)
    except ModuleNotFoundError:
        cmake_process = Popen(['cmake', '../'], env=clang_env)
    finally:
        if cmake_process.wait() != 0:
            print('Errors occurred during CMake.')
            exit(1)
        exit_code = call(['make', 'install', '-j1'])
        exit(exit_code)
else:
    try:
        import cython
        exit_code = call(['cmake', '../', '-DENABLE_CYTHON=ON'])
        if exit_code != 0:
            exit(exit_code)
    except ModuleNotFoundError:
        exit_code = call(['cmake', '../'])
        if exit_code != 0:
            exit(exit_code)
    finally:
        exit_code = call(['make', 'install', '-j4'])
        exit(exit_code)
