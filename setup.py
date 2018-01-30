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
    cmake_process = Popen(['cmake', '../'], env=clang_env)
    if cmake_process.wait() != 0:
        print('Errors occurred during CMake.')
        exit(1)
    call(['make', 'install', '-j1'])
else:
    call(['cmake', '../'])
    call(['make', 'install', '-j4'])
