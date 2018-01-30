from __future__ import print_function
import sys
from subprocess import call
import os

def help_text():
    print("This script must be called with `python setup.py install` or `python setup.py install --force`.")
    sys.exit(1)

if len(sys.argv) == 1:
    help_text()
if sys.argv[1] != 'install':
    help_text()
if len(sys.argv) == 3 and sys.argv[2] != '--force':
    help_text()
if len(sys.argv) > 3:
    help_text()

# Create and enter the build directory
call(["mkdir", "-p", "build"])
os.chdir("build");

# Run CMake, make install
call(["cmake", "../", "-DENABLE_CYTHON=on"])
call(["make", "install", "-j4"])
