from __future__ import division, print_function

import sys
from PySide import QtCore
from PySide import QtGui

## \package freud.qtmanager
#
# Manages the Qt application object and other settings in a unified way
#

## Global access to the app
# For use in later calls to the app
app = None;

## Initialize the QApplication
#
# Any module that uses Qt functionalities should call init_app() on module load. The first such call will initialize
# the QApplication, and subsequent calls will do nothing. initApp() will also detect when ipython initializes the
# application for us
#
def initApp():
    global app;
    
    # first, check if we have already initialized
    if QtCore.QCoreApplication.instance() is None:
        #app = QtGui.QApplication(sys.argv);
        app = QtCore.QCoreApplication(sys.argv);
        app.processEvents();
