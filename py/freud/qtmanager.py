from __future__ import division, print_function

import sys
import logging
from PySide import QtCore
from PySide import QtGui

## \package freud.qtmanager
#
# Manages the Qt application object and other settings in a unified way
#

## Global access to the app
# For use in later calls to the app
app = None;

## \internal
# \brief False if we initialized out own application instance
_own_app = False;

## \internal
# \brief Test if inside ipython
def _inIPython():
    try:
        __IPYTHON__
    except NameError:
        return False;
    else:
        return True;

## Initialize the QApplication
#
# Any module that uses Qt functionalities should call initApp() on module load. The first such call will initialize
# the QApplication, and subsequent calls will do nothing. initApp() will also detect when ipython initializes the
# application for us.
#
def initApp():
    global app;
    global _own_app;
    
    # first, check if we have already initialized
    if QtCore.QCoreApplication.instance() is None:
        app = QtGui.QApplication(sys.argv);
        _own_app = True;
        
        # this version doesn't open a window and would be useful in batch scripts
        #app = QtCore.QCoreApplication(sys.argv);
        app.processEvents();
        
        # if we are in ipython, warn the user that they ran ipython without the qt GUI
        if _inIPython():
            logging.warning('This script is run in ipython, but without --gui=qt');
    else:
        app = QtCore.QCoreApplication.instance();

## Run the Qt event loop
#
# User scripts should call this when they are done setting up windows and are ready to run the event loop. This method
# is smart enough to return right away if inside a ipython with the qt GUI enabled. When running in a standard
# python shell, or ipython without the qt GUI, this method blocks until the quit/exit signal is sent to qt.
#
def runEventLoop():
    global _own_app;
    
    if _inIPython() and not _own_app:
        return;
    else:
        app.exec_();
