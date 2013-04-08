from __future__ import division, print_function

import sys
import logging
logger = logging.getLogger(__name__);

# user code imports qt all the time, gracefully handle unavailable PySide
try:
    from PySide import QtCore, QtGui
except ImportError:
    QtGui = None;
    QtCore = None;
    logger.info('PySide is not available, init_app and run are disabled');

## \package freud.qtmanager
#
# Manages the Qt application object and other settings in a unified way
#

## \internal
# \brief Global access to the app
# For use in later calls to the app
_app = None;

## \internal
# \brief False if we initialized out own application instance
_own_app = False;

## \internal
# \brief Test if inside ipython
def _in_ipython():
    try:
        __IPYTHON__
    except NameError:
        return False;
    else:
        return True;

## Initialize the QApplication
#
# Any user script that uses Qt functionalities should call init_app() on start. The first such call will initialize
# the QApplication, and subsequent calls will do nothing. init_app() will also detect when ipython initializes the
# application. See the documentation for classes and modules in Freud to see whether they require init_app to be
# called before use.
#
# \warning If you fail to call init_app() when it is needed, expect to get error messages like
# `QWidget: Must construct a QApplication before a QPaintDevice` and seg faults. Some freud classes might perform
# checks and provide friendlier errors.
#
def init_app():
    global _app;
    global _own_app;
    
    # It is an error to try and initialize if QtCore or QtGui is not loaded
    if QtCore is None or QtGui is None:
        raise RuntimeError('Cannot init application when PySide is not installed');
    
    # first, check if we have already initialized
    if QtCore.QCoreApplication.instance() is None:
        _app = QtGui.QApplication(sys.argv);
        _own_app = True;
        
        # this version doesn't open a window and would be useful in batch scripts
        #_app = QtCore.QCoreApplication(sys.argv);
        _app.processEvents();
        
        # if we are in ipython, warn the user that they ran ipython without the qt GUI
        if _in_ipython():
            logger.warning('This script is run in ipython, but without --gui=qt');
    else:
        _app = QtCore.QCoreApplication.instance();

## Run the Qt event loop
#
# User scripts should call this when they are done setting up windows and are ready to run the event loop. This method
# is smart enough to return right away if inside a ipython with the qt GUI enabled. When running in a standard
# python shell, or ipython without the qt GUI, this method blocks until the quit/exit signal is sent to qt.
#
def run():
    global _app;
    global _own_app;
    
    if _app is None:
        raise RuntimeError('init_app must be called before run');
    
    if _in_ipython() and not _own_app:
        return;
    else:
        return _app.exec_();

## Test if the application is initialized
# \returns True if init_app() has already been called
#
def is_initialized():
    if _app is None:
        return False;
    else:
        return True;
