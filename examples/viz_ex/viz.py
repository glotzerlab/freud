import argparse
import sys
import os
import json
import itertools
import weakref
import cProfile
import pstats
import time
import re

import numpy

from freud import viz, qt, trajectory, density
from freud.shape import Polygon
from freud.viz import Outline

try:
    import OpenGL
    from PySide import QtCore, QtGui, QtOpenGL
except ImportError:
    logger.warning('Either PySide or pyopengl is not available, aborting rt initialization');
    raise ImportWarning('PySide or pyopengl not available')

import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4']='PySide'
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from util import loadShape
from util import getInfo

from vizUtil import calcRDF
from vizUtil import vizOptions
from vizUtil import vizParams
from vizUtil import trajGroup

def mkCapital(myString):
    return "{}{}".format(myString[0].upper(), myString[1:])

# Taken directly from Josh's code
class MatplotlibCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100, tight_layout=False, *args, **kwargs):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='w', tight_layout=tight_layout)
        FigureCanvas.__init__(self, self.fig, *args, **kwargs)
        self.axes = self.fig.add_subplot(111)
        self.axes.hold(False)
        # self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding);
        self.updateGeometry();


class Analysis(QtCore.QObject):
    # I need to base off the index not the other parameters
    def __init__(self,
                 *args,
                 **kwargs):
        super(Analysis, self).__init__(*args, **kwargs)
        # read in parameters
        self.params = vizParams.Params("params.json", "analysis.json")

        # set options
        self.options = vizOptions.Options(phiList=self.params.phiList,
                                          runList=self.params.runList,
                                          rdf=self.params.useRDF,
                                          ocf=self.params.useOCF,
                                          bocf=self.params.useOCF,
                                          angleSymmetry=self.params.angleSymmetry,
                                          rMax=self.params.rMax,
                                          ocfrMax=self.params.ocfrMax,
                                          kValue=self.params.k,
                                          outline=self.params.outline)
        # create connections for various GUI boxes
        self.options.phiComboBox.currentIndexChanged.connect(self.loadRun)
        self.options.runComboBox.currentIndexChanged.connect(self.loadRun)
        self.options.coloringComboBox.currentIndexChanged.connect(self.setColoringMode)
        self.options.angleLineEdit.returnPressed.connect(self.setAngleSymmetry)
        self.options.outlineLineEdit.returnPressed.connect(self.setOutline)
        self.options.kLineEdit.returnPressed.connect(self.setKValue)
        self.options.rdfLineEdit.returnPressed.connect(self.setRMax)
        self.options.ocfLineEdit.returnPressed.connect(self.setOCFRMax)
        self.options.rdfCheckBox.stateChanged.connect(self.setRDF)
        self.options.ocfCheckBox.stateChanged.connect(self.setOCF)
        self.options.bocfCheckBox.stateChanged.connect(self.setBOCF)
        self.options.forceReloadPushButton.released.connect(self.loadRun)
        self.options.recordMoviePushButton.released.connect(self.recordMovie)
        self.options.dumpPlotsPushButton.released.connect(self.dumpPlots)
        self.options.widthLineEdit.returnPressed.connect(self.setWidth)
        self.options.heightLineEdit.returnPressed.connect(self.setHeight)

        # initialize the plot widgets
        # plot for the rdf
        self.rdfPlot = MatplotlibCanvas()
        # plot for the cum rdf
        self.cumRdfPlot = MatplotlibCanvas()
        # plot for the OCF
        self.ocfPlot = MatplotlibCanvas()
        # plot for the BOCF
        self.bocfPlot = MatplotlibCanvas()
        # plot for tetratic order parameter
        self.psiPlot = MatplotlibCanvas(tight_layout=True)
        self.psiPlot.axes.set_axis_off()
        self.psiPlot.axes.get_xaxis().set_visible(False)
        self.psiPlot.axes.get_yaxis().set_visible(False)
        # plot for \psi(t)
        self.psiTimePlot = MatplotlibCanvas();

        # timer for property computation
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.analyzeNextFrame)

        # create the dock widgets
        self.progressBar = QtGui.QProgressBar()
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(0)
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(True)
        self.progressDock = QtGui.QDockWidget("Progress")
        self.progressDock.setWidget(self.progressBar)
        self.progressDock.setObjectName('Progress')

        self.rdfDock = QtGui.QDockWidget("RDF")
        self.rdfDock.setWidget(self.rdfPlot)
        self.rdfDock.setObjectName('RDF')

        self.cumRdfDock = QtGui.QDockWidget("CRDF")
        self.cumRdfDock.setWidget(self.cumRdfPlot)
        self.cumRdfDock.setObjectName('CRDF')

        self.ocfDock = QtGui.QDockWidget("ocf")
        self.ocfDock.setWidget(self.ocfPlot)
        self.ocfDock.setObjectName('ocf')

        self.bocfDock = QtGui.QDockWidget("bocf")
        self.bocfDock.setWidget(self.bocfPlot)
        self.bocfDock.setObjectName('bocf')

        self.psiDock = QtGui.QDockWidget("Psi histogram")
        self.psiDock.setWidget(self.psiPlot)
        self.psiDock.setObjectName('psiH')

        self.psiTimeDock = QtGui.QDockWidget("Psi vs. step")
        self.psiTimeDock.setWidget(self.psiTimePlot)
        self.psiTimeDock.setObjectName('psiT')

        self.optionsDock = QtGui.QDockWidget("Options")
        self.optionsDock.setWidget(self.options)
        self.optionsDock.setObjectName('options')

        # initialize the trajectory viewer
        self.viewer = viz.rt.TrajectoryViewer(dock_widgets=[self.progressDock,
                                                            self.optionsDock,
                                                            self.rdfDock,
                                                            self.cumRdfDock,
                                                            self.ocfDock,
                                                            self.bocfDock,
                                                            self.psiDock,
                                                            self.psiTimeDock],
                                                            immediate=True)
        self.viewer.frame_display.connect(self.analyzeFrame)
        self.viewer.show()

        self.currentFrame = 0
        self.loadRun()

    @QtCore.Slot()
    def loadRun(self):
        # load the run
        startTime = time.time()
        phi = float(self.options.phiComboBox.currentText())
        run = int(self.options.runComboBox.currentText())
        self.phi = phi
        self.run = run
        # get the path to the data
        # code your own function
        pth = getInfo.returnPath(phi=phi,run=run)
        # load the traj
        # could be offloaded to another function
        self.traj = trajectory.TrajectoryXMLDCD("{}/{}Init.xml".format(pth, self.params.fileName),
                                                "{}/{}.dcd".format(pth, self.params.fileName))
        self.numFrames = len(self.traj)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(self.numFrames-1)
        self.progressBar.setValue(0)
        try:
            dataDict = json.load(open("{}/{}Shape.json".format(pth, self.params.fileName)))
        shapeData = loadShape.shape(dataDict)
        if shapeData.nTypes == 1:
            self.nTypes = 1
            self.shapes = numpy.array([shapeData.verts["A"]])
        if shapeData.nTypes == 2:
            self.nTypes = 2
            self.shapes = numpy.array([shapeData.verts["A"], shapeData.verts["B"]])
        self.polyA = shapeData.area["A"]
        if shapeData.nTypes == 2:
            self.polyA += shapeData.area["B"]
            self.polyA /= 2.0

        # create the TrajGroup
        self.group = trajGroup.TrajGroup(trajectory=self.traj,analysis=self)
        self.setColoringMode()
        self.setAngleSymmetry()
        self.setOutline()
        self.setKValue()
        self.setRDF()
        self.setOCF()
        self.setBOCF()
        # self.setPairing()
        # set the camera; these settings for my data, could be adjusted (height)
        cam = viz.base.Camera(position=(0,0,1),
                              look_at=(0,0,0),
                              up=(0,1,0),
                              aspect=1,
                              height=numpy.sqrt(self.params.numParticles)/0.914)
        scene = viz.base.Scene(camera=cam, groups=[self.group])

        # clear axes and set to idle
        self.rdfPlot.axes.clear();
        self.rdfPlot.draw_idle();
        self.cumRdfPlot.axes.clear();
        self.cumRdfPlot.draw_idle();
        self.ocfPlot.axes.clear();
        self.ocfPlot.draw_idle();
        self.bocfPlot.axes.clear();
        self.bocfPlot.draw_idle();
        self.psiPlot.axes.clear();
        self.psiPlot.draw_idle();
        self.psiTimePlot.axes.clear();
        self.psiTimePlot.draw_idle();

        self.viewer.setScene(scene)
        loadTime = time.time() - startTime
        print("loadTime = {}".format(loadTime))
        # start recomputing the data at the next idle slot
        QtCore.QTimer.singleShot(0, self.recomputeData)

    @QtCore.Slot()
    def recomputeData(self):
        # called on a new data load; creates new objects and recomputes requested data
        box = self.traj[0].box
        self.orderParameter = calcPsi.OrderParameter(self.traj,
                                                     self.params.kMax,
                                                     self.params.k)

        if self.params.useRDF == True:
            self.rdf = calcRDF.RDF(self.traj,
                                   self.params.rMax,
                                   self.params.dr,
                                   df=10)
        else:
            self.rdf = None
        if self.params.useOCF == True:
            self.ocf = calcRDF.OCF(self.traj,
                                   self.params.ocfrMax,
                                   self.params.ocfdr,
                                   df=10)
            self.bocf = calcRDF.BOCF(self.traj,
                                     self.params.ocfrMax,
                                     self.params.kMax,
                                     self.params.ocfdr,
                                     df=10,
                                     isSplit=split)
        else:
            self.ocf = None
            self.bocf = None

        # start back at frame 0 and start the timer
        self.currentFrame = 0;
        self.timer.start();

    def closeEvent(self, event):
        # required for viz
        QtGui.QWidget.closeEvent(self, event);

    @QtCore.Slot()
    def analyzeFrame(self, frame):
        # plot and analyze those things that are required each frame
        if ((self.psiTimeDock.isVisible())):
            self.orderParameter.plotPsi(frame=frame, plot="time", canvas=self.psiTimePlot)
        if ((self.psiDock.isVisible())):
            self.orderParameter.plotPsi(frame=frame, plot="psi", canvas=self.psiPlot)

        self.psiPlot.draw_idle()
        self.psiTimePlot.draw_idle()

    @QtCore.Slot()
    def plotSummary(self):
        # plot the data that comes from final frames
        # plot N(r) if it exists
        startTime = time.time()
        if ((self.rdfDock.isVisible()) and self.rdf is not None):

            self.rdf.plotRDF(plot="rdf", canvas=self.rdfPlot)

            self.rdf.plotRDF(plot="cum", canvas=self.cumRdfPlot)

        if ((self.ocfDock.isVisible()) and self.ocf is not None):

            self.ocf.plotOCF(canvas=self.ocfPlot)

        if ((self.bocfDock.isVisible()) and self.bocf is not None):

            self.bocf.plotBOCF(canvas=self.bocfPlot)

        if ((self.psiTimeDock.isVisible())):

            self.orderParameter.plotPsi(canvas=self.psiTimePlot, plot="time")

        self.rdfPlot.draw_idle()
        self.cumRdfPlot.draw_idle()
        self.ocfPlot.draw_idle()
        self.bocfPlot.draw_idle()
        plotTime = time.time() - startTime
        print("plotTime = {}".format(plotTime))

    @QtCore.Slot()
    # analyze the next frame
    # could probably call a function but it's fine here
    def analyzeNextFrame(self):
        self.progressBar.setValue(self.currentFrame)
        print('Computing frame', self.currentFrame)
        startTime = time.time()
        self.orderParameter.calc(frame=self.currentFrame)
        stopTime = time.time()
        print("OrderParameter Time: {}".format(stopTime - startTime))
        if (self.currentFrame > 0):
            if (self.rdf is not None):
                startTime = time.time()
                self.rdf.calc(frame=self.currentFrame)
                stopTime = time.time()
                print("RDF Time: {}".format(stopTime - startTime))
            if (self.ocf is not None):
                startTime = time.time()
                self.ocf.calc(frame=self.currentFrame, symmetry=self.params.angleSymmetry)
                stopTime = time.time()
                print("OCF Time: {}".format(stopTime - startTime))
            if (self.bocf is not None):
                startTime = time.time()
                if self.orderParameter.psiRaw[self.currentFrame] is not None:
                    self.bocf.calc(frame=self.currentFrame, comp=self.orderParameter.psiRaw[self.currentFrame])
                else:
                    self.bocf.calc(frame=self.currentFrame)
                stopTime = time.time()
                print("BOCF Time: {}".format(stopTime - startTime))

        self.currentFrame = self.currentFrame + 1
        if self.currentFrame == len(self.traj):
            if (self.rdf is not None):
                self.rdf.calcRDF()
            if (self.ocf is not None):
                self.ocf.calcOCF()
            if (self.bocf is not None):
                self.bocf.calcBOCF()
            self.timer.stop()
            self.plotSummary()

    # slots for setting various variables using the GUI
    # required for it to work properly
    @QtCore.Slot()
    def setColoringMode(self):
        mode = str(self.options.coloringComboBox.currentText())
        self.group.setColorMode(mode)
        self.viewer.reloadFrame()

    @QtCore.Slot()
    def setAngleSymmetry(self):
        # symmetry = float(self.options.angleComboBox.currentText())
        symmetry = float(self.options.angleLineEdit.text())
        self.symmetry = symmetry
        self.group.setAngleSymmetry(symmetry)
        self.viewer.reloadFrame()

    @QtCore.Slot()
    def setOutline(self):
        # symmetry = float(self.options.angleComboBox.currentText())
        outline = float(self.options.outlineLineEdit.text())
        self.outline = outline
        self.group.setOutline(outline)
        self.viewer.reloadFrame()

    @QtCore.Slot()
    def setKValue(self):
        # symmetry = float(self.options.angleComboBox.currentText())
        k = float(self.options.kLineEdit.text())
        self.k = k
        self.group.setKValue(k)
        self.viewer.reloadFrame()

    @QtCore.Slot()
    def setRMax(self):
        # symmetry = float(self.options.angleComboBox.currentText())
        rMax = float(self.options.rdfLineEdit.text())
        self.rMax = rMax

    @QtCore.Slot()
    def setOCFRMax(self):
        # symmetry = float(self.options.angleComboBox.currentText())
        ocfrMax = float(self.options.ocfLineEdit.text())
        self.ocfrMax = ocfrMax

    @QtCore.Slot()
    def setRDF(self):
        myState = bool(self.options.rdfCheckBox.isChecked())
        self.params.useRDF = myState

    @QtCore.Slot()
    def setOCF(self):
        myState = bool(self.options.ocfCheckBox.isChecked())
        self.params.useOCF = myState

    @QtCore.Slot()
    def setBOCF(self):
        myState = bool(self.options.bocfCheckBox.isChecked())
        self.params.useBOCF = myState

    @QtCore.Slot()
    def setWidth(self):
        # symmetry = float(self.options.angleComboBox.currentText())
        width = float(self.options.widthLineEdit.text())
        self.params.movieWidth = width

    @QtCore.Slot()
    def setHeight(self):
        # symmetry = float(self.options.angleComboBox.currentText())
        height = float(self.options.heightLineEdit.text())
        self.params.movieHeight = Height

    @QtCore.Slot()
    def recordMovie(self):
        # creates movie directory if there isn't one
        if not os.path.isdir("./movie"):
            os.mkdir("./movie")
        # resizes the window
        self.viewer.glWidget.resize(self.params.movieWidth, self.params.movieHeight)

        # records movie in one shot
        # won't really be able to do anything while this happens
        if self.params.isRecording == False:
            self.params.isRecording = True
            self.viewer.gotoFirstFrame()
            for frame in range(self.numFrames-1):
                self.resizeWindow()
                filename = '{}.{:05}.{}'.format(self.params.fileName, frame, "png")
                path = os.path.join("./movie", filename)
                self.viewer.snapshot(filename=path)
                self.viewer.gotoNextFrame()
            self.viewer.restoreSettings()
            self.viewer.gotoFirstFrame()
        else:
            self.viewer.restoreSettings()
            self.viewer.gotoFirstFrame()
            self.params.isRecording = False

    @QtCore.Slot()
    def resizeWindow(self, _=True):
        # resizes the window for movie recording
        self.viewer.glWidget.resize(self.params.movieWidth, self.params.movieHeight)

    @QtCore.Slot()
    def dumpPlots(self):
        # function to dump specific plots
        # unclear if this is working...
        # can this be added to a function?
        if not os.path.isdir("./plots"):
            os.mkdir("./plots")

        if ((self.rdfDock.isVisible()) and self.rdf is not None):
            self.rdf.plotRDF(plot="rdf", path="plots", fileName=self.params.fileName)
            self.rdf.plotRDF(plot="cum", path="plots", fileName=self.params.fileName)
        if ((self.ocfDock.isVisible()) and self.ocf is not None):
            self.ocf.plotOCF(path="plots", fileName=self.params.fileName)
        if ((self.bocfDock.isVisible()) and self.bocf is not None):
            self.bocf.plotBOCF(path="plots", fileName=self.params.fileName)
        if ((self.psiTimeDock.isVisible())):
            self.orderParameter.plotPsi(path="plots", plot="time", fileName=self.params.fileName)
        if ((self.psiDock.isVisible())):
            self.orderParameter.plotPsi(path="plots", frame=(len(self.traj) - 1), plot="psi", fileName=self.params.fileName)

if __name__ == "__main__":
    # init app
    qt.init_app()
    # call the main function
    analysis = Analysis()
    # run
    qt.run()
