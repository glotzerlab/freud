from freud import trajectory, order, split
import os

import numpy
import json

import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4']='PySide'
from matplotlib import cm
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from . import calc

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100, tight_layout=False, *args, **kwargs):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='w', tight_layout=tight_layout)
        FigureCanvas.__init__(self, self.fig, *args, **kwargs)
        self.axes = self.fig.add_subplot(111)
        self.axes.hold(False)
        # self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding);
        self.updateGeometry()

class OrderParameter(calc.Calc):
    """OrderParameter object for calculating LocalDensity in viz.py"""
    def __init__(self, traj, maxR, k, isSplit=False):
        super(OrderParameter, self).__init__(traj=traj, df=1)
        self.maxR = maxR
        self.k = k
        self.orderParameterHandle = order.HexOrderParameter(self.box,
                                                            self.maxR,
                                                            self.k)
        self.split = isSplit
        if self.split == True:
            self.splitHandle = split.Split(self.box)
        self.psiHist = [None] * self.nFrames
        self.psiMag = [None] * self.nFrames
        self.psiRaw = [None] * self.nFrames

    def calc(self, frame, split=False):
        # only split if it's a rect/comp
        f = self.traj[frame]
        pos = numpy.copy(f.get("position"))
        pos[:,2] = 0
        ang = numpy.copy(f.get("position")[:,2])
        if self.split == True:
            self.splitHandle.compute(pos, ang, numpy.array([[0.25, 0, 0], [-0.25, 0, 0]], dtype=numpy.float32))
            self.orderParameterHandle.compute(self.splitHandle.shapePositions)
        else:
            self.orderParameterHandle.compute(pos)
        tmpPsi = self.orderParameterHandle.getPsi()
        self.psiMag[frame] = numpy.abs(numpy.mean(tmpPsi))
        if self.split == True:
            psi = numpy.zeros(shape=(self.numParticles), dtype=numpy.complex64)
            for i in range(self.numParticles):
                psi[i] = (tmpPsi[2*i] + tmpPsi[2*i + 1]) / 2.0
        else:
            psi = tmpPsi
        self.psiRaw[frame] = psi
        (self.psiHist[frame], ex, ey) = numpy.histogram2d(numpy.real(psi),
                                                          numpy.imag(psi),
                                                          range=[[-1.0,1.0], [-1.0,1.0]],
                                                          bins=[100,100])

    def plotPsi(self, frame=None, plot=None, canvas=None, path=None, fileName=None):
        if plot == "psi":
            self.plot2DHist(plotType="Psi",
                            dataArray=self.psiHist[frame],
                            canvas=canvas,
                            path=path,
                            fileName=fileName)
        elif plot == "time":
            if canvas is None:
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                ax.plot(self.psiMag)
                ax.set_xlabel('frame')
                ax.set_ylabel("|Psi_{}|".format(self.k))
                if path is not None:
                    plt.savefig(os.path.expanduser("./{}/{}.psi.time.png".format(path, fileName)), dpi=600, bbox_inches="tight")
                    plt.savefig(os.path.expanduser("./{}/{}.psi.time.svg".format(path, fileName)), bbox_inches="tight")
                else:
                    plt.show()
                plt.close()
            else:
                canvas.axes.clear()
                canvas.axes.hold(True)
                canvas.axes.plot(self.psiMag)
                canvas.axes.set_xlabel('frame')
                canvas.axes.set_ylabel("r$|\Psi_{}|$".format(self.k))
                if frame is not None:
                    canvas.axes.axvline(frame, color='r')
                canvas.draw_idle();
