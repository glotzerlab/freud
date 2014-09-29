import os
import re
import numpy
import json

import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4']='PySide'
from matplotlib import cm
import cubehelix
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100, tight_layout=False, *args, **kwargs):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='w', tight_layout=tight_layout)
        FigureCanvas.__init__(self, self.fig, *args, **kwargs)
        self.axes = self.fig.add_subplot(111)
        self.axes.hold(False)
        # self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding);
        self.updateGeometry()

class myPlot(object):
    def __init__(self):
        super(myPlot, self).__init__()
        self.fig = plt.figure(figsize=(20, 20), dpi=300, facecolor='w', tight_layout=False)
        self.axes = self.fig.add_subplot(111)

class Calc(object):
    """basic calculation object for freud viz.py"""
    def __init__(self, traj, df):
        super(Calc, self).__init__()
        self.traj = traj
        self.numParticles = self.traj.numParticles()
        self.df = df
        if self.df != 1:
            self.nFrames = int((len(self.traj) - 1) / self.df)
        else:
            self.nFrames = int(len(self.traj))
        self.box = self.traj[0].box

    def plot(self,
             plotType,
             xArray,
             yArray,
             xLabel,
             yLabel,
             xRange=None,
             yRange=None,
             xLog=False,
             yLog=False,
             canvas=None,
             path=None,
             fileName=None):
        if canvas is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            ax.set_xlabel(r"${}$".format(xLabel))
            ax.set_ylabel(r"${}$".format(yLabel))
            ax.plot(xArray, yArray, color="black")
            if xRange is not None:
                ax.set_xlim(left=xRange[0], right=xRange[1])
            if yRange is not None:
                ax.set_ylim(bottom=yRange[0], top=yRange[1])
            if xLog == True:
                ax.set_xscale('log')
            if yLog == True:
                ax.set_yscale('log')
            if path is not None:
                plt.savefig(os.path.expanduser("./{}/{}.{}.png".format(path, fileName, plotType)))
            else:
                plt.savefig(os.path.expanduser("./{}.{}.png".format(fileName, plotType)))
            plt.close()
        else:
            canvas.axes.clear()
            canvas.axes.hold(True)
            canvas.axes.set_xlabel(r"${}$".format(xLabel))
            canvas.axes.set_ylabel(r"${}$".format(yLabel))
            canvas.axes.plot(xArray, yArray, color="black")
            canvas.axes.set_xlim(left=0, right=self.maxR)
            if xRange is not None:
                canvas.axes.set_xlim(left=xRange[0], right=xRange[1])
            if yRange is not None:
                canvas.axes.set_ylim(bottom=yRange[0], top=yRange[1])
            if xLog == True:
                canvas.axes.set_xscale('log')
            if yLog == True:
                canvas.axes.set_yscale('log')

    def plot2DHist(self,
                   plotType,
                   dataArray,
                   xRange=None,
                   yRange=None,
                   zRange=None,
                   xLabel=None,
                   yLabel=None,
                   zLabel=None,
                   canvas=None,
                   path=None,
                   fileName=None,
                   colormap=cubehelix.cmap(start=1, rot=1.25),
                   cbarOffset=0.0,
                   interactive=False,
                   vertsX=None,
                   vertsY=None,
                   maskArray=None):
        if interactive == True:
            savePlots = False
            canvas = myPlot()
        else:
            if canvas is None:
                savePlots = True
                canvas = myPlot()
            else:
                savePlots = False
                canvas.axes.clear()
                canvas.axes.hold(True)
        if zRange is not None:
            vmin = zRange[0] - cbarOffset * (zRange[1] - zRange[0])
        # set up for plots; extra for pmf
        # right now this will never go because I am sending in more
        # than just pmf
        # if plotType == "pmf":
        # if bool(re.match(plotType, "^pmf")):
        if bool(re.match("^pmf", plotType)):
            canvas.axes.plot([0],[0], "o", color="black", linewidth=1)
            if maskArray is None:
                print("not using a mask to show infinite regions; consider using a mask")
            else:
                cax = canvas.axes.imshow(maskArray,
                                         extent=[xRange[0], xRange[1], yRange[0], yRange[1]],
                                         interpolation="nearest",
                                         vmin=zRange[0],
                                         vmax=zRange[1],
                                         cmap=cm.Greys)
            cax = canvas.axes.imshow(dataArray,
                                     extent=[xRange[0], xRange[1], yRange[0], yRange[1]],
                                     interpolation="nearest",
                                     vmin=vmin,
                                     vmax=zRange[1],
                                     cmap=colormap)
            # check that the verts won't break
            if (vertsX.shape != vertsY.shape):
                print("not using a central shape; consider providing verts")
            else:
                canvas.axes.plot(vertsX, vertsY, color="black", linewidth=2)
        else:
            cax = canvas.axes.imshow(dataArray, interpolation="nearest")
        if xLabel is not None:
            canvas.axes.set_xlabel(xLabel)
        if yLabel is not None:
            canvas.axes.set_ylabel(yLabel)
        canvas.axes.set_title("{}".format(plotType))
        if ((savePlots == True) or (interactive==True)):
            if zRange is not None:
                if (savePlots == True):
                    myShrink=0.77
                else:
                    myShrink=1.0
                cbar = canvas.fig.colorbar(cax, ticks=numpy.linspace(zRange[0], zRange[1], 5), shrink=myShrink)
                cbar.ax.set_ylim([cbar.norm(zRange[0]), cbar.norm(zRange[1])])
                cbar.outline.set_ydata([cbar.norm(zRange[0])] * 2 + [cbar.norm(zRange[1])] * 4 + [cbar.norm(zRange[0])] * 3)
                myTicks = [str(numpy.around(i, 2)) for i in numpy.linspace(zRange[0], zRange[1], 5)]
                cbar.ax.set_yticklabels(myTicks)# vertically oriented colorbar
                if zLabel is not None:
                    cbar.ax.set_ylabel(r"${}$".format(zLabel), fontsize=28)
        if (interactive == True):
            self.numrows, self.numcols = dataArray.shape
            canvas.axes.format_coord = self.format_coord
            plt.show()
            plt.close()
        if (savePlots == True):
            cbar.ax.tick_params(labelsize=24, length=15, width=1)
            canvas.axes.xaxis.label.set_fontsize(28)
            canvas.axes.yaxis.label.set_fontsize(28)
            canvas.axes.xaxis.set_tick_params(labelsize=24, length=15, width=1)
            canvas.axes.yaxis.set_tick_params(labelsize=24, length=15, width=1)
            if path is not None:
                canvas.fig.savefig(os.path.expanduser("./{}/{}.{}.png".format(path, fileName, plotType)), dpi=600, bbox_inches="tight")
                try:
                    cbar.solids.set_edgecolor("face")
                except:
                    pass
                canvas.fig.savefig(os.path.expanduser("./{}/{}.{}.svg".format(path, fileName, plotType)), bbox_inches="tight")
            else:
                canvas.fig.savefig(os.path.expanduser("./{}.{}.png".format(fileName, plotType)), dpi=600, bbox_inches="tight")
                try:
                    cbar.solids.set_edgecolor("face")
                except:
                    pass
                canvas.fig.savefig(os.path.expanduser("./{}.{}.svg".format(fileName, plotType)), bbox_inches="tight")
            plt.close()

    def format_coord(self, x, y):
        # this is a calc instance, so it should have these, need to be careful with enabling...
        # psi can't be interactive...for now
        col = numpy.floor(float(x+self.maxX) / self.dx)
        row = numpy.floor(float(y+self.maxY) / self.dy)
        if col>=0 and col<self.numcols and row>=0 and row<self.numrows:
            z = self.myPMFArray[row][col]
            return "x={:.4}, y={:.4}, z={:.4}".format(x, y, z)
        else:
            return "x={:.4}, y={:.4}".format(x, y)

