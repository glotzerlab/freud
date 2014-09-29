from freud import trajectory, density, order, split
import os
import sys

import numpy
import json

from . import calc
from . import calcPsi

class RDF(calc.Calc):
    """RDF object for calculating RDF in viz.py"""
    def __init__(self, traj, maxR, dr, df=1):
        super(RDF, self).__init__(traj=traj, df=df)
        self.maxR = maxR
        self.dr = dr
        self.rdfHandle = density.RDF(self.box,
                                     self.maxR,
                                     self.dr)
        self.rdfArray = None
        self.cumRDFArray = None
        self.rArray = None

    def calc(self, frame):
        if ((frame - 1) % self.df == 0):
            f = self.traj[frame]
            pos = numpy.copy(f.get("position"))
            pos[:,2] = 0
            self.rdfHandle.compute(pos, pos)
            if (frame == 1):
                self.rArray = self.rdfHandle.getR()
                self.rdfArray = numpy.copy(self.rdfHandle.getRDF())
                self.cumRDFArray = numpy.copy(self.rdfHandle.getNr())
            elif (frame > 1):
                self.rdfArray += numpy.copy(self.rdfHandle.getRDF())
                self.cumRDFArray += numpy.copy(self.rdfHandle.getNr())

    def calcRDF(self):
        self.rdfArray /= (self.nFrames)
        self.cumRDFArray /= (self.nFrames)

    def plotRDF(self, plot="rdf", canvas=None, path=None, fileName=None):
        if plot == "rdf":
            yArray = self.rdfArray
            yLabel = "RDF"
        elif plot == "cum":
            yArray = self.cumRDFArray
            yLabel = "CRDF"
        self.plot(plotType=plot,
                  xArray=self.rArray,
                  yArray=yArray,
                  xLabel="r",
                  yLabel=yLabel,
                  xRange=[0.0, self.maxR],
                  canvas=canvas,
                  path=path,
                  fileName=fileName)

class OCF(calc.Calc):
    """OCF object for calculating OCF in viz.py"""
    def __init__(self, traj, maxR, dr, df=1):
        super(OCF, self).__init__(traj=traj, df=df)
        self.maxR = maxR
        self.dr = dr
        self.ocfHandle = density.ComplexCF(self.box,
                                           self.maxR,
                                           self.dr)
        self.ocfArray = None
        self.rArray = None

    def calc(self, frame, symmetry=1.0):
        if ((frame - 1) % self.df == 0):
            f = self.traj[frame]
            pos = numpy.copy(f.get("position"))
            pos[:,2] = 0
            ang = numpy.copy(f.get('position')[:,2]) * symmetry
            comp = numpy.cos(ang) + 1j * numpy.sin(ang)
            conj = numpy.cos(ang) - 1j * numpy.sin(ang)
            self.ocfHandle.compute(pos, comp, pos, conj)
            if (frame == 1):
                self.rArray = self.ocfHandle.getR()
                self.ocfArray = numpy.real(numpy.copy(self.ocfHandle.getRDF()))
            elif (frame > 1):
                self.ocfArray += numpy.real(numpy.copy(self.ocfHandle.getRDF()))

    def calcOCF(self):
        self.ocfArray /= (self.nFrames)

    def plotOCF(self, canvas=None, path=None, fileName=None):
        self.plot(plotType="ocf",
                  xArray=self.rArray,
                  yArray=self.ocfArray,
                  xLabel="r",
                  yLabel="OCF",
                  xRange=[0.5, self.maxR],
                  yRange=[0.01, 1.1],
                  xLog=True,
                  yLog=True,
                  canvas=canvas,
                  path=path,
                  fileName=fileName)

class BOCF(calc.Calc):
    """BOCF object for calculating BOCF in viz.py"""
    def __init__(self, traj, maxR, maxPsiR, dr, df=1, isSplit=False):
        super(BOCF, self).__init__(traj=traj, df=df)
        self.maxR = maxR
        self.maxPsiR = maxPsiR
        self.dr = dr
        self.bocfHandle = density.ComplexCF(self.box,
                                            self.maxR,
                                            self.dr)
        self.orderHandle = calcPsi.OrderParameter(self.traj, self.maxPsiR, 4, isSplit)
        self.bocfArray = None
        self.rArray = None

    def calc(self, frame, comp=None):
        if ((frame - 1) % self.df == 0):
            f = self.traj[frame]
            pos = numpy.copy(f.get("position"))
            pos[:,2] = 0
            if comp is None:
                self.orderHandle.calc(frame)
                comp = self.orderHandle.psiRaw[frame]
            conj = numpy.conj(comp)
            self.bocfHandle.compute(pos, comp, pos, conj)
            if (frame == 1):
                self.rArray = self.bocfHandle.getR()
                self.bocfArray = numpy.real(numpy.copy(self.bocfHandle.getRDF()))
            elif (frame > 1):
                self.bocfArray += numpy.real(numpy.copy(self.bocfHandle.getRDF()))

    def calcBOCF(self):
        self.bocfArray /= (self.nFrames)

    def plotBOCF(self, canvas=None, path=None, fileName=None):
        self.plot(plotType="bocf",
                  xArray=self.rArray,
                  yArray=self.bocfArray,
                  xLabel="r",
                  yLabel="BOCF",
                  xRange=[0.5, self.maxR],
                  yRange=[0.01, 1.1],
                  xLog=True,
                  yLog=True,
                  canvas=canvas,
                  path=path,
                  fileName=fileName)
