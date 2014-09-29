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

from freud import viz, qt, trajectory, density, order
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

class TrajGroup(viz.base.GroupTrajectory):
    def __init__(self, trajectory, analysis):
        super(TrajGroup, self).__init__(trajectory)
        # Add in an option to control the outline thickness
        self.numParticles = self.trajectory.numParticles()
        self.analysisRef = weakref.ref(analysis)
        # set the default so that the whole thing doesn't die
        self.symmetry = 1.0
        self.mode = "normal"
        myAnalysis = self.analysisRef()
        self.nTypes = myAnalysis.nTypes
        self.shapeA = Polygon(myAnalysis.shapes[0])
        self.outlineA = Outline(self.shapeA, width=analysis.params.outline)
        if self.nTypes == 2:
            self.shapeB = Polygon(myAnalysis.shapes[1])
            self.outlineB = Outline(self.shapeB, width=analysis.params.outline)

        # set up colors
        self.blue = numpy.array([float(15), float(45), float(91), 1.0], dtype=numpy.float32)
        self.maize = numpy.array([float(255), float(196), float(37), 1.0], dtype=numpy.float32)
        self.red = numpy.array([float(255), float(0), float(0), 1.0], dtype=numpy.float32)
        self.green = numpy.array([float(0), float(255), float(0), 1.0], dtype=numpy.float32)
        # self.white = numpy.array([float(0), float(0), float(0), 0.5], dtype=numpy.float32)
        self.grey = numpy.array([float(0), float(0), float(0), 0.3], dtype=numpy.float32)
        self.blue[:3] = self.blue[:3]/255.0
        self.blue[:3] = self.blue[:3] * 1.5
        self.maize[:3] = self.maize[:3]/255.0
        self.red[:3] = self.red[:3]/255.0
        self.green[:3] = self.green[:3]/255.0
        self.grey[:3] = self.grey[:3]/255.0

        f = self.trajectory[0]
        self.positions = f.get("position")[:,0:2]
        self.angles = f.get("position")[:,2]
        self.primitives = []
        if self.nTypes == 1:
            self.colors = numpy.zeros((self.numParticles, 4), dtype=numpy.float32)
            self.colors[:] = self.maize
            self.primitives.append(viz.primitive.Polygons(
                positions=self.positions,
                orientations=self.angles,
                polygon=self.shapeA,
                colors=self.colors))
            self.primitives.append(viz.primitive.Polygons(
                positions=self.positions,
                orientations=self.angles,
                polygon=self.outlineA,
                color=[0,0,0,1]))
        else:
            self.colors = numpy.zeros((self.numParticles, 4), dtype=numpy.float32)
            typeA = [(t == 'A') for t in f.get("typename")]
            typeB = [(t == 'B') for t in f.get("typename")]
            posA = numpy.compress(typeA, self.positions, axis=0)
            posB = numpy.compress(typeB, self.positions, axis=0)
            angleA = numpy.compress(typeA, self.angles, axis=0)
            angleB = numpy.compress(typeB, self.angles, axis=0)
            self.colors[:] = self.maize
            colorsA = numpy.compress(typeA, self.colors, axis=0)
            self.colors[:] = self.blue
            colorsB = numpy.compress(typeB, self.colors, axis=0)
            self.primitives.append(viz.primitive.Polygons(
                positions=posA,
                orientations=angleA,
                polygon=self.shapeA,
                colors=colorsA
                ))
            self.primitives.append(viz.primitive.Polygons(
                positions=posA,
                orientations=angleA,
                polygon=self.outlineA,
                color=[0,0,0,1]
                ))
            self.primitives.append(viz.primitive.Polygons(
                positions=posB,
                orientations=angleB,
                polygon=self.shapeB,
                colors=colorsB
                ))
            self.primitives.append(viz.primitive.Polygons(
                positions=posB,
                orientations=angleB,
                polygon=self.outlineB,
                color=[0,0,0,1]
                ))

    def buildPrimitives(self, frame):
        analysis = self.analysisRef()
        if analysis is None:
            raise RunTimeError('GroupTrajectoryPolygons expected the Analysis to still exist!')
        f = self.trajectory[frame]
        self.positions = numpy.copy(f.get("position")[:,0:2])
        self.angles = numpy.copy(f.get("position")[:,2])
        if self.mode == 'normal':
            if self.nTypes == 1:
                self.colors = numpy.zeros((self.numParticles, 4), dtype=numpy.float32)
                self.colors[:] = self.maize
                self.primitives[0].update(positions=self.positions,
                    orientations=self.angles,
                    colors=self.colors)
                self.primitives[1].update(positions=self.positions,
                    orientations=self.angles,
                    color=[0,0,0,1])
            else:
                self.colors = numpy.zeros((self.numParticles, 4), dtype=numpy.float32)
                typeA = [(t == 'A') for t in f.get("typename")]
                typeB = [(t == 'B') for t in f.get("typename")]
                posA = numpy.compress(typeA, self.positions, axis=0)
                posB = numpy.compress(typeB, self.positions, axis=0)
                angleA = numpy.compress(typeA, self.angles, axis=0)
                angleB = numpy.compress(typeB, self.angles, axis=0)
                self.colors[:] = self.maize
                colorsA = numpy.compress(typeA, self.colors, axis=0)
                self.colors[:] = self.blue
                colorsB = numpy.compress(typeB, self.colors, axis=0)
                self.primitives[0].update(positions=posA,
                    orientations=angleA,
                    colors=colorsA
                    )
                self.primitives[1].update(positions=posA,
                    orientations=angleA,
                    color=[0,0,0,1]
                    )
                self.primitives[2].update(positions=posB,
                    orientations=angleB,
                    colors=colorsB
                    )
                self.primitives[3].update(positions=posB,
                    orientations=angleB,
                    color=[0,0,0,1]
                    )

        elif self.mode == 'angle':
            angleSymmetry = self.angles * self.symmetry % (2.0 * numpy.pi)
            if self.nTypes == 1:
                avgAngle = numpy.mean(angleSymmetry)
                self.colors = viz.colormap.hsv(angleSymmetry - avgAngle + 240.0*numpy.pi/180.0);
                self.primitives[0].update(positions=self.positions,
                    orientations=self.angles,
                    colors=self.colors)
                self.primitives[1].update(positions=self.positions,
                    orientations=self.angles,
                    color=[0,0,0,1])
            else:
                avgAngle = numpy.mean(angleSymmetry)
                self.colors = viz.colormap.hsv(angleSymmetry - avgAngle + 240.0*numpy.pi/180.0);
                typeA = [(t == 'A') for t in f.get("typename")]
                typeB = [(t == 'B') for t in f.get("typename")]
                posA = numpy.compress(typeA, self.positions, axis=0)
                posB = numpy.compress(typeB, self.positions, axis=0)
                angleA = numpy.compress(typeA, self.angles, axis=0)
                angleB = numpy.compress(typeB, self.angles, axis=0)
                colorsA = numpy.compress(typeA, self.colors, axis=0)
                colorsB = numpy.compress(typeB, self.colors, axis=0)
                self.primitives[0].update(positions=posA,
                    orientations=angleA,
                    colors=colorsA
                    )
                self.primitives[1].update(positions=posA,
                    orientations=angleA,
                    color=[0,0,0,1]
                    )
                self.primitives[2].update(positions=posB,
                    orientations=angleB,
                    colors=colorsB
                    )
                self.primitives[3].update(positions=posB,
                    orientations=angleB,
                    color=[0,0,0,1]
                    )

        elif self.mode == 'psi angle':
            # do I need to take into account particle symmetry?
            angle = numpy.angle(analysis.orderParameter.psiRaw[frame])
            avgAngle = numpy.angle(numpy.mean(analysis.orderParameter.psiRaw[frame]))
            self.colors = viz.colormap.hsv(angle - avgAngle + 240*numpy.pi/180)
            if self.nTypes == 1:
                self.primitives[0].update(positions=self.positions,
                    orientations=self.angles,
                    colors=self.colors)
                self.primitives[1].update(positions=self.positions,
                    orientations=self.angles,
                    color=[0,0,0,1])
            else:
                typeA = [(t == 'A') for t in f.get("typename")]
                typeB = [(t == 'B') for t in f.get("typename")]
                posA = numpy.compress(typeA, self.positions, axis=0)
                posB = numpy.compress(typeB, self.positions, axis=0)
                angleA = numpy.compress(typeA, self.angles, axis=0)
                angleB = numpy.compress(typeB, self.angles, axis=0)
                colorsA = numpy.compress(typeA, self.colors, axis=0)
                colorsB = numpy.compress(typeB, self.colors, axis=0)
                self.primitives[0].update(positions=posA,
                    orientations=angleA,
                    colors=colorsA
                    )
                self.primitives[1].update(positions=posA,
                    orientations=angleA,
                    color=[0,0,0,1]
                    )
                self.primitives[2].update(positions=posB,
                    orientations=angleB,
                    colors=colorsB
                    )
                self.primitives[3].update(positions=posB,
                    orientations=angleB,
                    color=[0,0,0,1]
                    )

        elif self.mode == 'psi magnitude':
            mag = 1.0 - numpy.abs(analysis.orderParameter.psiRaw[frame])
            self.colors = viz.colormap.grayscale(mag)
            if self.nTypes == 1:
                self.primitives[0].update(positions=self.positions,
                    orientations=self.angles,
                    colors=self.colors)
                self.primitives[1].update(positions=self.positions,
                    orientations=self.angles,
                    color=[0,0,0,1])
            else:
                typeA = [(t == 'A') for t in f.get("typename")]
                typeB = [(t == 'B') for t in f.get("typename")]
                posA = numpy.compress(typeA, self.positions, axis=0)
                posB = numpy.compress(typeB, self.positions, axis=0)
                angleA = numpy.compress(typeA, self.angles, axis=0)
                angleB = numpy.compress(typeB, self.angles, axis=0)
                colorsA = numpy.compress(typeA, self.colors, axis=0)
                colorsB = numpy.compress(typeB, self.colors, axis=0)
                self.primitives[0].update(positions=posA,
                    orientations=angleA,
                    colors=colorsA
                    )
                self.primitives[1].update(positions=posA,
                    orientations=angleA,
                    color=[0,0,0,1]
                    )
                self.primitives[2].update(positions=posB,
                    orientations=angleB,
                    colors=colorsB
                    )
                self.primitives[3].update(positions=posB,
                    orientations=angleB,
                    color=[0,0,0,1]
                    )

        else:
            raise RuntimeError("improper coloring spec'd")
        return self.primitives

    def setColorMode(self, mode):
        modes = ["normal",
                 "angle",
                 "psi angle",
                 "psi magnitude"]
        if mode in modes:
            self.mode = mode
        else:
            raise RuntimeError("unsupported coloring mode")

    def setAngleSymmetry(self, symmetry):
        self.symmetry = symmetry

    def setKValue(self, k):
        self.k = k

    def setOutline(self, outline):
        self.outline = outline
        self.outlineA = Outline(self.shapeA, width=self.outline)
        self.primitives[1].update(polygon=self.outlineA)
        if self.nTypes == 2:
            self.outlineB = Outline(self.shapeB, width=self.outline)
            self.primitives[3].update(polygon=self.outlineB)