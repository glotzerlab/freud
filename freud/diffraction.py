# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :class:`freud.diffraction` module provides functions for computing
diffraction for crystal structures.
"""

import freud
import numpy as np
import garnett
import matplotlib
import matplotlib.pyplot as plt

def readFile(fname):
	with garnett.read(fname) as traj:
		frame = traj[-1]
		box = freud.box.Box.from_box(frame.box)
		positions = frame.positions.copy()
	return box, positions

def findMaxBoxEdge(boxVectors):
	lengthBoxVect = np.linalg.norm(boxVectors, axis=0)
	return max(lengthBoxVect)


def main(fname):
	box, positions = readFile(fname)

	boxVectors = np.asarray([[box.Lx, box.Ly * box.xy, box.Lz * box.xz], [0, box.Ly, box.Lz * box.yz], [0, 0, box.Lz]]);
	maxEdge = findMaxBoxEdge(boxVectors)
	# dummy = boxVectors[0:2, 0:2] 

	# find two vectors whose area of parellelogram is largest
	area_01 = np.linalg.det(boxVectors[0:2, 0:2])
	area_02 = -np.linalg.det(boxVectors[-1:1, 0:2])
	area_12 = np.linalg.det(boxVectors[1:3, 0:2])
	
	# project the box down to xy plane
	largestIndex = np.argmax([area_01, area_02, area_12])
	projected_positions = np.zeros(positions.shape)
	
	if largestIndex == 0:
		projected_positions[:, 0] = positions[:, 0]
		projected_positions[:, 1] = positions[:, 1]
		projected_box = freud.box.Box.from_box(box)
		projected_box.is2D = True
	elif largestIndex == 1:
		projected_positions[:, 0] = positions[:, 0]
		projected_positions[:, 1] = positions[:, 2]
		projected_box = freud.box.Box(Lx=box.Lx, Ly=box.Lz, xy=0.0, xz=0.0, yz=0.0, is2D=True)
	elif largestIndex == 2:
		projected_positions[:, 0] = positions[:, 1]
		projected_positions[:, 1] = positions[:, 2]
		projected_box = freud.box.Box(Lx=box.Ly, Ly=box.Lz, xy=0.0, xz=0.0, yz=0.0, is2D=True)

	gd = freud.density.GaussianDensity(125, 2, 0.08)
	gd.compute(projected_box, projected_positions)
	density = gd.gaussian_density
	# using numpy fft implmentation, but msd has fft calculation
	fft = np.fft.fft2(density)

	plt.imshow(density)
	plt.colorbar()
	plt.show()
	fft_mag = np.absolute(np.fft.fftshift(fft))
	plt.imshow(np.log(fft_mag), cmap='afmhot', vmin=7, vmax=10)
	plt.colorbar

