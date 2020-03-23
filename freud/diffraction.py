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

	box = freud.box.Box.from_box(box)
	positions = frame.positions.copy()
	return box, positions

def main(box, positions):

	projected_positions = np.zeros(_positions.shape)
	projected_positions[:, 0] = _positions[:, 1]
	projected_positions[:, 1] = _positions[:, 2]
	projected_box = freud.box.Box.square(max(_box.L))


	gd = freud.density.GaussianDensity(125, 2, 0.08)
	gd.compute(projected_box, projected_positions)
	density = gd.gaussian_density
	# msd has fft calculation
	fft = np.fft.fft2(density)

	plt.imshow(density)
	plt.colorbar()
	plt.show()
	fft_mag = np.absolute(np.fft.fftshift(fft))
	plt.imshow(np.log(fft_mag), cmap='afmhot', vmin=7, vmax=10)
	plt.colorbar

