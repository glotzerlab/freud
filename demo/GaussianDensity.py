from freud import trajectory
from freud import locality
from freud import density
import scipy.io
import numpy
import math

#Begin by importing the trajectory, this file has not trajectory
traj = trajectory.TrajectoryXMLDCD('1sphere.xml', None)
f    = traj[0]

#set up the particle positions of each type
pos = f.get('position')

#calculate the Density
#GaussianDensity(box, n_bins[per side], r_cut, sigma)
#compute(array with positions)
gdens = density.GaussianDensity(f.box, 90, 2.15, 0.5)
gdens.compute(pos)
Density = gdens.getGaussianDensity()

scipy.io.savemat('Density.mat', dict(N=Density))
print('Done.')
