import numpy as np
import logging
logger = logging.getLogger(__name__)
try:
    from scipy.spatial import Voronoi
except ImportError:
    Voronoi = None
    msg = 'scipy.spatial.Voronoi is not available (requires scipy 0.12+), so freud.voronoi is not available.'
    logger.warning(msg)
    #raise ImportWarning(msg)

## Compute the Voronoi tesselation of a 2D or 3D system using qhull
# This essentially just wraps scipy.spatial.Voronoi, but accounts for
# periodic boundary conditions
class Voronoi:
  ##Initialize Voronoi
  # \param box The simulation box
  # \param buff The buffer of particles to be duplicated to simulated PBC, default=0.1
  def __init__(self,box,buff=0.1):
    self.box=box
    self.buff=buff

  ##Set the simulation box
  def setBox(self,box):
    self.box=box
  ##Set the box buffer width
  def setBufferWidth():
    self.buff=buff

  ##Compute Voronoi tesselation
  # \param box The simulation box
  # \param buff The buffer of particles to be duplicated to simulated PBC, default=0.1
  # \returns a list of ConvexPolyhedrons/ConvexPolygon describing the Voronoi tesselation
  def compute(self,positions,box=None,buff=None):
    #if box or buff is not specified, revert to object quantities
    if box is None:
      box=self.box
    if buff is None:
      buff=self.buff

    #Compute the buffer particles in c++
    box_inv = np.linalg.inv(box)
    buffer_particles = cpp_voronoi(positions,box_inv,buff)

    all_points = np.concatenate(positions,buff_particles)
    tes = vor(all_points)
