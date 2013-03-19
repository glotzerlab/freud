from __future__ import division, print_function
import numpy
import math

from freud.viz import colorutil

## \package freud.viz.colormap
#
# Color maps
#

## Grayscale colormap
# \param u A numpy array (or something that converts to one) of 0.0-1.0 linear values
# \param alpha The alpha value for the entire colormap is set to this value
# 
# The grayscale colormap maps 0 to black, 1 to white and linearly interpolates between shades of gray for intermediate
# values
#
# \note
# \a u can be any shape - e.g. a 1-element array, an N-length array an MxN array, an LxMxN 
# array .... 
#
# \returns
# A numpy array the same size as \a v , with an added dimension of size 4 containing r,g,b,a grayscale values in the
# sRGBA color space
#
def grayscale(u, alpha=1.0):
    # make a copy of v and convert to a numpy array if needed
    w = numpy.array(u, dtype=numpy.float32);
    newshape = list(w.shape);
    newshape.append(4);
    cmap = numpy.zeros(shape=tuple(newshape), dtype=numpy.float32);
        
    # unfold the array
    w_u = w.flatten();
    cmap_u = colorutil._unfold(cmap);
    
    # compute the grayscale colormap
    cmap_u[:,0] = w_u[:];
    cmap_u[:,1] = w_u[:];
    cmap_u[:,2] = w_u[:];
    cmap_u[:,3] = alpha;
    
    return colorutil.linearToSRGBA(cmap_u);
