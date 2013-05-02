from __future__ import division, print_function
import numpy
import math

from freud.viz import colorutil
import _freud;

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
    
    return colorutil.linearToSRGBA(cmap);

## HSV colormap
# \param theta numpy array (or something that converts to one) of 0.0-2*pi linear values (values outside this range
#              are wrapped back into it)
# \param s a single value or a numpy array (or something that converts to one) of 0.0-1.0 saturation values
# \param v a single value or a numpy array (or something that converts to one) of 0.0-1.0 linear intensity values
# \param alpha The alpha value for the entire colormap is set to this value
# 
# The HSV colormap maps theta (interpreted as an angle in radians) to the hue colorwheel in the hsv color space.
# s and v are fixed at 1.0.
#
# \note 
# \a theta can be any shape - e.g. a 1-element array, an N-length array an MxN array, an LxMxN 
# array .... 
# \note
# If \a s and/or \a v are numpy arrays, they must have the same shape as \a theta. If \a s and/or \a v are single values,
# then they are applied to all points.
#
# \returns
# A numpy array the same size as \a v , with an added dimension of size 4 containing r,g,b,a values in the
# sRGBA color space
#
def hsv(theta, s=1.0, v=1.0, alpha=1.0):
    # make a copy of v and convert to a numpy array if needed
    w = numpy.array(theta, dtype=numpy.float32);
    newshape = list(w.shape);
    newshape.append(4);
    cmap = numpy.zeros(shape=tuple(newshape), dtype=numpy.float32);
    
    # convert s and v to numpy arrays
    s_array = numpy.array(s, dtype=numpy.float32);
    v_array = numpy.array(v, dtype=numpy.float32);
    
    # promote single values to proper sized arrays
    if s_array.size == 1:
        s_array = numpy.ones(shape=w.shape, dtype=numpy.float32) * s;
    if v_array.size == 1:
        v_array = numpy.ones(shape=w.shape, dtype=numpy.float32) * v;
    
    # check that the size is correct
    if s_array.shape != w.shape:
        raise ValueError('s must have the same shape as theta');
    if v_array.shape != w.shape:
        raise ValueError('v must have the same shape as theta');
    
    # unfold the arrays
    w_u = w.flatten();
    cmap_u = colorutil._unfold(cmap);
    s_array_u = s_array.flatten();
    v_array_u = v_array.flatten();
    
    # compute the colormap
    _freud.hsv2RGBA(cmap, w_u, s_array_u, v_array_u, alpha);
    
    return colorutil.linearToSRGBA(cmap);
