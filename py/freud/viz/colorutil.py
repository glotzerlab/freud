from __future__ import division, print_function
import numpy
import math
import _freud;

## \package freud.viz.colorutil
#
# Color conversion functions
#

## \internal
# \brief Gamma value
# Consider this a predefined constant and do not change it, sRGB is approximately gamma 2.2 (close enough that no one
# will notice)
#
_gamma = 2.2;

## \internal
# \brief Unfold a MxNx...x4 array to Kx4
#
# This is a helper function used by many color conversion and colormap tools
#
# \returns a *view* of \a v with an unfolded shape
def _unfold(v):
    u = v.view();
    
    if v.ndim == 1:
        # handle the special case of a 1D array
        u.shape = (1,v.shape[-1]);
        return u;
    else:
        # start with the first dimension
        n = v.shape[0];
        # then loop over the remaining dimensions, skipping the last
        for m in v.shape[1:-1]:
            n = n * m;
        
        # reshape and return the array
        u.shape = (n, v.shape[-1]);
        return u

## Convert sRGBA colors to linear
# \param v A numpy array (or something that converts to one) of 0.0-1.0 sRGBA colors
#
# RGB values chosen in a color picker (for example) or in any image file are in sRGB format, which is pre-gamma
# corrected for display. This function undoes that correction and puts the colors back into a linear space for
# combination, lighting, etc...
#
# \note
# \a v can be any shape where the last dimension is 4 - e.g. a 4-element array, an Nx4 array an MxNx4 array, an LxMxNx4 
# array .... The last dimension specifies the r,g,b,a colors (in that order). This is useful for applying this function
# to per-particle colors sent into a primitive, or a colormapped image, or a colormapped volume, ...
#
# \returns
# A numpy array the same size as \a v with all r,g, and b values converted into a linear space. The alpha value is not
# modified (it is already linear)
#
def sRGBAtoLinear(v):
    global _gamma;
    
    # make a copy of v and convert to a numpy array if needed
    ret = numpy.array(v, dtype=numpy.float32);
    
    # check that the last dimension is indeed 4
    if ret.shape[-1] != 4:
        raise ValueError('sRGBAtoLinear expects the last dimension to be 4');
    
    # unfold the array
    u = _unfold(ret);
    
    # apply the correction to the colors
    u[:,0:3] = u[:,0:3]**_gamma;
    return ret

## Convert linear colors to sRGBA 
# \param u A numpy array (or something that converts to one) of 0.0-1.0 linear colors
#
# RGB values chosen in a color picker (for example) or in any image file are in sRGB format, which is pre-gamma
# corrected for display. This function applies that correction and takes colors from a linear space back into sRGBA.
#
# \note
# \a u can be any shape where the last dimension is 4 - e.g. a 4-element array, an Nx4 array an MxNx4 array, an LxMxNx4 
# array .... The last dimension specifies the r,g,b,a colors (in that order). This is useful for applying this function
# to per-particle colors sent into a primitive, or a colormapped image, or a colormapped volume, ...
#
# \returns
# A numpy array the same size as \a u with all r,g, and b values converted into the sRGBA space. The alpha value is not
# modified (it is always linear)
#
def linearToSRGBA(u):
    global _gamma;
    
    # make a copy of u and convert to a numpy array if needed
    ret = numpy.array(u, dtype=numpy.float32);
    
    # check that the last dimension is indeed 4
    if ret.shape[-1] != 4:
        raise ValueError('linearToSRGBA expects the last dimension to be 4');
    
    # unfold the array
    v = _unfold(ret);
    
    # apply the correction to the colors
    # v[:,0:3] = v[:,0:3]**(1.0/_gamma);
    # call fast implementation in C
    _freud.linearToSRGBA(v);
    
    return ret

## Convert sRGBA to ARGB32
# \param v A numpy array (or something that converts to one) of 0.0-1.0 sRGBA colors
#
# Image file formats don't store floats per channel, but instead use 8-bits per channel. This function converts an image
# from 0-1.0 floats to 0-255 uchar values. It is primarily intended for use in images output by QImage, but could also 
# be useful elsewhere.
#
# \note
# \a u can be any shape where the last dimension is 4 - e.g. a 4-element array, an Nx4 array an MxNx4 array, an LxMxNx4 
# array .... The last dimension specifies the r,g,b,a colors (in that order). This is useful for applying this function
# to per-particle colors sent into a primitive, or a colormapped image, or a colormapped volume, ...
#
# \returns
# A numpy array the same size as \a u with all r,g, and b values converted into 8-bit channels in the order a,r,g,b - 
# suitable for passing to a QImage
#
def sRGBAtoARGB32(v):
    # make a copy of v and convert to a numpy array if needed
    w = numpy.array(v, dtype=numpy.float32);
    ret = numpy.zeros(shape=w.shape, dtype=numpy.uint8);
    
    # check that the last dimension is indeed 4
    if w.shape[-1] != 4:
        raise ValueError('sRGBAtoARGB32 expects the last dimension to be 4');
    
    # unfold the arrays
    w_u = _unfold(w);
    ret_u = _unfold(ret);
    
    # Clamp to 0-1, then convert 0-1 to 0-255 and change the order of RGBA to ARGB
    w_u = numpy.minimum(w_u, 1.0);
    w_u = numpy.maximum(w_u, 0.0);
    
    # note that due to the way bytes are stored, this is actually in the order BGRA
    ret_u[:,3] = w_u[:,3]*255;   # A
    ret_u[:,0] = w_u[:,2]*255;   # B
    ret_u[:,1] = w_u[:,1]*255;   # G
    ret_u[:,2] = w_u[:,0]*255;   # R    
    return ret
