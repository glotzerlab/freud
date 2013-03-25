from __future__ import division, print_function
import numpy
import math
from ctypes import c_void_p
import OpenGL
#OpenGL.FULL_LOGGING = True
OpenGL.FORWARD_COMPATIBLE_ONLY = True
OpenGL.ERROR_ON_COPY = True
from OpenGL import GL

null = c_void_p(0)

## \package freud.viz.render.gl
#
# GL output for freud.viz
#

## DrawGL draws scenes using OpenGL
#
# Instantiating a DrawGLE loads shaders and performs other common init tasks. You can then call draw() as many times as
# you want to draw GL frames. resize() issues the GL commands to resize the display to match a given width and height.
#
# DrawGL uses the visitor pattern to handle output methods for different primitives. 
# The method used is described here: http://peter-hoffmann.com/2010/extrinsic-visitor-pattern-python-inheritance.html
#
# Internally, GL geometry differs from the raw primitive data. DrawGL generates this geometry on the fly as needed
# and stores it in a cache. The next draw call will reuse geometry data out of the cache for primitives that are
# identical. This is why primitives encourage recreation of primitives and not changing the data. However, this is
# only a temporary solution. For example, recreating the entire primitive is wasteful when just changing the outline
# width.
# 
# TODO - add some kind of dirty flag to primitives and set commands necessary to update values that set the flag.
#
class DrawGL:
    ## Initialize a DrawGL
    # \param width_cm Width of the output figure in cm
    # \note Height is determined from the aspect ratio
    def __init__(self):
        pass

    ## Draws a primitive to the GL context
    # \param prim Primitive to draw
    #
    def draw_Primitive(self, prim):
        raise RuntimeError('DrawGL encountered an unknown primitive type');

    ## Draw an entire scene
    # \param scene Scene to write
    #
    def draw_Scene(self, scene):
        # setup the camera matrix
        
        
        # loop through the render primitives and write out each one
        for i,group in enumerate(scene.groups):
            # apply the group transformation matrix
            for j,primitive in enumerate(group.primitives):
                self.write(primitive)

    ## Draw disks
    # \param disks Disks to write
    #
    def draw_Disks(self, disks):
        pass

    ## Draw repeated polygons
    # \param polygons Polygons to draw
    #
    def draw_RepeatedPolygons(self, polygons):
        pass

    ## Write out image
    # \param img Image to write
    #
    def draw_Image(self, img):
        pass
        
    ## Draw a viz element
    # \param obj Object to write
    #
    def write(self, obj):
        meth = None;
        for cls in obj.__class__.__mro__:
            meth_name = 'draw_'+cls.__name__;
            meth = getattr(self, meth_name, None);
            if meth is not None:
                break;

        if meth is None:
            raise RuntimeError('DrawGL does not know how to write a {0}'.format(obj.__class__.__name__));
        return meth(obj);

## Base class for cache items
#
# DrawGL stores openGL geometry in a cache. Each item stored in the cache derives from CacheItem and implements the same
# interface. There are a few requirements. 1) CacheItems are only created or access while the OpenGL context is active.
# 2) They initialize their geometry (or other OpenGL entities) on initialization. 3) There is a release() method
# that releases all of the OpenGL entities.
#
# Other than that, CacheItems are free-form and can be implemented however needed for the specific primitive. Typical
# use-cases will probably just store a few buffers as member variables to be directly accessed by DrawGL calls.
#
# Putting the code for geometry generation here keeps it compartmentalized separately from the code specific to drawing
# the actual geometry. Of course, drawing and the geometry format are tied closely and both bits of code will need to be
# updated in tandem.
#
class CacheItem:
    ## Initialize a cache item
    # \param prim Primitive to represent
    #
    def __init__(self, prim):
        pass
    
    ## Release a cache item
    # 
    # Frees all OpenGL resources used by the item
    #
    def release(self):
        pass

## Cache RepeatedPolygon geometry
#
# Store the OpenGL geometry for the RepeatedPolygon primitive
#
class CacheRepeatedPolygon:
    ## Initialize a cache item
    # \param prim Primitive to represent
    #
    def __init__(self, prim):
        pass
    
    ## Release a cache item
    # 
    # Frees all OpenGL resources used by the item
    #
    def release(self):
        pass  

## Cache Disk geometry
#
# Store the OpenGL geometry for the Disk primitive
#
class CacheDisk:
    ## Initialize a cache item
    # \param prim Primitive to represent
    #
    def __init__(self, prim):
        pass
    
    ## Release a cache item
    # 
    # Frees all OpenGL resources used by the item
    #
    def release(self):
        pass  
