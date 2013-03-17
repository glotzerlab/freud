from __future__ import division, print_function
import math
import numpy

## \package freud.viz
#
# Scriptable visualization tools in freud
#

## Scene container
#
# A Scene represents a collection of all objects necessary to render a given scene. It has a Camera, zero or more Light
# objects, and zero or more Group objects containing Primitive objects.
#
class Scene(object):
    ## Initialize a Scene
    # \param camera camera properties (of type Camera)
    # \param groups List of groups to render (of type Group, or inherited)
    # \param lights List of lights (of type Light, or inherited)
    #
    # If \a camera is left as \c None, then a default camera will be initialized.
    def __init__(self, camera = None, groups = [], lights = []):
        # initialize camera
        if camera is None:
            self.camera = Camera();
        else:
            self.camera = camera;
        
        self.lights = lights;
        self.groups = groups;
    
    ## Set the animation frame
    # \param frame Frame index to set
    def setFrame(self, frame):
        self.camera.setFrame(frame);
        
        for light in lights:
            light.setFrame(frame);
        
        for group in groups:
            group.setFrame(frame);
    
    ## Get the number of frames in the scene's animation
    # \returns The maximum number of frames in any constituent camera, light, or group
    def getNumFrames(self):
        num_frames = self.camera.getNumFrames();
        
        for light in lights:
            num_frames = max(num_frames, light.getNumFrames());
        
        for group in groups:
            num_frames = max(num_frames, group.getNumFrames());

## Base class for the simplest renderable items
#
# A Primitive specifies the simplest item that can be rendered. For performance, a Primitive should specify N such items
# (64000 spheres, for example). All N items in the primitive can be rendered in a single batch call. The same color
# and/or material parameters are applied to all items in the primitive, though primitives may specify per item
# quantities if they wish (for example, color).
#
# A primitive represents its visual aspect in the simplest pure form (for example, center and radius). Different 
# render implementations may take this data and produce specific information that they need.
#
class Primitive(object):
    ## Base class inits nothing
    def __init__(self):
        pass;

## Disk representation (2D)
#
# Represent N disks in 2D. Each has a given color and a global outline width is specified.
class Disks(Primitive):
    ## Initialize a disk representation
    # \param positions Nx2 array listing the positions of each disk (in distance units)
    # \param diameters N array listing the diameters of each disk (in distance units)
    # \param colors Nx4 array listing the colors (rgba 0.0-1.0) of each disk (in SRGB)
    # \param color 4 element iterable listing the color to be applied to every disk. (in SRGB)
    #              \a color overrides anything set by colors
    # \param outline Outline width (in distance units)
    #
    # When \a diameters is None, it defaults to 1.0 for each particle. When colors is none, it defaults to 
    # (0,0,0,1) for each particle.
    #
    # \note N **must** be the same for each array
    #
    # After initialization, the instance will have members positions, diameters, and colors, each being a numpy
    # array of the appropriate size and dtype float32. Users should not modify these directly, they are intended for
    # use only by renderers. Instead, users should create a new primitive from scratch to rebuild geometry.
    #
    def __init__(self, positions, diameters=None, colors=None, color=None, outline=0.1):
        # -----------------------------------------------------------------
        # set up positions
        # convert to a numpy array
        self.positions = numpy.array(positions, dtype=numpy.float32);
        # error check the input
        if len(self.positions.shape) != 2:
            raise TypeError("positions must be a Nx2 array");
        if self.positions.shape[1] != 2:
            raise ValueError("positions must be a Nx2 array");
        
        N = self.positions.shape[0];
        
        # -----------------------------------------------------------------
        # set up diameters
        if diameters is None:
            self.diameters = numpy.zeros(shape=(N,), dtype=numpy.float32);
            self.diameters[:] = 1;
        else:
            self.diameters = numpy.array(diameters);
        
        # error check diameters
        if len(self.diameters.shape) != 1:
            raise TypeError("diameters must be a single dimension array");
        if self.diameters.shape[0] != N:
            raise ValueError("diameters must have N the same as positions");
        
        # -----------------------------------------------------------------
        # set up colors
        if colors is None:
            self.colors = numpy.zeros(shape=(N,4), dtype=numpy.float32);
            self.colors[:,3] = 1;
        else:
            self.colors = numpy.array(colors);
        
        # error check colors
        if len(self.colors.shape) != 2:
            raise TypeError("colors must be a Nx4 array");
        if self.colors.shape[1] != 4:
            raise ValueError("colors must have N the same as positions");
        if self.colors.shape[0] != N:
            raise ValueError("colors must have N the same as positions");
        
        if color is not None:
            acolor = numpy.array(color);
            if len(acolor.shape) != 1:
                raise TypeError("color must be a 4 element array");
            if acolor.shape[0] != 4:
                raise ValueError("color must be a 4 element array");

            self.colors[:,:] = acolor;
    
        # -----------------------------------------------------------------
        # set up outline
        self.outline = outline;

## Line representation
#
# Represent N lines in 2D or 3D (2D specific renderers may simply ignore the z component).
class Lines(Primitive):
    pass

## Repeated polygons
#
# Represent N instances of the same polygon in 2D, each at a different position, orientation, and color. Black edges
# are drawn given a global outline width.
#
class RepeatedPolygons(Primitive):
    ## Initialize a disk representation
    # \param positions Nx2 array listing the positions of each polygon (in distance units)
    # \param angles N array listing the rotation of the polygon about its center (in radians)
    # \param polygon Kx2 array listing the coordinates of each polygon in its local frame (in distance units)
    # \param colors Nx4 array listing the colors (rgba 0.0-1.0) of each polygon (in SRGB)
    # \param color 4 element iterable listing the color to be applied to every polygon (in SRGB)
    #              \a color overrides anything set by colors
    # \param outline Outline width (in distance units)
    #
    # When colors is none, it defaults to (0,0,0,1) for each particle.
    #
    # \note N **must** be the same for each array
    #
    # After initialization, the instance will have members positions, angles, polygon and colors, each being a numpy
    # array of the appropriate size and dtype float32. Users should not modify these directly, they are intended for
    # use only by renderers. Instead, users should create a new primitive from scratch to rebuild geometry.
    #
    def __init__(self, positions, angles, polygon, colors=None, color=None, outline=0.1):        
        # -----------------------------------------------------------------
        # set up positions
        # convert to a numpy array
        self.positions = numpy.array(positions, dtype=numpy.float32);        
        # error check the input
        if len(self.positions.shape) != 2:
            raise TypeError("positions must be a Nx2 array");
        if self.positions.shape[1] != 2:
            raise ValueError("positions must be a Nx2 array");
        
        N = self.positions.shape[0];
        
        # -----------------------------------------------------------------
        # set up angles
        self.angles = numpy.array(angles);
        
        # error check angles
        if len(self.angles.shape) != 1:
            raise TypeError("angles must be a single dimension array");
        if self.angles.shape[0] != N:
            raise ValueError("angles must have N the same as positions");
        
        # -----------------------------------------------------------------
        # set up polygon
        self.polygon = numpy.array(polygon);
        
        # error check polygon
        if len(self.polygon.shape) != 2:
            raise TypeError("polygon must be a Kx2 array");
        if self.polygon.shape[0] < 3:
            raise ValueError("polygon must have at least 3 vertices");
        if self.polygon.shape[1] < 2:
            raise ValueError("polygon must be a Kx2 array");

        # -----------------------------------------------------------------
        # set up colors
        if colors is None:
            self.colors = numpy.zeros(shape=(N,4), dtype=numpy.float32);
            self.colors[:,3] = 1;
        else:
            self.colors = numpy.array(colors);
        
        # error check colors
        if len(self.colors.shape) != 2:
            raise TypeError("colors must be a Nx4 array");
        if self.colors.shape[1] != 4:
            raise ValueError("colors must have N the same as positions");
        if self.colors.shape[0] != N:
            raise ValueError("colors must have N the same as positions");
        
        if color is not None:
            acolor = numpy.array(color);
            if len(acolor.shape) != 1:
                raise TypeError("color must be a 4 element array");
            if acolor.shape[0] != 4:
                raise ValueError("color must be a 4 element array");

            self.colors[:,:] = acolor;
        
        # -----------------------------------------------------------------
        # set up outline
        self.outline = outline;

## Group collects zero or more primitives together
#
# A group may be positioned in world space. It collects together related primitives (for example, simulation box,
# particle primitives, and polygon overlays). Primitives need to be put in a group before they may be added to a
# Scene.  
class Group(object):
    ## Initialize a group
    # \param primitives List of primitives (of type Primitive or inherited) to include in the group
    def __init__(self, primitives = []):
        self.primitives = primitives;
    
    ## Sets the current animation frame
    # \param frame Animation frame index to set
    # The base class group does nothing
    def setFrame(self, frame):
        pass
    
    ## Get the number of animation frames
    # \returns Base class group only has 1 static frame
    def getNumFrames(self):
        return 1;
    
## Specify the camera from which the scene should be rendered
#
# A Camera has a position, look_at point, up vector, height (or vertical fov), and an aspect ratio.
# It can also be perspective or orthographic.
# \note Only orthographic is currently implemented
#
# When specified, the vertical field of view overrides the height setting with the height of the viewable image plane
# passing through the look_at point.
#
# In 2D, the look_at_point is ignored and height is interpreted directly. Position is used to set the middle of the view
# and width and height define the viewable region. 
#
# **TODO, use up-vector for rotations in 2D?** This could be done by using a standard matrix with look_at = position
# with a shifted z. Need to work out the math for clip space units to cm in GLE for this to work.... though maybe
# specifying a simple width and height for the final GLE image is fine. Will implement without rotation first and 
# see how it goes.
#
# **TODO** think about using focal lengths (equivalent to 35mm camera) instead of fov angles. These are more natural
# to use, and might interpolate better in animated cameras
#
# The width of the viewable image plane is derived from the height and aspect ratio: width = aspect * height.
#
# **TODO** Current methods are set up for 2D only. Expand with full 4x4 matrix generation method for general use (though
#          designed primarily for OpenGL.
# **TODO** add perspective projection camera
#
class Camera(object):
    ## Initialize a camera
    # \param position position of the camera (3-element iterable)
    # \param look_at point to look at in world space (3-element iterable)
    # \param up vector pointing up in world space (3-element iterable)
    # \param aspect aspect ratio of the viewable area: w/h
    # \param vfov vertical field of view (in radians) of the viewable plane passing through the look_at point
    # \param height height of the viewable plane passing through the look_at point in world space units (overrides vfov
    #        if set)
    #
    def __init__(self, position, look_at, up, aspect, vfov=math.pi/4, height=None):
        if len(position) != 3:
            raise TypeError('look_at must be a 3-element vector')
        if len(look_at) != 3:
            raise TypeError('look_at must be a 3-element vector')
        if len(up) != 3:
            raise TypeError('up must be a 3-element vector')
        
        self.position = numpy.array(position, dtype=numpy.float32);
        self.look_at = numpy.array(look_at, dtype=numpy.float32);
        self.up = numpy.array(up, dtype=numpy.float32);
        self.aspect = aspect;
        self.vfov = vfov;
        self.height = height;
    
    ## Get the height of the view plane
    # \returns height of the view plane passing through the look_at point
    def getHeight(self):
        if self.height is not None:
            return self.height;
        else:
            # tan(vfov/2) = h/2d
            # => h = 2d * ttan(vfov/2)
            # where d is the distance from the camera to look_at
            direction = self.look_at - self.position;
            d = math.sqrt(numpy.dot(direction, direction));
            return 2*d*math.tan(self.vfov / 2);
    
    ## Set the height of the view plane
    # \param height Height of the view plane passing through the look_at point
    # \note Setting height overrides any previous vfov setting
    def setHeight(self, height):
        self.height = height;
    
    ## Get the vertical field of view
    # \returns the vertical field of view
    def getVFOV(self):
        if self.height is not None:
            # tan(vfov/2) = h/2d
            # => vfov/2 = atan(h/2d)
            # => vfov = 2 * atan(h/2d)
            # where d is the distance from the camera to look_at
            direction = self.look_at - self.position;
            d = math.sqrt(numpy.dot(direction, direction));
            return 2*math.atan(self.height / (2 * d));
        else:
            return self.vfov;

    
    ## Set the vertical field of view
    # \param vfov Vertical field of view
    # \note Setting vfov overrides any previous height setting
    def setVFOV(self, vfov):
        self.height = None;
        self.vfov = vfov;
    
    ## Get the width of the view plane
    # \returns width of the view plane passing through the look_at point
    # The width is derived from the height and the aspect ratio
    def getWidth(self):
        # a = w/h
        # => w = a*h
        return self.getHeight() * self.aspect;
    
    ## Get the aspect ratio
    # \returns the Aspect ratio
    def getAspect(self):
        return self.aspect;
    
    ## Set the aspect ratio
    # \param aspect New aspect ratio to set (w/h)
    def setAspect(self, aspect):
        self.aspect = aspect;
    
    ## Get the up vector
    # \returns The up vector (as a 3-element numpy array)
    def getUp(self):
        return self.up;
    
    ## Set the up vector
    # \param up New up vector to set (3-element iterable)
    def setUp(self, up):
        if len(up) != 3:
            raise TypeError('up must be a 3-element vector')
        self.up = numpy.array(up);
    
    ## Get the look_at vector
    # \returns The look_at vector (as a 3-element numpy array)
    def getLookAt(self):
        return self.look_at;
    
    ## Set the look_at vector
    # \param look_at New look_at vector to set (3-element iterable)
    def setLookAt(self, look_at):
        if len(look_at) != 3:
            raise TypeError('look_at must be a 3-element vector')
        self.look_at = numpy.array(look_at);

    ## Get the position vector
    # \returns The position vector (as a 3-element numpy array)
    def getPosition(self):
        return self.position;
    
    ## Set the position vector
    # \param position New position vector to set (3-element iterable)
    def setPosition(self, position):
        if len(position) != 3:
            raise TypeError('position must be a 3-element vector')
        self.position = numpy.array(position);
    
## Specify the location and properties of a light in the scene
class Light(object):
    pass

## Specify material properties applied to a Primitive
class Material(object):
    pass
