from __future__ import division, print_function
import math
import numpy

## \package freud.viz.base
#
# Base classes for core viz functionality
#

_prim_id = 0;

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

        for light in self.lights:
            light.setFrame(frame);

        for group in self.groups:
            group.setFrame(frame);

    ## Get the number of frames in the scene's animation
    # \returns The maximum number of frames in any constituent camera, light, or group
    def getNumFrames(self):
        num_frames = self.camera.getNumFrames();

        for light in self.lights:
            num_frames = max(num_frames, light.getNumFrames());

        for group in self.groups:
            num_frames = max(num_frames, group.getNumFrames());

        return num_frames;

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
# Derived classes should call the super init which inserts a tracking identifier for use by the cache.
#
class Primitive(object):
    ## Base class inits nothing
    def __init__(self):
        global _prim_id;
        self.ident = _prim_id;
        _prim_id += 1;

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

## Group generated from a Trajectory
#
# GroupTrajectory is a Group that is closely tied with a single freud Trajectory. It selects a number of frames equal to the
# number of frames in the trajectory. It does not directly create geometry for each frame. Instead, a virtual function
# buildPrimitives() is provided for subclasses to implement. buildPrimitives() is called whenever it is needed to build
# primitives for a given frame index in the trajectory. This way, user code can generate whatever geometry it wishes to.
#
# \note buildPrimitives() output may be cached, so do not assume that a call to buildPrimitives() indicates a frame
#       change
#
class GroupTrajectory(Group):
    ## Initialize
    # \param primitives List of primitives (of type Primitive or inherited) to include in the group
    def __init__(self, trajectory):
        # default to no primitives
        self.trajectory = trajectory;
        self.primitives = [];

    ## Build primitives for the given frame
    # \param frame Frame index
    # \returns a list of primitives
    # \note derived classes should reference self.trajectory and build the corresponding list of primitives
    def buildPrimitives(self, frame):
        pass;

    ## Sets the current animation frame
    # \param frame Animation frame index to set
    # The base class group does nothing
    def setFrame(self, frame):
        frame = frame % len(self.trajectory);
        self.primitives = self.buildPrimitives(frame);

    ## Get the number of animation frames
    # \returns Base class group only has 1 static frame
    def getNumFrames(self):
        return len(self.trajectory);

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
# **TODO** convert all methods over to properties, and see how doxygen does at documenting them
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
    # \param resolution height of the viewable plane in display pixels
    #
    def __init__(self, position, look_at, up, aspect, vfov=math.pi/4, height=None, resolution=1080):
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
        self.resolution = resolution;

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

    ## Get the number of frames
    def getNumFrames(self):
        return 1;

    ## Set the current frame
    def setFrame(self, frame):
        pass;

    ## 2D orthographic camera matrix
    # \returns A 4x4 numpy array with the camera matrix
    #
    # When used in 2D rendering, only the following fields are used to generate the camera matrix (all others are
    # ignored).
    #   - position (x,y - z is ignored)
    #   - aspect
    #   - height
    #
    # https://en.wikipedia.org/wiki/Orthographic_projection_(geometry)
    @property
    def ortho_2d_matrix(self):
        l = self.position[0] - self.getWidth()/2;
        r = self.position[0] + self.getWidth()/2;
        b = self.position[1] - self.getHeight()/2;
        t = self.position[1] + self.getHeight()/2;
        n = -1;
        f = 1;

        mat = numpy.zeros(shape=(4,4), dtype=numpy.float32);
        # numpy matrices are row-major, so we index them [r,c]
        mat[0,0] = 2/(r-l);
        mat[1,1] = 2/(t-b);
        mat[2,2] = -2/(f-n);

        mat[0,3] = -(r+l)/(r-l);
        mat[1,3] = -(t+b)/(t-b);
        mat[2,3] = -(f+n)/(f-n);
        mat[3,3] = 1;

        return mat;

    ## Pixel size
    # \returns The size of a pixel (in distance units) at the view plane
    @property
    def pixel_size(self):
        return self.getHeight() / self.resolution;

## Specify the location and properties of a light in the scene
class Light(object):
    ## Get the number of frames
    def getNumFrames(self):
        return 1;

    ## Set the current frame
    def setFrame(self, frame):
        pass;

## Specify material properties applied to a Primitive
class Material(object):
    pass
