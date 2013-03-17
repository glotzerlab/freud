from __future__ import division, print_function
import numpy
import math

## \package freud.viz.gle
#
# GLE output for freud.viz
#

## WriteGLE writes scenes (2D) to GLE files
#
# Instantiating a WriteGLE enables settings. You can then call write() as many times as you want to write GLE files.
#
# WriteGLE uses the visitor pattern to handle output methods for different primitives. 
# The method used is described here: http://peter-hoffmann.com/2010/extrinsic-visitor-pattern-python-inheritance.html
#
# write can be used to write a Scene which will then recursively call write for all scene elements. Currently, this
# Scene write sets up scaling so that the scene fills the entire view. With some small tweaks, it should be possible
# to modify this in a way to enable multiple scenes in a single GLE file. Such modifications are left for a later date.
#
# TODO: once camera matrices are in, modify this to use them
class WriteGLE:
    ## Initialize a WriteGLE
    # \param width_cm Width of the output figure in cm
    # \note Height is determined from the aspect ratio
    def __init__(self, width_cm=8.0):
        self.width_cm = width_cm;

    ## Writes a Primitive out to the GLE file
    # \param out Output stream
    # \param prim Primitive to write
    #
    def write_Primitive(self, out, prim):
        raise RuntimeError('WriteGLE encountered an unknown primitive type');

    ## Write GLE output to a stream
    # \param out Output stream
    # \param scene Scene to write
    #
    def write_Scene(self, out, scene):
        # compute width and height
        width_sim = scene.camera.getWidth();
        height_sim = scene.camera.getHeight();
        self.view_pos = scene.camera.position[0:2];
        self.width_height = numpy.array([width_sim, height_sim], dtype=numpy.float32);
        self.sim_to_cm = self.width_cm / width_sim;
        self.height_cm = height_sim * self.sim_to_cm;
        
        out.write('size {0} {1}\n'.format(self.width_cm, self.height_cm));
        
        # loop through the render primitives and write out each one
        for i,group in enumerate(scene.groups):
            out.write('\n!Group {0}\n'.format(i));
            for j,primitive in enumerate(group.primitives):
                out.write('\n!Group {0}, primitive {1}\n'.format(i, j));
                self.write(out, primitive)

    ## Write out disks
    # \param out Output stream
    # \param disks Disks to write
    #
    def write_Disks(self, out, disks):
        out.write('set lwidth 0\n');
        
        for position,diameter,color in zip(disks.positions, disks.diameters, disks.colors):
            # map the position into the view space
            position = (position - self.view_pos + self.width_height/2.0) * self.sim_to_cm;
            diameter = diameter * self.sim_to_cm;
            
            # don't write out disks that are off the edge
            if position[0]+diameter/2 < 0 or position[0]-diameter/2 > self.width_cm:
                continue;
            if position[1]+diameter/2 < 0 or position[1]-diameter/2 > self.height_cm:
                continue;
            
            # This is how to draw a circle with an outline in GLE such that the circle and outline do not overlap
            # and the edge of the outline is entirely within the circle (a is the line width)
            # set color rgba(0,0,0,0.5)
            # begin path fill rgba(1,0,0,0.5)
            #    arc 2.0-a 0 360
            # end path
            # circle 2.0-a/2
            
            # compute outline width
            a = disks.outline * self.sim_to_cm;
            
            out.write('amove {0} {1}\n'.format(*position));

            out.write('begin path fill rgba({0}, {1}, {2}, {3})\n'.format(*color));
            out.write('    arc {0} 0 360\n'.format(diameter/2-a))
            out.write('end path\n');
            
            out.write('set lwidth {0}\n'.format(a));
            # for the outline color, chose black and the same alpha as the fill color
            out.write('set color rgba(0, 0, 0, {0})\n'.format(color[3]));
            out.write('circle {0}\n'.format(diameter/2-a/2));

    ## Write out repeated polygons
    # \param out Output stream
    # \param polygons Disks to polygons
    #
    def write_RepeatedPolygons(self, out, polygons):
        # initial implementation use stroke and fill on a path to draw polygons. This has 2 issues : 1) The stroke
        # goes outside the polygon and 2) The stroke overlaps the fill (looks bad with alpha rendering).
        # An improved implementation would need to compute an inner set of vertices for the polygon and use two paths,
        # one to fill and one to draw the edges, like so
        # sub draw_box
        # begin path fill rgba(0,0,0,1.0)
        #     rline 1 0
        #     rline 0 1
        #     rline -1 0
        #     rline 0 -1
        #     rmove w w
        #     rline 0 1-w*2
        #     rline 1-w*2 0
        #     rline 0 -(1-w*2)
        #     rline -(1-w*2) 0
        # end path
        # rmove w w
        # begin path fill rgba(1, 0, 0, 1.0)
        #     rline 1.0-w*2 0
        #     rline 0 1.0-w*2
        #     rline -(1-w*2) 0
        #     rline 0 -(1-w*2)
        # end path
        # end sub

        out.write('set lwidth {0}\n'.format(polygons.outline*self.sim_to_cm));
        
        # first, generate a string that writes the whole polygon
        fill_str = "amove {0} {1}\n".format(*polygons.polygon[0][:]*self.sim_to_cm);
        for vert in polygons.polygon[1:]:
            vert_cm = vert * self.sim_to_cm;
            fill_str += "aline {0} {1}\n".format(*vert_cm)
        fill_str += "aline {0} {1}\n".format(*polygons.polygon[0][:]*self.sim_to_cm);
        
        # compute the polygon's radius
        radius = 0;
        for vert in polygons.polygon:
            r = math.sqrt(numpy.dot(vert, vert));
            radius = max(radius, r);
        
        for position,angle,color in zip(polygons.positions, polygons.angles, polygons.colors):
            # map the position into the view space
            position = (position - self.view_pos + self.width_height/2.0) * self.sim_to_cm;
            
            # don't write out polygons that are off the edge
            if position[0]+radius < 0 or position[0]-radius > self.width_cm:
                continue;
            if position[1]+radius < 0 or position[1]-radius > self.height_cm:
                continue;
            
            out.write('begin translate {0} {1}\n'.format(*position));
            out.write('begin rotate {0}\n'.format(180*angle/math.pi));

            # for the outline color, chose black and the same alpha as the fill color
            out.write('set color rgba(0, 0, 0, {0})\n'.format(color[3]));

            out.write('begin path stroke fill rgba({0}, {1}, {2}, {3})\n'.format(*color));
            out.write(fill_str);
            out.write('closepath\n');
            out.write('end path\n');

            out.write('end rotate\n');
            out.write('end translate\n');
            
    ## Write a viz element to a GLE stream
    # \param out Output stream
    # \param obj Object to write
    #
    def write(self, out, obj):
        meth = None;
        for cls in obj.__class__.__mro__:
            meth_name = 'write_'+cls.__name__;
            meth = getattr(self, meth_name, None);
            if meth is not None:
                break;

        if meth is None:
            raise RuntimeError('WriteGLE does not know how to write a {0}'.format(obj.__class__.__name__));
        return meth(out, obj);
