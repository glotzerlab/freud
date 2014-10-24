from __future__ import division, print_function
import sys
import numpy
import math

## \package freud.viz.export.svg
#
# SVG output for freud.viz
#

## WriteSVG writes scenes (2D) to SVG files
#
# Instantiating a WriteSVG enables settings. You can then call write() as many times as you want to write SVG files.
#
# \internal
# WriteSVG uses the visitor pattern to handle output methods for different primitives.
# The method used is described here: http://peter-hoffmann.com/2010/extrinsic-visitor-pattern-python-inheritance.html
#
# write can be used to write a Scene which will then recursively call write for all scene elements. Currently, this
# Scene write sets up scaling so that the scene fills the entire view. With some small tweaks, it should be possible
# to modify this in a way to enable multiple scenes in a sinsvg SVG file. Such modifications are left for a later date.
#
# TODO: once camera matrices are in, modify this to use them
class WriteSVG(object):
    ## Initialize a WriteSVG
    # \param width_cm Width of the output figure in cm
    # \note Height is determined from the aspect ratio
    def __init__(self, width_cm=8.0):
        self.width_cm = width_cm;
        self.file_count = 0;

    ## \internal
    # \brief Writes a Primitive out to the SVG file
    # \param out Output stream
    # \param prim Primitive to write
    #
    def write_Primitive(self, out, prim):
        raise RuntimeError('WriteSVG encountered an unknown primitive type');

    ## \internal
    # \brief Write SVG output to a stream
    # \param out Output stream
    # \param scene Scene to write
    #
    def write_Scene(self, out, scene):
        # compute width and height
        width_sim = scene.camera.getWidth();
        height_sim = scene.camera.getHeight();
        self.width_height = numpy.array([width_sim, height_sim], dtype=numpy.float32);
        (self.width, self.height) = self.width_height;
        self.view_pos = scene.camera.position[0:2] - self.width_height/2;
        self.sim_to_cm = self.width_cm / width_sim;
        self.view_pos_to_cm = self.view_pos*self.sim_to_cm;
        self.height_cm = height_sim * self.sim_to_cm;

        out.write('<svg width="{}cm" height="{}cm" viewBox="{} {} {} {}" '
                  'xmlns="http://www.w3.org/2000/svg">\n'.format(
            self.width_cm, self.height_cm, 0, 0, width_sim, height_sim));

        # loop through the render primitives and write out each one
        for i,group in enumerate(scene.groups):
            out.write('<!-- Group {0} -->\n'.format(i));
            for j,primitive in enumerate(group.primitives):
                out.write('<!-- Group {0}, primitive {1} -->\n'.format(i, j));
                self.write(out, primitive)

        out.write('</svg>\n')

    ## \internal
    # \brief Write out arrows
    # \param out Output stream
    # \param arrows Arrows to write
    #
    def write_Arrows(self, out, arrows):
        vertices = arrows.vertices.copy();

        # Merge the three triangles composing the arrow into a single series
        # vertices was vertices :: Ntri, 3, 2
        # reshape to vertices :: 3 (stem0/stem1/tip), N, 3 (index in triangle), 2
        vertices = vertices.reshape((3, -1, 3, 2))
        # transpose to vertices :: N, 9 (index in triangle), 2
        vertices = numpy.transpose(vertices, (1, 0, 2, 3)).reshape((-1, 9, 2))
        # take the appropriate sequence to trace out an arrow, vertices :: (N, 7, 2)
        vertices = vertices[:, [1, 0, 5, 8, 7, 6, 2], :]

        # gather indices for any arrow which has a vertex inside the drawn box
        insideIndices = numpy.all([
            numpy.any(vertices[:, :, 0] > self.view_pos[0], axis=1),
            numpy.any(vertices[:, :, 1] > self.view_pos[1], axis=1),
            numpy.any(vertices[:, :, 0] < self.view_pos[0] + self.width, axis=1),
            numpy.any(vertices[:, :, 1] < self.view_pos[1] + self.height, axis=1)], axis=0);

        vertices = vertices[insideIndices] - self.view_pos;
        # vertically flip vertices
        vertices[:, :, 1] = self.height - vertices[:, :, 1];
        # grab the color from the first triangle
        colors = arrows.arrColors[insideIndices];
        colors[:, :3] *= 100;

        for (verts, color) in zip(vertices, colors):
            d = ('M {},{} '.format(verts[0][0], verts[0][1]) +
                 ' '.join('L {v[0]} {v[1]}'.format(v=v) for v in verts[1:]) +
                 'Z')
            out.write('<path d="{d}" fill="rgb({col[0]}%,{col[1]}%,{col[2]}%)" '
                      'fill-opacity="{col[3]}" />\n'.format(d=d, col=color));

    ## \internal
    # \brief Write out disks
    # \param out Output stream
    # \param disks Disks to write
    #
    def write_Disks(self, out, disks):
        # decrease diameters by the width of the outline so we don't
        # have to do any clipping. Also scale everything into physical
        # dimensions.
        outline = disks.outline;
        diameters = disks.diameters - disks.outline;
        positions = disks.positions.copy();

        # gather indices which are inside the drawn box
        halfDiam = diameters/2;
        insideIndices = numpy.all([
            positions[:, 0] + halfDiam > self.view_pos[0],
            positions[:, 1] + halfDiam > self.view_pos[1],
            positions[:, 0] - halfDiam < self.view_pos[0] + self.width,
            positions[:, 1] - halfDiam < self.view_pos[1] + self.height], axis=0);

        radii = diameters[insideIndices]/2;
        positions = positions[insideIndices] - self.view_pos;
        # vertically flip positions
        positions[:, 1] = self.height - positions[:, 1];
        # convert rgb colors to percent
        colors = disks.colors[insideIndices];
        colors[:, :3] *= 100;

        for (position, radius, color) in zip(positions, radii, colors):
            out.write('<circle cx="{pos[0]}" cy="{pos[1]}" r="{radius}" '
                      'stroke="#000000" stroke-width="{outline}" '
                      'fill="rgb({col[0]}%,{col[1]}%,{col[2]}%)" '
                      'fill-opacity="{col[3]}" stroke-opacity="{col[3]}"/>\n'.format(
                          pos=position, radius=radius, outline=outline, col=color));

    ## \internal
    # \brief Write out triangles
    # \param out Output stream
    # \param triangles Triangles to write
    #
    def write_Triangles(self, out, triangles):
        vertices = triangles.vertices.copy();

        # gather indices which are inside the drawn box
        insideIndices = numpy.all([
            numpy.any(vertices[:, :, 0] > self.view_pos[0], axis=1),
            numpy.any(vertices[:, :, 1] > self.view_pos[1], axis=1),
            numpy.any(vertices[:, :, 0] < self.view_pos[0] + self.width, axis=1),
            numpy.any(vertices[:, :, 1] < self.view_pos[1] + self.height, axis=1)], axis=0);

        vertices = vertices[insideIndices] - self.view_pos;
        # vertically flip vertices
        vertices[:, :, 1] = self.height - vertices[:, :, 1];
        # grab the color from the first vertex (currently triangles
        # only supports a single color per triangle)
        colors = triangles.colors[insideIndices][:, 0];
        colors[:, :3] *= 100;

        for (verts, color) in zip(vertices, colors):
            d = ('M {verts[0][0]},{verts[0][1]} L {verts[1][0]},{verts[1][1]} '
                'L {verts[2][0]},{verts[2][1]} Z').format(verts=verts);
            out.write('<path d="{d}" fill="rgb({col[0]}%,{col[1]}%,{col[2]}%)" '
                      'fill-opacity="{col[3]}" />'.format(d=d, col=color));

    # ## \internal
    # # \brief Write out repeated polygons
    # # \param out Output stream
    # # \param polygons Disks to polygons
    # #
    # I think I need to look at the triangle code maybe...
    def write_Polygons(self, out, polygons):

    # copy arrows, because this isn't correct
    # need to do rotation math, etc. here as it was all moved to the gpu in gl.

    #     # TODO: Update to do proper edge fills
    #     # initial implementation use stroke and fill on a path to draw polygons. This has 2 issues : 1) The stroke
    #     # goes outside the polygon and 2) The stroke overlaps the fill (looks bad with alpha rendering).
    #     # An improved implementation would need to compute an inner set of vertices for the polygon and use two paths,
    #     # one to fill and one to draw the edges, like so
    #     # sub draw_box
    #     # begin path fill rgba(0,0,0,1.0)
    #     #     rline 1 0
    #     #     rline 0 1
    #     #     rline -1 0
    #     #     rline 0 -1
    #     #     rmove w w
    #     #     rline 0 1-w*2
    #     #     rline 1-w*2 0
    #     #     rline 0 -(1-w*2)
    #     #     rline -(1-w*2) 0
    #     # end path
    #     # rmove w w
    #     # begin path fill rgba(1, 0, 0, 1.0)
    #     #     rline 1.0-w*2 0
    #     #     rline 0 1.0-w*2
    #     #     rline -(1-w*2) 0
    #     #     rline 0 -(1-w*2)
    #     # end path
    #     # end sub


        # out.write('set lwidth {0}\n'.format(polygons.outline*self.sim_to_cm));
        # out.write('set lwidth {0}\n'.format(0.1*self.sim_to_cm));

        # first, generate a string that writes the whole polygon
        # fill_str = "amove {0} {1}\n".format(*polygons.image[0,:]*self.sim_to_cm);
        # for vert in polygons.image[1:]:
        #     vert_cm = vert * self.sim_to_cm;
        #     fill_str += "aline {0} {1}\n".format(*vert_cm)
        # fill_str += "aline {0} {1}\n".format(*polygons.image[0,:]*self.sim_to_cm);

        # compute the polygon's radius
        # radius = 1.0;
        # this won't work as this is already triangulated
        # I will need to change this...maybe
        # for vert in polygons.image:
        #     r = math.sqrt(numpy.dot(vert, vert));
        #     radius = max(radius, r);

        # for idx, (position,orientation,color) in enumerate(zip(polygons.positions, polygons.orientations, polygons.colors)):
        #     print(len(polygons.positions))
        #     print(len(polygons.orientations))
        #     print(len(polygons.colors))
        #     sys.exit(0)
        #     # as of now each position is a position of a vertex of a triangle
        #     # map the position into the view space
        #     # position = (position - self.view_pos + self.width_height/2.0) * self.sim_to_cm;

        #     # # don't write out polygons that are off the edge
        #     # if position[0]+radius < 0 or position[0]-radius > self.width_cm:
        #     #     continue;
        #     # if position[1]+radius < 0 or position[1]-radius > self.height_cm:
        #     #     continue;

        #     # out.write('begin translate {0} {1}\n'.format(*position));
        #     # out.write('begin rotate {0}\n'.format(180*orientation/math.pi));

        #     # # for the outline color, chose black and the same alpha as the fill color
        #     # out.write('set color rgba(0, 0, 0, {0})\n'.format(color[3]));

        #     # out.write('begin path stroke fill rgba({0}, {1}, {2}, {3})\n'.format(*color));
        #     # out.write(fill_str);
        #     # out.write('closepath\n');
        #     # out.write('end path\n');

        #     # number of polygons
        #     Np = polygons.Np
        #     # number of triangles
        #     Nt = polygons.image.shape[0];
        #     print(polygons.images.shape)
        #     print(polygons.image.shape)
        #     print(Nt)
        #     # print(Np * Nt)

        #     for tri in range(Nt):
        #         verts = numpy.empty(shape=(3,2), dtype=numpy.float32)
        #         for vertex in range(3):
        #             # figure out where it barfs
        #             upperLimit = Np*Nt*3
        #             if idx > Np:
        #                 raise RuntimeError("idx > Np; idx = {}, Np = {}".format(idx, Np))
        #             verts[vertex] = polygons.images[idx*tri*vertex].copy();

        #         # gather indices which are inside the drawn box
        #         # print(verts[:, 0])
        #         insideIndices = numpy.all([
        #             numpy.any(verts[:, 0] > self.view_pos[0], axis=0),
        #             numpy.any(verts[:, 1] > self.view_pos[1], axis=0),
        #             numpy.any(verts[:, 0] < self.view_pos[0] + self.width, axis=0),
        #             numpy.any(verts[:, 1] < self.view_pos[1] + self.height, axis=0)], axis=0);

        #         vert = verts[insideIndices] - self.view_pos;
        #         # vertically flip vertices
        #         verts[:, 1] = self.height - verts[:, 1];
        #         # grab the color from the first vertex (currently polygons
        #         # only supports a single color per triangle)
        #         # colors = polygons.colors[insideIndices][:, 0];
        #         # colors[:, :3] *= 100;

        #         # for (verts, color) in zip(vertices, colors):
        #         d = ('M {verts[0][0]},{verts[0][1]} L {verts[1][0]},{verts[1][1]} '
        #             'L {verts[2][0]},{verts[2][1]} Z').format(verts=verts);
        #         out.write('<path d="{d}" fill="rgb({col[0]}%,{col[1]}%,{col[2]}%)" '
        #                   'fill-opacity="{col[3]}" />'.format(d=d, col=color));

        # for idx, (position,color) in enumerate(zip(polygons.positions, polygons.colors)):
        Nt = polygons.image.shape[0];
        for idx in range(polygons.Np):
            for tri in range(Nt):
                verts = numpy.empty(shape=(3,2), dtype=numpy.float32)
                for vertex in range(3):
                    verts[vertex] = polygons.positions[idx*tri*vertex].copy();
                    if vertex == 2:
                        color = polygons.colors[idx*tri*vertex].copy();

                # gather indices which are inside the drawn box
                # print(verts[:, 0])
                # insideIndices = numpy.all([
                #     numpy.any(verts[:, 0] > self.view_pos[0], axis=0),
                #     numpy.any(verts[:, 1] > self.view_pos[1], axis=0),
                #     numpy.any(verts[:, 0] < self.view_pos[0] + self.width, axis=0),
                #     numpy.any(verts[:, 1] < self.view_pos[1] + self.height, axis=0)], axis=0);
                verts[:, 0] += self.width
                verts[:, 1] += self.height

                verts = verts - self.view_pos;
                # vertically flip vertices
                verts[:, 1] = self.height - verts[:, 1];
                for vertex in verts:
                    if (vertex[0] < self.view_pos[0]) or (vertex[0] > self.view_pos[0] + self.width):
                        continue
                    if (vertex[1] < self.view_pos[1]) or (vertex[1] > self.view_pos[1] + self.height):
                        continue
                # grab the color from the first vertex (currently polygons
                # only supports a single color per triangle)
                # colors = polygons.colors[insideIndices][:, 0];
                # colors[:, :3] *= 100;

                # for (verts, color) in zip(vertices, colors):
                d = ('M {verts[0][0]},{verts[0][1]} L {verts[1][0]},{verts[1][1]} '
                    'L {verts[2][0]},{verts[2][1]} Z').format(verts=verts);
                out.write('<path d="{d}" fill="rgb({col[0]}%,{col[1]}%,{col[2]}%)" '
                          'fill-opacity="{col[3]}" />\n'.format(d=d, col=color));

        #     out.write('end rotate\n');
        #     out.write('end translate\n');

    # ## \internal
    # # \brief Write out image
    # # \param out Output stream
    # # \param img Image to write
    # #
    # def write_Image(self, out, img):
    #     # map the position into the view space
    #     position = (img.position - self.view_pos + self.width_height/2.0) * self.sim_to_cm;
    #     size = img.size * self.sim_to_cm;

    #     # don't write out images that are off the edge
    #     if position[0]+size[0]/2 < 0 or position[0]-size[0]/2 > self.width_cm:
    #         return;
    #     if position[1]+size[1]/2 < 0 or position[1]-size[1]/2 > self.height_cm:
    #         return;

    #     # save the image to a file
    #     if img.filename is not None:
    #         fname = img.filename;
    #     else:
    #         fname = 'img{0}.png'.format(self.file_count);
    #         self.file_count += 1;

    #     img.save(fname);

    #     # write out the SVG code to place the image
    #     out.write('amove {0} {1}\n'.format(*position));
    #     out.write('bitmap {0} {1} {2}\n'.format(fname, size[0], size[1]));

    ## Write a viz element to a SVG stream
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
            raise RuntimeError('WriteSVG does not know how to write a {0}'.format(obj.__class__.__name__));
        return meth(out, obj);
