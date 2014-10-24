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
    # # \param polygons polygons to draw
    # #
    def write_Polygons(self, out, polygons):

        for idx in range(polygons.Np):
            # center the polygon on the view position
            verts = numpy.flipud(polygons.polygon.vertices.copy()) - self.view_pos
            points = " ".join("{point[0]},{point[1]}".format(point=p) for p in verts)
            color = 100.0*polygons.colors[idx*3*(polygons.Nt + polygons.Nto)].copy()
            # math for the outline
            # find a characteristic length by which to scale
            L = polygons.polygon.rmax
            sf = (L / (L + polygons.outline.width))
            # move the polygon back to the view_pos
            scale_trans = numpy.array(-self.view_pos-(self.view_pos/sf), dtype=numpy.float32)
            # get the position of the polygon
            pos = (polygons.positions[idx*3*(polygons.Nt + polygons.Nto)].copy())
            pos[0] += 2.0*self.view_pos[0]
            pos[1] = 2.0*self.view_pos[1] - pos[1]
            angle = -180.0 * polygons.orientations[idx*3*(polygons.Nt + polygons.Nto)].copy()[0] / numpy.pi
            vp = numpy.array(-self.view_pos, dtype=numpy.float32)
            # pos[1] = self.height - pos[1];
            out.write('<polygon points="{points}" '
                      'fill="rgb({col[0]}%,{col[1]}%,{col[2]}%)" '
                      'fill-opacity="{col[3]}" stroke-width="{outline}" stroke="rgb(0%,0%,0%)" '
                      'transform="scale({scale}) translate({scale_trans[0]}, {scale_trans[1]}) translate({gp[0]},{gp[1]}) rotate({angle} {vp[0]} {vp[1]})" />\n'.format(points=points, col=color, outline=polygons.outline.width, vp=vp, scale_trans=scale_trans, scale=sf, angle=angle, gp=pos));

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
