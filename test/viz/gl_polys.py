from __future__ import division, print_function
import time
import math
import random
import numpy

from freud import viz, qt

# generate random positions
def gen_random_pos(N, w):
    positions = [];

    for i in range(0,N):
        pos = [random.random()*w - w/2, random.random()*w - w/2];
        positions.append(pos);

    return numpy.array(positions);

# generate oredered positions
def gen_ordered_pos(m):
    positions = [];

    for i in range(0,m):
        for j in range(0,m):
            pos = [i - m/2, j - m/2];
            positions.append(pos);

    return numpy.array(positions);

# generate oredered angles
def gen_ordered_angles(m):
    angles = [];

    for i in range(0,m*m):
        ang = 2*math.pi * (i / (m*m));
        angles.append(ang);

    return numpy.array(angles);

# generate random colors
def gen_random_colors(N):
    colors = [];

    for i in range(0,N):
        col = [random.random(), random.random(), random.random(), 1.0];
        colors.append(col);

    return numpy.array(colors);

if __name__ == '__main__':
    p = gen_ordered_pos(1000);
    a = gen_ordered_angles(1000);
    c = gen_random_colors(len(p));
    print("init complete")
    #triangle = [[-0.5, -0.5], [0.5, -0.5], [0, 0.5]];
    my_poly = numpy.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    # my_poly = []
    # my_poly.append([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]);
    # my_poly.append([[-0.5, -0.5], [0.5, -0.5], [0.0, 0.0], [0.5, 0.5], [-0.5, 0.5]]);
    # my_poly.append([[-0.5, -0.5], [0.0, -0.25], [0.5, -0.5], [0.5, 0.5], [0.0, 1.0], [-0.5, 0.5]]);
    # my_poly.append([[-0.5, -0.5], [0.0, -0.25], [0.5, -0.5], [0.5, 0.5], [0.0, 1.0], [-0.5, 0.5], [-1.0, 0.0]]);

    #for i in range(len(my_poly)):
    #print("{0} vertices".format(len(my_poly[i])))
    polys = viz.primitive.RepeatedPolygons(positions=p, angles=a, polygon=my_poly, colors=c, outline=0.05);
        # disks1 = viz.Disks(positions=gen_random_disks(100, 20), color=(0,0,1,1));

    group = viz.base.Group(primitives=[polys]); #, disks1]);
    cam = viz.base.Camera(position=(0,0,1), look_at=(0,0,0), up=(0,1,0), aspect=4/3, height=18);
    scene = viz.base.Scene(camera=cam, groups=[group]);

    qt.init_app();
    w = viz.rt.GLWidget(scene)
    w.show()

    qt.run();
