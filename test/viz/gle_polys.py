from __future__ import division, print_function
import math
import random

from freud import viz

# generate random positions
def gen_random_pos(N, w):
    positions = [];

    for i in range(0,N):
        pos = [random.random()*w - w/2, random.random()*w - w/2];
        positions.append(pos);
    
    return positions;

# generate oredered positions
def gen_ordered_pos(m):
    positions = [];

    for i in range(0,m):
        for j in range(0,m):
            pos = [i - m/2, j - m/2];
            positions.append(pos);
    
    return positions;

# generate oredered angles
def gen_ordered_angles(m):
    angles = [];

    for i in range(0,m*m):
        ang = 2*math.pi * (i / (m*m));
        angles.append(ang);
    
    return angles;

# generate random colors
def gen_random_colors(N):
    colors = [];

    for i in range(0,N):
        col = [random.random(), random.random(), random.random(), 1.0];
        colors.append(col);
    
    return colors;

if __name__ == '__main__':
    p = gen_ordered_pos(21);
    a = gen_ordered_angles(21);
    c = gen_random_colors(len(p));
    
    #triangle = [[-0.5, -0.5], [0.5, -0.5], [0, 0.5]];
    my_poly = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [0.0, 1.0], [-0.5, 0.5]];
    
    polys = viz.primitive.RepeatedPolygons(positions=p, angles=a, polygon=my_poly, colors=c, outline=0.05);
    # disks1 = viz.Disks(positions=gen_random_disks(100, 20), color=(0,0,1,1));
    
    group = viz.base.Group(primitives=[polys]); #, disks1]);
    cam = viz.base.Camera(position=(0,0,1), look_at=(0,0,0), up=(0,1,0), aspect=4/3, height=18);
    scene = viz.base.Scene(camera=cam, groups=[group]);
    
    writer = viz.export.gle.WriteGLE()
    writer.write(open('gle_polys.gle', 'wb'), scene);
