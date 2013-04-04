from __future__ import division, print_function
import random

from freud import viz
from freud.viz import window
from freud import qtmanager

# generate random disks
def gen_random_disks(N, w):
    positions = [];

    for i in range(0,N):
        pos = [random.random()*w - w/2, random.random()*w - w/2];
        positions.append(pos);
    
    return positions;

# generate oredered disks
def gen_ordered_disks(m):
    positions = [];

    for i in range(0,m):
        for j in range(0,m):
            pos = [i - m/2, j - m/2];
            positions.append(pos);
    
    return positions;

# generate random colors
def gen_random_colors(N):
    colors = [];

    for i in range(0,N):
        col = [random.random(), random.random(), random.random(), 1.0];
        colors.append(col);
    
    return colors;

# generate random diameters
def gen_random_diameters(N):
    diameters = [];

    for i in range(0,N):
        d = random.random()*0.9 + 0.1;
        diameters.append(d);
    
    return diameters;

if __name__ == '__main__':
    #p = gen_random_disks(100, 20);
    p = gen_ordered_disks(20);
    c = gen_random_colors(len(p));
    d = gen_random_diameters(len(p));
    
    disks = viz.primitive.Disks(positions=p, colors=c, diameters=d, outline=0.06);
    disks1 = viz.primitive.Disks(positions=gen_random_disks(100, 20), color=(0,0,1,1), outline=0.06);
    
    group = viz.base.Group(primitives=[disks, disks1]);
    cam = viz.base.Camera(position=(0,0,1), look_at=(0,0,0), up=(0,1,0), aspect=4/3, height=18);
    scene = viz.base.Scene(camera=cam, groups=[group]);
    
    w = window.GLWidget(scene)
    w.show()

qtmanager.runEventLoop();
