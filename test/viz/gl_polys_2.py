from __future__ import division, print_function
import math
import random

from freud import viz, qt

if __name__ == '__main__':
    my_poly = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
    
    p = [[0, 0]]
    a = [math.pi / 4.0]
    c = [[1.0, 0.0, 0.0, 1.0]]

    polys = viz.primitive.RepeatedPolygons(positions=p, angles=a, polygon=my_poly, colors=c);
    
    group = viz.base.Group(primitives=[polys]);
    cam = viz.base.Camera(position=(0,0,1), look_at=(0,0,0), up=(0,1,0), aspect=4/3, height=6);
    scene = viz.base.Scene(camera=cam, groups=[group]);
    
    qt.init_app();
    w = viz.rt.GLWidget(scene)
    w.show()

    qt.run();
