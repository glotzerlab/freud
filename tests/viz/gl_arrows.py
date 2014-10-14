from __future__ import division, print_function
import itertools
import math
import random

from freud import viz, qt

if __name__ == '__main__':
    positions = [[0, -5], [4, -0], [-10, -10]]

    # Make sure everything works properly when given both single
    # values and per-position values
    lengths = [1, [4, 2, 3]]
    angles = [0, [.25, -math.pi/4, 2]]
    widths = [.75, [.5, 1.75, 2]]
    color = [1,0,0,1]

    primitives = []

    for (length, angle, width) in itertools.product(lengths, angles, widths):
        arrows = viz.primitive.Arrows(positions, width, length, angle, color=color)
        primitives.append(arrows)

    group = viz.base.Group(primitives=primitives);
    cam = viz.base.Camera(position=(0,0,1), look_at=(0,0,0), up=(0,1,0), aspect=4/3, height=6);
    scene = viz.base.Scene(camera=cam, groups=[group]);

    qt.init_app();
    w = viz.rt.GLWidget(scene)
    w.show()

    qt.run();
