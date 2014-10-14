import os
import math
import random
import time
from hoomd_script import *

# parameters
n_particles = 6001
phi_p_init = 0.5

# derived parameters
L_init = math.sqrt(math.pi * 0.5**2 * n_particles / phi_p_init)
pw = int(math.sqrt(n_particles))+1
a = L_init / float(pw)
lo = - pw*a / 2.0;

# initialize the system
sysdef = init.create_empty(N=n_particles, box=(L_init, L_init, 1), n_particle_types=2)
sysdef.dimensions = 2

# temperature ramp
T = variant.linear_interp(points=[(0, 5.0),  (25000, 5.0), (45000, 0.5)]);

# place initial particles randomly
for p in sysdef.particles:
    (i, j) = (p.tag % pw, p.tag/pw % pw)
    if random.random() > 0.8:
        p.type='A';
    else:
        p.type='B'
    p.position = (lo + i*a + a/2, lo + j*a + a/2, 0)

# setup lj potential
lj = pair.lj(r_cut=2.5)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
lj.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0)
lj.pair_coeff.set('A', 'B', epsilon=1.0, sigma=1.0,
                            r_cut=2.0**(1.0/6.0))

all = group.all()
integrate.mode_standard(dt=0.005)
nve = integrate.nve(all, limit=0.05)
run(5000)

# switch over to nvt integration and equilibrate
nve.disable()
integrate.nvt(group=all, T=T, tau=0.1)

# save the relaxed state
xml = dump.xml();
xml.set_params(position=True, diameter=True, type=True)
xml.write(filename="start.xml")

# dump the trajectory
dump.dcd(filename="dump.dcd", period=1000, overwrite=True)

run(200e3)
