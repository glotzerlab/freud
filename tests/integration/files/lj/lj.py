# Copyright (c) 2010-2024 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import hoomd
from hoomd import md

hoomd.context.initialize()

# Create a 10x10x10 simple cubic lattice of particles with type name A
hoomd.init.create_lattice(unitcell=hoomd.lattice.sc(a=2.0, type_name="A"), n=10)

# Specify Lennard-Jones interactions between particle pairs
nl = md.nlist.cell()
lj = md.pair.lj(r_cut=3.0, nlist=nl)
lj.pair_coeff.set("A", "A", epsilon=1.0, sigma=1.0)

# Integrate at constant temperature
md.integrate.mode_standard(dt=0.005)
hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=1.2, seed=4)

hoomd.run(10e3)

hoomd.dump.dcd("lj.dcd", period=1000, overwrite=True)
hoomd.dump.gsd("lj.gsd", period=1000, group=hoomd.group.all(), overwrite=True)

# Run for 10,000 time steps
hoomd.run(10e3)
