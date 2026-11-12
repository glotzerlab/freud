# Copyright (c) 2010-2026 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
import numpy.typing as npt

import freud
import freud.plot


def check_input_types(box: freud.Box, points: npt.NDArray[np.float32]) -> None:
    """Type-check supported dictionary, sequence, and NumPy array inputs."""
    system = (box, points)
    query = freud.AABBQuery(box, points)
    radius: dict[str, float] = {"r_max": 2.0}
    nearest: dict[str, int] = {"num_neighbors": 4}
    mixed: dict[str, str | float] = {"mode": "ball", "r_max": 2.0}
    optional: dict[str, float | None] = {"r_max": 2.0, "r_min": None}
    query.query(points, radius)
    query.query(points, nearest)
    query.query(points, mixed)
    query.query(points, optional)
    freud.density.RDF(10, 2.0).compute(system, neighbors=radius)
    freud.cluster.Cluster().compute(system, neighbors=radius)
    freud.order.Hexatic().compute(system, neighbors=nearest)
    freud.environment.LocalDescriptors(3).compute(system, neighbors=nearest)
    freud.interface.Interface().compute(system, points, neighbors=radius)
    freud.pmft.PMFTXY(2, 2, 4).compute(system, np.zeros(len(points)), neighbors=radius)
    freud.locality.FilterSANN().compute(system, neighbors=radius)
    freud.locality.FilterRAD().compute(system, neighbors=radius)

    params = box.to_dict()
    freud.Box.from_box(params)
    lengths: dict[str, float] = {"Lx": 10.0, "Ly": 10.0, "Lz": 10.0}
    freud.Box.from_box(lengths)
    freud.AABBQuery(lengths, points)
    box.L = box.L
    box.L = np.array([8.0, 9.0, 10.0])
    freud.environment.BondOrder([10, 10])
    freud.order.Steinhardt((4, 6))

    values = np.arange(3)
    freud.plot.line_plot(values, values)
    freud.plot.bar_plot(values, values)
    freud.plot.bar_plot(["A", "B", "C"], values)
    freud.plot.clusters_plot(values, values)
    freud.plot.histogram_plot(values)
