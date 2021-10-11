import pytest

import freud


def helper_partial_structure_factor_arguments(sf):
    box, positions = freud.data.UnitCell.fcc().generate_system(4)
    with pytest.raises(ValueError):
        sf.compute((box, positions), query_points=positions)
    with pytest.raises(ValueError):
        sf.compute((box, positions), N_total=len(positions))
