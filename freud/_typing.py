# Copyright (c) 2010-2026 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

ArrayLike: TypeAlias = npt.ArrayLike
ShapeLike: TypeAlias = Sequence[int | None]
ScalarLike: TypeAlias = int | float | np.integer | np.floating
