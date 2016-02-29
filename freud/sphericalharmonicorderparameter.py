import warnings
warnings.simplefilter("always", DeprecationWarning)
warnings.warn("sphericalharmonicorderparameters have been moved to order. \
 This module is deprecated and will be deleted in the next version",
 DeprecationWarning)
from ._freud import LocalQl
from ._freud import LocalQlNear
from ._freud import LocalWl
from ._freud import LocalWlNear
