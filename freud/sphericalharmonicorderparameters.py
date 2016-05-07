import warnings
warnings.warn("sphericalharmonicorderparameters has been moved to order. \
 This module is deprecated and will be deleted in the next version",
 PendingDeprecationWarning)
from ._freud import LocalQl
from ._freud import LocalQlNear
from ._freud import LocalWl
from ._freud import LocalWlNear
