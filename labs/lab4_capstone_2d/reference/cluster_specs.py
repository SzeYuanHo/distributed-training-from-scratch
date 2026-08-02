"""Hardware profiles for the planner.

The canonical copy lives in ``common/cluster_specs.py`` because Lab 2's scaling
curve needs the same numbers and duplicating them would let the two drift. This
re-export exists so Lab 4's own code can say ``from reference import cluster_specs``
like every other lab does, and so the lab directory stays self-describing.
"""

from common.cluster_specs import (  # noqa: F401
    DEFAULT_ACHIEVABLE_FRACTION,
    PROFILES,
    achievable,
    describe,
)

__all__ = ["PROFILES", "describe", "achievable", "DEFAULT_ACHIEVABLE_FRACTION"]
