"""All PAM interactions with job dispatcher.

The pieces live in one class per concern; class JD combines them and is the
only name callers import. Mixins only work as part of JD.
"""

from .artifacts import JDArtifacts
from .cache_scores.publish import JDCacheScores
from .checkpoint import JDCheckpoint
from .jd_data import JDData
from .metrics import JDMetrics
from .submetrics import JDSubmetrics, SubmetricGroup
from .sync import JDSync


class JD(
    JDArtifacts,
    JDCacheScores,
    JDCheckpoint,
    JDData,
    JDMetrics,
    JDSubmetrics,
    JDSync,
):
    """The job dispatcher as PAM talks to it."""


__all__ = [
    "JD",
    "JDArtifacts",
    "JDCacheScores",
    "JDCheckpoint",
    "JDData",
    "JDMetrics",
    "JDSubmetrics",
    "JDSync",
    "SubmetricGroup",
]
