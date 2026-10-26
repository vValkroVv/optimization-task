"""The reduce-group vocabulary really mirrors the mesh-axis one.

``ReduceGroupName``'s docstring promises it stays in sync with
``MeshAxisName`` and covers every alias the reducer resolves — this is that
test. A mesh axis missing here would make a valid axis unusable in a metric
config; an alias resolving outside the vocabulary would pass config
validation and then miss the mesh mid-train.
"""

from torchtitan.callbacks.metrics.reducers import _GROUP_ALIASES
from torchtitan.config.schema.callbacks.metrics.sample import ReduceGroupName
from torchtitan.config.schema.protocols.types import MeshAxisName


def test_every_mesh_axis_is_a_reduce_group():
    assert {axis.value for axis in MeshAxisName} <= {
        group.value for group in ReduceGroupName
    }


def test_every_alias_and_its_target_are_in_the_vocabulary():
    names = {group.value for group in ReduceGroupName}
    assert set(_GROUP_ALIASES) <= names
    assert set(_GROUP_ALIASES.values()) <= names
