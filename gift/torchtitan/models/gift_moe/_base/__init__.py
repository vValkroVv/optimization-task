"""Shared GiftMoE base implementation, reused across versions (``gift_3_5``,
``gift_4``).

Submodules are imported directly (e.g. ``from gift_moe._base.model import
GiftMoEModel``) rather than re-exported here, to avoid a circular import with
the parent package during initialization.
"""
