"""Utility helpers for modelens."""

import numpy as np


def ensure_array(y):
    """Convert input to numpy array if not already."""
    return np.asarray(y)


def get_model_name(model):
    """Return human-readable model name."""
    return type(model).__name__