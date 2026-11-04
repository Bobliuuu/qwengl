"""Backends run a list of (kernel, input vector) pairs and return the output vectors."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


class NumpyBackend:
    """Golden semantics (integer relu/clamp).  Used as the oracle and for fast development."""
    name = "numpy"

    def __init__(self):
        self.runs = defaultdict(int)

    def run_many(self, items):
        groups = defaultdict(list)
        for idx, (k, x) in enumerate(items):
            groups[k].append(idx)
        out = [None] * len(items)
        for k, idxs in groups.items():
            Y = k.run_np(np.stack([items[i][1] for i in idxs]))
            self.runs[k.name.rstrip("0123456789")] += len(idxs)
            for i, y in zip(idxs, Y):
                out[i] = y
        return out
