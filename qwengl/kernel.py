"""Kernel IR: a stack of integer layers y = clip(relu(x @ W + b), 0, AMAX).

This is exactly what one qwenTPU v2 program computes (16-wide array, 5-bit unsigned activations,
5-bit signed weights, no shift, relu + saturating clamp), so every kernel here can be
handed to ``qwentpu.sequencer2.ProgramMachine`` unchanged.  A kernel has at most DEPTH_MAX layers.
"""
from __future__ import annotations

import numpy as np

AMAX = 31            # activations / outputs: 0..31 (5 bits)
WLO, WHI = -16, 15   # weights and biases: 5-bit two's complement
LANES = 15           # usable lanes between layers (lane 15 is the model's constant-1 lane)
OUT_LANES = 16       # the last layer may use all 16 columns
DEPTH_MAX = 2        # one qwenTPU v2 program (Dmax=2, n=16, u5s5, 32x32 packed array)
CONST = 15           # value of a "constant lane" (a lane whose weights are 0 and bias is 15)


class Layer:
    """W: (n_in + 1, n_out) integer matrix, last row = bias."""

    def __init__(self, n_in: int, n_out: int):
        self.n_in, self.n_out = n_in, n_out
        self.W = np.zeros((n_in + 1, n_out), dtype=np.int64)

    def w(self, i, j, v):
        self.W[i, j] = v
        return self

    def b(self, j, v):
        self.W[-1, j] = v
        return self

    def check(self, final: bool):
        assert self.n_in <= LANES, f"layer has {self.n_in} input lanes (max {LANES})"
        assert self.n_out <= (OUT_LANES if final else LANES), f"{self.n_out} output lanes"
        assert self.W.min() >= WLO and self.W.max() <= WHI, f"weight out of 5-bit range: {self.W.min()}..{self.W.max()}"


class Kernel:
    def __init__(self, name: str, layers: list[Layer]):
        assert 1 <= len(layers) <= DEPTH_MAX, f"{name}: depth {len(layers)}"
        for i, l in enumerate(layers):
            l.check(final=(i == len(layers) - 1))
            if i:
                assert l.n_in == layers[i - 1].n_out, f"{name}: layer {i} width mismatch"
        self.name, self.layers = name, layers
        self.n_in, self.n_out = layers[0].n_in, layers[-1].n_out
        self.Ws = [l.W for l in layers]
        self._key = (name, tuple(w.tobytes() for w in self.Ws))

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, o):
        return isinstance(o, Kernel) and self._key == o._key

    def run_np(self, X: np.ndarray) -> np.ndarray:
        """Golden numpy semantics.  X: (B, n_in) ints in 0..AMAX."""
        v = np.asarray(X, dtype=np.int64)
        assert v.ndim == 2 and v.shape[1] == self.n_in and v.min() >= 0 and v.max() <= AMAX, (self.name, v.shape, v.min(), v.max())
        for W in self.Ws:
            v = np.clip(np.maximum(v @ W[:-1] + W[-1], 0), 0, AMAX)
        return v
