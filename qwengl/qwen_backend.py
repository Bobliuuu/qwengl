"""Run kernels inside a real Qwen3.5 (qwenTPU v2: one prompt = one kernel; the model emits every output token).

Each run is one qwenTPU v2 program: the prompt holds the kernel's weights (CELL tokens) and the input lanes (ACT tokens);
the model then emits READ/ACT tokens for every layer and the OUT tokens of the last one.  We run the model's argmax over
the whole trace in one batched forward pass (teacher forcing with the *expected* continuation) and require that its
prediction at every position equals the expected token: for a deterministic argmax that is exactly what free-running
generation would emit.  The numbers handed to the next pipeline stage are decoded from the MODEL's predicted tokens.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np


class QwenBackend:
    name = "qwen"

    def __init__(self, qwen_dir: str, device: str = "cuda", batch: int = 24, readout: str = "code", graph: bool = False, free_run: bool = False):
        import torch
        from qwentpu import circuit as C
        from qwentpu.merge import build_merged, instantiate
        from qwentpu.sequencer2 import ProgramMachine
        self.torch, self.C, self.device, self.batch, self.free_run = torch, C, device, batch, free_run
        self.seq = ProgramMachine(2, 16, 5, 5).install()
        tc, sd, self.v0, lay, g = build_merged(qwen_dir, mute_qwen=False)
        self.model = instantiate(tc, sd, device)
        from qwentpu.driver import ChipPath, GraphedTrace
        self.chip = (GraphedTrace if graph else ChipPath)(self.model, self.v0, readout=readout)
        self.n_layers_total = tc["num_hidden_layers"]
        self.stats = dict(runs=0, unique_runs=0, tokens=0, positions=0, mismatches=0, seconds=0.0)
        self._pre = {}
        self.cache = {}                        # (kernel, x) -> output, identical runs are identical programs
        self.runs = defaultdict(int)

    # ---- prompt construction (weights are the expensive part: cache per kernel, patch the 16 input tokens) ----
    def _prompt(self, k, x):
        C, seq = self.C, self.seq
        if k not in self._pre:
            self._pre[k] = seq.program_prompt([W for W in k.Ws], np.zeros(k.n_in, dtype=int))
        p = list(self._pre[k])
        s, n = seq.depth - len(k.Ws), seq.width
        xs = list(int(v) for v in x) + [0] * (n - 1 - len(x)) + [1]
        p[-n:] = [C.tok_act(s * n + i, xs[i]) for i in range(n)]
        return p

    def _generate(self, prompts):
        """True autoregressive run, host = argmax + append until every row emits HALT.  The expected continuation is NOT fed."""
        torch, seq = self.torch, self.seq
        from transformers import DynamicCache
        past = DynamicCache(config=self.model.config)
        ids = torch.tensor([[self.v0 + t for t in p] for p in prompts], device=self.device)
        with torch.no_grad():
            nxt = self.chip.tokens(ids, past)[:, -1]
            outs = [nxt.cpu()]
            for _ in range(4096):
                if all(int(o[-1]) == seq.halt for o in [torch.stack(outs).T[b] for b in range(len(prompts))]): break
                nxt = self.chip.tokens((nxt + self.v0)[:, None], past)[:, -1]
                outs.append(nxt.cpu())
        T = torch.stack(outs).T.numpy()
        res = []
        for row in T:
            row = [int(t) for t in row]
            res.append(row[: row.index(seq.halt) + 1])
        return res

    def run_many(self, items):
        torch, seq = self.torch, self.seq
        out = [None] * len(items)
        todo = {}
        for i, (k, x) in enumerate(items):
            key = (k, np.asarray(x, dtype=np.int64).tobytes())
            self.stats["runs"] += 1
            self.runs[k.name.rstrip("0123456789")] += 1
            if key in self.cache:
                out[i] = self.cache[key]
            else:
                todo.setdefault(key, []).append(i)
        by_depth = defaultdict(list)
        for key, idxs in todo.items():
            k = items[idxs[0]][0]
            by_depth[len(k.Ws)].append((key, idxs))
        for depth, lst in by_depth.items():
            for lo in range(0, len(lst), self.batch):
                chunk = lst[lo: lo + self.batch]
                seqs, exps, plens = [], [], []
                for (key, idxs) in chunk:
                    k, x = items[idxs[0]]
                    p = self._prompt(k, x)
                    em = seq.expected(p)
                    seqs.append(p + em); exps.append(em + [seq.halt]); plens.append(len(p))
                if self.free_run:
                    t0 = time.time()
                    gen = self._generate([s[:l] for s, l in zip(seqs, plens)])
                    self.stats["seconds"] += time.time() - t0
                    self.stats["tokens"] += sum(plens) + sum(len(g) for g in gen)
                    for b, (key, idxs) in enumerate(chunk):
                        k = items[idxs[0]][0]
                        got = gen[b]
                        self.stats["positions"] += len(got)
                        self.stats["mismatches"] += int(got != exps[b])
                        y = np.array(seq.program_logits(got[:-1], k.n_out), dtype=np.int64)
                        self.stats["unique_runs"] += 1
                        self.cache[key] = y
                        for i in idxs:
                            out[i] = y
                    continue
                ids = torch.tensor([[self.v0 + t for t in s] for s in seqs], device=self.device)
                if self.device.startswith("cuda"): torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    pred = self.chip.tokens(ids, None).cpu().numpy() - 0
                if self.device.startswith("cuda"): torch.cuda.synchronize()
                self.stats["seconds"] += time.time() - t0
                self.stats["tokens"] += ids.numel()
                for b, (key, idxs) in enumerate(chunk):
                    k = items[idxs[0]][0]
                    got = [int(t) for t in pred[b, plens[b] - 1:]]
                    self.stats["positions"] += len(got)
                    bad = sum(g != e for g, e in zip(got, exps[b]))
                    self.stats["mismatches"] += bad
                    y = np.array(seq.program_logits(got[:-1], k.n_out), dtype=np.int64)   # decoded from the MODEL's tokens
                    self.stats["unique_runs"] += 1
                    self.cache[key] = y
                    for i in idxs:
                        out[i] = y
        return out
