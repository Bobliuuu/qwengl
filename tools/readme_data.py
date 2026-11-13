#!/usr/bin/env python3
"""Derive results/derived/readme_data.json (used by README.spec.yml) from the committed Qwen run logs:
results/*_qwen_*frames.jsonl (one line per frame) and results/*_free_run_frame.json. Nothing is typed by hand."""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [("cube", "Cube, 30x30", "cube_qwen_36frames.jsonl", "cube_free_run_frame.json"),
        ("scene", "Cube + octahedron + pyramid, 30x30", "scene_qwen_36frames.jsonl", "scene_free_run_frame.json"),
        ("dungeon", "Dungeon, 60x60", "dungeon_qwen_48frames.jsonl", "dungeon_free_run_frame.json")]


def main():
    runs, frames_dungeon, free = [], [], []
    for key, label, jl, fr in RUNS:
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "results", jl)) if l.strip()]
        kr = sum(sum(v for k, v in r["runs"].items() if k not in ("rop_skipped",)) for r in rows)
        runs.append(dict(key=key, scene=label, frames=len(rows), kernel_runs=kr, tokens_m=round(sum(r["tokens"] for r in rows) / 1e6, 2), tokens=sum(r["tokens"] for r in rows),
                         positions=sum(r["positions"] for r in rows), mismatches=sum(r["mismatches"] for r in rows), dropped=sum(r["dropped"] for r in rows),
                         seconds=round(sum(r["seconds"] for r in rows), 1), minutes=round(sum(r["seconds"] for r in rows) / 60, 1)))
        f = json.load(open(os.path.join(ROOT, "results", fr)))
        free.append(dict(key=key, scene=label, frame=f.get("frame"), kernel_runs=f["runs"], unique_runs=f["unique_runs"], tokens=f["tokens"], positions=f["positions"],
                         mismatches=f["mismatches"], identical=bool(f["identical_framebuffer"]), seconds=f["wall_seconds"]))
        if key == "dungeon":
            frames_dungeon = [dict(frame=r["frame"], seconds=r["seconds"], tokens_m=round(r["tokens"] / 1e6, 3), kernel_runs=sum(v for k, v in r["runs"].items() if k != "rop_skipped")) for r in rows]
    out = dict(runs=runs, free_run=free, dungeon_frames=frames_dungeon,
               totals=dict(frames=sum(r["frames"] for r in runs), mismatches=sum(r["mismatches"] for r in runs) + sum(f["mismatches"] for f in free),
                           tokens_m=round(sum(r["tokens"] for r in runs) / 1e6, 1), positions=sum(r["positions"] for r in runs), kernel_runs=sum(r["kernel_runs"] for r in runs),
                           free_frames_identical=sum(f["identical"] for f in free)))
    dst = os.path.join(ROOT, "results/derived/readme_data.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst); [print(r) for r in runs]; print(out["totals"])


if __name__ == "__main__":
    sys.exit(main())
