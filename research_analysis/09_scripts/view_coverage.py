"""
View-coverage / view-informativeness analysis on the CaMuViD-provided COCO
annotations for Wildtrack and MultiviewX.

This uses ONLY the annotation JSON files that ship inside P4/CaMuViD/data.
No model is run; no images are needed.  Every number below is computed
directly from ground-truth annotations.

Goal: answer, from data alone,
  (a) how many cameras actually see each pedestrian instance,
  (b) whether all cameras are equally informative,
  (c) how recall-upper-bound grows with the number of cameras used,
  (d) which camera subsets are redundant.
"""
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

BASE = r"c:\Users\Siddhartha\Desktop\M.tech project\Evaluation of codes\P4\CaMuViD\data"

DATASETS = {
    "Wildtrack": ["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
    "MultiviewX": ["C1", "C2", "C3", "C4", "C5", "C6"],
}


def load(ds, split):
    p = os.path.join(BASE, ds, f"{ds}_coco_{split}_anno_chunk_id.json")
    with open(p, "r") as fh:
        return json.load(fh)


def build(ds, split):
    """Return frame -> person_id -> set(cameras), plus per-camera box stats."""
    j = load(ds, split)
    img = {i["id"]: i["file_name"] for i in j["images"]}

    # file_name looks like "C1/00001800.png"
    frame_pid_cams = defaultdict(lambda: defaultdict(set))
    cam_boxes = defaultdict(list)
    n_ann = 0
    for a in j["annotations"]:
        fn = img[a["image_id"]]
        cam, stem = fn.split("/")
        frame = os.path.splitext(stem)[0]
        pid = a["id"]                      # in the *_id.json files, id == person ID
        frame_pid_cams[frame][pid].add(cam)
        x, y, w, h = a["bbox"]
        cam_boxes[cam].append((w, h, w * h))
        n_ann += 1
    return frame_pid_cams, cam_boxes, n_ann, len(j["images"])


def analyse(ds, split):
    cams = DATASETS[ds]
    N = len(cams)
    frame_pid_cams, cam_boxes, n_ann, n_img = build(ds, split)

    print(f"\n{'='*78}")
    print(f"{ds}  [{split}]   cameras={N}  images={n_img}  annotations={n_ann}")
    print(f"{'='*78}")

    # ---- sanity: is `id` really a person id (repeats across cameras)? ----
    n_frames = len(frame_pid_cams)
    total_inst = sum(len(v) for v in frame_pid_cams.values())
    print(f"frames={n_frames}   unique (frame,person) instances={total_inst}")
    print(f"annotations / instance = {n_ann/total_inst:.3f}   "
          f"(=1.0 would mean `id` is NOT a shared person id)")

    # ---- (a) distribution of #cameras that see each instance ----
    hist = defaultdict(int)
    for f, pids in frame_pid_cams.items():
        for pid, cs in pids.items():
            hist[len(cs)] += 1
    print("\n(a) How many cameras see each pedestrian instance?")
    print("    k cams | count  |  share  | cumulative(>=k)")
    cum = 0
    for k in sorted(hist, reverse=True):
        cum += hist[k]
    running = 0
    for k in sorted(hist):
        running += hist[k]
        print(f"    {k:>6} | {hist[k]:>6} | {100*hist[k]/total_inst:6.2f}% | "
              f"{100*(total_inst-running+hist[k])/total_inst:6.2f}%")
    mean_cov = sum(k * c for k, c in hist.items()) / total_inst
    print(f"    mean cameras per instance = {mean_cov:.2f} of {N}")

    # ---- (b) per-camera contribution ----
    print("\n(b) Per-camera statistics")
    print("    cam | instances | share of all | UNIQUE (only cam sees it) | mean box area(px)")
    cam_inst = defaultdict(int)
    cam_unique = defaultdict(int)
    for f, pids in frame_pid_cams.items():
        for pid, cs in pids.items():
            for c in cs:
                cam_inst[c] += 1
            if len(cs) == 1:
                cam_unique[next(iter(cs))] += 1
    for c in cams:
        areas = [a for (_, _, a) in cam_boxes[c]]
        ma = sum(areas) / len(areas) if areas else 0
        print(f"    {c:>3} | {cam_inst[c]:>9} | {100*cam_inst[c]/total_inst:11.2f}% | "
              f"{cam_unique[c]:>25} | {ma:>10.0f}")

    # ---- (c) recall upper bound vs. camera subset ----
    # An instance is "recoverable" by a subset S if at least one camera in S sees it.
    # This is exactly CaMuViD's TP criterion ("if an ID is assigned to at least one
    # detection it is counted as a TP"), so it upper-bounds achievable recall.
    def coverage(subset):
        s = set(subset)
        hit = 0
        for f, pids in frame_pid_cams.items():
            for pid, cs in pids.items():
                if cs & s:
                    hit += 1
        return hit / total_inst

    print("\n(c) Best / worst camera subset by ORACLE recall upper bound")
    print("    (fraction of pedestrian instances visible in >=1 chosen camera)")
    print("    k | best subset                    | best   | worst subset                   | worst  | greedy")
    greedy, greedy_sets = [], []
    chosen = []
    for k in range(1, N + 1):
        best, best_s = -1, None
        worst, worst_s = 2, None
        for sub in combinations(cams, k):
            c = coverage(sub)
            if c > best:
                best, best_s = c, sub
            if c < worst:
                worst, worst_s = c, sub
        # greedy incremental
        if k == 1:
            gbest, gs = -1, None
            for c_ in cams:
                v = coverage([c_])
                if v > gbest:
                    gbest, gs = v, c_
            chosen = [gs]
        else:
            gbest, gs = -1, None
            for c_ in cams:
                if c_ in chosen:
                    continue
                v = coverage(chosen + [c_])
                if v > gbest:
                    gbest, gs = v, c_
            chosen = chosen + [gs]
            gbest = coverage(chosen)
        greedy.append(gbest)
        greedy_sets.append(list(chosen))
        print(f"    {k} | {'+'.join(best_s):<30} | {100*best:6.2f}% | "
              f"{'+'.join(worst_s):<30} | {100*worst:6.2f}% | {100*gbest:6.2f}%")

    print("\n    greedy incremental order:", " -> ".join(chosen))
    print("    greedy marginal gain per added camera (pp):")
    prev = 0.0
    for k, g in enumerate(greedy, 1):
        print(f"      +{chosen[k-1]}  ->  {100*g:6.2f}%   (gain {100*(g-prev):+5.2f} pp)")
        prev = g

    # ---- (d) pairwise redundancy ----
    print("\n(d) Pairwise camera overlap (Jaccard of the instance sets they observe)")
    inst_of = {c: set() for c in cams}
    for f, pids in frame_pid_cams.items():
        for pid, cs in pids.items():
            for c in cs:
                inst_of[c].add((f, pid))
    hdr = "        " + "".join(f"{c:>7}" for c in cams)
    print(hdr)
    for a in cams:
        row = f"    {a:>3} "
        for b in cams:
            if a == b:
                row += f"{'-':>7}"
            else:
                inter = len(inst_of[a] & inst_of[b])
                union = len(inst_of[a] | inst_of[b])
                row += f"{100*inter/union:6.1f}%"
        print(row)


if __name__ == "__main__":
    for ds in ("Wildtrack", "MultiviewX"):
        for split in ("test",):
            analyse(ds, split)
