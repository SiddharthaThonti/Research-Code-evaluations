"""
Empirical test of MSMVD's (P2) motivating claim, using the ground-truth
annotations that ship with P4/CaMuViD.

MSMVD claims previous MVPD methods "struggle to detect pedestrians with
consistently small or large scales in views, or with vastly different scales
between views", and that this justifies multi-SCALE image features + a
multi-scale BEV feature pyramid.

P2 has no code, so we cannot run MSMVD.  We CAN, however, test whether the
premise holds in the data, since the same person ID appears in several cameras
and we know each camera's box for that person.

Metrics per (frame, person) instance seen in >= 2 cameras:
  - scale ratio  = max(box height) / min(box height) across the views that see it
  - absolute scale spread in pixels
Also: what fraction of instances would be assigned to DIFFERENT FPN levels
      in different views (an FPN level is assigned by sqrt(area), the standard
      RetinaNet/FPN rule: level = floor(4 + log2(sqrt(area)/224)) clipped to P3..P7).
"""
import json, os, math, statistics
from collections import defaultdict

BASE = r"c:\Users\Siddhartha\Desktop\M.tech project\Evaluation of codes\P4\CaMuViD\data"

def fpn_level(area):
    if area <= 0:
        return None
    lvl = 4 + math.log2(math.sqrt(area) / 224.0)
    return max(3, min(7, int(math.floor(lvl))))

def analyse(ds, split="test"):
    with open(os.path.join(BASE, ds, f"{ds}_coco_{split}_anno_chunk_id.json")) as fh:
        j = json.load(fh)
    img = {i["id"]: i["file_name"] for i in j["images"]}
    inst = defaultdict(dict)          # (frame,pid) -> cam -> (w,h,area)
    for a in j["annotations"]:
        cam, stem = img[a["image_id"]].split("/")
        frame = os.path.splitext(stem)[0]
        x, y, w, h = a["bbox"]
        inst[(frame, a["id"])][cam] = (w, h, w * h)

    ratios, lvl_spread, multi = [], [], 0
    for key, cams in inst.items():
        if len(cams) < 2:
            continue
        multi += 1
        hs = [v[1] for v in cams.values()]
        ratios.append(max(hs) / max(1e-9, min(hs)))
        lvls = {fpn_level(v[2]) for v in cams.values()}
        lvl_spread.append(len(lvls))

    ratios.sort()
    def pct(p):
        return ratios[int(p / 100 * (len(ratios) - 1))]

    print(f"\n{'='*72}")
    print(f"{ds} [{split}]  -  cross-view SCALE variation of the SAME person")
    print(f"{'='*72}")
    print(f"instances seen in >=2 cameras : {multi}")
    print(f"box-height ratio max/min across views:")
    print(f"   median {statistics.median(ratios):5.2f}x   mean {statistics.mean(ratios):5.2f}x")
    for p in (25, 50, 75, 90, 95, 99):
        print(f"   p{p:<2} = {pct(p):5.2f}x")
    print(f"   max  = {ratios[-1]:5.2f}x")
    for t in (1.5, 2.0, 3.0, 4.0):
        n = sum(1 for r in ratios if r >= t)
        print(f"   instances with >= {t:.1f}x cross-view scale ratio: "
              f"{n:5d}  ({100*n/len(ratios):5.1f}%)")

    print(f"\n number of DISTINCT FPN levels the same person falls on, across views:")
    hist = defaultdict(int)
    for s in lvl_spread:
        hist[s] += 1
    for k in sorted(hist):
        print(f"   {k} level(s): {hist[k]:5d}  ({100*hist[k]/len(lvl_spread):5.1f}%)")
    n_multi_lvl = sum(v for k, v in hist.items() if k >= 2)
    print(f"   => {100*n_multi_lvl/len(lvl_spread):.1f}% of pedestrians would be handled by"
          f" DIFFERENT pyramid levels in different views")

    # per-camera absolute scale profile
    print(f"\n per-camera box-height distribution (px):")
    percam = defaultdict(list)
    for key, cams in inst.items():
        for c, (w, h, a) in cams.items():
            percam[c].append(h)
    for c in sorted(percam):
        v = sorted(percam[c])
        print(f"   {c}: n={len(v):5d}  p10={v[len(v)//10]:4.0f}  median={statistics.median(v):5.0f}"
              f"  p90={v[9*len(v)//10]:4.0f}  max={v[-1]:4.0f}")

for ds in ("Wildtrack", "MultiviewX"):
    analyse(ds)
