"""
Cross-check CaMuViD's published camera-elimination ablation (Table 3 of the
CVPR-2025 paper / README) against the ORACLE recall ceiling that is implied by
the ground-truth annotations shipped in P4/CaMuViD/data.

CaMuViD's TP rule (paper Sec. 4.2): "If an ID is assigned to at least one
detection, it is counted as a TP; otherwise, it is considered an FN."
=> recall is measured per (frame, person-ID), over ALL 7 cameras' GT, even when
   some cameras have been blanked to black (verified in camuvid.py L243-248 and
   L328-330 where the `if view in blank_views: continue` skip is commented out).

Therefore, for a camera subset S, the maximum achievable recall is:
    oracle(S) = |{(frame,pid) : visible in at least one camera of S}| / |all (frame,pid)|

The paper eliminates cameras in index order, keeping the prefix C1..Ck.
"""
import json, os
from collections import defaultdict

BASE = r"c:\Users\Siddhartha\Desktop\M.tech project\Evaluation of codes\P4\CaMuViD\data"

# Reported by CaMuViD (README table + paper Table 3), Wildtrack, prefix C1..Ck kept
REPORTED = {           # k : (MODA, MODP, Precision, Recall, F1)
    1: (60.1, 65.0, 99.8, 60.2, 75.1),
    2: (77.8, 71.5, 99.8, 77.9, 87.5),
    3: (90.6, 76.4, 98.4, 92.1, 95.1),
    4: (93.8, 79.4, 98.3, 95.5, 96.9),
    5: (93.8, 80.0, 96.8, 96.9, 97.0),
    6: (95.6, 80.2, 96.6, 99.3, 98.0),
    7: (95.0, 80.9, 96.3, 98.6, 97.4),
}

def build(ds, split, cams):
    with open(os.path.join(BASE, ds, f"{ds}_coco_{split}_anno_chunk_id.json")) as fh:
        j = json.load(fh)
    img = {i["id"]: i["file_name"] for i in j["images"]}
    inst = defaultdict(set)
    for a in j["annotations"]:
        cam, stem = img[a["image_id"]].split("/")
        frame = os.path.splitext(stem)[0]
        inst[(frame, a["id"])].add(cam)
    return inst

def oracle(inst, subset):
    s = set(subset)
    return 100.0 * sum(1 for cs in inst.values() if cs & s) / len(inst)

cams = ["C1","C2","C3","C4","C5","C6","C7"]
for split in ("test", "train"):
    inst = build("Wildtrack", split, cams)
    print(f"\n{'='*94}")
    print(f"Wildtrack [{split}] - {len(inst)} (frame,person) instances")
    print(f"{'='*94}")
    print(" k | cameras kept          | oracle recall | CaMuViD recall | achieved/oracle | CaMuViD MODA")
    print("-" * 94)
    for k in range(1, 8):
        sub = cams[:k]
        o = oracle(inst, sub)
        if split == "test":
            moda, modp, prec, rec, f1 = REPORTED[k]
            print(f" {k} | {'+'.join(sub):<21} | {o:12.2f}% | {rec:13.1f}% |"
                  f" {100*rec/o:14.1f}% | {moda:11.1f}")
        else:
            print(f" {k} | {'+'.join(sub):<21} | {o:12.2f}% | {'-':>13} | {'-':>15} | {'-':>11}")

    if split == "test":
        print("\n  Marginal ORACLE gain of each added camera (index order):")
        prev = 0.0
        for k in range(1, 8):
            o = oracle(inst, cams[:k])
            r = REPORTED[k][3]
            pr = REPORTED[k-1][3] if k > 1 else 0.0
            print(f"    +{cams[k-1]}: oracle {o:6.2f}% ({o-prev:+5.2f} pp)   "
                  f"CaMuViD recall {r:5.1f}% ({r-pr:+5.1f} pp)   "
                  f"MODA {REPORTED[k][0]:5.1f} ({REPORTED[k][0]-(REPORTED[k-1][0] if k>1 else 0):+5.1f})")
            prev = o

# MultiviewX, index order, for reference
print(f"\n{'='*94}")
mcams = ["C1","C2","C3","C4","C5","C6"]
inst = build("MultiviewX", "test", mcams)
print(f"MultiviewX [test] - {len(inst)} instances - oracle recall by prefix")
print(f"{'='*94}")
prev = 0.0
for k in range(1, 7):
    o = oracle(inst, mcams[:k])
    print(f"  k={k}  {'+'.join(mcams[:k]):<20} oracle {o:6.2f}%  ({o-prev:+5.2f} pp)")
    prev = o
