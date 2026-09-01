"""Parse the final per-class evaluation table out of each ImGeoNet log."""
import re, os, statistics

LOGS = r"c:\Users\Siddhartha\Desktop\M.tech project\Evaluation of codes\P3\ImGeoNet\logs"

ROW = re.compile(r"^\|\s*([A-Za-z0-9_ '\-/]+?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*$")

def last_table(path):
    lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()
    starts = [i for i, l in enumerate(lines) if "| classes" in l and "AP_0.25" in l]
    s = starts[-1]
    rows = []
    for l in lines[s + 1:]:
        m = ROW.match(l)
        if m:
            name = m.group(1).strip()
            rows.append((name, float(m.group(2)), float(m.group(3)),
                         float(m.group(4)), float(m.group(5))))
        elif rows and l.startswith("2024") or (rows and l.startswith("2023")):
            break
    return rows

for f in ("scannet.txt", "arkit.txt", "scannet200.txt"):
    rows = last_table(os.path.join(LOGS, f))
    cls = [r for r in rows if r[0].lower() != "overall"]
    overall = [r for r in rows if r[0].lower() == "overall"]
    print(f"\n{'='*70}\n{f}: {len(cls)} classes")
    if overall:
        o = overall[0]
        print(f"  Overall  mAP@0.25={o[1]:.4f}  mAR@0.25={o[2]:.4f}  "
              f"mAP@0.50={o[3]:.4f}  mAR@0.50={o[4]:.4f}")
    ap25 = [r[1] for r in cls]
    ar25 = [r[2] for r in cls]
    ap50 = [r[3] for r in cls]
    z25 = [r[0] for r in cls if r[1] == 0.0]
    zr25 = [r[0] for r in cls if r[2] == 0.0]
    z50 = [r[0] for r in cls if r[3] == 0.0]
    print(f"  AP@0.25 == 0 : {len(z25):3d} classes ({100*len(z25)/len(cls):.1f}%)")
    print(f"  AR@0.25 == 0 : {len(zr25):3d} classes ({100*len(zr25)/len(cls):.1f}%)  "
          f"<- never even RECALLED")
    print(f"  AP@0.50 == 0 : {len(z50):3d} classes ({100*len(z50)/len(cls):.1f}%)")
    print(f"  AP@0.25 median={statistics.median(ap25):.4f}  mean={statistics.mean(ap25):.4f}")
    print(f"  mean AP@0.50 / AP@0.25 drop = "
          f"{100*(1 - statistics.mean(ap50)/max(1e-9,statistics.mean(ap25))):.1f}%")
    worst = sorted(cls, key=lambda r: r[1])[:10]
    print("  10 worst classes by AP@0.25:")
    for n, a25, r25, a50, r50 in worst:
        print(f"     {n:<28} AP25={a25:.4f} AR25={r25:.4f} AP50={a50:.4f}")
    best = sorted(cls, key=lambda r: -r[1])[:8]
    print("  8 best classes by AP@0.25:")
    for n, a25, r25, a50, r50 in best:
        print(f"     {n:<28} AP25={a25:.4f} AR25={r25:.4f} AP50={a50:.4f}")
    # classes that are found (AR>0) but badly localised (big AP25->AP50 collapse)
    coll = [(n, a25, a50, 1 - a50 / a25) for n, a25, r25, a50, r50 in cls if a25 > 0.30]
    coll.sort(key=lambda t: -t[3])
    print("  worst LOCALISATION collapse (AP25>0.30, largest relative AP25->AP50 drop):")
    for n, a25, a50, d in coll[:8]:
        print(f"     {n:<28} AP25={a25:.4f} -> AP50={a50:.4f}  ({100*d:.1f}% drop)")
