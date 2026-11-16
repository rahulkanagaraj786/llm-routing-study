"""Create the split manifest. Uses prompt text and benchmark names only (no outcomes)."""
import sys, json, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router import data as D, splits as S

if S.MANIFEST_PATH.exists() and "--force" not in sys.argv:
    sys.exit(f"{S.MANIFEST_PATH} already exists; the manifest is frozen. Use --force only before any model has been fit.")
rows = D.read_paired()
manifest, info = S.build_manifest(rows)
meta = S.write_manifest(manifest, dict(near_duplicates=info))
print(json.dumps(info))
print("manifest sha256", meta["manifest_sha256"])
tab = collections.defaultdict(collections.Counter)
for m in manifest:
    if m["eligible"]:
        tab[m["benchmark"]][m["split"]] += 1
print(f"{'benchmark':14s} {'role':15s} {'train':>6s} {'val':>6s} {'test':>6s} {'fresh':>6s} {'total':>6s}")
tot = collections.Counter()
for b in D.BENCHMARKS:
    t = tab[b]; tot.update(t)
    print(f"{b:14s} {D.BENCHMARKS[b]['role']:15s} {t['train']:6d} {t['val']:6d} {t['test']:6d} {t['fresh_sealed']:6d} {sum(t.values()):6d}")
print(f"{'TOTAL':30s} {tot['train']:6d} {tot['val']:6d} {tot['test']:6d} {tot['fresh_sealed']:6d} {sum(tot.values()):6d}")
folds = collections.Counter((m["benchmark"], m["cv_fold"]) for m in manifest if m["eligible"] and m["role"] == "selection_pool" and m["split"] in ("train", "val"))
print("selection-pool dev rows per CV fold:", {f: sum(v for (b, ff), v in folds.items() if ff == f) for f in range(S.N_FOLDS)})
# leakage checks
by_cluster = collections.defaultdict(set)
for m in manifest: by_cluster[m["cluster_id"]].add(m["split"])
print("clusters spanning more than one split:", sum(len(v) > 1 for v in by_cluster.values()))
