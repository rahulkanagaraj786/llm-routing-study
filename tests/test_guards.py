"""The sealed-data guard must refuse the locked test split, holdout benchmarks and fresh sets."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from router import splits as S


def refuses(**kw):
    try:
        S.load_rows("guard_test", **kw)
    except PermissionError:
        return True
    return False


def test_sealed_rows_are_refused():
    assert refuses(splits=("test",))
    assert refuses(splits=("fresh_sealed",))
    assert refuses(splits=("train",), benchmarks=("hle",))
    assert refuses(splits=("train", "val"), benchmarks=("livecodebench", "aime", "livemathbench"))
    assert refuses(splits=("train",))  # all benchmarks includes the holdouts


def test_source_pool_dev_is_allowed_filtered_and_disjoint_from_test():
    pool = ("simpleqa", "mmlupro", "gpqa")
    tab, rows = S.load_rows("guard_test", splits=("train", "val"), benchmarks=pool, exclude=lambda r: not S.in_source_pool(r))
    dev, folds = S.load_selection_dev("guard_test")
    assert set(tab.qid) == set(dev.qid) and len(dev) == 5640
    unfiltered, _ = S.load_selection_dev("guard_test", filtered=False)
    assert len(unfiltered) == 6011 and set(dev.qid) < set(unfiltered.qid)
    man = S.read_manifest()
    assert all(man[q]["split"] in ("train", "val") and man[q]["role"] == "selection_pool" for q in dev.qid)
    assert not any(r["benchmark"] == "mmlupro" and r["subject"] in ("math", "computer science") for r in rows)
    assert set(folds.tolist()) == {0, 1, 2, 3, 4}
    assert len(S.boundary_duplicate_qids()) == 1 and not S.boundary_duplicate_qids() & set(dev.qid)


def test_rq1_dev_covers_all_benchmarks_and_never_the_test_split():
    dev, folds = S.load_rq1_dev("guard_test")
    man = S.read_manifest()
    assert len(dev) == 6216 + 2091
    assert set(dev.benchmark.tolist()) == {"simpleqa", "mmlupro", "gpqa", "hle", "livecodebench", "aime", "livemathbench"}
    assert all(man[q]["split"] in ("train", "val") for q in dev.qid)
    assert set(folds.tolist()) == {0, 1, 2, 3, 4}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
