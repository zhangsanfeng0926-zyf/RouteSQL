import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


def sample_indices_by_db(items, db_allow_set, per_db, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for idx, item in enumerate(items):
        db_id = item["db_id"]
        if db_allow_set is None or db_id in db_allow_set:
            groups[db_id].append(idx)

    selected = []
    for db_id in sorted(groups.keys()):
        ids = groups[db_id]
        if len(ids) <= per_db:
            chosen = ids
        else:
            chosen = rng.sample(ids, per_db)
        selected.extend(chosen)

    selected.sort()
    return selected


def subset_json_by_indices(path_in, path_out, indices):
    data = json.loads(Path(path_in).read_text(encoding="utf-8"))
    sub = [data[i] for i in indices]
    Path(path_out).parent.mkdir(parents=True, exist_ok=True)
    Path(path_out).write_text(json.dumps(sub, ensure_ascii=False, indent=2), encoding="utf-8")


def subset_lines_by_indices(path_in, path_out, indices):
    lines = Path(path_in).read_text(encoding="utf-8").splitlines()
    sub = [lines[i] for i in indices]
    Path(path_out).parent.mkdir(parents=True, exist_ok=True)
    Path(path_out).write_text("\n".join(sub) + ("\n" if sub else ""), encoding="utf-8")


def subset_jsonl_by_indices(path_in, path_out, indices):
    if not Path(path_in).exists():
        return
    lines = [l for l in Path(path_in).read_text(encoding="utf-8").splitlines() if l.strip()]
    sub = [lines[i] for i in indices]
    Path(path_out).parent.mkdir(parents=True, exist_ok=True)
    Path(path_out).write_text("\n".join(sub) + ("\n" if sub else ""), encoding="utf-8")


def copy_db_dirs(src_db_root, dst_db_root, db_ids):
    Path(dst_db_root).mkdir(parents=True, exist_ok=True)
    for db_id in sorted(db_ids):
        src = Path(src_db_root) / db_id
        dst = Path(dst_db_root) / db_id
        if not src.exists():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="dataset/spider", help="Source spider directory")
    parser.add_argument("--dst", default="dataset/dataset_min/spider", help="Destination mini spider directory")
    parser.add_argument("--dev_per_db", type=int, default=8, help="Sample size per db_id for dev set")
    parser.add_argument(
        "--train_per_db",
        type=int,
        default=30,
        help="Sample size per db_id for train set",
    )
    parser.add_argument(
        "--train_scope",
        choices=["all", "dev"],
        default="all",
        help="Use all train db categories or only dev db categories when sampling train",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    dev_json = json.loads((src / "dev.json").read_text(encoding="utf-8"))
    train_json = json.loads((src / "train_spider_and_others.json").read_text(encoding="utf-8"))

    dev_db_ids = sorted({x["db_id"] for x in dev_json})

    dev_indices = sample_indices_by_db(dev_json, db_allow_set=None, per_db=args.dev_per_db, seed=args.seed)
    train_allow = None if args.train_scope == "all" else set(dev_db_ids)
    train_indices = sample_indices_by_db(train_json, db_allow_set=train_allow, per_db=args.train_per_db, seed=args.seed)

    selected_dev = [dev_json[i] for i in dev_indices]
    selected_train = [train_json[i] for i in train_indices]

    used_db_ids = sorted({x["db_id"] for x in selected_dev} | {x["db_id"] for x in selected_train})

    # Core dataset files with unchanged filenames/structure.
    subset_json_by_indices(src / "dev.json", dst / "dev.json", dev_indices)
    subset_lines_by_indices(src / "dev_gold.sql", dst / "dev_gold.sql", dev_indices)

    subset_json_by_indices(src / "train_spider_and_others.json", dst / "train_spider_and_others.json", train_indices)
    subset_lines_by_indices(src / "train_gold.sql", dst / "train_gold.sql", train_indices)

    # Optional linking files (keep alignment with subset indices).
    subset_jsonl_by_indices(src / "enc/test_schema-linking.jsonl", dst / "enc/test_schema-linking.jsonl", dev_indices)
    subset_jsonl_by_indices(src / "enc/train_schema-linking.jsonl", dst / "enc/train_schema-linking.jsonl", train_indices)

    # tables.json restricted to used dbs.
    tables = json.loads((src / "tables.json").read_text(encoding="utf-8"))
    tables_sub = [t for t in tables if t.get("db_id") in set(used_db_ids)]
    (dst / "tables.json").parent.mkdir(parents=True, exist_ok=True)
    (dst / "tables.json").write_text(json.dumps(tables_sub, ensure_ascii=False, indent=2), encoding="utf-8")

    # Keep database folder structure unchanged, but only copy required db directories.
    copy_db_dirs(src / "database", dst / "database", used_db_ids)

    summary = {
        "seed": args.seed,
        "dev_per_db": args.dev_per_db,
        "train_per_db": args.train_per_db,
        "train_scope": args.train_scope,
        "dev_total": len(selected_dev),
        "train_total": len(selected_train),
        "used_db_total": len(used_db_ids),
        "used_db_ids": used_db_ids,
        "src": str(src),
        "dst": str(dst),
    }
    (dst / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
