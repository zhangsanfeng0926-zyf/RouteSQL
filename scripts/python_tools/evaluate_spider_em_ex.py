import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import List, Tuple


def normalize_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";")
    sql = re.sub(r"\s+", " ", sql)
    return sql.lower()


def load_gold(gold_path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with gold_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" not in line:
                raise ValueError(f"Invalid gold line (missing tab): {line}")
            sql, db_id = line.rsplit("\t", 1)
            pairs.append((sql, db_id))
    return pairs


def load_pred(pred_path: Path) -> List[str]:
    with pred_path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def db_file(db_dir: Path, db_id: str) -> Path:
    # Spider convention: database/<db_id>/<db_id>.sqlite
    return db_dir / db_id / f"{db_id}.sqlite"


def execute_sql(db_path: Path, sql: str):
    conn = sqlite3.connect(str(db_path))
    try:
        conn.text_factory = lambda b: b.decode(errors="ignore") if isinstance(b, (bytes, bytearray)) else b
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        # Ignore output order by using multiset comparison.
        normalized_rows = [tuple(str(v) for v in row) for row in rows]
        return True, Counter(normalized_rows)
    except Exception:
        return False, Counter()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Prediction file, one SQL per line")
    parser.add_argument("--gold", default="dataset/spider/dev_gold.sql", help="Gold file with SQL<TAB>db_id")
    parser.add_argument("--db_dir", default="dataset/spider/database", help="Spider database directory")
    parser.add_argument("--out", default="results/stage3_eval_metrics.json", help="Output metrics json path")
    args = parser.parse_args()

    pred_path = Path(args.pred)
    gold_path = Path(args.gold)
    db_dir = Path(args.db_dir)
    out_path = Path(args.out)

    preds = load_pred(pred_path)
    gold = load_gold(gold_path)

    if len(preds) != len(gold):
        raise ValueError(f"Line count mismatch: pred={len(preds)} gold={len(gold)}")

    total = len(gold)
    em_hit = 0
    ex_hit = 0
    exec_errors = 0

    for pred_sql, (gold_sql, db_id) in zip(preds, gold):
        if normalize_sql(pred_sql) == normalize_sql(gold_sql):
            em_hit += 1

        db_path = db_file(db_dir, db_id)
        ok_pred, pred_res = execute_sql(db_path, pred_sql)
        ok_gold, gold_res = execute_sql(db_path, gold_sql)

        if not ok_pred or not ok_gold:
            exec_errors += 1
            continue

        if pred_res == gold_res:
            ex_hit += 1

    metrics = {
        "total": total,
        "em_count": em_hit,
        "ex_count": ex_hit,
        "execution_error_count": exec_errors,
        "em": em_hit / total,
        "ex": ex_hit / total,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"TOTAL={total}")
    print(f"EM={metrics['em']:.4f} ({em_hit}/{total})")
    print(f"EX={metrics['ex']:.4f} ({ex_hit}/{total})")
    print(f"EXEC_ERROR={exec_errors}")
    print(f"METRICS_FILE={out_path}")


if __name__ == "__main__":
    main()
