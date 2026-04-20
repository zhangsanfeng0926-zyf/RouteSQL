import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.runtime_setup import configure_local_runtime

configure_local_runtime()

from scripts.python_tools.ask_llm import (
    analyze_question_complexity,
    build_verifier_report,
    build_semantic_subspace,
    enrich_fuzzy_hints_with_join_paths,
    extract_values_from_question,
    extract_question_text,
    execute_sql_with_stats,
    fuzzy_match_schema,
    load_schema_catalog,
    normalize_sql_output,
    score_sql_against_question,
    sql_structure_features,
)

def db_file(db_dir: str, db_id: str) -> str:
    return os.path.join(db_dir, db_id, f"{db_id}.sqlite")


def load_pred(path: str):
    return [line.rstrip("\n") for line in Path(path).read_text(encoding="utf-8").splitlines()]


def load_candidate_trace(path: str):
    trace_path = Path(path)
    if not trace_path.exists():
        return None
    lines = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def score_with_stage_bias(question_text, sql, exec_ok, exec_info, fuzzy_hints, source):
    complexity = analyze_question_complexity(question_text)
    features = sql_structure_features(sql)
    source_bias = 0.0

    # For complex questions we trust stage2 more when it carries richer structure.
    if source == "b" and exec_ok and complexity["extra_hard_like"]:
        source_bias += 0.8
        if features["has_join"]:
            source_bias += 0.4
        if features["has_subquery"]:
            source_bias += 0.4
        if features["has_group"] or features["has_set_ops"]:
            source_bias += 0.3

    # For simpler questions, stage1 remains a useful conservative anchor.
    if source == "a" and exec_ok and not complexity["extra_hard_like"]:
        source_bias += 0.2

    return score_sql_against_question(question_text, sql, exec_ok, exec_info, fuzzy_hints, source_bias=source_bias)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--pred_a", required=True)
    parser.add_argument("--pred_b", required=True)
    parser.add_argument("--db_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trace_out", required=True)
    parser.add_argument("--pred_b_candidates", default="")
    parser.add_argument("--prefer", choices=["a", "b"], default="a")
    args = parser.parse_args()

    questions_json = json.load(open(Path(args.questions) / "questions.json", "r", encoding="utf-8"))
    prompts = [q["prompt"] for q in questions_json["questions"]]
    db_ids = [q["db_id"] for q in questions_json["questions"]]
    pred_a = load_pred(args.pred_a)
    pred_b = load_pred(args.pred_b)
    pred_b_candidates = load_candidate_trace(args.pred_b_candidates) if args.pred_b_candidates else None

    if not (len(prompts) == len(pred_a) == len(pred_b)):
        raise ValueError(f"Line count mismatch: prompts={len(prompts)} a={len(pred_a)} b={len(pred_b)}")
    if pred_b_candidates is not None and len(pred_b_candidates) != len(prompts):
        raise ValueError(f"Candidate trace count mismatch: prompts={len(prompts)} cand={len(pred_b_candidates)}")

    merged = []
    traces = []
    for idx, (prompt, db_id, sql_a, sql_b) in enumerate(zip(prompts, db_ids, pred_a, pred_b)):
        question_text = extract_question_text(prompt)
        db_path = db_file(args.db_dir, db_id)
        catalog = load_schema_catalog(db_path)
        fuzzy_hints = fuzzy_match_schema(question_text, catalog, top_k_tables=4)
        fuzzy_hints = enrich_fuzzy_hints_with_join_paths(args.db_dir, db_id, question_text, fuzzy_hints)
        fuzzy_hints["value_candidates"] = extract_values_from_question(question_text)
        fuzzy_hints = build_semantic_subspace(args.db_dir, db_id, question_text, fuzzy_hints)

        sql_a = normalize_sql_output(sql_a)
        sql_b = normalize_sql_output(sql_b)
        ok_a, info_a = execute_sql_with_stats(db_path, sql_a)
        ok_b, info_b = execute_sql_with_stats(db_path, sql_b)
        score_a = score_with_stage_bias(question_text, sql_a, ok_a, info_a, fuzzy_hints, "a")
        score_b = score_with_stage_bias(question_text, sql_b, ok_b, info_b, fuzzy_hints, "b")
        verifier_a = build_verifier_report(question_text, sql_a, ok_a, info_a, fuzzy_hints)
        verifier_b = build_verifier_report(question_text, sql_b, ok_b, info_b, fuzzy_hints)

        # When stage2 is executable and meaningfully more structured for complex questions,
        # let it win even if stage1 is slightly closer under generic heuristics.
        complexity = analyze_question_complexity(question_text)
        if ok_b and complexity["extra_hard_like"]:
            struct_a = sql_structure_features(sql_a)
            struct_b = sql_structure_features(sql_b)
            richness_delta = sum([
                int(struct_b["has_join"]) - int(struct_a["has_join"]),
                int(struct_b["has_subquery"]) - int(struct_a["has_subquery"]),
                int(struct_b["has_group"]) - int(struct_a["has_group"]),
                int(struct_b["has_set_ops"]) - int(struct_a["has_set_ops"]),
            ])
            if richness_delta >= 1 and score_b >= score_a - 0.6:
                score_b = score_a + 0.01

        candidate_pool = [
            {
                "sql": sql_a,
                "label": "stage1",
                "score": score_a,
                "ok": ok_a,
                "exec_info": info_a,
                "verifier_report": verifier_a,
            },
            {
                "sql": sql_b,
                "label": "stage2",
                "score": score_b,
                "ok": ok_b,
                "exec_info": info_b,
                "verifier_report": verifier_b,
            },
        ]

        if pred_b_candidates is not None:
            trace_item = pred_b_candidates[idx]
            for cand in trace_item.get("ranked_candidates", [])[:8]:
                sql_c = normalize_sql_output(cand.get("sql", ""))
                if not sql_c:
                    continue
                ok_c, info_c = execute_sql_with_stats(db_path, sql_c)
                # Keep pool arbitration conservative: stage2 pool candidates should
                # compete mostly on semantic fit and executability, not on extra
                # source-specific bonuses that can over-favor structurally richer
                # but semantically drifted SQL.
                score_c = score_sql_against_question(question_text, sql_c, ok_c, info_c, fuzzy_hints, source_bias=0.0)
                candidate_pool.append({
                    "sql": sql_c,
                    "label": "stage2_pool",
                    "score": score_c,
                    "ok": ok_c,
                    "exec_info": info_c,
                    "verifier_report": build_verifier_report(question_text, sql_c, ok_c, info_c, fuzzy_hints),
                })

        dedup = {}
        for cand in candidate_pool:
            sql = cand["sql"]
            prev = dedup.get(sql)
            if prev is None or cand["score"] > prev["score"]:
                dedup[sql] = cand
        candidate_pool = sorted(dedup.values(), key=lambda x: (x["ok"], x["score"]), reverse=True)
        best = candidate_pool[0]
        chosen = best["sql"]
        choose_a = best["label"] == "stage1"
        merged.append(chosen)
        traces.append({
            "db_id": db_id,
            "question": question_text,
            "sql_a": sql_a,
            "sql_b": sql_b,
            "ok_a": ok_a,
            "ok_b": ok_b,
            "score_a": score_a,
            "score_b": score_b,
            "verifier_a": verifier_a,
            "verifier_b": verifier_b,
            "chosen": "a" if choose_a else "b",
            "final_sql": chosen,
            "final_verifier": best.get("verifier_report", {}),
            "pool_top": candidate_pool[:6],
        })

    out_path = Path(args.out)
    trace_path = Path(args.trace_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(merged) + "\n", encoding="utf-8")
    with trace_path.open("w", encoding="utf-8") as f:
        for item in traces:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")
    print(f"wrote {trace_path}")


if __name__ == "__main__":
    main()
