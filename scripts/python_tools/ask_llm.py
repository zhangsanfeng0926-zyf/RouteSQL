import argparse
import os
import sys
import json
import re
import sqlite3
import ast
import difflib
from typing import Dict, Any, List, Tuple
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.runtime_setup import configure_local_runtime

configure_local_runtime()

import openai
from tqdm import tqdm

from llm.chatgpt import init_chatgpt, ask_llm
from utils.enums import LLM
from utils.schema_path_utils import (
    build_path_graph_subspace,
    build_candidate_schema_subspace,
    get_ranked_join_paths,
    infer_table_json_path,
    score_sql_with_graph_consistency,
    sql_matches_join_path,
    sql_path_consistency,
    sql_path_coverage,
    sql_path_minimality,
)
from torch.utils.data import DataLoader

from utils.post_process import process_duplication, get_sqls

QUESTION_FILE = "questions.json"
INVALID_SQL_SENTINEL = "SELECT __INVALID__ FROM __INVALID__"
ENABLE_EM_CANONICALIZATION = True


def sql_structural_normality(sql: str) -> float:
    low = " ".join((sql or "").strip().lower().split())
    score = 0.0
    if " count(1)" not in low and " count(0)" not in low:
        score += 0.2
    if re.search(r"\b(from|join)\s+[a-zA-Z_][\w]*\s+as\s+t\d+\b", low):
        score += 0.3
    if "select distinct" in low:
        score += 0.1
    if " order by " in low and " limit " in low:
        score += 0.1
    if " join " in low and " on " in low:
        score += 0.2
    return score


def _split_top_level_csv(text: str) -> List[str]:
    parts = []
    buf = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _sort_sql_list_clause(sql: str, clause_regex: str, joiner: str = ", ") -> str:
    match = re.search(clause_regex, sql, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return sql
    clause = match.group(1).strip()
    items = _split_top_level_csv(clause)
    if len(items) <= 1:
        return sql
    sorted_clause = joiner.join(sorted(items, key=lambda x: x.lower()))
    start, end = match.span(1)
    return sql[:start] + sorted_clause + sql[end:]


def _normalize_count_star(sql: str) -> str:
    sql = re.sub(r"(?is)count\s*\(\s*1\s*\)", "COUNT(*)", sql)
    sql = re.sub(r"(?is)count\s*\(\s*0\s*\)", "COUNT(*)", sql)
    return sql


def _canonicalize_aliases_and_joins(sql: str) -> str:
    alias_pattern = re.compile(r"(?is)\b(from|join)\s+([a-zA-Z_][\w]*)\s+(?:as\s+)?([a-zA-Z_][\w]*)")
    matches = list(alias_pattern.finditer(sql))
    if not matches:
        return sql

    tables = []
    alias_map = {}
    for m in matches:
        table = m.group(2)
        alias = m.group(3)
        tables.append(table)
        alias_map[alias] = table
    sorted_tables = sorted(dict.fromkeys(tables), key=lambda x: x.lower())
    new_alias_map = {table: f"t{i+1}" for i, table in enumerate(sorted_tables)}

    # Replace old aliases in SQL with canonical aliases.
    for old_alias, table in sorted(alias_map.items(), key=lambda x: -len(x[0])):
        new_alias = new_alias_map.get(table, old_alias)
        sql = re.sub(rf"(?<![\w]){re.escape(old_alias)}\.", f"{new_alias}.", sql)

    # Rewrite FROM/JOIN table aliases in original appearance order.
    def repl(match):
        keyword = match.group(1).upper()
        table = match.group(2)
        alias = new_alias_map.get(table, match.group(3))
        return f"{keyword} {table} AS {alias}"

    sql = alias_pattern.sub(repl, sql)
    return sql


def canonicalize_sql_for_em(sql: str, enabled: bool = True, mode: str = "safe") -> str:
    if not enabled:
        return " ".join((sql or "").split()).strip()
    sql = " ".join((sql or "").split())
    if not sql:
        return sql
    sql = _normalize_count_star(sql)
    sql = _canonicalize_aliases_and_joins(sql)
    if mode in {"safe", "aggressive"}:
        sql = _sort_sql_list_clause(sql, r"(?is)\bselect\b\s+(.*?)\s+\bfrom\b")
    if mode == "aggressive":
        sql = _sort_sql_list_clause(sql, r"(?is)\bgroup\s+by\b\s+(.*?)(?=\border\s+by\b|\blimit\b|$)")
        sql = _sort_sql_list_clause(sql, r"(?is)\border\s+by\b\s+(.*?)(?=\blimit\b|$)")
    sql = re.sub(r"(?is)(group by [^ ](?:.*?))order by", r"\1 ORDER BY", sql)
    sql = re.sub(r"\s+", " ", sql).strip()
    return sql


def is_placeholder_sql(sql: str) -> bool:
    s = sql.lower()
    placeholders = [
        "table_name",
        "column_name",
        "column1",
        "column2",
        "column3",
        "your_table",
        "your_column",
    ]
    return any(token in s for token in placeholders)


def normalize_sql_output(raw_text: str) -> str:
    raw_text = raw_text or ""

    # Prefer direct SQL lines from reasoning-style outputs.
    for line in raw_text.splitlines():
        candidate = line.strip().strip("`")
        if re.match(r"(?is)^(select|with)\b", candidate):
            if is_placeholder_sql(candidate):
                continue
            text = " ".join(candidate.split())
            if ";" in text:
                text = text.split(";", 1)[0]
            return text

    # Accept both closed and unclosed sql code blocks.
    code_block = re.search(r"```(?:sql)?\s*(.*?)(?:```|$)", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if code_block:
        block = " ".join(code_block.group(1).split())
        if re.match(r"(?is)^(select|with)\b", block) and not is_placeholder_sql(block):
            if ";" in block:
                block = block.split(";", 1)[0]
            return block.strip("` \t\r\n")

    text = " ".join(raw_text.replace("\n", " ").split())
    text = process_duplication(text)

    code_block = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, flags=re.IGNORECASE)
    if code_block:
        text = " ".join(code_block.group(1).split())

    # Prefer well-formed SQL snippets that include FROM (or CTE with WITH).
    cte_candidates = re.findall(r"(?is)\bwith\b.*?\bselect\b.*?(?:;|$)", text)
    for cand in cte_candidates:
        cand = " ".join(cand.split()).strip()
        if " from " in cand.lower() or cand.lower().startswith("with"):
            text = cand
            break

    if text == " ".join(raw_text.replace("\n", " ").split()) or " from " not in text.lower():
        select_candidates = re.findall(r"(?is)\bselect\b.*?\bfrom\b.*?(?:;|$)", text)
        if select_candidates:
            text = " ".join(select_candidates[0].split()).strip()

    lower = text.lower()
    start = lower.find("select ")
    if start == -1:
        start = lower.find("with ")
    if start != -1:
        text = text[start:]

    if ";" in text:
        text = text.split(";", 1)[0]

    text = text.strip("` \t\r\n")

    if text.upper().startswith("SELECT") or text.upper().startswith("WITH"):
        return canonicalize_sql_for_em(text, ENABLE_EM_CANONICALIZATION)
    if text.strip() == "":
        return INVALID_SQL_SENTINEL
    if text.startswith(" "):
        return canonicalize_sql_for_em("SELECT" + text, ENABLE_EM_CANONICALIZATION)
    return canonicalize_sql_for_em("SELECT " + text, ENABLE_EM_CANONICALIZATION)


def normalize_framework_output(raw_text: str) -> str:
    raw_text = raw_text or ""

    for line in raw_text.splitlines():
        candidate = line.strip().strip("`")
        if re.match(r"(?is)^(select|with)\b", candidate):
            text = " ".join(candidate.split())
            if ";" in text:
                text = text.split(";", 1)[0]
            return text

    code_block = re.search(r"```(?:sql)?\s*(.*?)(?:```|$)", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if code_block:
        block = " ".join(code_block.group(1).split())
        if re.match(r"(?is)^(select|with)\b", block):
            if ";" in block:
                block = block.split(";", 1)[0]
            return block.strip("` \t\r\n")

    text = " ".join(raw_text.replace("\n", " ").split()).strip("` \t\r\n")
    lower = text.lower()
    start = lower.find("select ")
    if start == -1:
        start = lower.find("with ")
    if start != -1:
        text = text[start:]
    if ";" in text:
        text = text.split(";", 1)[0]

    if text.upper().startswith("SELECT") or text.upper().startswith("WITH"):
        return canonicalize_sql_for_em(text, ENABLE_EM_CANONICALIZATION)
    return "SELECT <COLUMN_1> FROM <TABLE_1>"


def normalize_identifier(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()


def extract_question_text(original_prompt: str) -> str:
    text = original_prompt or ""

    comment_matches = re.findall(
        r"/\*\s*Answer the following:\s*(.*?)\s*\*/",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if comment_matches:
        return " ".join(comment_matches[-1].split())

    input_matches = re.findall(
        r"Input Question:\s*(.*?)(?:\n|$)",
        text,
        flags=re.IGNORECASE,
    )
    if input_matches:
        return " ".join(input_matches[-1].split())

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i in range(len(lines) - 1, -1, -1):
        ln = lines[i]
        low = ln.lower()
        if low.startswith("question:"):
            return ln.split(":", 1)[1].strip()

    if lines and lines[-1].strip().lower() == "select":
        for ln in reversed(lines[:-1]):
            if ln and not ln.lower().startswith(("create table", "foreign key", "primary key", "/*", "*/")):
                return ln

    return lines[-1] if lines else ""


def extract_values_from_question(question_text: str) -> List[str]:
    values = []
    for m in re.finditer(r"'([^']+)'|\"([^\"]+)\"", question_text):
        val = m.group(1) if m.group(1) is not None else m.group(2)
        if val and len(val) <= 64:
            values.append(val)
    for m in re.finditer(r"\b\d+(?:\.\d+)?\b", question_text):
        values.append(m.group(0))
    # Keep order while deduplicating.
    seen = set()
    uniq = []
    for v in values:
        if v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq[:8]


def load_schema_catalog(db_path: str) -> Dict[str, List[str]]:
    if not os.path.exists(db_path):
        return {}
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [r[0] for r in cur.fetchall()]
        catalog = {}
        for t in tables:
            try:
                cur.execute(f"PRAGMA table_info('{t}')")
                cols = [r[1] for r in cur.fetchall()]
            except Exception:
                cols = []
            catalog[t] = cols
        return catalog
    finally:
        conn.close()


def fuzzy_match_schema(question_text: str, catalog: Dict[str, List[str]], top_k_tables: int = 4) -> Dict[str, Any]:
    q_norm = normalize_identifier(question_text)
    q_tokens = q_norm.split()
    table_scores = []
    column_hits = {}

    for table, cols in catalog.items():
        t_norm = normalize_identifier(table)
        score = difflib.SequenceMatcher(None, q_norm, t_norm).ratio()
        token_bonus = 0.0
        for tok in q_tokens:
            if tok and tok in t_norm:
                token_bonus += 0.05
        score += min(token_bonus, 0.25)

        col_rank = []
        for c in cols:
            c_norm = normalize_identifier(c)
            c_score = difflib.SequenceMatcher(None, q_norm, c_norm).ratio()
            c_token_bonus = 0.0
            for tok in q_tokens:
                if tok and tok in c_norm:
                    c_token_bonus += 0.05
            c_score += min(c_token_bonus, 0.2)
            col_rank.append((c, c_score))
        col_rank.sort(key=lambda x: x[1], reverse=True)
        column_hits[table] = [x[0] for x in col_rank[:6]]
        table_scores.append((table, score + (col_rank[0][1] * 0.3 if col_rank else 0.0)))

    table_scores.sort(key=lambda x: x[1], reverse=True)
    chosen_tables = [t for t, _ in table_scores[:top_k_tables]]
    return {
        "candidate_tables": chosen_tables,
        "candidate_columns": {t: column_hits.get(t, []) for t in chosen_tables},
    }


def enrich_fuzzy_hints_with_join_paths(
    db_dir: str,
    db_id: str,
    question_text: str,
    fuzzy_hints: Dict[str, Any],
) -> Dict[str, Any]:
    hints = dict(fuzzy_hints or {})
    try:
        table_json_path = infer_table_json_path(db_dir)
        ranked_paths = get_ranked_join_paths(
            table_json_path=table_json_path,
            db_id=db_id,
            question_text=question_text,
            candidate_tables=hints.get("candidate_tables", []),
            candidate_columns=hints.get("candidate_columns", {}),
            max_hops=2,
            top_k=4,
        )
    except Exception:
        ranked_paths = []
    hints["join_path_candidates"] = ranked_paths
    return hints


def build_semantic_subspace(
    db_dir: str,
    db_id: str,
    question_text: str,
    fuzzy_hints: Dict[str, Any],
) -> Dict[str, Any]:
    hints = dict(fuzzy_hints or {})
    table_json_path = infer_table_json_path(db_dir)
    subspace = build_candidate_schema_subspace(
        table_json_path=table_json_path,
        db_id=db_id,
        question_text=question_text,
        candidate_tables=hints.get("candidate_tables", []),
        candidate_columns=hints.get("candidate_columns", {}),
        value_candidates=hints.get("value_candidates", []),
        max_hops=2,
        top_k_paths=4,
    )
    hints["schema_subspace"] = subspace
    hints["candidate_tables"] = subspace.get("top_tables", hints.get("candidate_tables", []))
    hints["candidate_columns"] = subspace.get("top_columns", hints.get("candidate_columns", {}))
    hints["join_path_candidates"] = subspace.get("top_join_paths", hints.get("join_path_candidates", []))
    hints["value_candidates"] = subspace.get("top_value_bindings", hints.get("value_candidates", []))
    hints["path_graph_subspace"] = build_path_graph_subspace(
        table_json_path=table_json_path,
        db_id=db_id,
        question_text=question_text,
        candidate_tables=hints.get("candidate_tables", []),
        candidate_columns=hints.get("candidate_columns", {}),
        value_candidates=hints.get("value_candidates", []),
        max_hops=2,
        top_k_paths=4,
    )
    return hints


def rewrite_question_with_subspace(question_text: str, fuzzy_hints: Dict[str, Any]) -> str:
    complexity = analyze_question_complexity(question_text)
    subspace = (fuzzy_hints or {}).get("schema_subspace", {})
    top_tables = subspace.get("top_tables", [])
    top_values = subspace.get("top_value_bindings", [])
    top_paths = subspace.get("top_join_paths", [])
    q_norm = normalize_identifier(question_text)
    targets = []
    returns = []
    filters = []
    if question_mentions_count(question_text):
        returns.append("COUNT")
    if question_mentions_distinct(question_text):
        returns.append("DISTINCT")
    if complexity["needs_aggregation"]:
        returns.append("AGGREGATION")
    if complexity["needs_order_limit"]:
        returns.append("ORDER_OR_RANK")
    if complexity["needs_group"]:
        returns.append("GROUPING")
    if complexity["needs_nested"]:
        returns.append("NESTED")
    if complexity["needs_set_ops"]:
        returns.append("SET_OP")
    if complexity["needs_filter"]:
        filters.append("FILTER")

    targets.extend(top_tables[:3])
    best_path = " -> ".join(top_paths[0].get("tables", [])) if top_paths else ""
    value_text = ", ".join(str(v) for v in top_values[:6]) if top_values else ""

    slot_lines = [
        f"QUESTION={question_text.strip()}",
        f"QUERY_TARGET={', '.join(targets) if targets else 'UNKNOWN'}",
        f"RETURN_TARGET={', '.join(returns) if returns else 'ROWSET'}",
        f"FILTERS={', '.join(filters) if filters else 'NONE'}",
        f"VALUE_BINDINGS={value_text if value_text else 'NONE'}",
        f"PREFERRED_PATH={best_path if best_path else 'NONE'}",
        f"SEMANTIC_SUBSPACE_TABLES={', '.join(top_tables[:4]) if top_tables else 'NONE'}",
        f"NORMALIZED_QUESTION={q_norm}",
    ]
    return " | ".join(slot_lines)


def parse_framework_json(raw_text: str) -> Dict[str, Any]:
    raw_text = raw_text or ""
    candidates = [raw_text.strip()]

    m_code = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if m_code:
        candidates.append(m_code.group(1).strip())

    m_obj = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if m_obj:
        candidates.append(m_obj.group(0).strip())

    for cand in candidates:
        if not cand:
            continue
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            try:
                obj = ast.literal_eval(cand)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue

    sql_template = normalize_framework_output(raw_text)
    if sql_template and sql_template != "SELECT <COLUMN_1> FROM <TABLE_1>":
        return build_spec_from_template(sql_template)

    return {
        "intent": "fallback",
        "tables": ["<TABLE_1>"],
        "joins": [],
        "select": ["<COLUMN_1>"],
        "where": [],
        "group_by": [],
        "order_by": [],
        "set_ops": [],
        "subqueries": [],
        "sql_template": "SELECT <COLUMN_1> FROM <TABLE_1>",
        "confidence": 0.1,
        "alternatives": [],
    }


def build_spec_from_template(sql_template: str) -> Dict[str, Any]:
    text = " ".join(sql_template.split())
    lower = text.lower()

    table_matches = re.findall(r"\bfrom\s+([a-zA-Z_][\w]*)|\bjoin\s+([a-zA-Z_][\w]*)", text, flags=re.IGNORECASE)
    tables = []
    for a, b in table_matches:
        t = a or b
        if t and t not in tables:
            tables.append(t)

    joins = []
    for m in re.finditer(r"\bjoin\s+([a-zA-Z_][\w]*)\s+(?:as\s+)?([a-zA-Z_][\w]*)?\s*on\s+(.*?)(?=\bjoin\b|\bwhere\b|\bgroup\s+by\b|\border\s+by\b|\blimit\b|$)", text, flags=re.IGNORECASE):
        joins.append({
            "table": m.group(1),
            "alias": m.group(2) or "",
            "on": m.group(3).strip(),
            "type": "INNER",
        })

    select_clause = ""
    m_select = re.search(r"\bselect\b\s+(.*?)\s+\bfrom\b", text, flags=re.IGNORECASE)
    if m_select:
        select_clause = m_select.group(1).strip()

    where_clause = ""
    m_where = re.search(r"\bwhere\b\s+(.*?)(?=\bgroup\s+by\b|\border\s+by\b|\blimit\b|$)", text, flags=re.IGNORECASE)
    if m_where:
        where_clause = m_where.group(1).strip()

    group_clause = ""
    m_group = re.search(r"\bgroup\s+by\b\s+(.*?)(?=\border\s+by\b|\blimit\b|$)", text, flags=re.IGNORECASE)
    if m_group:
        group_clause = m_group.group(1).strip()

    order_clause = ""
    m_order = re.search(r"\border\s+by\b\s+(.*?)(?=\blimit\b|$)", text, flags=re.IGNORECASE)
    if m_order:
        order_clause = m_order.group(1).strip()

    set_ops = []
    for op in ["union", "intersect", "except"]:
        if re.search(rf"\b{op}\b", lower):
            set_ops.append(op.upper())

    subqueries = re.findall(r"\(\s*select\b", lower)

    return {
        "intent": "sql-template-derived",
        "tables": tables if tables else ["<TABLE_1>"],
        "joins": joins,
        "select": [select_clause] if select_clause else ["<COLUMN_1>"],
        "where": [where_clause] if where_clause else [],
        "group_by": [group_clause] if group_clause else [],
        "order_by": [order_clause] if order_clause else [],
        "set_ops": set_ops,
        "subqueries": ["SUBQUERY"] * len(subqueries),
        "sql_template": text,
        "confidence": 0.5,
        "alternatives": [],
    }


def normalize_framework_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        spec = {}
    spec = dict(spec)
    spec.setdefault("intent", "fallback")
    spec.setdefault("tables", ["<TABLE_1>"])
    spec.setdefault("joins", [])
    spec.setdefault("select", ["<COLUMN_1>"])
    spec.setdefault("where", [])
    spec.setdefault("group_by", [])
    spec.setdefault("order_by", [])
    spec.setdefault("set_ops", [])
    spec.setdefault("subqueries", [])
    spec.setdefault("sql_template", "SELECT <COLUMN_1> FROM <TABLE_1>")
    spec.setdefault("confidence", 0.3)
    spec.setdefault("confidence_breakdown", {})
    spec.setdefault("alternatives", [])
    try:
        spec["confidence"] = float(spec.get("confidence", 0.3))
    except Exception:
        spec["confidence"] = 0.3
    spec["confidence"] = max(0.0, min(1.0, spec["confidence"]))
    if not isinstance(spec.get("alternatives"), list):
        spec["alternatives"] = []
    if not isinstance(spec.get("confidence_breakdown"), dict):
        spec["confidence_breakdown"] = {}
    return spec


def derive_framework_confidence(
    framework_spec: Dict[str, Any],
    question_text: str,
    fuzzy_hints: Dict[str, Any],
) -> Dict[str, float]:
    complexity = analyze_question_complexity(question_text)
    sql_template = get_framework_template(framework_spec).lower()
    schema_fit = 0.3
    path_fit = 0.3
    structure_fit = 0.3

    for table in (fuzzy_hints or {}).get("candidate_tables", [])[:3]:
        if table.lower() in sql_template:
            schema_fit += 0.15

    if complexity["needs_multi_table"]:
        path_fit += 0.2 if " join " in sql_template else -0.15
    if complexity["needs_nested"]:
        structure_fit += 0.2 if "(select " in sql_template else -0.15
    if complexity["needs_group"]:
        structure_fit += 0.15 if " group by " in sql_template else -0.1
    if complexity["needs_set_ops"]:
        structure_fit += 0.15 if any(op in sql_template for op in [" union ", " intersect ", " except "]) else -0.15

    graph_scores = score_sql_with_graph_consistency(sql_template, (fuzzy_hints or {}).get("path_graph_subspace", {}))
    path_fit += 0.2 * graph_scores["coverage"] + 0.1 * graph_scores["consistency"]

    schema_fit = max(0.0, min(1.0, schema_fit))
    path_fit = max(0.0, min(1.0, path_fit))
    structure_fit = max(0.0, min(1.0, structure_fit))
    total = max(0.0, min(1.0, 0.4 * schema_fit + 0.3 * path_fit + 0.3 * structure_fit))
    return {
        "total": total,
        "schema_fit": schema_fit,
        "path_fit": path_fit,
        "structure_fit": structure_fit,
    }


def get_framework_template(spec: Dict[str, Any]) -> str:
    template = spec.get("sql_template", "")
    if not isinstance(template, str) or not template.strip():
        return "SELECT <COLUMN_1> FROM <TABLE_1>"
    return normalize_framework_output(template)


def build_complexity_hint_block(question_text: str) -> str:
    complexity = analyze_question_complexity(question_text)
    hints = []
    if complexity["needs_multi_table"]:
        hints.append("likely multi-table join")
    if complexity["needs_aggregation"]:
        hints.append("likely aggregation")
    if complexity["needs_group"]:
        hints.append("likely grouping or per-entity aggregation")
    if complexity["needs_nested"]:
        hints.append("likely nested query or IN/EXISTS pattern")
    if complexity["needs_set_ops"]:
        hints.append("likely UNION / INTERSECT / EXCEPT")
    if complexity["needs_order_limit"]:
        hints.append("likely ranking / order by / limit")
    if complexity["extra_hard_like"]:
        hints.append("treat as extra-hard: do not oversimplify")
    return ", ".join(hints) if hints else "no strong complexity signals"


def build_framework_prompt_advanced(original_prompt: str, question_text: str) -> str:
    few_shot = """
Few-shot Example 1 (single table)
Input Question: How many singers are there?
Output JSON:
{
  "intent": "count all rows",
  "tables": ["singer"],
  "joins": [],
  "select": ["COUNT(*)"],
  "where": [],
  "group_by": [],
  "order_by": [],
  "set_ops": [],
  "subqueries": [],
  "sql_template": "SELECT COUNT(*) FROM <TABLE_1>"
}

Few-shot Example 2 (multi-table join + aggregation)
Input Question: List each stadium name and number of concerts held there.
Output JSON:
{
  "intent": "join stadium with concert and aggregate",
  "tables": ["stadium", "concert"],
  "joins": [
    {"left": "<TABLE_1>.<FK_1>", "right": "<TABLE_2>.<PK_1>", "type": "INNER"}
  ],
  "select": ["<TABLE_1>.<NAME_COL>", "COUNT(<TABLE_2>.<ID_COL>)"],
  "where": [],
  "group_by": ["<TABLE_1>.<NAME_COL>"],
  "order_by": [],
  "set_ops": [],
  "subqueries": [],
  "sql_template": "SELECT <TABLE_1>.<NAME_COL>, COUNT(<TABLE_2>.<ID_COL>) FROM <TABLE_1> JOIN <TABLE_2> ON <TABLE_1>.<FK_1> = <TABLE_2>.<PK_1> GROUP BY <TABLE_1>.<NAME_COL>"
}

Few-shot Example 3 (nested query)
Input Question: Find singer names whose song sales are above the average sales.
Output JSON:
{
    "intent": "nested aggregate filter",
    "tables": ["singer", "song"],
    "joins": [
        {"left": "<TABLE_1>.<PK_1>", "right": "<TABLE_2>.<FK_1>", "type": "INNER"}
    ],
    "select": ["<TABLE_1>.<NAME_COL>"],
    "where": ["<TABLE_2>.<SALES_COL> > (SELECT AVG(<SALES_COL>) FROM <TABLE_2>)"],
    "group_by": [],
    "order_by": [],
    "set_ops": [],
    "subqueries": ["avg sales subquery"],
    "sql_template": "SELECT <TABLE_1>.<NAME_COL> FROM <TABLE_1> JOIN <TABLE_2> ON <TABLE_1>.<PK_1> = <TABLE_2>.<FK_1> WHERE <TABLE_2>.<SALES_COL> > (SELECT AVG(<SALES_COL>) FROM <TABLE_2>)"
}

Few-shot Example 4 (set operation)
Input Question: Find districts that have shops with products < 3000 and also > 10000.
Output JSON:
{
    "intent": "intersect two filtered sets",
    "tables": ["shop"],
    "joins": [],
    "select": ["<DISTRICT_COL>"],
    "where": [],
    "group_by": [],
    "order_by": [],
    "set_ops": ["INTERSECT"],
    "subqueries": [],
    "sql_template": "SELECT <DISTRICT_COL> FROM <TABLE_1> WHERE <NUM_COL> < <VALUE_1> INTERSECT SELECT <DISTRICT_COL> FROM <TABLE_1> WHERE <NUM_COL> > <VALUE_2>"
}
""".strip()

    return (
        "You are an expert SQL planner for complex Text-to-SQL.\n"
        "Task: produce a high-quality SQL framework as STRICT JSON only.\n"
        "The framework must support multi-table joins, nested subqueries, set operations (UNION/INTERSECT/EXCEPT), grouping and ordering.\n"
        "Output must be valid JSON object with keys exactly:\n"
        "intent, tables, joins, select, where, group_by, order_by, set_ops, subqueries, sql_template, confidence, alternatives\n"
        "Rules:\n"
        "1) Use placeholders in sql_template where helpful: <TABLE_1>, <COLUMN_1>, <VALUE_1>, <FK_1>, <PK_1>.\n"
        "2) Preserve the full logical structure needed by the question.\n"
        "3) Do not oversimplify to a single-table or COUNT(*) query unless strongly justified.\n"
        "4) If confident, sql_template may be near-complete SQL instead of a very abstract skeleton.\n"
        "5) Treat the framework as a best-faith hypothesis of semantics, not a minimal sketch.\n"
        "6) confidence must be a number between 0 and 1.\n"
        "7) alternatives must be a short list of plausible alternative structures, especially when the question is ambiguous or hard.\n"
        "8) NEVER output or imply trivial SQL like SELECT 1.\n"
        "9) Do NOT output markdown or extra text.\n\n"
        f"{few_shot}\n\n"
        "Complexity hints inferred from the question:\n"
        f"{build_complexity_hint_block(question_text)}\n\n"
        "Now process the following task context:\n"
        f"{original_prompt}\n"
    )


def build_fuzzy_hint_block(hints: Dict[str, Any]) -> str:
    return json.dumps(hints, ensure_ascii=False, indent=2)


def build_join_path_hint_text(fuzzy_hints: Dict[str, Any]) -> str:
    paths = (fuzzy_hints or {}).get("join_path_candidates", [])
    if not paths:
        return "No ranked join-path candidates available."
    lines = []
    for idx, path in enumerate(paths[:4], start=1):
        tables = " -> ".join(path.get("tables", []))
        edges = []
        for edge in path.get("edges", []):
            edges.append(f"{edge['from_table']}.{edge['from_col']} = {edge['to_table']}.{edge['to_col']}")
        edge_text = "; ".join(edges) if edges else "no explicit edge"
        lines.append(f"{idx}. score={path.get('score', 0):.2f} path={tables} joins={edge_text}")
    return "\n".join(lines)


def top_join_path_strength(fuzzy_hints: Dict[str, Any]) -> float:
    paths = (fuzzy_hints or {}).get("join_path_candidates", [])
    if not paths:
        return 0.0
    return float(paths[0].get("score", 0.0))


def build_schema_subspace_text(fuzzy_hints: Dict[str, Any]) -> str:
    subspace = (fuzzy_hints or {}).get("schema_subspace", {})
    if not subspace:
        return "No schema subspace available."
    return json.dumps(subspace, ensure_ascii=False, indent=2)


def augment_prompt_with_rewrite(base_prompt: str, rewritten_question: str) -> str:
    if not rewritten_question:
        return base_prompt
    return (
        f"{base_prompt}\n"
        "Structured rewritten question:\n"
        f"{rewritten_question}\n"
    )


def build_fill_prompt_advanced(
    original_prompt: str,
    framework_spec: Dict[str, Any],
    fuzzy_hints: Dict[str, Any],
    question_text: str,
    framework_weak: bool = False,
) -> str:
    framework_json = json.dumps(framework_spec, ensure_ascii=False, indent=2)
    few_shot = """
Few-shot Fill Example A
Framework sql_template: SELECT COUNT(*) FROM <TABLE_1>
Filled SQL: SELECT COUNT(*) FROM singer

Few-shot Fill Example B
Framework sql_template: SELECT <TABLE_1>.<NAME_COL>, COUNT(<TABLE_2>.<ID_COL>) FROM <TABLE_1> JOIN <TABLE_2> ON <TABLE_1>.<FK_1> = <TABLE_2>.<PK_1> GROUP BY <TABLE_1>.<NAME_COL>
Filled SQL: SELECT s.Name, COUNT(c.concert_ID) FROM stadium AS s JOIN concert AS c ON s.Stadium_ID = c.Stadium_ID GROUP BY s.Name

Few-shot Fill Example C
Framework sql_template: SELECT <TABLE_1>.<DISTRICT_COL> FROM <TABLE_1> WHERE <NUM_COL> < <VALUE_1> INTERSECT SELECT <TABLE_1>.<DISTRICT_COL> FROM <TABLE_1> WHERE <NUM_COL> > <VALUE_2>
Fuzzy hints values: ["3000", "10000"]
Filled SQL: SELECT District FROM shop WHERE Number_products < 3000 INTERSECT SELECT District FROM shop WHERE Number_products > 10000
""".strip()

    return (
        "You are an expert SQL generator.\n"
        "Given original task context and a framework JSON, produce ONE executable SQL query.\n"
        "Guidelines:\n"
        "1) Preserve framework semantics when they are consistent with the question and schema hints.\n"
        "2) If the framework is incomplete or slightly wrong, you MAY revise joins, grouping, nesting, filters, or set operations to better match the task.\n"
        "3) Replace placeholders with schema-valid tables/columns/values from the provided context.\n"
        "4) Prefer explicit aliases when multi-table joins are used.\n"
        "5) NEVER output trivial fallback SQL (e.g., SELECT 1).\n"
        "6) Output SQL only, no explanation and no markdown fences.\n\n"
        f"{few_shot}\n\n"
        "Complexity hints inferred from the question:\n"
        f"{build_complexity_hint_block(question_text)}\n\n"
        f"Framework confidence: {'weak, treat as a soft hint only' if framework_weak else 'usable, but still revisable when needed'}\n\n"
        "Candidate schema subspace:\n"
        f"{build_schema_subspace_text(fuzzy_hints)}\n\n"
        "Fuzzy schema/value hints:\n"
        f"{build_fuzzy_hint_block(fuzzy_hints)}\n\n"
        "Ranked join-path candidates:\n"
        f"{build_join_path_hint_text(fuzzy_hints)}\n\n"
        "Framework JSON:\n"
        f"{framework_json}\n\n"
        "Original task context:\n"
        f"{original_prompt}\n"
    )


def build_repair_prompt(
    original_prompt: str,
    framework_spec: Dict[str, Any],
    failed_sql: str,
    error_msg: str,
    question_text: str,
    fuzzy_hints: Dict[str, Any],
    framework_weak: bool = False,
) -> str:
    framework_json = json.dumps(framework_spec, ensure_ascii=False, indent=2)
    error_type = classify_sql_error(error_msg)
    join_error_hint = ""
    err_low = (error_msg or "").lower()
    if any(key in err_low for key in ["no such column", "no such table", "ambiguous column", "cannot join", "syntax error"]):
        join_error_hint = (
            "The error likely comes from wrong table/column selection or join path. "
            "Prefer the ranked join-path candidates when reconstructing the SQL.\n\n"
        )
    typed_hint = build_typed_repair_hint(error_type)
    return (
        "You are a senior SQL debugger.\n"
        "Fix the SQL using the execution error while preserving the intended semantics.\n"
        "If the framework is misleading, you may rewrite substantial parts of the SQL instead of making a tiny patch.\n"
        "Prefer semantic correctness over literal adherence to the framework.\n"
        "Return one executable SQL query only.\n"
        "Never return trivial fallback SQL like SELECT 1.\n"
        "Do not output explanation or markdown.\n\n"
        "Complexity hints inferred from the question:\n"
        f"{build_complexity_hint_block(question_text)}\n\n"
        f"Framework confidence: {'weak, framework may be discarded if needed' if framework_weak else 'moderate, but still revisable'}\n\n"
        f"{join_error_hint}"
        f"Repair type: {error_type}\n"
        f"Repair guidance: {typed_hint}\n\n"
        "Candidate schema subspace:\n"
        f"{build_schema_subspace_text(fuzzy_hints)}\n\n"
        "Schema/value hints:\n"
        f"{json.dumps(fuzzy_hints, ensure_ascii=False, indent=2)}\n\n"
        "Ranked join-path candidates:\n"
        f"{build_join_path_hint_text(fuzzy_hints)}\n\n"
        "Execution error:\n"
        f"{error_msg}\n\n"
        "Failed SQL:\n"
        f"{failed_sql}\n\n"
        "Framework JSON:\n"
        f"{framework_json}\n\n"
        "Original task context:\n"
        f"{original_prompt}\n"
    )


def simple_sql_sanity_check(sql: str) -> Tuple[bool, str]:
    s = (sql or "").strip()
    if not s:
        return False, "empty sql"
    low = s.lower()
    if is_trivial_fallback_sql(s):
        return False, "trivial fallback sql"
    if not (low.startswith("select") or low.startswith("with")):
        return False, "sql must start with select or with"
    if low.startswith("select") and " from " not in f" {low} ":
        return False, "missing FROM clause"
    if s.count("(") != s.count(")"):
        return False, "unbalanced parentheses"
    if s.count("'") % 2 != 0:
        return False, "unbalanced single quote"
    return True, ""


def classify_sql_error(error_msg: str) -> str:
    msg = (error_msg or "").lower()
    if not msg:
        return "unknown"
    if "no such column" in msg or "ambiguous column" in msg:
        return "column_resolution"
    if "no such table" in msg:
        return "table_resolution"
    if "misuse of aggregate" in msg or "aggregate" in msg or "group by" in msg:
        return "aggregation"
    if "syntax error" in msg or "near" in msg:
        return "syntax"
    if "select __invalid__" in msg or "__invalid__" in msg:
        return "invalid_placeholder"
    if "empty sql" in msg or "missing from clause" in msg:
        return "incomplete_sql"
    return "generic"


def build_typed_repair_hint(error_type: str) -> str:
    hints = {
        "column_resolution": "Re-check selected columns against candidate tables and prefer columns that appear on the top join paths.",
        "table_resolution": "Reconstruct the FROM/JOIN structure using the strongest join-path candidates and avoid unsupported tables.",
        "aggregation": "Re-align SELECT, GROUP BY, HAVING and aggregate functions. If grouping is intended, keep group keys explicit.",
        "syntax": "Rewrite the SQL into a complete, executable statement. Prefer a clean reconstruction over a small patch.",
        "invalid_placeholder": "Replace placeholders with real schema items, or regenerate the SQL from schema hints and join paths.",
        "incomplete_sql": "Complete the missing clauses using schema/value hints; do not keep fragmentary SQL.",
        "generic": "Prefer semantically correct SQL using schema hints and join paths, even if you must rewrite large parts.",
        "unknown": "Use the schema hints and join paths to produce the most plausible executable SQL.",
    }
    return hints.get(error_type, hints["generic"])


def repair_strategy_decider(
    sql: str,
    error_msg: str,
    framework_spec: Dict[str, Any],
    question_text: str,
    fuzzy_hints: Dict[str, Any],
    exec_ok: bool = False,
) -> str:
    error_type = classify_sql_error(error_msg)
    struct = sql_structure_features(sql)
    graph = score_sql_with_graph_consistency(sql, (fuzzy_hints or {}).get("path_graph_subspace", {}))
    fw_conf = derive_framework_confidence(framework_spec, question_text, fuzzy_hints)

    if exec_ok and graph["coverage"] >= 0.75 and graph["consistency"] >= 0.75:
        return "minimal_patch"
    if error_type in {"table_resolution", "column_resolution"} and graph["coverage"] < 0.5:
        return "path_rebuild"
    if error_type == "aggregation":
        return "aggregation_rebuild"
    if error_type in {"syntax", "invalid_placeholder", "incomplete_sql"}:
        return "full_regenerate"
    if fw_conf["total"] < 0.35 and not struct["has_join"] and top_join_path_strength(fuzzy_hints) > 1.5:
        return "path_rebuild"
    return "minimal_patch"


def repair_simple_sql_error(
    original_prompt: str,
    framework_spec: Dict[str, Any],
    sql: str,
    error_msg: str,
    model: str,
    question_text: str = "",
    fuzzy_hints: Dict[str, Any] = None,
    framework_weak: bool = False,
) -> str:
    prompt = build_repair_prompt(
        original_prompt,
        framework_spec,
        sql,
        error_msg,
        question_text,
        fuzzy_hints or {},
        framework_weak,
    )
    try:
        res = ask_llm(model, [prompt], 0.0, 1)
        fixed = normalize_sql_output(res["response"][0])
        if is_trivial_fallback_sql(fixed):
            return sql
        return fixed
    except Exception:
        return sql


def repair_sql_by_error_type(
    original_prompt: str,
    framework_spec: Dict[str, Any],
    sql: str,
    error_msg: str,
    model: str,
    question_text: str,
    fuzzy_hints: Dict[str, Any],
    db_path: str,
    framework_weak: bool = False,
) -> str:
    error_type = classify_sql_error(error_msg)
    strategy = repair_strategy_decider(
        sql,
        error_msg,
        framework_spec,
        question_text,
        fuzzy_hints,
        exec_ok=False,
    )

    if strategy == "full_regenerate":
        repaired = repair_sql_for_syntax(
            original_prompt,
            framework_spec,
            model,
            question_text,
            fuzzy_hints,
            db_path,
        )
        if repaired:
            return repaired

    if strategy == "aggregation_rebuild":
        repaired = repair_sql_for_aggregation(
            original_prompt,
            sql,
            error_msg,
            model,
            question_text,
            fuzzy_hints,
        )
        if repaired:
            return repaired

    if strategy == "path_rebuild":
        repaired = repair_sql_for_join_path(
            original_prompt,
            framework_spec,
            sql,
            error_msg,
            model,
            question_text,
            fuzzy_hints,
        )
        if repaired:
            return repaired

    if error_type == "column_resolution" and top_join_path_strength(fuzzy_hints) >= 1.5:
        join_repaired = repair_sql_for_join_path(
            original_prompt,
            framework_spec,
            sql,
            error_msg,
            model,
            question_text,
            fuzzy_hints,
        )
        if join_repaired:
            return join_repaired

    if error_type == "column_resolution":
        repaired = repair_sql_for_column_resolution(
            original_prompt,
            framework_spec,
            sql,
            error_msg,
            model,
            question_text,
            fuzzy_hints,
        )
        if repaired:
            return repaired

    if error_type == "table_resolution":
        repaired = repair_sql_for_table_resolution(
            original_prompt,
            framework_spec,
            sql,
            error_msg,
            model,
            question_text,
            fuzzy_hints,
        )
        if repaired:
            return repaired

    if error_type == "aggregation":
        repaired = repair_sql_for_aggregation(
            original_prompt,
            sql,
            error_msg,
            model,
            question_text,
            fuzzy_hints,
        )
        if repaired:
            return repaired

    if error_type in {"syntax", "incomplete_sql", "invalid_placeholder"}:
        repaired = repair_sql_for_syntax(
            original_prompt,
            framework_spec,
            model,
            question_text,
            fuzzy_hints,
            db_path,
        )
        if repaired:
            return repaired

    if error_type in {"column_resolution", "table_resolution"}:
        schema_first = build_schema_first_sql_prompt(original_prompt, fuzzy_hints, question_text)
        try:
            res = ask_llm(model, [schema_first], 0.2, 1)
            fixed = normalize_sql_output(res["response"][0])
            if not is_trivial_fallback_sql(fixed):
                return fixed
        except Exception:
            pass

    return repair_simple_sql_error(
        original_prompt,
        framework_spec,
        sql,
        error_msg,
        model,
        question_text,
        fuzzy_hints,
        framework_weak,
    )


def repair_sql_for_join_path(
    original_prompt: str,
    framework_spec: Dict[str, Any],
    sql: str,
    error_msg: str,
    model: str,
    question_text: str,
    fuzzy_hints: Dict[str, Any],
) -> str:
    prompt = (
        "You are repairing a SQL query whose join path is likely wrong.\n"
        "Reconstruct the SQL around the strongest join path candidates.\n"
        "Use the candidate schema subspace as a hard preference, and keep the query semantically faithful.\n"
        "Return SQL only.\n\n"
        "Candidate schema subspace:\n"
        f"{json.dumps((fuzzy_hints or {}).get('schema_subspace', {}), ensure_ascii=False, indent=2)}\n\n"
        "Ranked join-path candidates:\n"
        f"{build_join_path_hint_text(fuzzy_hints)}\n\n"
        "Framework:\n"
        f"{json.dumps(framework_spec, ensure_ascii=False, indent=2)}\n\n"
        "Failed SQL:\n"
        f"{sql}\n\n"
        "Execution error:\n"
        f"{error_msg}\n\n"
        "Task context:\n"
        f"{original_prompt}\n"
    )
    try:
        res = ask_llm(model, [prompt], 0.0, 1)
        fixed = normalize_sql_output(res["response"][0])
        if not is_trivial_fallback_sql(fixed):
            return fixed
    except Exception:
        pass
    return ""


def repair_sql_for_column_resolution(
    original_prompt: str,
    framework_spec: Dict[str, Any],
    sql: str,
    error_msg: str,
    model: str,
    question_text: str,
    fuzzy_hints: Dict[str, Any],
) -> str:
    prompt = (
        "You are fixing a SQL query with wrong or ambiguous column references.\n"
        "Stay inside the candidate schema subspace and prefer columns attached to the strongest join paths.\n"
        "Return SQL only.\n\n"
        "Candidate schema subspace:\n"
        f"{build_schema_subspace_text(fuzzy_hints)}\n\n"
        "Ranked join-path candidates:\n"
        f"{build_join_path_hint_text(fuzzy_hints)}\n\n"
        "Failed SQL:\n"
        f"{sql}\n\n"
        "Error:\n"
        f"{error_msg}\n\n"
        "Task context:\n"
        f"{original_prompt}\n"
    )
    try:
        res = ask_llm(model, [prompt], 0.0, 1)
        fixed = normalize_sql_output(res["response"][0])
        if not is_trivial_fallback_sql(fixed):
            return fixed
    except Exception:
        pass
    return ""


def repair_sql_for_table_resolution(
    original_prompt: str,
    framework_spec: Dict[str, Any],
    sql: str,
    error_msg: str,
    model: str,
    question_text: str,
    fuzzy_hints: Dict[str, Any],
) -> str:
    prompt = (
        "You are fixing a SQL query with wrong tables or wrong FROM/JOIN structure.\n"
        "Rebuild the query using only the candidate schema subspace and the best join-path candidates.\n"
        "Return SQL only.\n\n"
        "Candidate schema subspace:\n"
        f"{build_schema_subspace_text(fuzzy_hints)}\n\n"
        "Ranked join-path candidates:\n"
        f"{build_join_path_hint_text(fuzzy_hints)}\n\n"
        "Failed SQL:\n"
        f"{sql}\n\n"
        "Error:\n"
        f"{error_msg}\n\n"
        "Task context:\n"
        f"{original_prompt}\n"
    )
    try:
        res = ask_llm(model, [prompt], 0.0, 1)
        fixed = normalize_sql_output(res["response"][0])
        if not is_trivial_fallback_sql(fixed):
            return fixed
    except Exception:
        pass
    return ""


def repair_sql_for_aggregation(
    original_prompt: str,
    sql: str,
    error_msg: str,
    model: str,
    question_text: str,
    fuzzy_hints: Dict[str, Any],
) -> str:
    prompt = (
        "You are fixing an aggregation SQL.\n"
        "Repair SELECT, GROUP BY, HAVING, ORDER BY, and aggregate functions so they are structurally consistent.\n"
        "Return SQL only.\n\n"
        "Candidate schema subspace:\n"
        f"{build_schema_subspace_text(fuzzy_hints)}\n\n"
        "Question complexity hints:\n"
        f"{build_complexity_hint_block(question_text)}\n\n"
        "Failed SQL:\n"
        f"{sql}\n\n"
        "Error:\n"
        f"{error_msg}\n\n"
        "Task context:\n"
        f"{original_prompt}\n"
    )
    try:
        res = ask_llm(model, [prompt], 0.0, 1)
        fixed = normalize_sql_output(res["response"][0])
        if not is_trivial_fallback_sql(fixed):
            return fixed
    except Exception:
        pass
    return ""


def repair_sql_for_syntax(
    original_prompt: str,
    framework_spec: Dict[str, Any],
    model: str,
    question_text: str,
    fuzzy_hints: Dict[str, Any],
    db_path: str,
) -> str:
    try:
        rescued = rescue_sql_with_relaxed_generation(
            model,
            original_prompt,
            framework_spec,
            fuzzy_hints,
            question_text,
            db_path,
            0.2,
            3,
        )
        if rescued:
            return rescued
    except Exception:
        pass
    return ""


def db_file(db_dir: str, db_id: str) -> str:
    return os.path.join(db_dir, db_id, f"{db_id}.sqlite")


def execute_sql_once(db_path: str, sql: str) -> Tuple[bool, str]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        cur.fetchall()
        return True, ""
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def to_candidate_list(responses: Any) -> List[str]:
    if isinstance(responses, list) and len(responses) > 0 and isinstance(responses[0], list):
        return [str(x) for x in responses[0]]
    if isinstance(responses, list):
        return [str(x) for x in responses]
    return [str(responses)]


def dedup_keep_order(sqls: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in sqls:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def make_candidate_record(
    sql: str,
    source: str,
    route: str,
    framework_confidence: float = 0.0,
    metadata: Dict[str, Any] = None,
) -> Dict[str, Any]:
    rec = {
        "sql": normalize_sql_output(sql),
        "source": source,
        "route": route,
        "framework_confidence": framework_confidence,
    }
    if metadata:
        rec.update(metadata)
    return rec


def dedup_candidate_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = {}
    ordered = []
    for rec in records:
        sql = normalize_sql_output(rec.get("sql", ""))
        if not sql:
            continue
        if sql not in merged:
            new_rec = dict(rec)
            new_rec["sql"] = sql
            new_rec["sources"] = [rec.get("source", "unknown")]
            new_rec["routes"] = [rec.get("route", "unknown")]
            merged[sql] = new_rec
            ordered.append(new_rec)
        else:
            cur = merged[sql]
            source = rec.get("source", "unknown")
            route = rec.get("route", "unknown")
            if source not in cur["sources"]:
                cur["sources"].append(source)
            if route not in cur["routes"]:
                cur["routes"].append(route)
            cur["framework_confidence"] = max(cur.get("framework_confidence", 0.0), rec.get("framework_confidence", 0.0))
    return ordered


def rank_candidate_records(
    records: List[Dict[str, Any]],
    db_path: str,
    question_text: str,
    fuzzy_hints: Dict[str, Any],
    prefer_richer: bool = False,
) -> List[Dict[str, Any]]:
    ranked = []
    for rec in records:
        sql = normalize_sql_output(rec.get("sql", ""))
        ok, exec_info = execute_sql_with_stats(db_path, sql)
        source_bias = 0.0
        if prefer_richer and any(route in rec.get("routes", []) for route in ["schema-first", "direct-relaxed", "framework-fill"]):
            source_bias += 0.3
        source_bias += min(0.5, rec.get("framework_confidence", 0.0) * 0.4)
        route_support = len(rec.get("routes", []))
        source_support = len(rec.get("sources", []))
        source_bias += min(0.8, 0.25 * max(0, route_support - 1))
        source_bias += min(0.4, 0.2 * max(0, source_support - 1))
        top_path = top_join_path_strength(fuzzy_hints)
        if any(route in rec.get("routes", []) for route in ["schema-first", "schema-first-sanity-repair"]) and top_path >= 1.5:
            source_bias += 0.5
        if any(route in rec.get("routes", []) for route in ["framework-fill", "framework-fill-sanity-repair"]) and top_path >= 2.5:
            if " join " not in sql.lower():
                source_bias -= 0.6
        if any(route in rec.get("routes", []) for route in ["candidate-vote"]) and route_support >= 2:
            source_bias += 0.4
        score = score_sql_against_question(question_text, sql, ok, exec_info, fuzzy_hints, source_bias=source_bias)
        new_rec = dict(rec)
        new_rec["sql"] = sql
        new_rec["exec_ok"] = ok
        new_rec["exec_info"] = exec_info
        new_rec["path_match"] = any(sql_matches_join_path(sql, path) for path in fuzzy_hints.get("join_path_candidates", [])[:3])
        new_rec["route_support"] = route_support
        new_rec["source_support"] = source_support
        new_rec["score"] = score
        ranked.append(new_rec)
    for rec in ranked:
        rec["pairwise_wins"] = 0.0
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            pref = pairwise_candidate_preference(ranked[i], ranked[j], question_text, fuzzy_hints)
            if pref > 0:
                ranked[i]["pairwise_wins"] += 1
            elif pref < 0:
                ranked[j]["pairwise_wins"] += 1
    for rec in ranked:
        rec["score"] += 0.15 * rec["pairwise_wins"]
    ranked.sort(key=lambda x: (x["exec_ok"], x["score"], x["pairwise_wins"]), reverse=True)
    return ranked


def pairwise_candidate_preference(
    cand_a: Dict[str, Any],
    cand_b: Dict[str, Any],
    question_text: str,
    fuzzy_hints: Dict[str, Any],
) -> int:
    a_score = cand_a.get("score", 0.0)
    b_score = cand_b.get("score", 0.0)
    if cand_a.get("exec_ok") and not cand_b.get("exec_ok"):
        return 1
    if cand_b.get("exec_ok") and not cand_a.get("exec_ok"):
        return -1

    a_graph = score_sql_with_graph_consistency(cand_a.get("sql", ""), (fuzzy_hints or {}).get("path_graph_subspace", {}))
    b_graph = score_sql_with_graph_consistency(cand_b.get("sql", ""), (fuzzy_hints or {}).get("path_graph_subspace", {}))
    a_struct = sql_structure_features(cand_a.get("sql", ""))
    b_struct = sql_structure_features(cand_b.get("sql", ""))

    a_pref = a_score + 0.6 * a_graph["coverage"] + 0.4 * a_graph["consistency"] + 0.2 * cand_a.get("route_support", 1)
    b_pref = b_score + 0.6 * b_graph["coverage"] + 0.4 * b_graph["consistency"] + 0.2 * cand_b.get("route_support", 1)
    a_pref += 0.35 * sql_structural_normality(cand_a.get("sql", ""))
    b_pref += 0.35 * sql_structural_normality(cand_b.get("sql", ""))

    complexity = analyze_question_complexity(question_text)
    if complexity["extra_hard_like"]:
        a_pref += 0.25 * int(a_struct["has_subquery"]) + 0.2 * int(a_struct["has_join"])
        b_pref += 0.25 * int(b_struct["has_subquery"]) + 0.2 * int(b_struct["has_join"])

    if abs(a_pref - b_pref) < 0.15:
        return 0
    return 1 if a_pref > b_pref else -1


def is_trivial_fallback_sql(sql: str) -> bool:
    s = " ".join((sql or "").strip().lower().rstrip(";").split())
    return s in {"select 1", "select 1.0", "select '1'", 'select "1"'}


def filter_non_trivial_sqls(sqls: List[str]) -> List[str]:
    return [s for s in sqls if not is_trivial_fallback_sql(s)]


def enforce_non_trivial_sql(sql: str) -> str:
    if is_trivial_fallback_sql(sql):
        return INVALID_SQL_SENTINEL
    return sql


def is_invalid_sentinel_sql(sql: str) -> bool:
    return " ".join((sql or "").strip().lower().split()) == "select __invalid__ from __invalid__"


def is_bad_final_sql(sql: str) -> bool:
    s = (sql or "").strip()
    if not s:
        return True
    if is_trivial_fallback_sql(s):
        return True
    if is_invalid_sentinel_sql(s):
        return True
    if is_placeholder_sql(s):
        return True
    return False


def question_mentions_count(question_text: str) -> bool:
    q = (question_text or "").lower()
    return any(
        key in q
        for key in [
            "how many",
            "number of",
            "count ",
            "total number",
            "total amount",
        ]
    )


def question_mentions_distinct(question_text: str) -> bool:
    q = (question_text or "").lower()
    return any(key in q for key in ["distinct", "different", "unique"])


def question_mentions_superlative(question_text: str) -> Tuple[str, bool]:
    q = (question_text or "").lower()
    max_keys = ["highest", "largest", "maximum", "max", "most", "latest", "newest", "greatest", "biggest"]
    min_keys = ["lowest", "smallest", "minimum", "min", "least", "fewest", "earliest", "oldest", "shortest"]
    for key in max_keys:
        if key in q:
            return "desc", True
    for key in min_keys:
        if key in q:
            return "asc", True
    return "desc", False


def question_mentions_listing(question_text: str) -> bool:
    q = (question_text or "").lower()
    return any(
        key in q
        for key in [
            "list ",
            "show ",
            "find ",
            "which ",
            "what ",
            "give ",
            "return ",
            "name",
            "names",
        ]
    )


def infer_sql_order_direction(sql: str) -> str:
    low = " ".join((sql or "").strip().lower().split())
    match = re.search(r"\border by\b\s+.*?\b(desc|asc)\b", low)
    if match:
        return match.group(1)
    if " max(" in f" {low} " or low.startswith("select max("):
        return "desc"
    if " min(" in f" {low} " or low.startswith("select min("):
        return "asc"
    return ""


def build_verifier_report(
    question_text: str,
    sql: str,
    exec_ok: bool,
    exec_info: Any,
    fuzzy_hints: Dict[str, Any],
) -> Dict[str, Any]:
    low = " ".join((sql or "").strip().lower().split())
    complexity = analyze_question_complexity(question_text)
    features = sql_structure_features(sql)
    issues = []
    score = 0.0

    def add_issue(kind: str, message: str, penalty: float, critical: bool = False):
        nonlocal score
        issues.append({
            "kind": kind,
            "message": message,
            "penalty": penalty,
            "critical": critical,
        })
        score -= penalty

    score += 0.6 if exec_ok else -1.2

    if complexity["needs_count"] and not features["has_count"]:
        add_issue("count_mismatch", "Question asks for COUNT-style output but SQL lacks COUNT aggregation.", 0.7, True)

    if complexity["needs_set_ops"] and not features["has_set_ops"]:
        add_issue("missing_set_op", "Question likely needs UNION/INTERSECT/EXCEPT but SQL does not use set operations.", 0.8, True)

    if complexity["needs_multi_table"] and top_join_path_strength(fuzzy_hints) > 1.5 and not features["has_join"]:
        add_issue("missing_join", "Question likely needs multi-table reasoning but SQL has no JOIN.", 0.8, True)

    expected_dir, has_superlative = question_mentions_superlative(question_text)
    observed_dir = infer_sql_order_direction(sql)
    if has_superlative and complexity["needs_order_limit"] and not (features["has_order"] or features["has_agg"]):
        add_issue("missing_order", "Question likely needs ranking or superlative resolution but SQL lacks ORDER BY or equivalent aggregation.", 0.6, True)
    elif has_superlative and observed_dir and expected_dir and observed_dir != expected_dir:
        add_issue("order_direction", "SQL order/aggregate direction looks inconsistent with the question superlative.", 0.75, True)

    join_paths = (fuzzy_hints or {}).get("join_path_candidates", [])[:2]
    if features["has_join"] and join_paths and not any(sql_matches_join_path(sql, path) for path in join_paths):
        add_issue("join_path_mismatch", "SQL JOIN structure is weakly aligned with the strongest join-path candidates.", 0.55, True)

    if re.search(r"\bselect\s+\*\b", low) and not question_mentions_listing(question_text):
        add_issue("select_star", "SQL uses SELECT * even though the question does not ask for a broad listing.", 0.4)

    if exec_ok and isinstance(exec_info, dict):
        row_count = exec_info.get("row_count", 0)
        empty_result = exec_info.get("empty_result", False)
        if question_mentions_count(question_text) and row_count != 1:
            add_issue("count_shape", "Count-style question should usually return a single row.", 0.35)
        if question_mentions_listing(question_text) and empty_result:
            add_issue("empty_listing", "Listing-style question returned an empty result set.", 0.2)

    if not issues:
        score += 0.25

    return {
        "score": round(score, 3),
        "issues": issues,
        "critical_issue_count": sum(1 for item in issues if item["critical"]),
        "summary": "clean" if not issues else "; ".join(item["kind"] for item in issues[:4]),
    }


def analyze_question_complexity(question_text: str) -> Dict[str, Any]:
    q = (question_text or "").lower()
    needs_count = question_mentions_count(question_text)
    needs_distinct = question_mentions_distinct(question_text)
    order_dir, needs_superlative = question_mentions_superlative(question_text)

    aggregate_keys = [
        "average", "avg", "sum", "total", "maximum", "minimum", "highest",
        "lowest", "most", "least", "count", "number of",
    ]
    group_keys = ["for each", "each", "per ", "group by", "how many"]
    nested_keys = [
        "above average", "below average", "greater than average", "less than average",
        "not in", "exists", "all ", "any ", "at least one", "more than the average",
    ]
    setop_keys = ["both", "either", "also", "intersect", "except", "union"]
    join_keys = [
        "with their", "together with", "for each", "who have", "that have",
        "whose", "and their", "along with",
    ]
    filter_keys = [
        "before ", "after ", "between ", "older than", "younger than",
        "at least", "at most", "more than", "less than", "greater than",
        "not ", "without ", "with ",
    ]

    needs_aggregation = needs_count or any(key in q for key in aggregate_keys)
    needs_group = any(key in q for key in group_keys)
    needs_nested = any(key in q for key in nested_keys)
    needs_set_ops = any(key in q for key in setop_keys)
    needs_multi_table = any(key in q for key in join_keys)
    needs_filter = any(key in q for key in filter_keys)
    needs_order_limit = needs_superlative or any(key in q for key in ["top ", "highest", "lowest", "first", "last", "latest"])

    complexity_score = 0
    for flag, weight in [
        (needs_multi_table, 1.5),
        (needs_aggregation, 1.5),
        (needs_group, 1.2),
        (needs_nested, 1.8),
        (needs_set_ops, 2.0),
        (needs_order_limit, 1.0),
        (needs_filter, 0.6),
        (needs_distinct, 0.5),
    ]:
        if flag:
            complexity_score += weight

    extra_hard_like = (
        complexity_score >= 4.0
        or needs_set_ops
        or (needs_nested and needs_group)
        or (needs_multi_table and needs_aggregation and needs_order_limit)
    )

    return {
        "needs_count": needs_count,
        "needs_distinct": needs_distinct,
        "needs_superlative": needs_superlative,
        "order_dir": order_dir,
        "needs_aggregation": needs_aggregation,
        "needs_group": needs_group,
        "needs_nested": needs_nested,
        "needs_set_ops": needs_set_ops,
        "needs_multi_table": needs_multi_table,
        "needs_filter": needs_filter,
        "needs_order_limit": needs_order_limit,
        "extra_hard_like": extra_hard_like,
        "complexity_score": complexity_score,
    }


def sql_structure_features(sql: str) -> Dict[str, Any]:
    low = " ".join((sql or "").strip().lower().split())
    return {
        "has_join": " join " in low,
        "has_group": " group by " in low,
        "has_having": " having " in low,
        "has_order": " order by " in low,
        "has_limit": " limit " in low,
        "has_subquery": "(select " in low or " in (select " in low or " exists (" in low,
        "has_set_ops": any(f" {op} " in low for op in [" union ", " intersect ", " except "]),
        "has_distinct": "select distinct" in low,
        "has_count": "count(" in low,
        "has_agg": any(fn in low for fn in ["count(", "avg(", "sum(", "max(", "min("]),
        "has_where": " where " in low,
        "is_count_only": low.startswith("select count(*) from"),
        "length": len(low),
    }


def execute_sql_with_stats(db_path: str, sql: str) -> Tuple[bool, Any]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        row_count = len(rows)
        col_count = len(rows[0]) if rows else 0
        normalized_rows = [tuple(str(v) for v in row) for row in rows]
        distinct_count = len(set(normalized_rows)) if normalized_rows else 0
        distinct_ratio = (distinct_count / row_count) if row_count > 0 else 0.0
        return True, {
            "row_count": row_count,
            "column_count": col_count,
            "empty_result": row_count == 0,
            "distinct_count": distinct_count,
            "distinct_ratio": distinct_ratio,
        }
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def score_sql_against_question(
    question_text: str,
    sql: str,
    exec_ok: bool,
    exec_info: Any,
    fuzzy_hints: Dict[str, Any] = None,
    source_bias: float = 0.0,
) -> float:
    low = " ".join((sql or "").strip().lower().split())
    complexity = analyze_question_complexity(question_text)
    features = sql_structure_features(sql)
    score = 0.0

    if exec_ok:
        score += 5.0
    else:
        score -= 5.0

    if complexity["needs_count"]:
        score += 2.2 if features["has_count"] else -2.0
    elif features["is_count_only"]:
        score -= 4.0

    if complexity["needs_distinct"]:
        score += 1.4 if features["has_distinct"] else -0.4

    if complexity["needs_multi_table"]:
        score += 2.0 if features["has_join"] else -1.5

    if complexity["needs_group"]:
        score += 1.8 if features["has_group"] else -1.0

    if complexity["needs_nested"]:
        score += 2.0 if features["has_subquery"] else -1.6

    if complexity["needs_set_ops"]:
        score += 2.5 if features["has_set_ops"] else -2.0

    if complexity["needs_order_limit"]:
        if features["has_order"]:
            score += 1.0
        else:
            score -= 0.6
        if features["has_limit"]:
            score += 0.8

    if complexity["needs_filter"]:
        score += 0.8 if features["has_where"] else -0.3

    if question_mentions_listing(question_text):
        if features["has_count"] and not complexity["needs_count"]:
            score -= 2.2
        if any(key in low for key in [" name", ".name", "title", "fname", "lname"]):
            score += 1.2

    if complexity["extra_hard_like"]:
        if features["has_join"]:
            score += 0.8
        if features["has_subquery"]:
            score += 0.8
        if features["has_set_ops"]:
            score += 0.8
        if features["has_group"]:
            score += 0.6

    value_candidates = (fuzzy_hints or {}).get("value_candidates", [])
    for value in value_candidates[:6]:
        sval = str(value).lower()
        if sval and sval in low:
            score += 0.6

    for table in (fuzzy_hints or {}).get("candidate_tables", [])[:3]:
        if table.lower() in low:
            score += 0.5

    join_paths = (fuzzy_hints or {}).get("join_path_candidates", [])[:3]
    for join_path in join_paths:
        if sql_matches_join_path(sql, join_path):
            score += min(1.5, 0.6 + join_path.get("score", 0.0) * 0.2)
            if complexity["needs_multi_table"]:
                score += 0.4
            break
    else:
        if complexity["needs_multi_table"] and top_join_path_strength(fuzzy_hints) > 1.5:
            score -= 1.0

    graph_scores = score_sql_with_graph_consistency(sql, (fuzzy_hints or {}).get("path_graph_subspace", {}))
    coverage = graph_scores["coverage"]
    consistency = graph_scores["consistency"]
    minimality = graph_scores["minimality"]
    violation_penalty = graph_scores["violation_penalty"]
    if join_paths:
        score += 1.6 * coverage
        score += 1.1 * consistency
        score += 0.8 * minimality
        score -= 0.7 * violation_penalty
        if complexity["needs_multi_table"] and coverage < 0.4:
            score -= 1.0

    if exec_ok and isinstance(exec_info, dict):
        row_count = exec_info.get("row_count", 0)
        col_count = exec_info.get("column_count", 0)
        empty_result = exec_info.get("empty_result", False)
        distinct_ratio = exec_info.get("distinct_ratio", 0.0)

        if question_mentions_listing(question_text) and empty_result:
            score -= 0.8
        if question_mentions_listing(question_text) and row_count == 1 and not complexity["needs_count"]:
            score -= 0.6
        if question_mentions_listing(question_text) and row_count > 50:
            score += 0.2
        if complexity["needs_count"] and row_count == 1:
            score += 0.8
        if complexity["needs_group"] and row_count <= 1 and not features["has_limit"]:
            score -= 0.7
        if complexity["needs_multi_table"] and row_count > 2000:
            score -= 0.8
        if complexity["needs_distinct"] and distinct_ratio > 0.9 and row_count > 0:
            score += 0.4
        if complexity["needs_aggregation"] and col_count == 1 and not complexity["needs_count"] and not features["has_group"]:
            score -= 0.4

    score += 0.35 * sql_structural_normality(sql)
    score += source_bias
    return score


def decide_route_plan(
    question_text: str,
    framework_spec: Dict[str, Any],
    framework_weak: bool,
    fuzzy_hints: Dict[str, Any],
    base_fill_candidates: int,
    base_vote_n: int,
) -> Dict[str, Any]:
    complexity = analyze_question_complexity(question_text)
    top_path = top_join_path_strength(fuzzy_hints)
    framework_sql = get_framework_template(framework_spec).lower()
    framework_has_join = " join " in framework_sql
    subspace = (fuzzy_hints or {}).get("schema_subspace", {})
    subspace_table_count = len(subspace.get("top_tables", []))
    subspace_is_narrow = subspace_table_count <= 2 and top_path >= 1.8

    direct_sample_count = max(2, base_fill_candidates, base_vote_n)
    schema_first_count = max(3, base_fill_candidates)
    use_schema_first = False
    prefer_framework_fill = False

    if complexity["extra_hard_like"]:
        direct_sample_count += 2
        schema_first_count += 2
        use_schema_first = True
    if framework_weak:
        direct_sample_count += 2
        schema_first_count += 2
        use_schema_first = True
    if complexity["needs_multi_table"] and top_path >= 2.5:
        schema_first_count += 2
        use_schema_first = True
    if complexity["needs_multi_table"] and not framework_has_join and top_path >= 1.5:
        direct_sample_count += 1
        schema_first_count += 2
        use_schema_first = True
    if subspace_is_narrow and complexity["needs_multi_table"]:
        schema_first_count += 2
        direct_sample_count = max(2, direct_sample_count - 1)
        use_schema_first = True
    if (not framework_weak) and framework_has_join and top_path >= 1.5:
        prefer_framework_fill = True
    if subspace_table_count >= 4 and top_path < 1.5:
        direct_sample_count += 2
        use_schema_first = False

    return {
        "direct_sample_count": direct_sample_count,
        "schema_first_count": schema_first_count,
        "use_schema_first": use_schema_first,
        "top_join_path_strength": top_path,
        "subspace_table_count": subspace_table_count,
        "subspace_is_narrow": subspace_is_narrow,
        "prefer_framework_fill": prefer_framework_fill,
    }


def framework_spec_is_weak(
    framework_spec: Dict[str, Any],
    question_text: str,
    fuzzy_hints: Dict[str, Any],
) -> bool:
    complexity = analyze_question_complexity(question_text)
    sql_template = get_framework_template(framework_spec).lower()
    tables = framework_spec.get("tables", []) if isinstance(framework_spec, dict) else []
    if framework_spec.get("intent") == "fallback":
        return True
    if sql_template.strip() in {"select <column_1> from <table_1>", "select count(*) from <table_1>"}:
        return True
    if complexity["needs_multi_table"] and " join " not in sql_template and len(tables) <= 1:
        return True
    if complexity["needs_nested"] and "(select " not in sql_template:
        return True
    if complexity["needs_set_ops"] and not any(op in sql_template for op in [" union ", " intersect ", " except "]):
        return True
    if complexity["needs_group"] and " group by " not in sql_template and " having " not in sql_template:
        return True
    candidate_tables = (fuzzy_hints or {}).get("candidate_tables", [])
    if candidate_tables and len(tables) == 1 and tables[0] == "<TABLE_1>" and complexity["extra_hard_like"]:
        return True
    return False


def choose_fallback_table(
    catalog: Dict[str, List[str]],
    fuzzy_hints: Dict[str, Any],
) -> str:
    candidate_tables = fuzzy_hints.get("candidate_tables", []) if isinstance(fuzzy_hints, dict) else []
    for table in candidate_tables:
        if table in catalog:
            return table
    return sorted(catalog.keys())[0] if catalog else ""


def score_column_name(col: str, question_text: str) -> float:
    c = normalize_identifier(col)
    q = normalize_identifier(question_text)
    score = difflib.SequenceMatcher(None, q, c).ratio()
    if any(tok in c for tok in q.split()):
        score += 0.3
    return score


def choose_name_like_column(columns: List[str], question_text: str) -> str:
    preferred_keys = [
        "name", "title", "fname", "lname", "first_name", "last_name",
        "nickname", "type", "code", "id",
    ]
    best_col = ""
    best_score = -1.0
    for col in columns:
        low = col.lower()
        score = score_column_name(col, question_text)
        for idx, key in enumerate(preferred_keys):
            if key in low:
                score += 2.0 - (idx * 0.1)
        if score > best_score:
            best_col = col
            best_score = score
    return best_col or (columns[0] if columns else "")


def choose_metric_like_column(columns: List[str], question_text: str) -> str:
    preferred_keys = [
        "age", "year", "date", "time", "score", "salary", "price", "weight",
        "height", "population", "rating", "duration", "size", "amount", "count",
        "number", "total",
    ]
    best_col = ""
    best_score = -1.0
    for col in columns:
        low = col.lower()
        score = score_column_name(col, question_text)
        for idx, key in enumerate(preferred_keys):
            if key in low:
                score += 2.2 - (idx * 0.08)
        if score > best_score:
            best_col = col
            best_score = score
    return best_col or choose_name_like_column(columns, question_text)


def fallback_sql_from_catalog(
    catalog: Dict[str, List[str]],
    question_text: str = "",
    fuzzy_hints: Dict[str, Any] = None,
) -> str:
    if not catalog:
        return "SELECT 0"

    table = choose_fallback_table(catalog, fuzzy_hints or {})
    columns = catalog.get(table, [])
    name_col = choose_name_like_column(columns, question_text)
    metric_col = choose_metric_like_column(columns, question_text)
    wants_count = question_mentions_count(question_text)
    wants_distinct = question_mentions_distinct(question_text)
    order_dir, wants_superlative = question_mentions_superlative(question_text)
    wants_listing = question_mentions_listing(question_text)

    if wants_count:
        return f"SELECT COUNT(*) FROM {table}"

    if wants_superlative and metric_col:
        select_col = name_col or metric_col
        if wants_distinct and select_col:
            return f"SELECT DISTINCT {select_col} FROM {table} ORDER BY {metric_col} {order_dir.upper()} LIMIT 1"
        return f"SELECT {select_col} FROM {table} ORDER BY {metric_col} {order_dir.upper()} LIMIT 1"

    if wants_listing and name_col:
        if wants_distinct:
            return f"SELECT DISTINCT {name_col} FROM {table}"
        return f"SELECT {name_col} FROM {table}"

    if name_col:
        return f"SELECT {name_col} FROM {table}"
    if columns:
        return f"SELECT {columns[0]} FROM {table}"
    return f"SELECT COUNT(*) FROM {table}"


def build_relaxed_direct_sql_prompt(
    original_prompt: str,
    framework_spec: Dict[str, Any],
    fuzzy_hints: Dict[str, Any],
    question_text: str,
) -> str:
    return (
        "You are an expert SQL generator.\n"
        "Generate one executable SQL query for the task context.\n"
        "Use framework and hints as soft guidance, but you may deviate when needed to keep SQL valid.\n"
        "Constraints:\n"
        "1) Output only one SQL statement.\n"
        "2) Do not output placeholders like <TABLE_1> or __INVALID__.\n"
        "3) Avoid trivial fallback SQL such as SELECT 1.\n\n"
        "Complexity hints inferred from the question:\n"
        f"{build_complexity_hint_block(question_text)}\n\n"
        "Candidate schema subspace:\n"
        f"{build_schema_subspace_text(fuzzy_hints)}\n\n"
        "Framework (soft):\n"
        f"{json.dumps(framework_spec, ensure_ascii=False, indent=2)}\n\n"
        "Schema/value hints (soft):\n"
        f"{json.dumps(fuzzy_hints, ensure_ascii=False, indent=2)}\n\n"
        "Task context:\n"
        f"{original_prompt}\n"
    )


def build_schema_first_sql_prompt(
    original_prompt: str,
    fuzzy_hints: Dict[str, Any],
    question_text: str,
) -> str:
    return (
        "You are an expert Text-to-SQL model solving a difficult query.\n"
        "Generate one executable SQL query directly from the task context and schema hints.\n"
        "Do not force a simplified skeleton if it hurts correctness.\n"
        "If the question appears complex, prefer the semantically complete SQL even if it is longer.\n"
        "Output SQL only, with no explanation and no markdown.\n\n"
        "Complexity hints inferred from the question:\n"
        f"{build_complexity_hint_block(question_text)}\n\n"
        "Candidate schema subspace:\n"
        f"{build_schema_subspace_text(fuzzy_hints)}\n\n"
        "Schema/value hints:\n"
        f"{json.dumps(fuzzy_hints, ensure_ascii=False, indent=2)}\n\n"
        "Task context:\n"
        f"{original_prompt}\n"
    )


def rescue_sql_with_relaxed_generation(
    model: str,
    original_prompt: str,
    framework_spec: Dict[str, Any],
    fuzzy_hints: Dict[str, Any],
    question_text: str,
    db_path: str,
    temperature: float,
    samples: int,
) -> str:
    prompt = build_relaxed_direct_sql_prompt(original_prompt, framework_spec, fuzzy_hints, question_text)
    try:
        res = ask_llm(model, [prompt], max(0.1, temperature), max(1, samples))
        cands = [normalize_sql_output(x) for x in to_candidate_list(res["response"])]
        cands = dedup_keep_order(cands)
        cands = [s for s in cands if not is_bad_final_sql(s)]
        for cand in cands:
            ok, _ = execute_sql_once(db_path, cand)
            if ok:
                return cand
        if cands:
            return cands[0]
    except Exception:
        pass
    return ""


def collect_relaxed_direct_candidates(
    model: str,
    original_prompt: str,
    framework_spec: Dict[str, Any],
    fuzzy_hints: Dict[str, Any],
    question_text: str,
    temperature: float,
    samples: int,
) -> List[str]:
    prompt = build_relaxed_direct_sql_prompt(original_prompt, framework_spec, fuzzy_hints, question_text)
    try:
        res = ask_llm(model, [prompt], max(0.1, temperature), max(1, samples))
        cands = [normalize_sql_output(x) for x in to_candidate_list(res["response"])]
        cands = dedup_keep_order(cands)
        return [s for s in cands if not is_bad_final_sql(s)]
    except Exception:
        return []


def collect_schema_first_candidates(
    model: str,
    original_prompt: str,
    fuzzy_hints: Dict[str, Any],
    question_text: str,
    temperature: float,
    samples: int,
) -> List[str]:
    prompt = build_schema_first_sql_prompt(original_prompt, fuzzy_hints, question_text)
    try:
        res = ask_llm(model, [prompt], max(0.1, temperature), max(1, samples))
        cands = [normalize_sql_output(x) for x in to_candidate_list(res["response"])]
        cands = dedup_keep_order(cands)
        return [s for s in cands if not is_bad_final_sql(s)]
    except Exception:
        return []


def pick_best_candidate(
    candidates: List[str],
    db_path: str,
    question_text: str,
    fuzzy_hints: Dict[str, Any],
    prefer_richer: bool = False,
) -> Tuple[str, bool, str]:
    best_executable_sql = ""
    best_executable_score = None
    best_non_trivial_failed_sql = ""
    best_non_trivial_failed_err = ""
    trivial_executable_sql = ""
    for cand in candidates:
        norm = normalize_sql_output(cand)
        ok, err = execute_sql_with_stats(db_path, norm)
        if ok and not is_trivial_fallback_sql(norm):
            source_bias = 0.4 if prefer_richer else 0.0
            score = score_sql_against_question(question_text, norm, True, err, fuzzy_hints, source_bias=source_bias)
            if best_executable_score is None or score > best_executable_score:
                best_executable_sql = norm
                best_executable_score = score
            continue
        if ok and is_trivial_fallback_sql(norm) and not trivial_executable_sql:
            trivial_executable_sql = norm
        if (not ok) and (not is_trivial_fallback_sql(norm)) and not best_non_trivial_failed_sql:
            best_non_trivial_failed_sql = norm
            best_non_trivial_failed_err = err

    if best_executable_sql:
        return best_executable_sql, True, ""
    if best_non_trivial_failed_sql:
        return best_non_trivial_failed_sql, False, best_non_trivial_failed_err
    if trivial_executable_sql:
        return trivial_executable_sql, False, "trivial fallback sql rejected"
    if candidates:
        return normalize_sql_output(candidates[0]), False, "no executable non-trivial candidate"
    return "SELECT 1", False, "empty candidate set"


def build_framework_prompt(original_prompt: str) -> str:
    return (
        "You are a Text-to-SQL planner.\\n"
        "Generate one SQL framework only.\\n"
        "Rules:\\n"
        "1) Output one single SQL statement and nothing else.\\n"
        "2) Keep SQL structure (SELECT/FROM/JOIN/WHERE/GROUP BY/ORDER BY/LIMIT/subqueries).\\n"
        "3) Use placeholders for concrete schema/value tokens, for example <TABLE_1>, <COLUMN_1>, <VALUE_1>.\\n"
        "4) Never output SELECT 1 or similar trivial SQL.\\n"
        "5) Do not output markdown fences or explanations.\\n\\n"
        "Original task context:\\n"
        f"{original_prompt}\\n"
    )


def build_fill_prompt(original_prompt: str, framework_sql: str) -> str:
    return (
        "You are a Text-to-SQL engine.\\n"
        "Given an original Text-to-SQL task and a SQL framework, fill placeholders with concrete tables/columns/values.\\n"
        "Never output trivial fallback SQL like SELECT 1.\\n"
        "Return one executable SQL query only. No explanation, no markdown fences.\\n\\n"
        "Original task context:\\n"
        f"{original_prompt}\\n\\n"
        "SQL framework:\\n"
        f"{framework_sql}\\n"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str)
    parser.add_argument("--openai_api_key", type=str)
    parser.add_argument("--openai_group_id", type=str, default="")
    parser.add_argument("--openai_base_url", type=str, default="")
    parser.add_argument("--model", type=str, default=LLM.GLM_45_AIR)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=1000000)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--mini_index_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n", type=int, default=5, help="Size of self-consistent set")
    parser.add_argument("--db_dir", type=str, default="dataset/spider/database")
    parser.add_argument("--two_stage_framework", action="store_true",
                        help="Generate SQL framework first, then fill placeholders into final SQL")
    parser.add_argument("--framework_temperature", type=float, default=0.0,
                        help="Temperature used in SQL framework generation stage")
    parser.add_argument("--fill_candidates", type=int, default=3,
                        help="Number of SQL candidates sampled in fill stage when two_stage_framework is enabled")
    parser.add_argument("--repair_rounds", type=int, default=2,
                        help="Maximum auto-repair rounds using execution feedback in two-stage mode")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Append suffix to output filenames to avoid overwriting existing results")
    parser.add_argument("--result_output_dir", type=str, default="results",
                        help="Directory to write final SQL outputs")
    parser.add_argument("--write_framework_outputs", action="store_true",
                        help="Whether to write framework/spec/trace outputs")
    args = parser.parse_args()

    # check args
    assert args.model in LLM.BATCH_FORWARD or \
           args.model not in LLM.BATCH_FORWARD and args.batch_size == 1, \
        f"{args.model} doesn't support batch_size > 1"
    if args.two_stage_framework:
        assert args.batch_size == 1, "two_stage_framework currently requires batch_size == 1"

    questions_json = json.load(open(os.path.join(args.question, QUESTION_FILE), "r"))
    questions = [_["prompt"] for _ in questions_json["questions"]]
    db_ids = [_["db_id"] for _ in questions_json["questions"]]

    # init openai api
    if args.openai_base_url:
        os.environ["OPENAI_API_BASE"] = args.openai_base_url
    init_chatgpt(args.openai_api_key, args.openai_group_id, args.model)

    if args.start_index == 0:
        mode = "w"
    else:
        mode = "a"

    if args.mini_index_path:
        mini_index = json.load(open(args.mini_index_path, 'r'))
        questions = [questions[i] for i in mini_index]
        db_ids = [db_ids[i] for i in mini_index]
        base_out = f"{args.result_output_dir}/RESULTS_MODEL-{args.model}_MINI.txt"
    else:
        base_out = f"{args.result_output_dir}/RESULTS_MODEL-{args.model}.txt"

    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    if suffix:
        if args.mini_index_path:
            out_file = f"{args.result_output_dir}/RESULTS_MODEL-{args.model}_MINI{suffix}.txt"
        else:
            out_file = f"{args.result_output_dir}/RESULTS_MODEL-{args.model}{suffix}.txt"
    else:
        out_file = base_out

    question_loader = DataLoader(questions, batch_size=args.batch_size, shuffle=False, drop_last=False)

    token_cnt = 0
    os.makedirs(args.result_output_dir, exist_ok=True)
    framework_out_file = f"{args.result_output_dir}/FRAMEWORK_MODEL-{args.model}{suffix}.txt"
    framework_spec_out_file = f"{args.result_output_dir}/FRAMEWORK_SPEC_MODEL-{args.model}{suffix}.jsonl"
    repair_trace_out_file = f"{args.result_output_dir}/REPAIR_TRACE_MODEL-{args.model}{suffix}.jsonl"
    candidate_trace_out_file = f"{args.result_output_dir}/CANDIDATE_TRACE_MODEL-{args.model}{suffix}.jsonl"

    with open(out_file, mode) as f:
        ir_fw = None
        fw = None
        fw_spec = None
        repair_trace = None
        candidate_trace = None
        if args.two_stage_framework and args.write_framework_outputs:
            fw = open(framework_out_file, mode)
            fw_spec = open(framework_spec_out_file, mode)
            repair_trace = open(repair_trace_out_file, mode)
            candidate_trace = open(candidate_trace_out_file, mode)
        for i, batch in enumerate(tqdm(question_loader)):
            if i < args.start_index:
                continue
            if i >= args.end_index:
                break
            if args.two_stage_framework:
                original_prompt = batch[0]
                db_id = db_ids[i]
                db_path = db_file(args.db_dir, db_id)
                question_text = extract_question_text(original_prompt)
                complexity_info = analyze_question_complexity(question_text)
                schema_catalog = load_schema_catalog(db_path)
                fuzzy_hints = fuzzy_match_schema(question_text, schema_catalog, top_k_tables=4)
                fuzzy_hints = enrich_fuzzy_hints_with_join_paths(args.db_dir, db_id, question_text, fuzzy_hints)
                fuzzy_hints["value_candidates"] = extract_values_from_question(question_text)
                fuzzy_hints = build_semantic_subspace(args.db_dir, db_id, question_text, fuzzy_hints)
                rewritten_question = rewrite_question_with_subspace(question_text, fuzzy_hints)

                framework_prompt = build_framework_prompt_advanced(
                    augment_prompt_with_rewrite(original_prompt, rewritten_question),
                    question_text,
                )
                try:
                    framework_res = ask_llm(args.model, [framework_prompt], args.framework_temperature, 1)
                    framework_raw = framework_res["response"][0]
                    framework_spec = normalize_framework_spec(parse_framework_json(framework_raw))
                    framework_sql = get_framework_template(framework_spec)
                except Exception as e:
                    err_name = e.__class__.__name__
                    if err_name in ["InvalidRequestError", "BadRequestError"]:
                        framework_spec = normalize_framework_spec({
                            "intent": "fallback",
                            "tables": ["<TABLE_1>"],
                            "joins": [],
                            "select": ["<COLUMN_1>"],
                            "where": [],
                            "group_by": [],
                            "order_by": [],
                            "set_ops": [],
                            "subqueries": [],
                            "sql_template": "SELECT <COLUMN_1> FROM <TABLE_1>",
                            "confidence": 0.05,
                            "alternatives": [],
                        })
                        framework_sql = "SELECT <COLUMN_1> FROM <TABLE_1>"
                        framework_res = {"total_tokens": 0}
                    else:
                        raise

                framework_weak = framework_spec_is_weak(framework_spec, question_text, fuzzy_hints)

                if fw is not None:
                    fw.write(framework_sql + "\n")
                    fw.flush()
                if fw_spec is not None:
                    fw_spec.write(json.dumps({
                        "db_id": db_id,
                        "framework": framework_spec,
                        "fuzzy_hints": fuzzy_hints,
                        "framework_weak": framework_weak,
                        "complexity": complexity_info,
                        "top_join_path_strength": top_join_path_strength(fuzzy_hints),
                    }, ensure_ascii=False) + "\n")
                    fw_spec.flush()

                # Route A: framework-guided fill generation.
                fill_prompt = build_fill_prompt_advanced(
                    augment_prompt_with_rewrite(original_prompt, rewritten_question),
                    framework_spec,
                    fuzzy_hints,
                    question_text,
                    framework_weak,
                )
                try:
                    fill_res = ask_llm(args.model, [fill_prompt], args.temperature, max(1, args.fill_candidates, args.n))
                except Exception as e:
                    err_name = e.__class__.__name__
                    if err_name in ["InvalidRequestError", "BadRequestError"]:
                        print(f"The {i}-th question has too much tokens! Return \"SELECT\" instead")
                        fill_res = {"response": [""], "total_tokens": 0}
                    else:
                        raise

                candidate_records = [
                    make_candidate_record(
                        x,
                        source="stage2",
                        route="framework-fill",
                        framework_confidence=framework_spec.get("confidence", 0.0),
                    )
                    for x in to_candidate_list(fill_res["response"])
                ]

                # Route B: relaxed direct generation; merge with framework route.
                route_plan = decide_route_plan(
                    question_text,
                    framework_spec,
                    framework_weak,
                    fuzzy_hints,
                    args.fill_candidates,
                    args.n,
                )
                direct_sample_count = route_plan["direct_sample_count"]
                direct_candidates = collect_relaxed_direct_candidates(
                    args.model,
                    augment_prompt_with_rewrite(original_prompt, rewritten_question),
                    framework_spec,
                    fuzzy_hints,
                    question_text,
                    max(0.2, args.temperature),
                    direct_sample_count,
                )
                candidate_records.extend(
                    make_candidate_record(
                        x,
                        source="stage2",
                        route="direct-relaxed",
                        framework_confidence=framework_spec.get("confidence", 0.0) * 0.5,
                    )
                    for x in direct_candidates
                )

                # Route C: schema-first direct generation for extra-hard or weak-framework cases.
                if route_plan["use_schema_first"]:
                    schema_first_candidates = collect_schema_first_candidates(
                        args.model,
                        augment_prompt_with_rewrite(original_prompt, rewritten_question),
                        fuzzy_hints,
                        question_text,
                        max(0.3, args.temperature),
                        route_plan["schema_first_count"],
                    )
                    candidate_records.extend(
                        make_candidate_record(
                            x,
                            source="stage2",
                            route="schema-first",
                            framework_confidence=0.0,
                        )
                        for x in schema_first_candidates
                    )

                candidate_records = dedup_candidate_records(candidate_records)
                candidates = [rec["sql"] for rec in candidate_records]
                candidates = filter_non_trivial_sqls(candidates)

                # Re-enable voting/self-consistency on two-stage outputs.
                voted_sql = None
                if args.n > 1 and candidates:
                    try:
                        voted = get_sqls([{"db_id": db_id, "p_sqls": candidates}], args.n, args.db_dir)
                        if voted:
                            voted_sql = normalize_sql_output(voted[0])
                    except Exception:
                        voted_sql = None
                if voted_sql:
                    if not is_trivial_fallback_sql(voted_sql):
                        candidate_records.insert(0, make_candidate_record(
                            voted_sql,
                            source="stage2",
                            route="candidate-vote",
                            framework_confidence=framework_spec.get("confidence", 0.0),
                        ))
                        candidate_records = dedup_candidate_records(candidate_records)
                        candidates = [rec["sql"] for rec in candidate_records]

                candidates = filter_non_trivial_sqls(candidates)
                if not candidates:
                    # Force a non-trivial failing SQL to trigger repair, never fallback to SELECT 1.
                    candidates = [INVALID_SQL_SENTINEL]
                    candidate_records = [make_candidate_record(
                        INVALID_SQL_SENTINEL,
                        source="stage2",
                        route="forced-invalid",
                        framework_confidence=0.0,
                    )]

                # Syntax-level correction before execution-level filtering.
                corrected_candidate_records = []
                for rec in candidate_records:
                    cand = rec["sql"]
                    ok_sanity, sanity_err = simple_sql_sanity_check(cand)
                    if ok_sanity:
                        corrected_candidate_records.append(rec)
                    else:
                        fixed = repair_sql_by_error_type(
                            original_prompt,
                            # Preserve the rewritten structure in repair prompts too.
                            framework_spec,
                            cand,
                            sanity_err,
                            args.model,
                            question_text,
                            fuzzy_hints,
                            db_path,
                            framework_weak,
                        )
                        new_rec = dict(rec)
                        new_rec["sql"] = fixed
                        new_rec["route"] = f"{rec.get('route','unknown')}-sanity-repair"
                        corrected_candidate_records.append(new_rec)
                candidate_records = dedup_candidate_records(corrected_candidate_records)
                candidate_records = [rec for rec in candidate_records if not is_trivial_fallback_sql(rec["sql"])]
                candidates = [rec["sql"] for rec in candidate_records]
                if not candidates:
                    candidate_records = [make_candidate_record(
                        INVALID_SQL_SENTINEL,
                        source="stage2",
                        route="post-sanity-invalid",
                        framework_confidence=0.0,
                    )]
                    candidates = [INVALID_SQL_SENTINEL]

                ranked_candidates = rank_candidate_records(
                    candidate_records,
                    db_path,
                    question_text,
                    fuzzy_hints,
                    prefer_richer=complexity_info["extra_hard_like"] or framework_weak,
                )
                if ranked_candidates:
                    best_rec = ranked_candidates[0]
                    best_sql = best_rec["sql"]
                    ok = best_rec["exec_ok"]
                    err_msg = "" if ok else str(best_rec["exec_info"])
                else:
                    best_sql, ok, err_msg = pick_best_candidate(
                        candidates,
                        db_path,
                        question_text,
                        fuzzy_hints,
                        prefer_richer=complexity_info["extra_hard_like"] or framework_weak,
                    )

                repair_logs = []
                for _ in range(max(0, args.repair_rounds)):
                    if ok:
                        break
                    repair_prompt = build_repair_prompt(
                        original_prompt,
                        # Preserve the rewritten structure in repair prompts too.
                        framework_spec,
                        best_sql,
                        err_msg,
                        question_text,
                        fuzzy_hints,
                        framework_weak,
                    )
                    try:
                        repair_res = ask_llm(args.model, [repair_prompt], 0.0, 1)
                        repaired_sql = normalize_sql_output(repair_res["response"][0])
                    except Exception:
                        break

                    ok, new_err = execute_sql_once(db_path, repaired_sql)
                    if ok and is_trivial_fallback_sql(repaired_sql):
                        ok = False
                        new_err = "trivial fallback sql rejected"
                    repair_logs.append({
                        "failed_sql": best_sql,
                        "error": err_msg,
                        "repaired_sql": repaired_sql,
                        "repaired_ok": ok,
                    })
                    best_sql = repaired_sql
                    err_msg = new_err

                if is_trivial_fallback_sql(best_sql):
                    forced_sql = repair_sql_by_error_type(
                        original_prompt,
                        framework_spec,
                        best_sql,
                        "trivial fallback sql rejected",
                        args.model,
                        question_text,
                        fuzzy_hints,
                        db_path,
                        framework_weak,
                    )
                    if not is_trivial_fallback_sql(forced_sql):
                        best_sql = forced_sql

                best_sql = enforce_non_trivial_sql(best_sql)
                if is_bad_final_sql(best_sql):
                    rescued_sql = rescue_sql_with_relaxed_generation(
                        args.model,
                        augment_prompt_with_rewrite(original_prompt, rewritten_question),
                        framework_spec,
                        fuzzy_hints,
                        question_text,
                        db_path,
                        args.temperature,
                        max(3, direct_sample_count),
                    )
                    if rescued_sql:
                        best_sql = rescued_sql
                if is_bad_final_sql(best_sql):
                    best_sql = fallback_sql_from_catalog(schema_catalog, question_text, fuzzy_hints)

                if candidate_trace is not None:
                    candidate_trace.write(json.dumps({
                        "db_id": db_id,
                        "question": question_text,
                        "complexity": complexity_info,
                        "route_plan": route_plan,
                        "framework_confidence": framework_spec.get("confidence", 0.0),
                        "framework_weak": framework_weak,
                        "ranked_candidates": ranked_candidates[:8],
                        "selected_sql": best_sql,
                    }, ensure_ascii=False) + "\n")
                    candidate_trace.flush()

                if repair_trace is not None:
                    repair_trace.write(json.dumps({
                        "db_id": db_id,
                        "selected_sql": best_sql,
                        "selected_ok": ok,
                        "final_error": err_msg,
                        "repair_logs": repair_logs,
                    }, ensure_ascii=False) + "\n")
                    repair_trace.flush()

                f.write(best_sql + "\n")
                f.flush()

                token_cnt += framework_res.get("total_tokens", 0)
                token_cnt += fill_res.get("total_tokens", 0)
                continue
            else:
                try:
                    res = ask_llm(args.model, batch, args.temperature, args.n)
                except Exception as e:
                    err_name = e.__class__.__name__
                    if err_name in ["InvalidRequestError", "BadRequestError"]:
                        print(f"The {i}-th question has too much tokens! Return \"SELECT\" instead")
                        res = {"response": [""], "total_tokens": 0}
                    else:
                        raise

            # parse result
            token_cnt += res["total_tokens"]
            if args.n == 1:
                cur_db_ids = db_ids[i * args.batch_size: i * args.batch_size + len(batch)]
                for sql, db_id, original_prompt in zip(res["response"], cur_db_ids, batch):
                    sql = normalize_sql_output(sql)
                    sql = enforce_non_trivial_sql(sql)
                    if is_bad_final_sql(sql):
                        db_path = db_file(args.db_dir, db_id)
                        schema_catalog = load_schema_catalog(db_path)
                        question_text = extract_question_text(original_prompt)
                        fuzzy_hints = fuzzy_match_schema(question_text, schema_catalog, top_k_tables=4)
                        fuzzy_hints = enrich_fuzzy_hints_with_join_paths(args.db_dir, db_id, question_text, fuzzy_hints)
                        fuzzy_hints["value_candidates"] = extract_values_from_question(question_text)
                        fuzzy_hints = build_semantic_subspace(args.db_dir, db_id, question_text, fuzzy_hints)
                        rescued_sql = rescue_sql_with_relaxed_generation(
                            args.model,
                            augment_prompt_with_rewrite(original_prompt, rewritten_question),
                            {},
                            fuzzy_hints,
                            question_text,
                            db_path,
                            args.temperature,
                            max(2, args.n),
                        )
                        sql = rescued_sql if rescued_sql else fallback_sql_from_catalog(schema_catalog, question_text, fuzzy_hints)
                    f.write(sql + "\n")
                f.flush()
            else:
                results = []
                cur_db_ids = db_ids[i * args.batch_size: i * args.batch_size + len(batch)]
                for sqls, db_id, original_prompt in zip(res["response"], cur_db_ids, batch):
                    processed_sqls = []
                    for sql in sqls:
                        sql = normalize_sql_output(sql)
                        if is_trivial_fallback_sql(sql):
                            continue
                        processed_sqls.append(sql)
                    if not processed_sqls:
                        processed_sqls = [INVALID_SQL_SENTINEL]
                    result = {
                        'db_id': db_id,
                        'p_sqls': processed_sqls
                    }
                    final_sqls = get_sqls([result], args.n, args.db_dir)

                    for sql in final_sqls:
                        sql = enforce_non_trivial_sql(normalize_sql_output(sql))
                        if is_bad_final_sql(sql):
                            db_path = db_file(args.db_dir, db_id)
                            schema_catalog = load_schema_catalog(db_path)
                            question_text = extract_question_text(original_prompt)
                            fuzzy_hints = fuzzy_match_schema(question_text, schema_catalog, top_k_tables=4)
                            fuzzy_hints = enrich_fuzzy_hints_with_join_paths(args.db_dir, db_id, question_text, fuzzy_hints)
                            fuzzy_hints["value_candidates"] = extract_values_from_question(question_text)
                            fuzzy_hints = build_semantic_subspace(args.db_dir, db_id, question_text, fuzzy_hints)
                            rewritten_question = rewrite_question_with_subspace(question_text, fuzzy_hints)
                            rescued_sql = rescue_sql_with_relaxed_generation(
                                args.model,
                                augment_prompt_with_rewrite(original_prompt, rewritten_question),
                                {},
                                fuzzy_hints,
                                question_text,
                                db_path,
                                args.temperature,
                                max(2, args.n),
                            )
                            sql = rescued_sql if rescued_sql else fallback_sql_from_catalog(schema_catalog, question_text, fuzzy_hints)
                        f.write(sql + "\n")
                f.flush()

        if ir_fw is not None:
            ir_fw.close()
        if fw is not None:
            fw.close()
        if fw_spec is not None:
            fw_spec.close()
        if repair_trace is not None:
            repair_trace.close()
        if candidate_trace is not None:
            candidate_trace.close()
