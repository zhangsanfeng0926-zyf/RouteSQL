import collections
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple


def infer_table_json_path(db_dir: str) -> str:
    db_path = Path(db_dir).resolve()
    return str(db_path.parent / "tables.json")


@lru_cache(maxsize=8)
def load_tables_metadata(table_json_path: str) -> Dict[str, Dict[str, Any]]:
    path = Path(table_json_path)
    if not path.exists():
        return {}
    items = json.loads(path.read_text(encoding="utf-8"))
    return {item["db_id"]: item for item in items}


def _normalize_name(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in (text or "")).strip()


def build_schema_graph(schema_entry: Dict[str, Any]) -> Dict[str, Any]:
    tables = schema_entry.get("table_names_original", [])
    cols = schema_entry.get("column_names_original", [])
    graph = collections.defaultdict(list)
    col_meta = {}

    for idx, (table_idx, col_name) in enumerate(cols):
        if table_idx < 0:
            continue
        table_name = tables[table_idx]
        col_meta[idx] = {
            "table_idx": table_idx,
            "table_name": table_name,
            "column_name": col_name,
        }

    for left_col, right_col in schema_entry.get("foreign_keys", []):
        if left_col not in col_meta or right_col not in col_meta:
            continue
        left = col_meta[left_col]
        right = col_meta[right_col]
        graph[left["table_idx"]].append({
            "to": right["table_idx"],
            "from_table": left["table_name"],
            "to_table": right["table_name"],
            "from_col": left["column_name"],
            "to_col": right["column_name"],
        })
        graph[right["table_idx"]].append({
            "to": left["table_idx"],
            "from_table": right["table_name"],
            "to_table": left["table_name"],
            "from_col": right["column_name"],
            "to_col": left["column_name"],
        })

    return {
        "tables": tables,
        "graph": graph,
    }


def _table_name_to_idx(tables: List[str]) -> Dict[str, int]:
    out = {}
    for idx, name in enumerate(tables):
        out[name] = idx
        out[_normalize_name(name)] = idx
    return out


def enumerate_join_paths(
    schema_entry: Dict[str, Any],
    candidate_tables: List[str],
    max_hops: int = 2,
) -> List[Dict[str, Any]]:
    if not schema_entry:
        return []

    graph_info = build_schema_graph(schema_entry)
    tables = graph_info["tables"]
    graph = graph_info["graph"]
    name_to_idx = _table_name_to_idx(tables)
    seeds = []
    for table in candidate_tables:
        if table in name_to_idx:
            seeds.append(name_to_idx[table])
        else:
            norm = _normalize_name(table)
            if norm in name_to_idx:
                seeds.append(name_to_idx[norm])
    seeds = list(dict.fromkeys(seeds))
    if len(seeds) <= 1:
        return []

    paths = []
    for start in seeds:
        queue = collections.deque([(start, [], {start})])
        while queue:
            node, edges, visited = queue.popleft()
            if len(edges) > max_hops:
                continue
            if node in seeds and node != start and edges:
                paths.append({
                    "tables": [tables[start]] + [edge["to_table"] for edge in edges],
                    "edges": edges,
                    "hop_count": len(edges),
                })
            for edge in graph.get(node, []):
                nxt = edge["to"]
                if nxt in visited:
                    continue
                queue.append((nxt, edges + [edge], visited | {nxt}))

    dedup = {}
    for path in paths:
        key = (
            tuple(path["tables"]),
            tuple((e["from_table"], e["from_col"], e["to_table"], e["to_col"]) for e in path["edges"]),
        )
        dedup[key] = path
    return list(dedup.values())


def score_join_paths(
    question_text: str,
    candidate_paths: List[Dict[str, Any]],
    candidate_tables: List[str],
    candidate_columns: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    q = _normalize_name(question_text)
    ranked = []
    for path in candidate_paths:
        score = 0.0
        score += max(0.0, 2.5 - 0.7 * path.get("hop_count", 0))
        for table in path.get("tables", []):
            if _normalize_name(table) in q:
                score += 1.0
            if table in candidate_tables:
                score += 0.6
            for col in candidate_columns.get(table, [])[:4]:
                if _normalize_name(col) in q:
                    score += 0.4
        for edge in path.get("edges", []):
            if _normalize_name(edge["from_col"]) in q or _normalize_name(edge["to_col"]) in q:
                score += 0.5
        new_path = dict(path)
        new_path["score"] = score
        ranked.append(new_path)
    ranked.sort(key=lambda x: (x["score"], -x["hop_count"]), reverse=True)
    return ranked


def get_ranked_join_paths(
    table_json_path: str,
    db_id: str,
    question_text: str,
    candidate_tables: List[str],
    candidate_columns: Dict[str, List[str]],
    max_hops: int = 2,
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    meta = load_tables_metadata(table_json_path).get(db_id)
    if not meta:
        return []
    paths = enumerate_join_paths(meta, candidate_tables, max_hops=max_hops)
    ranked = score_join_paths(question_text, paths, candidate_tables, candidate_columns)
    return ranked[:top_k]


def build_candidate_schema_subspace(
    table_json_path: str,
    db_id: str,
    question_text: str,
    candidate_tables: List[str],
    candidate_columns: Dict[str, List[str]],
    value_candidates: List[str],
    max_hops: int = 2,
    top_k_paths: int = 4,
) -> Dict[str, Any]:
    join_paths = get_ranked_join_paths(
        table_json_path=table_json_path,
        db_id=db_id,
        question_text=question_text,
        candidate_tables=candidate_tables,
        candidate_columns=candidate_columns,
        max_hops=max_hops,
        top_k=top_k_paths,
    )

    top_tables = []
    table_provenance = collections.defaultdict(list)
    for path in join_paths:
        for table in path.get("tables", []):
            if table not in top_tables:
                top_tables.append(table)
            table_provenance[table].append("join_path")

    for table in candidate_tables:
        if table not in top_tables:
            top_tables.append(table)
        table_provenance[table].append("table_match")

    table_to_columns = {}
    column_provenance = {}
    for table in top_tables:
        table_to_columns[table] = candidate_columns.get(table, [])[:6]
        column_provenance[table] = {col: ["column_match"] for col in table_to_columns[table]}

    value_provenance = {str(v): ["question_value"] for v in value_candidates[:8]}

    return {
        "top_tables": top_tables[:6],
        "top_columns": table_to_columns,
        "top_join_paths": join_paths,
        "top_value_bindings": value_candidates[:8],
        "provenance": {
            "tables": {k: sorted(set(v)) for k, v in table_provenance.items()},
            "columns": column_provenance,
            "values": value_provenance,
        },
    }


def build_path_graph_subspace(
    table_json_path: str,
    db_id: str,
    question_text: str,
    candidate_tables: List[str],
    candidate_columns: Dict[str, List[str]],
    value_candidates: List[str],
    max_hops: int = 2,
    top_k_paths: int = 4,
) -> Dict[str, Any]:
    subspace = build_candidate_schema_subspace(
        table_json_path=table_json_path,
        db_id=db_id,
        question_text=question_text,
        candidate_tables=candidate_tables,
        candidate_columns=candidate_columns,
        value_candidates=value_candidates,
        max_hops=max_hops,
        top_k_paths=top_k_paths,
    )
    nodes = {
        "tables": subspace.get("top_tables", []),
        "columns": subspace.get("top_columns", {}),
        "values": subspace.get("top_value_bindings", []),
    }
    edges = {
        "join_paths": subspace.get("top_join_paths", []),
    }
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "table_count": len(nodes["tables"]),
            "path_count": len(edges["join_paths"]),
            "value_count": len(nodes["values"]),
        },
        "provenance": subspace.get("provenance", {}),
    }


def sql_matches_join_path(sql: str, join_path: Dict[str, Any]) -> bool:
    low = " ".join((sql or "").strip().lower().split())
    tables = join_path.get("tables", [])
    if not tables:
        return False
    table_hits = sum(1 for table in tables if table.lower() in low)
    edge_hits = 0
    for edge in join_path.get("edges", []):
        if edge["from_col"].lower() in low and edge["to_col"].lower() in low:
            edge_hits += 1
    return table_hits >= max(2, len(tables) - 1) or edge_hits >= 1


def sql_path_coverage(sql: str, join_paths: List[Dict[str, Any]]) -> float:
    if not join_paths:
        return 0.0
    best = 0.0
    low = " ".join((sql or "").strip().lower().split())
    for path in join_paths:
        tables = path.get("tables", [])
        edges = path.get("edges", [])
        if not tables:
            continue
        table_hits = sum(1 for table in tables if table.lower() in low)
        edge_hits = sum(1 for edge in edges if edge["from_col"].lower() in low and edge["to_col"].lower() in low)
        table_cov = table_hits / max(1, len(tables))
        edge_cov = edge_hits / max(1, len(edges)) if edges else 0.0
        best = max(best, 0.6 * table_cov + 0.4 * edge_cov)
    return best


def sql_path_consistency(sql: str, join_paths: List[Dict[str, Any]]) -> float:
    if not join_paths:
        return 0.0
    low = " ".join((sql or "").strip().lower().split())
    penalty = 0.0
    candidate_tables = set()
    for path in join_paths:
        for table in path.get("tables", []):
            candidate_tables.add(table.lower())
    sql_tables = {table for table in candidate_tables if table in low}
    if not sql_tables:
        return 0.0
    outside_hits = 0
    tokens = low.replace(",", " ").split()
    for tok in tokens:
        if tok in candidate_tables:
            continue
    if outside_hits:
        penalty += 0.3
    return max(0.0, 1.0 - penalty)


def sql_path_minimality(sql: str, join_paths: List[Dict[str, Any]]) -> float:
    if not join_paths:
        return 0.0
    low = " ".join((sql or "").strip().lower().split())
    join_count = low.count(" join ")
    best_hops = min(path.get("hop_count", 99) for path in join_paths)
    if join_count <= best_hops:
        return 1.0
    extra = join_count - best_hops
    return max(0.0, 1.0 - 0.25 * extra)


def score_sql_with_graph_consistency(sql: str, path_graph_subspace: Dict[str, Any]) -> Dict[str, float]:
    join_paths = ((path_graph_subspace or {}).get("edges", {}) or {}).get("join_paths", [])
    coverage = sql_path_coverage(sql, join_paths)
    consistency = sql_path_consistency(sql, join_paths)
    minimality = sql_path_minimality(sql, join_paths)
    violation_penalty = max(0.0, 1.0 - consistency)
    return {
        "coverage": coverage,
        "consistency": consistency,
        "minimality": minimality,
        "violation_penalty": violation_penalty,
    }
