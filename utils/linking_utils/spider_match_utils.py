import collections
import difflib
import re
import string

import nltk.corpus
from nltk.stem import PorterStemmer
from utils.runtime_setup import configure_local_runtime


configure_local_runtime()

try:
    STOPWORDS = set(nltk.corpus.stopwords.words("english"))
except LookupError:
    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
        "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were",
        "what", "when", "where", "which", "who", "with",
    }
PUNKS = set(a for a in string.punctuation)
STEMMER = PorterStemmer()

CELL_EXACT_MATCH_FLAG = "EXACTMATCH"
CELL_PARTIAL_MATCH_FLAG = "PARTIALMATCH"
COL_PARTIAL_MATCH_FLAG = "CPM"
COL_EXACT_MATCH_FLAG = "CEM"
TAB_PARTIAL_MATCH_FLAG = "TPM"
TAB_EXACT_MATCH_FLAG = "TEM"

# A small alias map helps recall on medium/hard questions that use common
# abbreviations instead of literal schema names.
COMMON_SCHEMA_ALIASES = {
    "num": "number",
    "nums": "number",
    "qty": "quantity",
    "avg": "average",
    "dept": "department",
    "dob": "birth",
    "fname": "first",
    "lname": "last",
    "id": "identifier",
    "ids": "identifier",
}


def _split_identifier(text):
    if text is None:
        return []
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ")
    pieces = []
    for token in text.split():
        token = token.strip(string.punctuation).lower()
        if token:
            pieces.append(token)
    return pieces


def _normalize_token(token):
    token = token.strip(string.punctuation).lower()
    if not token:
        return ""
    token = COMMON_SCHEMA_ALIASES.get(token, token)
    if len(token) > 3 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def _normalize_tokens(tokens):
    normalized = []
    for token in tokens:
        for part in _split_identifier(token):
            norm = _normalize_token(part)
            if norm:
                normalized.append(norm)
    return normalized


def _stem_tokens(tokens):
    return [STEMMER.stem(tok) for tok in tokens if tok]


def _token_overlap_score(a_tokens, b_tokens):
    if not a_tokens or not b_tokens:
        return 0.0
    a_counter = collections.Counter(a_tokens)
    b_counter = collections.Counter(b_tokens)
    overlap = sum((a_counter & b_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(len(a_tokens), 1)
    recall = overlap / max(len(b_tokens), 1)
    return 2 * precision * recall / max(precision + recall, 1e-8)


def _is_meaningful_span(tokens):
    filtered = [
        tok for tok in tokens
        if tok and tok not in STOPWORDS and tok not in PUNKS
    ]
    return len(filtered) > 0


def _score_span_to_schema(span_tokens, schema_tokens):
    span_norm = _normalize_tokens(span_tokens)
    schema_norm = _normalize_tokens(schema_tokens)
    if not _is_meaningful_span(span_norm) or not schema_norm:
        return 0.0, False

    span_join = " ".join(span_norm)
    schema_join = " ".join(schema_norm)
    span_stems = _stem_tokens(span_norm)
    schema_stems = _stem_tokens(schema_norm)

    exact = span_join == schema_join
    if exact:
        return 10.0, True

    score = 0.0
    if span_join and schema_join and re.search(rf"\b{re.escape(span_join)}\b", schema_join):
        score += 5.0
    if schema_join and span_join and re.search(rf"\b{re.escape(schema_join)}\b", span_join):
        score += 3.0

    score += 3.5 * _token_overlap_score(span_norm, schema_norm)
    score += 2.5 * _token_overlap_score(span_stems, schema_stems)
    score += 1.5 * difflib.SequenceMatcher(None, span_join, schema_join).ratio()

    if len(span_norm) == 1 and len(schema_norm) > 1 and span_norm[0] in schema_norm:
        score += 1.0
    if len(span_stems) == 1 and len(schema_stems) > 1 and span_stems[0] in schema_stems:
        score += 0.8

    return score, False


def _update_match(match_dict, score_dict, key, flag, score):
    prev_score = score_dict.get(key, -1.0)
    priority = {
        COL_EXACT_MATCH_FLAG: 3,
        TAB_EXACT_MATCH_FLAG: 3,
        COL_PARTIAL_MATCH_FLAG: 2,
        TAB_PARTIAL_MATCH_FLAG: 2,
    }
    prev_flag = match_dict.get(key)
    prev_priority = priority.get(prev_flag, -1)
    new_priority = priority.get(flag, -1)
    if new_priority > prev_priority or (new_priority == prev_priority and score > prev_score):
        match_dict[key] = flag
        score_dict[key] = score


def _matching_qids(span_tokens, schema_tokens, start_idx):
    span_norm = _normalize_tokens(span_tokens)
    schema_norm = set(_normalize_tokens(schema_tokens))
    schema_stems = set(_stem_tokens(schema_norm))
    q_ids = []

    for offset, raw_token in enumerate(span_tokens):
        token_parts = _normalize_tokens([raw_token])
        if not token_parts:
            continue
        keep = False
        for part in token_parts:
            if part in schema_norm or STEMMER.stem(part) in schema_stems:
                keep = True
                break
        if keep:
            q_ids.append(start_idx + offset)

    if q_ids:
        return q_ids
    return [start_idx + offset for offset, tok in enumerate(span_norm) if tok and tok not in STOPWORDS]


def _quote_identifier(identifier):
    return '"' + str(identifier).replace('"', '""') + '"'


def _normalize_value_text(value):
    if value is None:
        return ""
    value = str(value).strip().lower()
    return re.sub(r"\s+", " ", value)


def _build_schema_value_cache(schema, max_distinct=200):
    cache = getattr(schema, "_value_link_cache", None)
    if cache is not None:
        return cache

    cache = {}
    conn = getattr(schema, "connection", None)
    if conn is None:
        setattr(schema, "_value_link_cache", cache)
        return cache

    for col_id, column in enumerate(schema.columns):
        if col_id == 0 or column.table is None:
            continue
        if column.type not in ["text", "time", "number"]:
            continue

        table_name = _quote_identifier(column.table.orig_name)
        col_name = _quote_identifier(column.orig_name)
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT DISTINCT {col_name} FROM {table_name} "
                f"WHERE {col_name} IS NOT NULL LIMIT {max_distinct}"
            )
            rows = cursor.fetchall()
        except Exception:
            rows = []

        normalized_values = set()
        token_index = collections.defaultdict(set)
        for row in rows:
            value = row[0]
            text = _normalize_value_text(value)
            if not text or len(text) > 80:
                continue
            normalized_values.add(text)
            for token in _normalize_tokens(text.split()):
                if token and token not in STOPWORDS:
                    token_index[token].add(text)

        cache[col_id] = {
            "type": column.type,
            "values": normalized_values,
            "token_index": token_index,
        }

    setattr(schema, "_value_link_cache", cache)
    return cache


def _is_date_like(text):
    text = text.strip().lower()
    if not text:
        return False
    if re.fullmatch(r"\d{4}", text):
        return True
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", text):
        return True
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", text):
        return True
    month_names = {
        "jan", "january", "feb", "february", "mar", "march", "apr", "april",
        "may", "jun", "june", "jul", "july", "aug", "august", "sep", "sept",
        "september", "oct", "october", "nov", "november", "dec", "december",
    }
    return any(part in month_names for part in _split_identifier(text))


# schema linking, upgraded from literal matching to a hybrid lexical scorer.
def compute_schema_linking(question, column, table):
    q_col_match = {}
    q_tab_match = {}
    col_scores = {}
    tab_scores = {}

    col_id2list = {
        col_id: col_item
        for col_id, col_item in enumerate(column)
        if col_id != 0
    }
    tab_id2list = {
        tab_id: tab_item
        for tab_id, tab_item in enumerate(table)
    }

    max_n = min(6, len(question))
    for n in range(max_n, 0, -1):
        for i in range(len(question) - n + 1):
            span_tokens = question[i:i + n]
            if not _is_meaningful_span(_normalize_tokens(span_tokens)):
                continue

            for col_id, col_tokens in col_id2list.items():
                score, is_exact = _score_span_to_schema(span_tokens, col_tokens)
                if score >= 7.5 or is_exact:
                    flag = COL_EXACT_MATCH_FLAG
                elif score >= 4.2:
                    flag = COL_PARTIAL_MATCH_FLAG
                else:
                    continue
                matched_qids = _matching_qids(span_tokens, col_tokens, i)
                for q_id in matched_qids:
                    _update_match(q_col_match, col_scores, f"{q_id},{col_id}", flag, score)

            for tab_id, tab_tokens in tab_id2list.items():
                score, is_exact = _score_span_to_schema(span_tokens, tab_tokens)
                if score >= 7.2 or is_exact:
                    flag = TAB_EXACT_MATCH_FLAG
                elif score >= 4.0:
                    flag = TAB_PARTIAL_MATCH_FLAG
                else:
                    continue
                matched_qids = _matching_qids(span_tokens, tab_tokens, i)
                for q_id in matched_qids:
                    _update_match(q_tab_match, tab_scores, f"{q_id},{tab_id}", flag, score)

    return {"q_col_match": q_col_match, "q_tab_match": q_tab_match}


def compute_cell_value_linking(tokens, schema):
    def isnumber(word):
        try:
            float(word)
            return True
        except Exception:
            return False

    num_date_match = {}
    cell_match = {}
    value_cache = _build_schema_value_cache(schema)

    # Numeric and date grounding.
    for q_id, word in enumerate(tokens):
        stripped = word.strip()
        if not stripped:
            continue
        if isnumber(stripped):
            for col_id, column in enumerate(schema.columns):
                if col_id == 0 or column.table is None:
                    continue
                if column.type in ["number", "time"]:
                    num_date_match[f"{q_id},{col_id}"] = column.type.upper()
        elif _is_date_like(stripped):
            for col_id, column in enumerate(schema.columns):
                if col_id == 0 or column.table is None:
                    continue
                if column.type == "time":
                    num_date_match[f"{q_id},{col_id}"] = "TIME"

    # Cache-backed textual value linking.
    max_span = min(4, len(tokens))
    for n in range(max_span, 0, -1):
        for i in range(len(tokens) - n + 1):
            span_tokens = tokens[i:i + n]
            norm_span_tokens = _normalize_tokens(span_tokens)
            if not _is_meaningful_span(norm_span_tokens):
                continue

            span_text = " ".join(norm_span_tokens)
            span_token_set = set(tok for tok in norm_span_tokens if tok not in STOPWORDS)
            if not span_text or not span_token_set:
                continue

            for col_id, cached in value_cache.items():
                if cached["type"] not in ["text", "time"]:
                    continue

                flag = None
                if span_text in cached["values"]:
                    flag = CELL_EXACT_MATCH_FLAG
                elif n >= 2:
                    for candidate in cached["values"]:
                        cand_tokens = set(_normalize_tokens(candidate.split()))
                        if span_token_set and span_token_set.issubset(cand_tokens):
                            flag = CELL_PARTIAL_MATCH_FLAG
                            break
                elif len(span_text) >= 4 and span_text in cached["token_index"]:
                    flag = CELL_PARTIAL_MATCH_FLAG

                if flag is None:
                    continue

                for q_id in range(i, i + n):
                    key = f"{q_id},{col_id}"
                    prev_flag = cell_match.get(key)
                    if prev_flag == CELL_EXACT_MATCH_FLAG:
                        continue
                    if prev_flag == CELL_PARTIAL_MATCH_FLAG and flag == CELL_PARTIAL_MATCH_FLAG:
                        continue
                    cell_match[key] = flag

    cv_link = {"num_date_match": num_date_match, "cell_match": cell_match}
    return cv_link


def match_shift(q_col_match, q_tab_match, cell_match):

    q_id_to_match = collections.defaultdict(list)
    for match_key in q_col_match.keys():
        q_id = int(match_key.split(",")[0])
        c_id = int(match_key.split(",")[1])
        match_type = q_col_match[match_key]
        q_id_to_match[q_id].append((match_type, c_id))
    for match_key in q_tab_match.keys():
        q_id = int(match_key.split(",")[0])
        t_id = int(match_key.split(",")[1])
        match_type = q_tab_match[match_key]
        q_id_to_match[q_id].append((match_type, t_id))
    relevant_q_ids = list(q_id_to_match.keys())

    priority = []
    for q_id in q_id_to_match.keys():
        q_id_to_match[q_id] = list(set(q_id_to_match[q_id]))
        priority.append((len(q_id_to_match[q_id]), q_id))
    priority.sort()
    matches = []
    new_q_col_match, new_q_tab_match = dict(), dict()
    for _, q_id in priority:
        if not list(set(matches) & set(q_id_to_match[q_id])):
            exact_matches = []
            for match in q_id_to_match[q_id]:
                if match[0] in [COL_EXACT_MATCH_FLAG, TAB_EXACT_MATCH_FLAG]:
                    exact_matches.append(match)
            if exact_matches:
                res = exact_matches
            else:
                res = q_id_to_match[q_id]
            matches.extend(res)
        else:
            res = list(set(matches) & set(q_id_to_match[q_id]))
        for match in res:
            match_type, c_t_id = match
            if match_type in [COL_PARTIAL_MATCH_FLAG, COL_EXACT_MATCH_FLAG]:
                new_q_col_match[f"{q_id},{c_t_id}"] = match_type
            if match_type in [TAB_PARTIAL_MATCH_FLAG, TAB_EXACT_MATCH_FLAG]:
                new_q_tab_match[f"{q_id},{c_t_id}"] = match_type

    new_cell_match = dict()
    for match_key in cell_match.keys():
        q_id = int(match_key.split(",")[0])
        if q_id in relevant_q_ids:
            continue
        new_cell_match[match_key] = cell_match[match_key]

    return new_q_col_match, new_q_tab_match, new_cell_match
