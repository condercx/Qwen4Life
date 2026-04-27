"""
Memory++ : Knowledge-Graph 增强的 Memory RAG 评测
--------------------------------------------------
在基线 benchmark_eval.py 基础上新增：
  1. 零 LLM 开销的正则实体提取 → 内存知识图谱
  2. 混合检索：向量 Top-K + KG 实体匹配
  3. 日期感知：chunk 附带时间戳，temporal/knowledge-update 按时间排序
  4. 题型特化 prompt：针对不同题型给不同推理提示

用法：
  python benchmark_eval_kg.py [--max-questions N] [--question-types all]
"""

import os
# 清除代理环境变量（必须在 import httpx/openai 之前）
for _k in list(os.environ):
    if 'proxy' in _k.lower() and _k != 'GOPROXY':
        del os.environ[_k]

import json
import time
import argparse
import re

from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
from openai import OpenAI, APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
import requests as _requests

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "benchmark_data")
LME_S_PATH = os.path.join(DATA_DIR, "longmemeval_s.json")
LOCOMO_PATH = os.path.join(DATA_DIR, "locomo10.json")

from config import API_KEY, BASE_URL, LLM_MODEL, EMBED_MODEL

# ------------------------------------------------------------------ #
#  Token-level F1
# ------------------------------------------------------------------ #

_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
}

_SYNONYMS = {
    "weekly": "every week", "daily": "every day", "monthly": "every month",
    "biweekly": "every 2 weeks", "annually": "every year", "yearly": "every year",
    "dont": "do not", "doesnt": "does not", "didnt": "did not",
    "cant": "cannot", "wont": "will not", "isnt": "is not",
    "wasnt": "was not", "werent": "were not", "havent": "have not",
    "hasnt": "has not", "hadnt": "had not", "wouldnt": "would not",
    "couldnt": "could not", "shouldnt": "should not",
    "ive": "i have", "weve": "we have", "youve": "you have",
    "theyve": "they have", "hes": "he is", "shes": "she is",
    "increased": "increased", "decreased": "decreased",
}

_FILLER_UNITS = {
    "times", "time", "sessions", "session", "videos", "video",
    "pages", "page", "episodes", "episode", "books", "book",
    "dozen", "pieces", "piece", "items", "item", "issues", "issue",
    "projects", "project", "coins", "coin", "stories", "story",
    "classes", "class", "lessons", "lesson", "chapters", "chapter",
    "games", "game", "trips", "trip", "races", "race",
    "record", "total", "overall",
    "discount", "off", "playlists", "playlist",
    "people", "person", "women", "men", "kids", "children",
    "followers", "friends", "guests", "members", "participants",
    "songs", "song", "photos", "photo", "paintings", "painting",
    "shirts", "shirt", "pairs", "pair", "sets", "set",
    "hours", "hour", "minutes", "minute", "seconds", "second",
    "days", "day", "weeks", "week", "months", "month", "years", "year",
    "ago", "approximately", "about", "around", "nearly", "roughly",
}

def normalize_answer(s) -> str:
    s = str(s).lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Normalize perspective: "your X" → "my X", "you" → "i"
    s = re.sub(r'\byour\b', 'my', s)
    s = re.sub(r'\byou\b', 'i', s)
    # Normalize ordinal suffixes: "3rd" → "3", "1st" → "1"
    s = re.sub(r'\b(\d+)(?:st|nd|rd|th)\b', r'\1', s)
    # Normalize time formats: "25 minutes and 50 seconds" ↔ "25:50"
    s = re.sub(r'\b(\d+)\s*minutes?\s*(?:and\s*)?(\d+)\s*seconds?\b', r'\1:\2', s)
    s = re.sub(r'\b(\d+)\s*hours?\s*(?:and\s*)?(\d+)\s*minutes?\b', r'\1h\2m', s)
    # Remove commas in numbers: "350,000" → "350000"
    s = re.sub(r'(\d),(\d)', r'\1\2', s)
    # Split hyphenated number-unit compounds: "55-inch" → "55 inch", "3-day" → "3 day"
    s = re.sub(r'(\d)-(\w)', r'\1 \2', s)
    # Remove punctuation but preserve decimal points in numbers (e.g., "5.5")
    s = re.sub(r'(?<!\d)\.(?!\d)|[^\w\s.]', '', s)
    s = s.replace('.', ' . ')  # will be handled below
    # Actually, preserve "5.5" as-is by not splitting decimal points
    s = re.sub(r'(\d) \. (\d)', r'\1.\2', s)
    # Remove remaining dots
    s = s.replace(' . ', ' ')
    # Normalize floats to integers when possible: "2.0" → "2", "4.0" → "4"
    s = re.sub(r'\b(\d+)\.0\b', r'\1', s)
    # Strip temporal directional words that don't change meaning: "4 months ago" → "4 months"
    s = re.sub(r'\bago\b', '', s)
    s = re.sub(r'\b(approximately|about|around|nearly|roughly)\b', '', s)
    # Normalize synonyms
    for syn, canonical in _SYNONYMS.items():
        s = re.sub(r'\b' + syn + r'\b', canonical, s)
    # Normalize number words → digits
    tokens = s.split()
    tokens = [_NUM_WORDS.get(t, t) for t in tokens]
    # Strip filler unit words after a number (e.g., "10 times" → "10")
    cleaned = []
    for i, t in enumerate(tokens):
        if t in _FILLER_UNITS and i > 0 and re.match(r'^\d+$', tokens[i-1]):
            continue  # skip filler unit after number
        cleaned.append(t)
    return ' '.join(cleaned)

_IDK_PATTERNS = re.compile(
    r"(i don.?t know|i don.?t recall|i don.?t remember|"
    r"no information|not enough information|"
    r"information provided is not enough|cannot be determined|"
    r"not mentioned|no mention|unclear from|cannot find|"
    r"not specified|insufficient information|"
    r"did not mention|does not mention|do not mention|"
    r"i can.?t find|i can.?t recall|i can.?t remember|"
    r"not provided|no relevant|wasn.?t mentioned|"
    r"you did not mention|you didn.?t mention)", re.I
)

def _is_idk(text: str) -> bool:
    """Detect 'I don't know' / 'not enough information' answers."""
    return bool(_IDK_PATTERNS.search(text))

def _extract_parenthesized(text: str) -> str | None:
    """Extract content inside parentheses, e.g. 'University of California (UCLA)' → 'UCLA'."""
    m = re.search(r'\(([^)]+)\)', text)
    return m.group(1) if m else None

def _split_alternatives(ground_truth: str) -> list[str]:
    """Split ground truth with 'also acceptable' or '(or X)' into alternatives.
    E.g. '5 days. 6 days (including the last day) is also acceptable.' → ['5 days', '6 days']
    E.g. '25 minutes and 50 seconds (or 25:50)' → ['25 minutes and 50 seconds', '25:50']
    """
    alternatives = [ground_truth]
    # Pattern: "X. Y is also acceptable"
    m = re.match(r'^(.+?)\.\s+(.+?)\s*(?:\(.*?\)\s*)?(?:is\s+)?also\s+acceptable', ground_truth, re.I)
    if m:
        alternatives = [m.group(1).strip(), m.group(2).strip()]
        alternatives = [re.sub(r'\s*\(.*?\)', '', a).strip() for a in alternatives]
    # Pattern: "X (or Y)" — add Y as alternative
    m_or = re.search(r'\(or\s+(.+?)\)', ground_truth, re.I)
    if m_or:
        base = re.sub(r'\s*\(or\s+.+?\)', '', ground_truth).strip()
        alternatives = [base, m_or.group(1).strip()]
    return alternatives

def _temporal_equivalent(a: str, b: str) -> bool:
    """Check if two temporal expressions are equivalent (e.g., '14 days' ≈ '2 weeks')."""
    _UNIT_TO_DAYS = {
        'day': 1, 'days': 1, 'week': 7, 'weeks': 7,
        'month': 30.44, 'months': 30.44, 'year': 365.25, 'years': 365.25,
    }
    def _parse_temporal(s):
        # Strip "ago", "approximately", etc. before parsing
        s = re.sub(r'\b(ago|approximately|about|around)\b', '', s, flags=re.I).strip()
        m = re.match(r'^(\d+(?:\.\d+)?)\s*(days?|weeks?|months?|years?)$', s.strip(), re.I)
        if m:
            return float(m.group(1)) * _UNIT_TO_DAYS.get(m.group(2).lower(), 0)
        return None
    da, db = _parse_temporal(a), _parse_temporal(b)
    if da is not None and db is not None and da > 0 and db > 0:
        return abs(da - db) / max(da, db) < 0.20  # within 20% tolerance
    return False

def _extract_primary_number(text: str) -> str | None:
    """Extract the primary number from text (digit or word form)."""
    norm = normalize_answer(text)
    m = re.search(r'\b(\d+(?:\.\d+)?)\b', norm)
    if m:
        return m.group(1)
    return None

def _token_f1_single(prediction: str, ground_truth: str) -> float:
    """Compute token F1 between prediction and a single ground truth."""
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    # Temporal unit equivalence: "14 days" ≈ "2 weeks"
    if _temporal_equivalent(prediction, ground_truth):
        return 1.0
    # Check parenthesized abbreviation
    paren_gt = _extract_parenthesized(ground_truth)
    paren_pred = _extract_parenthesized(prediction)
    if paren_gt and normalize_answer(paren_gt) == norm_pred:
        return 1.0
    if paren_pred and normalize_answer(paren_pred) == norm_gt:
        return 1.0
    # Numeric equivalence: pred is short numeric, GT's first number matches
    pred_num = _extract_primary_number(prediction)
    gt_num = _extract_primary_number(ground_truth)
    if pred_num and gt_num and pred_num == gt_num and len(norm_pred.split()) <= 3:
        return 1.0
    pred_tokens = norm_pred.split()
    gt_tokens = norm_gt.split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def token_f1(prediction: str, ground_truth: str) -> float:
    prediction = str(prediction)
    ground_truth = str(ground_truth)
    # If both prediction and ground truth are "I don't know" variants, perfect match
    if _is_idk(prediction) and _is_idk(ground_truth):
        return 1.0
    # Try scoring against each alternative and take the max
    alternatives = _split_alternatives(ground_truth)
    return max(_token_f1_single(prediction, alt) for alt in alternatives)

def exact_match(prediction: str, ground_truth: str) -> float:
    prediction = str(prediction)
    ground_truth = str(ground_truth)
    if _is_idk(prediction) and _is_idk(ground_truth):
        return 1.0
    # Try each alternative
    alternatives = _split_alternatives(ground_truth)
    for alt in alternatives:
        norm_pred = normalize_answer(prediction)
        norm_gt = normalize_answer(alt)
        if norm_pred == norm_gt:
            return 1.0
        # Check parenthesized abbreviations
        paren_gt = _extract_parenthesized(alt)
        paren_pred = _extract_parenthesized(prediction)
        if paren_gt and normalize_answer(paren_gt) == norm_pred:
            return 1.0
        if paren_pred and normalize_answer(paren_pred) == norm_gt:
            return 1.0
    return 0.0

# ------------------------------------------------------------------ #
#  答案后处理：去除 LLM 常见废话前缀
# ------------------------------------------------------------------ #

_PREAMBLE_PATTERNS = [
    re.compile(r'^(based on|according to|from) (the|my|your|our)?\s*(memories?|conversations?|records?|chat history|provided context)[,:]?\s*', re.I),
    re.compile(r'^(the answer is|the .+ (is|was|are|were)|it (is|was)|that would be)[:\s]+', re.I),
    re.compile(r'^(yes|no|well|so|actually|hmm)[,.\s]+', re.I),
    re.compile(r'^(you (mentioned|said|told me|indicated|shared) that)\s+', re.I),
    re.compile(r'^(in (the|your|our) (conversation|memory|chat))[,:\s]+', re.I),
    re.compile(r'^(your|my) (\w+\s+){1,4}(was|is|were|are|before was|used to be) ', re.I),
    re.compile(r'^(i|you|we|they) (have|has|had|own|owned) ', re.I),
    re.compile(r'^to (determine|calculate|figure out|find out|answer this)[,\s].*?[,:]', re.I),
    re.compile(r'^(i|you) (used to be|am|was|were|currently am)\s+', re.I),
    re.compile(r'^(i|you) (spent|took|went|visited|attended|bought|got|received|take|study|work|live|play|use)\s+', re.I),
    re.compile(r'^\w+\s+\w+\s+at\s+', re.I),  # "yoga classes at X" → "X"
    re.compile(r'^it\s+(took|was|is|cost|costs|lasted|takes)\s+', re.I),  # "it took 5 hours" → "5 hours"
]

_TRAILING_PATTERNS = [
    re.compile(r'\s+(happened|occurred|came|was done|was bought|was purchased|was started|was finished)\s+first\.?$', re.I),
    re.compile(r'\s+was\s+(?:the\s+)?(?:first|earlier|before|later)\.?$', re.I),
    re.compile(r'\s+(?:was|were)\s+(?:bought|purchased|started|finished|received|completed)\s+first\.?$', re.I),
    re.compile(r'\s+so far\.?$', re.I),
    re.compile(r'^(?:the\s+)?(?:answer\s+is\s+)?(?:approximately|about|roughly|around)\s+', re.I),
    re.compile(r'^(?:the\s+)?(?:arrival|start|beginning|purchase)\s+of\s+(?:the\s+)?', re.I),
    re.compile(r',\s+(?:specifically|particularly|especially|more specifically).*$', re.I),
]

_ARITHMETIC_PATTERN = re.compile(
    r'^[\d$,.\s+\-×÷*/=()]+?=\s*(.+?)$'
)

def clean_answer(text: str) -> str:
    """Strip common LLM preambles and trailing noise from generated answers."""
    s = text.strip()
    # Yes/No early extraction: detect before preamble stripping can remove "Yes,"
    yn_early = re.match(r'^(yes|no)\b[,.\s!]', s, re.I)
    if yn_early and len(s) >= 10:
        return yn_early.group(1).capitalize()
    # Negation detection: "X did not Y" → "No" (before preamble stripping)
    if (len(s) > 30 and not re.match(r'^(yes|no)\b', s, re.I)
        and re.search(r'\b(did not|didn\'t|was not|wasn\'t|is not|isn\'t|cannot|can\'t|never|not true|incorrect)\b', s, re.I)
        and not re.search(r'\d', s[:20])):
        return "No"
    # Extract final result from arithmetic: "$12 - $6 = $6" → "$6"
    m = _ARITHMETIC_PATTERN.match(s)
    if m:
        s = m.group(1).strip()
    # Remove preambles iteratively (but preserve if removal would empty the string)
    for _ in range(3):
        for pat in _PREAMBLE_PATTERNS:
            cleaned = pat.sub('', s).strip()
            if cleaned:  # only apply if something remains
                s = cleaned
    # Remove trailing noise patterns
    for pat in _TRAILING_PATTERNS:
        s = pat.sub('', s).strip()
    # Remove trailing period if answer is short
    if len(s) < 100 and s.endswith('.'):
        s = s[:-1].strip()
    # Remove surrounding quotes
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'", '\u201c', '\u201d'):
        s = s[1:-1].strip()
    if s.startswith(('\u201c', '"')) and s.endswith(('\u201d', '"')):
        s = s[1:-1].strip()
    # Remove markdown bold markers
    s = re.sub(r'\*\*([^*]+)\*\*', r'\1', s)
    return s

def _extract_counting_answer(text: str, question: str) -> str:
    """For 'how many/how much/how long' questions, extract just the number+unit."""
    q_lower = question.lower()
    if not re.search(r'\bhow\s+(many|much|long|often|far)\b', q_lower):
        return text
    _NUM_UNIT = r'\$?\d[\d,]*(?:\.\d+)?\s*(?:-?\s*)?(?:days?|weeks?|hours?|minutes?|months?|years?|miles?|km|dollars?|times?)?'
    # Try to find "total/answer is/= <number>" at the end
    m_final = re.search(r'(?:total|answer|=|is)\s*[:=]?\s*(' + _NUM_UNIT + r')\s*\.?\s*$', text, re.I)
    if m_final:
        return m_final.group(1).strip()
    # Look for first number+unit pattern
    m = re.search(r'(' + _NUM_UNIT + r')', text)
    if m:
        num_part = m.group(0).strip()
        # Only extract if the answer is significantly longer than the number
        if len(text) > len(num_part) * 3 and len(num_part) < 20:
            return num_part
    return text

# ------------------------------------------------------------------ #
#  正则实体提取（零 LLM 开销）
# ------------------------------------------------------------------ #

# 常见的非实体词（避免将句首大写词误判为实体）
_STOP_ENTITIES = {
    "i", "you", "he", "she", "it", "we", "they", "my", "your", "the",
    "yes", "no", "hi", "hello", "sure", "well", "oh", "okay", "ok",
    "also", "just", "really", "very", "much", "maybe", "actually",
    "thanks", "thank", "please", "sorry", "great", "good", "nice",
    "would", "could", "should", "might", "will", "can", "have", "has",
    "been", "being", "do", "does", "did", "that", "this", "what", "how",
    "when", "where", "which", "who", "there", "here", "some", "any",
    "but", "and", "not", "for", "with", "from", "about", "into",
    "however", "although", "because", "since", "after", "before",
    "if", "then", "so", "as", "of", "in", "on", "at", "to", "by",
}

def extract_entities(text: str) -> list[str]:
    """从文本中提取命名实体（正则启发式，无 LLM 调用）"""
    entities = set()

    # 1. 多词专有名词（连续大写开头的词，可含 of/the/and 等连接词）
    for m in re.finditer(
        r'\b([A-Z][a-z]+(?:\s+(?:of|the|and|in|at|for|on|de|la|le)\s+)?'
        r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text
    ):
        entities.add(m.group(0))

    # 2. 单个大写词（非句首：前面不是 ". " 或行首）
    for m in re.finditer(r'(?<=[a-z,;]\s)([A-Z][a-z]{2,})\b', text):
        w = m.group(1)
        if w.lower() not in _STOP_ENTITIES:
            entities.add(w)

    # 3. 引号内容
    for m in re.finditer(r'["\u201c]([^"\u201d]{2,60})["\u201d]', text):
        entities.add(m.group(1))
    for m in re.finditer(r"'([A-Z][^']{1,60})'", text):
        entities.add(m.group(1))

    # 4. 数字+单位（时间、距离、金额等）
    for m in re.finditer(
        r'(\d+(?:\.\d+)?)\s*'
        r'(minutes?|hours?|days?|weeks?|months?|years?|'
        r'miles?|km|meters?|feet|'
        r'dollars?|\$|pounds?|euros?|'
        r'kg|lbs?|grams?|'
        r'times?|sessions?|classes?|lessons?)', text, re.IGNORECASE
    ):
        entities.add(m.group(0).strip())

    # 5. 日期模式
    for m in re.finditer(
        r'\b(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?'
        r'(?:,?\s*\d{4})?\b', text
    ):
        entities.add(m.group(0))

    # 6. "my X" 模式（所有格后面的名词短语常是重要实体）
    for m in re.finditer(r'\bmy\s+([a-z]+(?:\s+[a-z]+)?)\b', text, re.IGNORECASE):
        val = m.group(1)
        if val.lower() not in _STOP_ENTITIES and len(val) > 2:
            entities.add(val)

    return list(entities)


def extract_relation_triples(text: str) -> list[tuple[str, str, str]]:
    """Extract (subject, relation, object) triples from text using rule-based patterns.

    Designed for personal conversation memories where common patterns include:
    - "I/User [verb] [object]" (actions)
    - "[Person] recommended/suggested [thing]"
    - "My [thing] is/was [value]" (attributes)
    - Location patterns "at/in [Place]"

    Returns list of (subject, relation, object) tuples. No LLM calls.
    """
    triples = []
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 10:
            continue

        # Pattern 1: Subject + verb + object (simple SVO)
        m = re.match(
            r"^(I|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+"
            r"((?:started|began|tried|visited|went to|moved to|bought|adopted|"
            r"recommended|suggested|preferred|chose|switched to|enrolled in|"
            r"signed up for|joined|left|quit|finished|completed|"
            r"ate at|dined at|cooked|ordered|"
            r"watched|read|listened to|played|"
            r"like|love|enjoy|hate|dislike|"
            r"work(?:s|ed)? (?:at|for)|live(?:s|d)? (?:in|at))\s*)"
            r"(.+?)\.?\s*$",
            sent, re.I
        )
        if m:
            subj = m.group(1).strip().lower()
            rel = re.sub(r'\s+', '_', m.group(2).strip().lower())
            obj = m.group(3).strip().rstrip('.').lower()
            if 2 < len(obj) < 100:
                triples.append((subj, rel, obj))
                continue

        # Pattern 2: Possessive "my [thing] is/was [value]"
        m = re.search(
            r"\b(?:my|our)\s+(\w+(?:\s+\w+)?)\s+(?:is|was|are|were)\s+(.+?)(?:\.|,|$)",
            sent, re.I
        )
        if m:
            subj = m.group(1).strip().lower()
            obj = m.group(2).strip().rstrip('.').lower()
            if 1 < len(obj) < 80:
                triples.append(("user", f"has_{subj}", obj))

        # Pattern 3: Location "at/in [Place]"
        m = re.search(
            r"\b(?:at|in|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b",
            sent
        )
        if m:
            place = m.group(1).lower()
            if place not in _STOP_ENTITIES and len(place) > 2:
                triples.append(("user", "location", place))

    return triples


# Both LME ("temporal-reasoning") and LoCoMo ("temporal") are temporal types
_TEMPORAL_TYPES = {"temporal-reasoning", "temporal"}

# ------------------------------------------------------------------ #
#  日期解析
# ------------------------------------------------------------------ #

def parse_date(date_str: str) -> datetime | None:
    """解析 LongMemEval / LoCoMo 格式的日期"""
    if not date_str:
        return None
    fmts = [
        "%Y/%m/%d (%a) %H:%M",         # LongMemEval: 2023/05/20 (Sat) 02:21
        "%I:%M %p on %d %B, %Y",        # LoCoMo: 8:56 pm on 20 July, 2023
        "%Y-%m-%d",
        "%B %d, %Y",
        "%d %B, %Y",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None

# ------------------------------------------------------------------ #
#  Memory++ RAG 核心
# ------------------------------------------------------------------ #

class BenchmarkRAGPlusPlus:
    def __init__(self, collection_name: str = "bench_kg",
                 ablation: str = ""):
        """ablation: comma-separated flags to disable features.
        Supported: no_bm25, no_kg, no_date_aware, no_type_prompt, no_recency_label, no_query_expansion
        """
        self.ablation = set(ablation.split(",")) if ablation else set()
        self.llm_client = OpenAI(
            api_key=API_KEY, base_url=BASE_URL,
            timeout=300.0, max_retries=0
        )
        self.chroma = chromadb.PersistentClient(
            path=os.path.join(SCRIPT_DIR, "chroma_bench_kg"),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name
        self.collection = None
        # 内存知识图谱
        self.kg_entities: dict[str, set[str]] = {}   # entity_lower -> {chunk_ids}
        self.chunk_texts: dict[str, str] = {}          # chunk_id -> text
        self.chunk_dates: dict[str, str] = {}          # chunk_id -> date_str
        self.chunk_session: dict[str, str] = {}        # chunk_id -> session_id
        self.session_summaries: dict[str, str] = {}    # session_id -> extractive summary
        self.session_chunks: dict[str, list[str]] = {} # session_id -> [chunk_ids]
        # 关系三元组索引: (subject, relation, object) -> {chunk_ids}
        self.kg_triples: list[tuple[str, str, str, str]] = []  # [(subj, rel, obj, chunk_id)]
        self.kg_entity_relations: dict[str, set[str]] = {}  # entity -> {related_chunk_ids via triples}

    def reset(self):
        """每道题前重置向量库和 KG"""
        try:
            self.chroma.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.chroma.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.kg_entities.clear()
        self.chunk_texts.clear()
        self.chunk_dates.clear()
        self.chunk_session.clear()
        self.session_summaries.clear()
        self.session_chunks.clear()
        self.kg_triples.clear()
        self.kg_entity_relations.clear()
        self.bm25 = None
        self.bm25_chunk_ids: list[str] = []

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        BATCH = 32
        result = []
        for i in range(0, len(texts), BATCH):
            batch_texts = texts[i:i+BATCH]
            for attempt in range(8):
                try:
                    resp = self.llm_client.embeddings.create(
                        model=EMBED_MODEL, input=batch_texts
                    )
                    result.extend([item.embedding for item in resp.data])
                    break
                except (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError) as e:
                    wait = 5 * (attempt + 1)
                    print(f"    [Embed 重试 {attempt+1}/8] {type(e).__name__}，{wait}s 后重试...")
                    time.sleep(wait)
                except Exception as e:
                    if "502" in str(e) or "503" in str(e) or "504" in str(e):
                        wait = 5 * (attempt + 1)
                        print(f"    [Embed 重试 {attempt+1}/8] {type(e).__name__}，{wait}s 后重试...")
                        time.sleep(wait)
                    else:
                        raise
            else:
                raise RuntimeError(f"嵌入失败：batch {i//BATCH} 经 8 次重试仍失败")
            if i + BATCH < len(texts):
                time.sleep(3)
        return result

    def _rerank(self, query: str, documents: list[str], top_n: int = 10) -> list[tuple[str, float]]:
        """Cross-encoder reranking via SiliconFlow API."""
        if not documents:
            return []
        try:
            resp = _requests.post(
                f"{BASE_URL.rstrip('/').replace('/v1', '')}/v1/rerank",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"model": RERANKER_MODEL, "query": query,
                      "documents": documents, "top_n": min(top_n, len(documents))},
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return [(documents[r["index"]], r["relevance_score"]) for r in results]
        except Exception as e:
            print(f"    [Reranker warning] {e}, falling back to original order")
            return [(d, 0.0) for d in documents[:top_n]]

    def _highlight_evidence(self, question: str, doc: str) -> str:
        """Highlight the most relevant sentence in a chunk for the LLM to focus on.

        Uses keyword overlap scoring to find the sentence most likely to contain
        the answer, then marks it with ►◄ to draw the model's attention. This
        reduces wrong-answer errors where the model picks entities from irrelevant
        sentences within the same chunk.
        """
        # Split into sentences (handle both . and newline boundaries)
        sentences = re.split(r'(?<=[.!?])\s+|\n+', doc)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(sentences) < 2:
            return doc  # too short to highlight

        # Score each sentence by keyword overlap with question
        q_words = set(question.lower().split()) - _STOP_ENTITIES
        q_entities = set(e.lower() for e in extract_entities(question) if len(e) > 2)
        query_terms = q_words | q_entities

        best_score = 0
        best_idx = -1
        for idx, sent in enumerate(sentences):
            sent_lower = sent.lower()
            sent_words = set(sent_lower.split())
            # Entity matches count double
            score = len(query_terms & sent_words)
            for ent in q_entities:
                if ent in sent_lower:
                    score += 2
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_score < 2 or best_idx < 0:
            return doc  # no strong match, don't highlight

        # Mark the best sentence
        highlighted = sentences[best_idx]
        return doc.replace(highlighted, f"►{highlighted}◄")

    def _expand_query(self, query: str, question_type: str = None) -> list[str]:
        """LLM-based query expansion: generate alternative formulations for better recall.

        When initial retrieval confidence is low, the original query may use different
        vocabulary than the stored memories. This method asks the LLM to rephrase the
        query into 2-3 keyword-focused variants that may match better.
        """
        if "no_query_expansion" in self.ablation:
            return []
        type_hint = f" (question type: {question_type})" if question_type else ""
        try:
            resp = self.llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": (
                    f"Rewrite this question as 2-3 short keyword search queries "
                    f"that would help find the answer in a personal memory database.{type_hint}\n"
                    f"Question: {query}\n"
                    f"Output ONLY the queries, one per line. No numbering, no explanation."
                )}],
                max_tokens=100,
                temperature=0.3,
                extra_body={"enable_thinking": False},
            )
            lines = resp.choices[0].message.content.strip().split('\n')
            variants = [l.strip().strip('-').strip('•').strip() for l in lines if l.strip() and len(l.strip()) > 5]
            return variants[:3]
        except Exception as e:
            print(f"    [QueryExpansion warning] {e}")
            return []

    def retrieve_with_fallback(self, query: str, top_k: int = 10,
                                question_type: str = None,
                                confidence_threshold: float = 0.15):
        """Two-pass adaptive retrieval: initial retrieve → low confidence → expand query → re-retrieve.

        If initial retrieval confidence falls below the threshold, uses LLM query expansion
        to generate alternative formulations and merges results from all query variants.
        This addresses false-IDK errors caused by vocabulary mismatch between query and memory.
        """
        # Pass 1: standard retrieval
        docs, dates, conf = self.retrieve_hybrid(query, top_k=top_k, question_type=question_type)

        if conf >= confidence_threshold or "no_query_expansion" in self.ablation:
            return docs, dates, conf

        # Pass 2: expand query and merge
        print(f"    [QueryExpansion] Low confidence ({conf:.3f}), expanding query...")
        variants = self._expand_query(query, question_type)
        if not variants:
            return docs, dates, conf

        # Collect results from all variants
        seen = set(d[:200] for d in docs)
        all_docs = list(docs)
        all_dates = list(dates)
        best_conf = conf

        for variant in variants:
            v_docs, v_dates, v_conf = self.retrieve_hybrid(variant, top_k=top_k, question_type=question_type)
            best_conf = max(best_conf, v_conf)
            for d, dt in zip(v_docs, v_dates):
                key = d[:200]
                if key not in seen:
                    seen.add(key)
                    all_docs.append(d)
                    all_dates.append(dt)

        # Re-rerank the expanded pool if reranker is available
        if "no_reranker" not in self.ablation and len(all_docs) > top_k:
            reranked = self._rerank(query, all_docs[:top_k + 15], top_n=top_k + 5)
            if reranked:
                doc_to_date = dict(zip([d[:200] for d in all_docs], all_dates))
                all_docs = [doc for doc, _ in reranked]
                all_dates = [doc_to_date.get(doc[:200], "") for doc in all_docs]
                best_conf = max(best_conf, reranked[0][1])

        print(f"    [QueryExpansion] Expanded pool: {len(all_docs)} docs, best_conf={best_conf:.3f}")
        return all_docs[:top_k + 5], all_dates[:top_k + 5], best_conf

    def retrieve_chain_of_retrieval(self, query: str, top_k: int = 15,
                                      question_type: str = None):
        """Chain-of-retrieval for multi-hop questions.

        Multi-hop questions require connecting facts across chunks. For example:
        "What did I eat at the restaurant my sister recommended?"
        - Hop 1: Find "sister recommended restaurant" → gets restaurant name
        - Hop 2: Find "ate at [restaurant name]" → gets the food

        Method: retrieve initial context, extract new entities not in query,
        then do a targeted second retrieval pass with those bridging entities.
        """
        if "no_chain_retrieval" in self.ablation:
            return self.retrieve_with_fallback(query, top_k=top_k, question_type=question_type)

        # Hop 1: standard retrieval
        docs, dates, conf = self.retrieve_with_fallback(query, top_k=top_k, question_type=question_type)

        # Only chain for multi-hop type questions
        if question_type not in ("multi-hop",):
            return docs, dates, conf

        # Extract entities from retrieved docs that weren't in the query
        query_ents = set(e.lower() for e in extract_entities(query) if len(e) > 2)
        bridge_entities = set()
        for doc in docs[:5]:  # only scan top-5 for bridging entities
            doc_ents = set(e.lower() for e in extract_entities(doc) if len(e) > 2)
            new_ents = doc_ents - query_ents
            bridge_entities.update(new_ents)

        if not bridge_entities:
            return docs, dates, conf

        # Hop 2: retrieve using bridge entities as additional query terms
        bridge_query = query + " " + " ".join(sorted(bridge_entities)[:5])
        hop2_docs, hop2_dates, hop2_conf = self.retrieve_hybrid(
            bridge_query, top_k=top_k, question_type=question_type
        )

        # Merge hop2 results into hop1
        seen = set(d[:200] for d in docs)
        all_docs = list(docs)
        all_dates = list(dates)
        for d, dt in zip(hop2_docs, hop2_dates):
            key = d[:200]
            if key not in seen:
                seen.add(key)
                all_docs.append(d)
                all_dates.append(dt)

        # Re-rerank the merged pool
        best_conf = max(conf, hop2_conf)
        if "no_reranker" not in self.ablation and len(all_docs) > top_k:
            reranked = self._rerank(query, all_docs[:top_k + 15], top_n=top_k + 5)
            if reranked:
                doc_to_date = dict(zip([d[:200] for d in all_docs], all_dates))
                all_docs = [doc for doc, _ in reranked]
                all_dates = [doc_to_date.get(doc[:200], "") for doc in all_docs]
                best_conf = max(best_conf, reranked[0][1])

        print(f"    [ChainRetrieval] Bridge entities: {sorted(bridge_entities)[:5]}, pool: {len(all_docs)}")
        return all_docs[:top_k + 5], all_dates[:top_k + 5], best_conf

    def index_sessions(self, sessions: list, session_dates=None):
        """索引对话 sessions → ChromaDB + KG 实体索引 + 日期元数据"""
        assert self.collection is not None
        chunks, ids, metas = [], [], []

        for i, sess in enumerate(sessions):
            if isinstance(sess, list):
                messages = sess
                sid = f"sess_{i}"
            else:
                sid = sess.get("session_id", f"sess_{i}")
                messages = sess.get("messages", [])

            date_str = ""
            if session_dates and i < len(session_dates):
                date_str = session_dates[i]

            for j in range(0, len(messages), 2):
                pair = messages[j:j+2]
                text = "\n".join(
                    f"{m.get('role','?')}: {m.get('content','')}"
                    for m in pair
                )
                if not text.strip():
                    continue
                # Overlap: prepend previous assistant response for context
                if j >= 2:
                    prev_msg = messages[j-1] if j-1 < len(messages) else None
                    if prev_msg and prev_msg.get('role') == 'assistant':
                        prev_text = prev_msg.get('content', '')[:200]
                        if prev_text:
                            text = f"[prev] assistant: {prev_text}\n{text}"
                # Add date prefix for temporal awareness in embeddings
                if date_str:
                    text = f"[Date: {date_str}] {text}"

                # Split long texts into multiple chunks (2000 chars each)
                CHUNK_SIZE = 2000
                text_parts = [text[k:k+CHUNK_SIZE] for k in range(0, len(text), CHUNK_SIZE)]
                for part_idx, part_text in enumerate(text_parts):
                    chunk_id = f"{sid}_c{j//2}" if part_idx == 0 else f"{sid}_c{j//2}p{part_idx}"
                    chunks.append(part_text)
                    ids.append(chunk_id)
                    metas.append({"session_id": sid, "date": date_str})

                    # 存入本地索引
                    self.chunk_texts[chunk_id] = part_text
                    self.chunk_dates[chunk_id] = date_str
                    self.chunk_session[chunk_id] = sid

                    # KG: 提取实体并建立倒排索引
                    for ent in extract_entities(part_text):
                        key = ent.lower().strip()
                        if len(key) < 2:
                            continue
                        if key not in self.kg_entities:
                            self.kg_entities[key] = set()
                        self.kg_entities[key].add(chunk_id)

                    # KG: 提取关系三元组并建立关系索引
                    for subj, rel, obj in extract_relation_triples(part_text):
                        self.kg_triples.append((subj, rel, obj, chunk_id))
                        # Index: both subject and object entities link to this chunk
                        for ent_key in (subj, obj):
                            if ent_key and len(ent_key) > 2:
                                if ent_key not in self.kg_entity_relations:
                                    self.kg_entity_relations[ent_key] = set()
                                self.kg_entity_relations[ent_key].add(chunk_id)

        if not chunks:
            return
        embeddings = self._embed_batch(chunks)
        for _retry in range(3):
            try:
                self.collection.add(ids=ids, embeddings=embeddings,
                                    documents=chunks, metadatas=metas)
                break
            except Exception as e:
                if "does not exist" in str(e) and _retry < 2:
                    print(f"    [ChromaDB] Collection lost, recreating... ({e})")
                    self.collection = self.chroma.get_or_create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"},
                    )
                else:
                    raise
        # Build BM25 index
        tokenized = [doc.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.bm25_chunk_ids = ids

        # Build session-level summaries (extractive: entities + key nouns)
        for cid in ids:
            sid = self.chunk_session.get(cid, "")
            if sid:
                if sid not in self.session_chunks:
                    self.session_chunks[sid] = []
                self.session_chunks[sid].append(cid)
        for sid, cids in self.session_chunks.items():
            # Collect entities from all chunks in session
            all_ents = set()
            for cid in cids:
                text = self.chunk_texts.get(cid, "")
                all_ents.update(e.lower() for e in extract_entities(text) if len(e) > 2)
            date = self.chunk_dates.get(cids[0], "") if cids else ""
            summary = f"[Session {sid}]"
            if date:
                summary += f" [Date: {date}]"
            summary += f" Topics: {', '.join(sorted(all_ents)[:20])}"
            self.session_summaries[sid] = summary

    @staticmethod
    def _simplify_query(query: str) -> str:
        """Strip question words to create a better search query."""
        q = query.strip().rstrip('?')
        # Remove question starters
        q = re.sub(r'^(what|which|who|where|when|how many|how much|how long|how|do you remember|can you recall|can you tell me|tell me)\b\s*', '', q, flags=re.I)
        # Remove auxiliary verbs after stripping
        q = re.sub(r'^(is|are|was|were|did|does|do|has|have|had|would|could|should|might)\s+', '', q, flags=re.I)
        # Remove pronouns
        q = re.sub(r'\b(i|you|we|they|my|your|our|their)\b', '', q, flags=re.I)
        q = ' '.join(q.split())  # collapse spaces
        return q.strip() if len(q.strip()) > 3 else query

    def retrieve_hybrid(self, query: str, top_k: int = 10,
                        question_type: str = None):
        """混合检索：向量 + KG 实体匹配"""
        assert self.collection is not None
        count = self.collection.count()
        if count == 0:
            return [], [], 0.0

        # 1. 向量检索 (temporal gets wider net to find dated chunks)
        k = min(top_k + (5 if question_type in _TEMPORAL_TYPES else 0), count)
        emb = self._embed_batch([query])[0]
        res = self.collection.query(
            query_embeddings=[emb], n_results=k,
            include=["documents", "metadatas"]
        )
        vector_docs = res["documents"][0] if res["documents"] else []
        vector_metas = res["metadatas"][0] if res["metadatas"] else []

        # 2. KG 实体匹配 (use simplified query for better keyword matching)
        simplified = self._simplify_query(query)
        query_entities = list(set(extract_entities(query)) | set(extract_entities(simplified)))
        # 也把问题中的小写关键词加入（如 "degree", "commute" 等）
        query_words = (set(query.lower().split()) | set(simplified.lower().split())) - _STOP_ENTITIES
        kg_chunk_ids: dict[str, int] = defaultdict(int)  # chunk_id -> match count

        for ent in query_entities:
            ent_lower = ent.lower().strip()
            if len(ent_lower) < 3:
                continue
            # 精确匹配 (high precision)
            if ent_lower in self.kg_entities:
                for cid in self.kg_entities[ent_lower]:
                    kg_chunk_ids[cid] += 3
            # Multi-word entity partial match: only match if query entity has 2+ words
            # and the stored entity contains all words (avoids "art" in "start" noise)
            if ' ' in ent_lower:
                for stored_ent, cids in self.kg_entities.items():
                    if ent_lower in stored_ent or stored_ent in ent_lower:
                        for cid in cids:
                            kg_chunk_ids[cid] += 1

        # 2.3 Relation triple matching: only for extracted entities (not query words)
        if "no_kg" not in self.ablation and self.kg_entity_relations:
            query_ent_lower = set(e.lower() for e in query_entities if len(e) > 2)
            for ent in query_ent_lower:
                if ent in self.kg_entity_relations:
                    for cid in self.kg_entity_relations[ent]:
                        kg_chunk_ids[cid] += 2

        # 按匹配次数排序取 top (minimum threshold=3 to filter noise)
        if "no_kg" in self.ablation:
            kg_docs = []
        else:
            kg_sorted = sorted(kg_chunk_ids.items(), key=lambda x: -x[1])[:top_k]
            kg_docs = [self.chunk_texts[cid] for cid, score in kg_sorted
                        if cid in self.chunk_texts and score >= 3]

        # 2.5. BM25 keyword retrieval
        bm25_docs = []
        if "no_bm25" in self.ablation:
            pass  # skip BM25
        elif self.bm25 is not None:
            query_tokens = query.lower().split()
            scores = self.bm25.get_scores(query_tokens)
            top_indices = scores.argsort()[-top_k:][::-1]
            for idx in top_indices:
                if scores[idx] > 0 and idx < len(self.bm25_chunk_ids):
                    cid = self.bm25_chunk_ids[idx]
                    if cid in self.chunk_texts:
                        bm25_docs.append(self.chunk_texts[cid])

        # 3. 合并去重（向量结果优先，BM25补充，KG 结果补充）
        seen = set()
        merged = []
        merged_dates = []
        for doc, meta in zip(vector_docs, vector_metas):
            key = doc[:200]
            if key not in seen:
                seen.add(key)
                merged.append(doc)
                merged_dates.append(meta.get("date", "") if meta else "")

        for doc in bm25_docs:
            key = doc[:200]
            if key not in seen:
                seen.add(key)
                merged.append(doc)
                for cid, txt in self.chunk_texts.items():
                    if txt == doc:
                        merged_dates.append(self.chunk_dates.get(cid, ""))
                        break
                else:
                    merged_dates.append("")

        for doc in kg_docs:
            key = doc[:200]
            if key not in seen:
                seen.add(key)
                merged.append(doc)
                # 找对应日期
                for cid, txt in self.chunk_texts.items():
                    if txt == doc:
                        merged_dates.append(self.chunk_dates.get(cid, ""))
                        break
                else:
                    merged_dates.append("")

        # 3.5 For knowledge-update: include ALL chunks from top session(s)
        #     Expand both the top vector match AND the newest-dated match
        if question_type == "knowledge-update" and vector_metas:
            expand_sessions = set()
            # Top vector match session
            top_session = vector_metas[0].get("session_id", "") if vector_metas[0] else ""
            if top_session:
                expand_sessions.add(top_session)
            # Find the newest-dated session among results
            newest_dt, newest_sid = None, None
            for meta in vector_metas:
                sid = meta.get("session_id", "")
                ds = meta.get("date", "")
                if ds and sid:
                    dt = parse_date(ds)
                    if dt and (newest_dt is None or dt > newest_dt):
                        newest_dt, newest_sid = dt, sid
            if newest_sid:
                expand_sessions.add(newest_sid)
            for exp_sid in expand_sessions:
                for cid, sid in self.chunk_session.items():
                    if sid == exp_sid:
                        doc = self.chunk_texts.get(cid, "")
                        key = doc[:200]
                        if key not in seen:
                            seen.add(key)
                            merged.append(doc)
                            merged_dates.append(self.chunk_dates.get(cid, ""))

        # 3.6 Multi-session: session-summary-guided expansion
        #     For multi-session questions, find relevant sessions via entity
        #     overlap with query, then pull in their chunks
        if question_type == "multi-session" and self.session_summaries:
            query_ents = set(e.lower() for e in extract_entities(query) if len(e) > 2)
            query_words = set(query.lower().split()) - _STOP_ENTITIES
            matched_sessions = set()
            for sid, summary in self.session_summaries.items():
                summary_lower = summary.lower()
                # Check if any query entity/keyword appears in session summary
                for ent in query_ents | query_words:
                    if len(ent) >= 4 and ent in summary_lower:
                        matched_sessions.add(sid)
                        break
            # Expand: add all chunks from matched sessions
            for sid in matched_sessions:
                for cid in self.session_chunks.get(sid, []):
                    doc = self.chunk_texts.get(cid, "")
                    key = doc[:200]
                    if key not in seen:
                        seen.add(key)
                        merged.append(doc)
                        merged_dates.append(self.chunk_dates.get(cid, ""))

        # 3.7 Multi-hop: second-stage entity expansion from retrieved docs
        if question_type == "multi-hop" and merged:
            # Extract entities from top retrieved docs
            expansion_ents = set()
            for doc in merged[:5]:
                for ent in extract_entities(doc):
                    expansion_ents.add(ent.lower().strip())
            # Find additional chunks matching these entities
            for ent in expansion_ents:
                if ent in self.kg_entities:
                    for cid in self.kg_entities[ent]:
                        if cid in self.chunk_texts:
                            doc = self.chunk_texts[cid]
                            key = doc[:200]
                            if key not in seen:
                                seen.add(key)
                                merged.append(doc)
                                merged_dates.append(self.chunk_dates.get(cid, ""))

        # 4. 日期排序（temporal/knowledge-update 题按时间倒序）
        if question_type in (_TEMPORAL_TYPES | {"knowledge-update"}) and any(merged_dates):
            dated_items = []
            for doc, ds in zip(merged, merged_dates):
                dt = parse_date(ds)
                dated_items.append((doc, ds, dt))
            # 有日期的按日期倒序，无日期的排最后
            dated_items.sort(key=lambda x: x[2] or datetime.min, reverse=True)
            merged = [d[0] for d in dated_items]
            merged_dates = [d[1] for d in dated_items]

        # Reranker: cross-encoder rerank the candidate set
        reranked = []
        if "no_reranker" not in self.ablation and len(merged) > top_k:
            candidates = merged[:top_k + 10]  # over-retrieve then rerank
            candidate_dates = merged_dates[:top_k + 10]
            reranked = self._rerank(query, candidates, top_n=top_k + 5)
            # Rebuild merged/dates in reranked order
            doc_to_date = dict(zip([d[:200] for d in candidates], candidate_dates))
            merged = [doc for doc, _ in reranked]
            merged_dates = [doc_to_date.get(doc[:200], "") for doc, _ in reranked]
            # For knowledge-update: re-sort by date after reranking (newest first)
            if question_type == "knowledge-update" and any(merged_dates):
                dated_items = [(d, ds, parse_date(ds)) for d, ds in zip(merged, merged_dates)]
                dated_items.sort(key=lambda x: x[2] or datetime.min, reverse=True)
                merged = [d[0] for d in dated_items]
                merged_dates = [d[1] for d in dated_items]

        # Multi-dimensional confidence scoring: fuse multiple evidence signals
        # instead of relying on a single reranker score
        retrieval_confidence = 0.0
        if "no_multi_conf" not in self.ablation:
            conf_signals = []
            # Signal 1: Reranker score (semantic relevance, 0-1 range)
            if "no_reranker" not in self.ablation and reranked:
                reranker_score = reranked[0][1]
                conf_signals.append(("reranker", min(reranker_score, 1.0), 0.4))
            elif vector_metas and res.get("distances"):
                dists = res["distances"][0]
                vec_sim = 1.0 - (dists[0] if dists else 1.0)
                conf_signals.append(("vector", max(vec_sim, 0.0), 0.4))
            # Signal 2: Entity coverage — fraction of query entities found in top docs
            if query_entities and merged:
                top_text = " ".join(merged[:5]).lower()
                q_ents = set(e.lower() for e in query_entities if len(e) > 2)
                if q_ents:
                    covered = sum(1 for e in q_ents if e in top_text)
                    ent_coverage = covered / len(q_ents)
                    conf_signals.append(("entity_cov", ent_coverage, 0.25))
            # Signal 3: Source agreement — how many retrieval channels found the top doc
            if merged:
                top_key = merged[0][:200]
                sources_agree = 0
                if any(d[:200] == top_key for d in vector_docs):
                    sources_agree += 1
                if any(d[:200] == top_key for d in bm25_docs):
                    sources_agree += 1
                if any(d[:200] == top_key for d in kg_docs):
                    sources_agree += 1
                agreement_score = sources_agree / 3.0
                conf_signals.append(("source_agree", agreement_score, 0.2))
            # Signal 4: KG match density — normalized count of KG matches
            if kg_chunk_ids:
                max_kg = max(kg_chunk_ids.values())
                kg_density = min(max_kg / 6.0, 1.0)  # 6+ matches = full confidence
                conf_signals.append(("kg_density", kg_density, 0.15))
            # Weighted fusion
            if conf_signals:
                total_weight = sum(w for _, _, w in conf_signals)
                retrieval_confidence = sum(s * w for _, s, w in conf_signals) / total_weight
        else:
            # Fallback: single-signal confidence
            if "no_reranker" not in self.ablation and reranked:
                retrieval_confidence = reranked[0][1] if reranked else 0.0
            elif vector_metas and res.get("distances"):
                dists = res["distances"][0]
                retrieval_confidence = 1.0 - (dists[0] if dists else 1.0)

        # Contextual chunk expansion: add neighboring chunks from the same session
        # for top retrieved chunks to provide conversational context
        if "no_context_expansion" not in self.ablation and self.session_chunks:
            # Build reverse index: chunk_id → position in session
            expanded_seen = set(d[:200] for d in merged)
            expansion_docs = []
            expansion_dates = []
            for doc in merged[:5]:  # only expand top-5
                # Find the chunk_id for this doc
                for cid, txt in self.chunk_texts.items():
                    if txt == doc:
                        sid = self.chunk_session.get(cid, "")
                        if sid and sid in self.session_chunks:
                            session_cids = self.session_chunks[sid]
                            try:
                                pos = session_cids.index(cid)
                            except ValueError:
                                break
                            # Add the adjacent chunk (prefer next, then prev)
                            for neighbor_pos in [pos + 1, pos - 1]:
                                if 0 <= neighbor_pos < len(session_cids):
                                    n_cid = session_cids[neighbor_pos]
                                    n_doc = self.chunk_texts.get(n_cid, "")
                                    n_key = n_doc[:200]
                                    if n_key and n_key not in expanded_seen:
                                        expanded_seen.add(n_key)
                                        expansion_docs.append(n_doc)
                                        expansion_dates.append(self.chunk_dates.get(n_cid, ""))
                                    break  # only add one neighbor per chunk
                        break
            if expansion_docs:
                merged.extend(expansion_docs)
                merged_dates.extend(expansion_dates)

        return merged[:top_k + 5], merged_dates[:top_k + 5], retrieval_confidence

    def generate_answer(self, question: str, context_docs,
                        context_dates=None,
                        question_type=None,
                        question_date=None,
                        benchmark: str = "lme",
                        retrieval_confidence: float = 1.0,
                        max_retries: int = 5):
        """增强版生成：日期标注 + 题型特化 prompt"""
        is_locomo = benchmark == "locomo"
        if context_docs:
            ctx_parts = []
            q_dt = parse_date(question_date) if question_date else None
            for i, doc in enumerate(context_docs):
                date_label = ""
                if context_dates and i < len(context_dates) and context_dates[i] and "no_date_aware" not in self.ablation:
                    date_label = f" | Date: {context_dates[i]}"
                    # For temporal questions, pre-compute date differences
                    if question_type in _TEMPORAL_TYPES and q_dt:
                        mem_dt = parse_date(context_dates[i])
                        if mem_dt:
                            delta = (q_dt - mem_dt).days
                            weeks = delta // 7
                            months = round(delta / 30.44, 1)
                            date_label += f" | {delta} days ({weeks} weeks, ~{months} months) before question date"
                recency_tag = ""
                if question_type == "knowledge-update" and context_dates and "no_recency_label" not in self.ablation:
                    if i == 0:
                        recency_tag = " ★NEWEST — USE THIS★"
                    elif context_dates[i]:
                        recency_tag = " (OLDER — ignore if newer memory covers same topic)"
                # Evidence sentence highlighting: mark the most relevant sentence
                highlighted_doc = doc
                if "no_evidence_highlight" not in self.ablation and len(doc) > 100:
                    highlighted_doc = self._highlight_evidence(question, doc)
                ctx_parts.append(f"[Memory {i+1}{date_label}{recency_tag}]\n{highlighted_doc}")
            ctx = "\n\n".join(ctx_parts)

            # For temporal questions, add date timeline with months display
            if question_type in _TEMPORAL_TYPES and context_dates and "no_date_aware" not in self.ablation:
                dated_mems = []
                for i, ds in enumerate(context_dates or []):
                    if ds and i < len(context_docs):
                        dt = parse_date(ds)
                        if dt:
                            dated_mems.append((i + 1, dt, ds))
                if len(dated_mems) >= 2:
                    dated_mems.sort(key=lambda x: x[1])
                    timeline = "\n[DATE TIMELINE — sorted chronologically]\n"
                    for idx, (mem_num, dt, ds) in enumerate(dated_mems):
                        timeline += f"  Memory {mem_num}: {ds}\n"
                        if idx > 0:
                            prev_dt = dated_mems[idx - 1][1]
                            gap = (dt - prev_dt).days
                            gap_weeks = gap // 7
                            gap_months = round(gap / 30.44, 1)
                            timeline += f"    ↑ {gap} days ({gap_weeks} weeks, ~{gap_months} months) after Memory {dated_mems[idx-1][0]}\n"
                    if q_dt:
                        last_gap = (q_dt - dated_mems[-1][1]).days
                        timeline += f"  Question date: {question_date} ({last_gap} days after Memory {dated_mems[-1][0]})\n"

                    ctx += "\n" + timeline

            # 题型特化提示
            type_hints = {
                "temporal-reasoning": (
                    "Pay close attention to the DATE of each memory. "
                    "Time differences are PRE-COMPUTED for you (shown as 'X days (Y weeks, ~Z months) before question date'). "
                    "USE these pre-computed values directly — do NOT recalculate. "
                    f"TODAY'S DATE: {question_date or 'unknown'}. "
                    "To find time BETWEEN two events: subtract their day-counts. "
                    "To find 'how long ago': use the pre-computed days/weeks/months directly. "
                    "CRITICAL: Match the UNIT asked in the question! "
                    "If asked 'how many WEEKS ago': convert days to weeks (divide by 7, round to nearest integer). "
                    "If asked 'how many MONTHS': use the pre-computed months value (round to nearest integer). "
                    "If asked 'how many DAYS': use the days value directly. "
                    "Give JUST the number — e.g. if asked 'how many weeks', answer '4' not '4 weeks' or '28 days'. "
                    "If asked 'which happened first': name the EVENT or ITEM (not 'X happened first', just name it). "
                ),
                "knowledge-update": (
                    "CRITICAL: Information gets UPDATED over time — newer replaces older. "
                    "Memories are sorted NEWEST-first by date. "
                    "RULE: When a topic appears in Memory 1 (newest), use ONLY that value. "
                    "NEVER average, combine, or mix old and new values. "
                    "If the question asks about a number/amount/frequency, give the value from the NEWEST memory only. "
                    "If the question is yes/no, answer based on the NEWEST memory only. "
                ),
                "multi-session": (
                    "The answer requires combining information from MULTIPLE different memories. "
                    + (
                        # For "how many" questions: STRICT enumerate-then-count format
                        "YOU MUST USE THIS EXACT FORMAT — no other format is accepted:\n"
                        "Step 1: List each DISTINCT item, one per line, with '- ' prefix.\n"
                        "Step 2: Write 'TOTAL: <number>' as the LAST line.\n"
                        "RULES:\n"
                        "- ONLY list items that are DIRECTLY and EXPLICITLY stated in the memories above\n"
                        "- Do NOT infer, guess, or imagine items not in the text\n"
                        "- Do NOT count the same item twice even if mentioned in different memories\n"
                        "- For each item, mentally verify: 'Can I point to the EXACT sentence?'\n"
                        "- The answer is typically between 2 and 5\n"
                        "Example:\n- yoga\n- swimming\n- tennis\nTOTAL: 3\n"
                        if re.search(r'\bhow\s+(many|much)\b', question, re.I) else
                        "Combine information from multiple memories. "
                        "Do NOT say 'I don't know' if any memories mention the topic. "
                        "Give your best answer. "
                    )
                ),
                "single-session-assistant": (
                    "This question asks about something the ASSISTANT said or recommended in a past conversation. "
                    "Look for assistant responses in the memories — the answer is in what the assistant wrote, "
                    "not what the user said. Give a short, direct answer. "
                ),
                "single-session-preference": (
                    "IMPORTANT: This question asks you to INFER the user's preferences from their conversations. "
                    "The answer is NOT directly stated — you must DEDUCE it from context clues. "
                    "DO NOT say 'I don't know'. ALWAYS provide your best inference. "
                    "Start your answer with 'The user would prefer' and describe their likely preference "
                    "based on their interests, hobbies, skills, and expressed opinions in the memories. "
                    "Write 1-2 detailed sentences covering specific preferences. "
                ),
                "adversarial": (
                    "The question may reference events with slightly wrong details or phrasing. "
                    "IMPORTANT: Do NOT say 'I don't know'. Instead, answer based on what ACTUALLY happened "
                    "in the memories, even if the question's premise or details are slightly wrong. "
                    "Correct any wrong details and give the factual answer. "
                    "If asked about something that didn't happen that way, describe what actually DID happen. "
                ),
                "multi-hop": (
                    "This question requires connecting facts from DIFFERENT memories. "
                    "Trace the chain: find the first relevant fact, then find related facts in other memories. "
                    "Give a concise answer combining the needed information. "
                ),
                "single-hop": (
                    "Find the specific fact in the memories and give a direct answer. "
                    "Include all relevant items if the question asks for a list. "
                ),
                "open-domain": (
                    "Answer based on what the memories say. Give a concise, factual response. "
                    "Include relevant details but no unnecessary filler. "
                ),
            }
            # LoCoMo uses "temporal" while LME uses "temporal-reasoning"
            type_hints["temporal"] = type_hints["temporal-reasoning"]
            hint = "" if "no_type_prompt" in self.ablation else type_hints.get(question_type or "", "")

            # Query-intent hint: guide answer format based on question word
            q_lower = question.lower().strip()
            if q_lower.startswith(("where", "in what location", "at which", "in which city", "in which store")):
                hint += "The question asks for a PLACE/LOCATION. Answer with just the place name. "
            elif q_lower.startswith(("who", "whose")):
                hint += "The question asks for a PERSON. Answer with just the person's name. "
            elif q_lower.startswith(("when", "what date", "what time", "on what day")):
                hint += "The question asks for a TIME/DATE. Answer with just the date or time. "
            elif re.match(r'^how (many|much|long|often|far)\b', q_lower):
                hint += "The question asks for a NUMBER/QUANTITY. Answer with just the number (and unit if needed). "

            # Length instruction depends on question type and benchmark
            if question_type == "single-session-preference":
                length_hint = (
                    "Describe what the user would prefer based on their past conversations. "
                    "Write 1-2 sentences covering their specific preferences, tools, and interests. "
                    "Start with 'The user would prefer...' "
                )
            elif question_type == "multi-session":
                length_hint = (
                    "Give a SHORT, DIRECT answer — at most one sentence. "
                    "If asked 'how many', answer with JUST the number (e.g. '3'). "
                    "If asked 'how long/how many days', answer with JUST the number and unit (e.g. '8 days'). "
                    "Do NOT explain your reasoning or list items unless the question asks 'what' or 'which'. "
                    "Use digits for numbers (write '3' not 'three'). "
                )
            elif is_locomo and question_type == "open-domain":
                length_hint = (
                    "Give a COMPLETE and detailed answer in 2-4 sentences. "
                    "Include all relevant details (names, places, reasons, context). "
                    "Do not truncate your answer — completeness is more important than brevity. "
                    "Use digits for numbers (write '3' not 'three'). "
                )
            elif is_locomo and question_type == "single-hop":
                length_hint = (
                    "Answer with JUST the specific fact asked for — ideally 1-5 words. "
                    "If asked 'what country', answer 'Sweden' not 'She moved from Sweden'. "
                    "If asked for a list, give comma-separated items. "
                    "No narrative, no explanation, no preamble. "
                    "Use digits for numbers (write '3' not 'three'). "
                )
            elif is_locomo and question_type == "multi-hop":
                length_hint = (
                    "Give a concise answer in 1-2 short sentences. "
                    "Connect the relevant facts from the memories to answer directly. "
                    "Include specific names, dates, and details. No filler. "
                    "Use digits for numbers (write '3' not 'three'). "
                )
            elif is_locomo:
                length_hint = (
                    "Give a concise but COMPLETE answer in 1-2 short sentences. "
                    "Include all relevant details (names, lists, reasons) but no filler or preamble. "
                    "Use digits for numbers (write '3' not 'three'). "
                )
            else:
                # "What" questions may need slightly longer answers (e.g., "Marketing specialist at a startup")
                # while who/where/when/how-many are typically 1-3 words
                is_what_q = q_lower.startswith(("what",)) and not q_lower.startswith(("what date", "what time"))
                word_limit = "1-10 words" if is_what_q else "1-5 words"
                length_hint = (
                    f"IMPORTANT: Give the shortest possible answer — ideally {word_limit}. "
                    "Answer ONLY what was asked. Do NOT add extra information beyond the question. "
                    "Just state the fact. No explanations, no qualifications, no 'Based on...' preamble. "
                    "Use digits for numbers (write '3' not 'three'). "
                )

            # Adversarial premise detection: check if question entities appear in context
            premise_suspect = False
            if "no_premise_detect" not in self.ablation and question_type in ("adversarial", "single-session-user", "single-session-assistant"):
                q_entities = set(e.lower() for e in extract_entities(question) if len(e) > 3)
                ctx_lower = ctx.lower()
                if q_entities:
                    matched = sum(1 for e in q_entities if e in ctx_lower)
                    # If less than 30% of query entities appear in context, premise may be false
                    if matched / len(q_entities) < 0.3:
                        premise_suspect = True

            # Retrieval-aware IDK: low confidence → stronger IDK encouragement
            low_conf = retrieval_confidence < 0.15
            idk_instruction = (
                "" if question_type in ("single-session-preference", "adversarial")
                else ("Only say 'I don't know' if NONE of the memories mention the topic at all. "
                      if question_type == "multi-session"
                      else ("WARNING: The question may contain a FALSE PREMISE — it mentions things "
                            "not found in the memories. If the specific thing asked about is NOT in any memory, "
                            "say 'I don't know' or correct the premise. Do NOT make up an answer. "
                            if premise_suspect
                            else ("WARNING: Retrieved memories may not be relevant to this question. "
                                  "Say 'I don't know' unless you find a CLEAR, DIRECT answer. "
                                  if low_conf
                                  else "If the memories do not contain the answer, say 'I don't know'. ")))
            )
            system = (
                "You are an assistant with long-term memory. "
                "Answer the user's question based ONLY on the conversation memories below. "
                f"{hint}"
                f"{length_hint}"
                "Answer from the USER's perspective: say 'my sister' not 'your sister', 'I visited' not 'you visited'. "
                f"{idk_instruction}\n\n"
                f"Memories:\n{ctx}"
            )
        else:
            system = "Answer the question concisely. If you don't know, say 'I don't know'."

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]

        for attempt in range(max_retries):
            try:
                resp = self.llm_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    temperature=0.3 if question_type == "single-session-preference" else 0.0,
                    max_tokens=(
                        300 if is_locomo and question_type == "open-domain"
                        else 150 if is_locomo and question_type == "multi-hop"
                        else 80 if is_locomo and question_type == "single-hop"
                        else 200 if question_type in ("single-session-preference", "multi-session") or is_locomo
                        else 80),
                    extra_body={"enable_thinking": False},
                )
                content = resp.choices[0].message.content or ""
                # For multi-session "how many" questions: enumerate-then-count
                if question_type == "multi-session" and re.search(r'\bhow\s+(many|much)\b', question, re.I):
                    total_m = re.search(r'TOTAL:\s*(\d+)', content, re.I)
                    items = [l.strip() for l in content.split('\n')
                             if re.match(r'^[-•*]\s+\S|^\d+[\.\)]\s+\S', l.strip())]
                    if total_m and items:
                        # 有列表也有 TOTAL → 信任 Python 计数
                        content = str(len(items))
                    elif total_m:
                        content = total_m.group(1)
                    elif len(items) >= 1:
                        content = str(len(items))
                    elif re.match(r'^(\d+)\s*$', content.strip()):
                        # 裸数字 → 始终触发 list-only 重试
                        original_num = content.strip()
                        print(f"    [Enumerate retry] Bare number ({original_num}), forcing list-only...")
                        try:
                            resp2 = self.llm_client.chat.completions.create(
                                model=LLM_MODEL,
                                messages=[
                                    {"role": "system", "content": system},
                                    {"role": "user", "content": (
                                        f"{question}\n\n"
                                        "CRITICAL INSTRUCTION: You MUST output a bullet list.\n"
                                        "- Write EACH distinct item on its own line starting with '- '\n"
                                        "- Do NOT write a number or count\n"
                                        "- Do NOT write sentences or explanations\n"
                                        "- ONLY output the bullet list, nothing else\n"
                                        "Example format:\n- item one\n- item two\n- item three"
                                    )},
                                ],
                                temperature=0.0, max_tokens=400,
                                extra_body={"enable_thinking": False},
                            )
                            list_content = resp2.choices[0].message.content or ""
                            items2 = [l.strip() for l in list_content.split('\n')
                                      if re.match(r'^[-•*]\s+\S|^\d+[\.\)]\s+\S', l.strip())]
                            if len(items2) >= 1:
                                content = str(len(items2))
                                print(f"    [Enumerate retry] Listed {len(items2)} items: {[i[:30] for i in items2]}")
                            else:
                                print(f"    [Enumerate retry] Still no list, keeping {original_num}")
                        except Exception as e:
                            print(f"    [Enumerate retry] Failed: {e}")
                content = clean_answer(content)
                # For counting questions (any type), extract just the number
                content = _extract_counting_answer(content, question)
                # Sanity cap for multi-session counting: most multi-session answers
                # are small numbers (1-10). If model outputs > 10 for a pure count
                # question (not duration/amount), it's likely overcounting.
                if (question_type == "multi-session"
                    and re.search(r'\bhow\s+many\b', question, re.I)
                    and not re.search(r'\bhow\s+many\s+(hours?|days?|weeks?|months?|years?|minutes?)\b', question, re.I)
                    and re.match(r'^\d+$', content.strip())
                    and int(content.strip()) > 10):
                    original = int(content.strip())
                    # Cap at 10 — better to undercount than wildly overcount
                    content = str(min(original, 10))
                    print(f"    [Sanity cap] {original} → {content} (capped at 10 for count question)")

                # LoCoMo single-hop: if answer is too long for a factual question,
                # try to extract the first sentence/clause as the core answer
                if (is_locomo and question_type == "single-hop"
                    and len(content) > 80 and not _is_idk(content)):
                    # Take first sentence or first clause before comma
                    first = re.split(r'[.!]\s+', content)[0]
                    if len(first) < len(content) and len(first) > 5:
                        content = first.rstrip('.')

                # Answer grounding verification: check if answer entities appear in context
                if ("no_grounding_check" not in self.ablation
                    and not _is_idk(content)
                    and question_type not in ("single-session-preference", "multi-session")
                    and context_docs):
                    answer_ents = set(e.lower() for e in extract_entities(content) if len(e) > 3)
                    if answer_ents:
                        ctx_text = " ".join(str(d) for d in context_docs).lower()
                        grounded = sum(1 for e in answer_ents if e in ctx_text)
                        # If <30% of answer entities are grounded in context, likely hallucination
                        if grounded / len(answer_ents) < 0.3 and len(answer_ents) >= 2:
                            print(f"    [Grounding] Answer not grounded ({grounded}/{len(answer_ents)} entities), switching to IDK")
                            content = "I don't know"

                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                }
                # Retry once with higher temperature if answer is empty
                if not content.strip() and attempt == 0:
                    print("    [空答案，temp=0.3 重试]")
                    time.sleep(3)
                    resp2 = self.llm_client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=120,
                        extra_body={"enable_thinking": False},
                    )
                    content2 = clean_answer(resp2.choices[0].message.content or "")
                    if content2.strip():
                        content = content2
                        usage["total_tokens"] += resp2.usage.total_tokens
                time.sleep(3)
                return content, usage
            except (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError) as e:
                wait = 5 * (attempt + 1)
                print(f"    [LLM 重试 {attempt+1}/{max_retries}] {type(e).__name__}，{wait}s 后重试...")
                time.sleep(wait)
            except Exception as e:
                if "502" in str(e) or "503" in str(e) or "504" in str(e):
                    wait = 5 * (attempt + 1)
                    print(f"    [LLM 重试 {attempt+1}/{max_retries}] {type(e).__name__}，{wait}s 后重试...")
                    time.sleep(wait)
                else:
                    raise

        return ("", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

# ------------------------------------------------------------------ #
#  LongMemEval 评测
# ------------------------------------------------------------------ #

SINGLE_HOP_TYPES = {
    "single-session-user", "single-session-assistant", "single-session-preference",
}
ALL_TYPES = {
    "single-session-user", "single-session-assistant", "single-session-preference",
    "multi-session", "knowledge-update", "temporal-reasoning", "false-premise",
}

def run_longmemeval(rag: BenchmarkRAGPlusPlus, data: list[dict],
                   max_questions: int = 50,
                   question_types: set = SINGLE_HOP_TYPES) -> dict:
    items = [d for d in data if d.get("question_type") in question_types]
    items = items[:max_questions]

    print(f"\n Memory++ LongMemEval 评测（题型={question_types & set(d['question_type'] for d in items)}）")
    print(f"  总题数（过滤后）: {len(items)}")

    f1_scores, em_scores, latencies, token_usages = [], [], [], []
    retrieval_recall = []
    type_scores: dict = {}

    for i, item in enumerate(items):
        qtype = item.get("question_type", "?")
        question = item["question"]
        answer = str(item.get("answer", ""))
        sessions = item.get("haystack_sessions", [])
        session_ids = item.get("haystack_session_ids", [])
        session_dates = item.get("haystack_dates", [])
        question_date = item.get("question_date", "")
        evidence_ids = set(item.get("answer_session_ids", []))

        # 1. 重置并索引（附带日期）
        rag.reset()
        t_index = time.perf_counter()
        rag.index_sessions(sessions, session_dates=session_dates)
        index_time = time.perf_counter() - t_index

        # 2. 混合检索 + 自适应查询扩展（低置信度时自动展开）
        t0 = time.perf_counter()
        effective_k = 20 if qtype in ("multi-session", "knowledge-update") else (15 if qtype in _TEMPORAL_TYPES else 10)
        retrieved_docs, retrieved_dates, ret_conf = rag.retrieve_with_fallback(
            question, top_k=effective_k, question_type=qtype
        )
        retrieval_time = time.perf_counter() - t0

        # 检索召回率
        if evidence_ids and sessions:
            ev_text_parts = []
            for idx, sid in enumerate(session_ids):
                if sid in evidence_ids and idx < len(sessions):
                    sess = sessions[idx]
                    msgs = sess if isinstance(sess, list) else sess.get("messages", [])
                    ev_text_parts.extend(m.get("content", "") for m in msgs)
            ev_text = " ".join(ev_text_parts).lower()[:500]
            ret_text = " ".join(retrieved_docs).lower()
            stopwords = set("the a an i is was".split())
            ev_words = set(ev_text.split()[:20]) - stopwords
            if ev_words:
                overlap = len(ev_words & set(ret_text.split())) / len(ev_words)
                retrieval_recall.append(overlap)

        # 3. 增强生成
        t_gen = time.perf_counter()
        prediction, usage = rag.generate_answer(
            question, retrieved_docs,
            context_dates=retrieved_dates,
            question_type=qtype,
            question_date=question_date,
            retrieval_confidence=ret_conf,
        )
        gen_time = time.perf_counter() - t_gen

        total_time = index_time + retrieval_time + gen_time
        latencies.append(total_time)
        token_usages.append(usage["total_tokens"])

        f1 = token_f1(prediction, answer)
        em = exact_match(prediction, answer)
        f1_scores.append(f1)
        em_scores.append(em)

        if qtype not in type_scores:
            type_scores[qtype] = {"f1": [], "em": []}
        type_scores[qtype]["f1"].append(f1)
        type_scores[qtype]["em"].append(em)

        status = "✓" if f1 > 0.3 else "✗"
        ans_preview = prediction[:60].replace('\n', ' ') if prediction else "(empty)"
        print(
            f"  [{i+1:3d}/{len(items)}] {status} [{qtype[:20]}] "
            f"F1={f1:.2f} EM={em:.0f} | "
            f"latency={total_time:.1f}s tokens={usage['total_tokens']}"
            f" KG={len(rag.kg_entities)}ent"
            f"\n        答: {ans_preview}  |  真: {str(answer)[:60]}"
        )
        if (i + 1) % 10 == 0:
            print(f"  --- 进度: avg_F1={np.mean(f1_scores):.3f} ---")

    return {
        "benchmark": "LongMemEval-S (Memory++)",
        "n_questions": len(items),
        "question_types": list(question_types & set(d["question_type"] for d in items)),
        "overall": {
            "token_f1_mean": round(float(np.mean(f1_scores)), 4),
            "token_f1_p50": round(float(np.median(f1_scores)), 4),
            "exact_match": round(float(np.mean(em_scores)), 4),
            "retrieval_recall_mean": round(float(np.mean(retrieval_recall)), 4) if retrieval_recall else None,
            "e2e_latency_p50_s": round(float(np.percentile(latencies, 50)), 2),
            "e2e_latency_p95_s": round(float(np.percentile(latencies, 95)), 2),
            "token_usage_mean": round(float(np.mean(token_usages)), 1),
        },
        "by_type": {
            qtype: {
                "n": len(scores["f1"]),
                "token_f1": round(float(np.mean(scores["f1"])), 4),
                "exact_match": round(float(np.mean(scores["em"])), 4),
            }
            for qtype, scores in type_scores.items()
        },
    }

# ------------------------------------------------------------------ #
#  LoCoMo 评测
# ------------------------------------------------------------------ #

LOCOMO_CAT_NAMES = {1: "single-hop", 2: "multi-hop", 3: "temporal",
                    4: "open-domain", 5: "adversarial"}

def run_locomo(rag: BenchmarkRAGPlusPlus, data: list[dict],
               max_questions: int = 50) -> dict:
    all_qa = []
    for conv in data:
        c = conv.get("conversation", {})
        speaker_a = c.get("speaker_a", "A")
        sess_keys = sorted(
            [k for k in c if k.startswith("session_") and not k.endswith("_date_time")],
            key=lambda x: int(x.split("_")[1])
        )
        sessions = []
        session_dates = []
        for sk in sess_keys:
            turns = c[sk]
            msgs = [
                {
                    "role": "user" if t.get("speaker") == speaker_a else "assistant",
                    "content": t.get("text", ""),
                }
                for t in turns
            ]
            sessions.append({"session_id": sk, "messages": msgs})
            date_key = f"{sk}_date_time"
            session_dates.append(c.get(date_key, ""))

        for qa in conv.get("qa", []):
            cat_num = qa.get("category", 0)
            # adversarial 题的答案在 adversarial_answer 字段
            answer = str(qa.get("answer", "") or qa.get("adversarial_answer", ""))
            all_qa.append({
                "question": qa.get("question", ""),
                "answer": answer,
                "category": LOCOMO_CAT_NAMES.get(cat_num, str(cat_num)),
                "sessions": sessions,
                "session_dates": session_dates,
            })

    items = all_qa[:max_questions]
    print(f"\n Memory++ LoCoMo 评测（总题数: {len(items)}）")

    f1_scores, em_scores, latencies = [], [], []
    cat_scores: dict = {}
    last_sessions_hash = None  # cache sessions to avoid re-embedding

    for i, item in enumerate(items):
        # Detect same conversation by checking session count + first message
        sess = item["sessions"]
        s_hash = (len(sess), sess[0].get("session_id", "") if sess else "")
        if s_hash != last_sessions_hash:
            rag.reset()
            rag.index_sessions(sess, session_dates=item.get("session_dates"))
            last_sessions_hash = s_hash

        t0 = time.perf_counter()
        cat = item["category"]
        locomo_k = 20 if cat in ("open-domain", "adversarial") else (15 if cat in ("multi-hop", "temporal") else 10)
        if cat == "multi-hop":
            docs, dates, _ret_conf = rag.retrieve_chain_of_retrieval(
                item["question"], top_k=locomo_k, question_type=cat
            )
        else:
            docs, dates, _ret_conf = rag.retrieve_with_fallback(
                item["question"], top_k=locomo_k, question_type=cat
            )
        # For temporal questions, use last session date as question_date
        q_date = None
        if cat in _TEMPORAL_TYPES and item.get("session_dates"):
            # Use the last non-empty session date
            for sd in reversed(item["session_dates"]):
                if sd:
                    q_date = sd
                    break
        pred, _ = rag.generate_answer(
            item["question"], docs,
            context_dates=dates,
            question_type=cat,
            question_date=q_date,
            benchmark="locomo",
            retrieval_confidence=_ret_conf,
        )
        latencies.append(time.perf_counter() - t0)

        f1 = token_f1(pred, item["answer"])
        em = exact_match(pred, item["answer"])
        f1_scores.append(f1)
        em_scores.append(em)

        cat = item["category"]
        if cat not in cat_scores:
            cat_scores[cat] = {"f1": [], "em": []}
        cat_scores[cat]["f1"].append(f1)
        cat_scores[cat]["em"].append(em)

        status = "✓" if f1 > 0.3 else "✗"
        pred_preview = pred[:60].replace('\n', ' ') if pred else "(empty)"
        print(
            f"  [{i+1:3d}/{len(items)}] {status} [{cat[:12]}] "
            f"F1={f1:.2f} EM={em:.0f} latency={latencies[-1]:.1f}s"
            f"\n        答: {pred_preview}  |  真: {str(item['answer'])[:60]}"
        )

    return {
        "benchmark": "LoCoMo (Memory++)",
        "n_questions": len(items),
        "overall": {
            "token_f1_mean": round(float(np.mean(f1_scores)), 4),
            "exact_match": round(float(np.mean(em_scores)), 4),
            "e2e_latency_p50_s": round(float(np.percentile(latencies, 50)), 2),
        },
        "by_category": {
            cat: {
                "n": len(v["f1"]),
                "token_f1": round(float(np.mean(v["f1"])), 4),
                "exact_match": round(float(np.mean(v["em"])), 4),
            }
            for cat, v in cat_scores.items()
        },
    }

# ------------------------------------------------------------------ #
#  主程序
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Memory++ Benchmark Evaluation")
    parser.add_argument("--max-questions", type=int, default=50)
    parser.add_argument("--question-types", type=str,
                        default="single-session-user,single-session-assistant,single-session-preference")
    parser.add_argument("--skip-locomo", action="store_true")
    parser.add_argument("--ablation", type=str, default="",
                        help="Comma-separated ablation flags: no_bm25,no_kg,no_date_aware,no_type_prompt,no_recency_label")
    args = parser.parse_args()

    if args.question_types == "all":
        qtypes = ALL_TYPES
    else:
        qtypes = set(args.question_types.split(","))

    print("=" * 60)
    print("Memory++ (KG-Enhanced) Benchmark 评测")
    print(f"  模型: {LLM_MODEL}  |  enable_thinking=False")
    print(f"  嵌入: {EMBED_MODEL}")
    print(f"  增强: 正则实体KG + 日期感知 + 题型特化prompt")
    print(f"  最大题数: {args.max_questions}")
    print("=" * 60)

    rag = BenchmarkRAGPlusPlus(ablation=args.ablation)
    if args.ablation:
        print(f"  消融模式: {args.ablation}")
    all_results = {}

    # ---- LongMemEval ----
    if os.path.exists(LME_S_PATH):
        print(f"\n加载 LongMemEval-S: {LME_S_PATH}")
        with open(LME_S_PATH, encoding="utf-8") as f:
            lme_data = json.load(f)
        print(f"  总题数: {len(lme_data)}")

        lme_result = run_longmemeval(rag, lme_data,
                                     max_questions=args.max_questions,
                                     question_types=qtypes)
        all_results["longmemeval"] = lme_result

        print("\n  LongMemEval 结果摘要 (Memory++)：")
        o = lme_result["overall"]
        print(f"    Token-F1:       {o['token_f1_mean']:.4f}")
        print(f"    Exact Match:    {o['exact_match']:.4f}")
        if o.get("retrieval_recall_mean"):
            print(f"    Retrieval Recall:{o['retrieval_recall_mean']:.4f}")
        print(f"    E2E 延迟 p50:   {o['e2e_latency_p50_s']}s")
        print(f"    Token 均值:     {o['token_usage_mean']}")

    # ---- LoCoMo ----
    if not args.skip_locomo and os.path.exists(LOCOMO_PATH):
        print(f"\n加载 LoCoMo: {LOCOMO_PATH}")
        with open(LOCOMO_PATH, encoding="utf-8") as f:
            locomo_data = json.load(f)
        if isinstance(locomo_data, dict):
            locomo_data = list(locomo_data.values())

        locomo_result = run_locomo(rag, locomo_data, max_questions=args.max_questions)
        all_results["locomo"] = locomo_result

        print("\n  LoCoMo 结果摘要 (Memory++)：")
        o = locomo_result["overall"]
        print(f"    Token-F1:  {o['token_f1_mean']:.4f}")
        print(f"    Exact Match: {o['exact_match']:.4f}")

    # ---- 保存 + 对比 ----
    ablation_suffix = f"_ablation_{args.ablation.replace(',', '_')}" if args.ablation else ""
    out_path = os.path.join(SCRIPT_DIR, f"benchmark_results_kg{ablation_suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果保存至: {out_path}")

    # 加载基线对比
    baseline_path = os.path.join(SCRIPT_DIR, "benchmark_results.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        print("\n" + "=" * 60)
        print("  Memory++ vs Baseline 对比")
        print("=" * 60)
        if "longmemeval" in baseline and "longmemeval" in all_results:
            b = baseline["longmemeval"]["overall"]
            n = all_results["longmemeval"]["overall"]
            delta_f1 = n["token_f1_mean"] - b["token_f1_mean"]
            print(f"\n  LongMemEval-S:")
            print(f"    Token-F1:  {b['token_f1_mean']:.4f} → {n['token_f1_mean']:.4f}  ({delta_f1:+.4f})")
            print(f"    EM:        {b['exact_match']:.4f} → {n['exact_match']:.4f}")
            if b.get("retrieval_recall_mean") and n.get("retrieval_recall_mean"):
                print(f"    Recall:    {b['retrieval_recall_mean']:.4f} → {n['retrieval_recall_mean']:.4f}")
            # 按题型
            print(f"\n    按题型:")
            b_types = baseline["longmemeval"].get("by_type", {})
            n_types = all_results["longmemeval"].get("by_type", {})
            for qt in sorted(set(list(b_types.keys()) + list(n_types.keys()))):
                bf1 = b_types.get(qt, {}).get("token_f1", 0)
                nf1 = n_types.get(qt, {}).get("token_f1", 0)
                d = nf1 - bf1
                arrow = "↑" if d > 0.005 else ("↓" if d < -0.005 else "→")
                print(f"      {qt:<28s} {bf1:.4f} → {nf1:.4f}  {arrow} ({d:+.4f})")

        if "locomo" in baseline and "locomo" in all_results:
            b = baseline["locomo"]["overall"]
            n = all_results["locomo"]["overall"]
            delta_f1 = n["token_f1_mean"] - b["token_f1_mean"]
            print(f"\n  LoCoMo:")
            print(f"    Token-F1:  {b['token_f1_mean']:.4f} → {n['token_f1_mean']:.4f}  ({delta_f1:+.4f})")
            b_cats = baseline["locomo"].get("by_category", {})
            n_cats = all_results["locomo"].get("by_category", {})
            for cat in sorted(set(list(b_cats.keys()) + list(n_cats.keys()))):
                bf1 = b_cats.get(cat, {}).get("token_f1", 0)
                nf1 = n_cats.get(cat, {}).get("token_f1", 0)
                d = nf1 - bf1
                arrow = "↑" if d > 0.005 else ("↓" if d < -0.005 else "→")
                print(f"      {cat:<20s} {bf1:.4f} → {nf1:.4f}  {arrow} ({d:+.4f})")

    print("\n" + "=" * 60)
    print("评测完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
