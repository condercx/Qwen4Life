"""
答案评分：Token-F1、Exact Match、答案后处理。
"""

import re
from collections import Counter

# ---------- 数字词 → 数字 ----------

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


# ------------------------------------------------------------------ #
#  答案规范化
# ------------------------------------------------------------------ #

def normalize_answer(s: str) -> str:
    """规范化答案字符串，用于 F1/EM 比较。"""
    s = str(s).lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'\byour\b', 'my', s)
    s = re.sub(r'\byou\b', 'i', s)
    s = re.sub(r'\b(\d+)(?:st|nd|rd|th)\b', r'\1', s)
    s = re.sub(r'\b(\d+)\s*minutes?\s*(?:and\s*)?(\d+)\s*seconds?\b', r'\1:\2', s)
    s = re.sub(r'\b(\d+)\s*hours?\s*(?:and\s*)?(\d+)\s*minutes?\b', r'\1h\2m', s)
    s = re.sub(r'(\d),(\d)', r'\1\2', s)
    s = re.sub(r'(\d)-(\w)', r'\1 \2', s)
    s = re.sub(r'(?<!\d)\.(?!\d)|[^\w\s.]', '', s)
    s = s.replace('.', ' . ')
    s = re.sub(r'(\d) \. (\d)', r'\1.\2', s)
    s = s.replace(' . ', ' ')
    s = re.sub(r'\b(\d+)\.0\b', r'\1', s)
    s = re.sub(r'\bago\b', '', s)
    s = re.sub(r'\b(approximately|about|around|nearly|roughly)\b', '', s)
    for syn, canonical in _SYNONYMS.items():
        s = re.sub(r'\b' + syn + r'\b', canonical, s)
    tokens = s.split()
    tokens = [_NUM_WORDS.get(t, t) for t in tokens]
    cleaned = []
    for i, t in enumerate(tokens):
        if t in _FILLER_UNITS and i > 0 and re.match(r'^\d+$', tokens[i - 1]):
            continue
        cleaned.append(t)
    return ' '.join(cleaned)


# ------------------------------------------------------------------ #
#  IDK 检测
# ------------------------------------------------------------------ #

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


def is_idk(text: str) -> bool:
    """检测 'I don't know' 类回答。"""
    return bool(_IDK_PATTERNS.search(text))


# ------------------------------------------------------------------ #
#  时间等价判断
# ------------------------------------------------------------------ #

_UNIT_TO_DAYS = {
    'day': 1, 'days': 1, 'week': 7, 'weeks': 7,
    'month': 30.44, 'months': 30.44, 'year': 365.25, 'years': 365.25,
}


def _parse_temporal(s: str) -> float | None:
    s = re.sub(r'\b(ago|approximately|about|around)\b', '', s, flags=re.I).strip()
    m = re.match(r'^(\d+(?:\.\d+)?)\s*(days?|weeks?|months?|years?)$', s.strip(), re.I)
    if m:
        return float(m.group(1)) * _UNIT_TO_DAYS.get(m.group(2).lower(), 0)
    return None


def temporal_equivalent(a: str, b: str) -> bool:
    """判断两个时间表达式是否等价 (20% 容差)，如 '14 days' ≈ '2 weeks'。"""
    da, db = _parse_temporal(a), _parse_temporal(b)
    if da is not None and db is not None and da > 0 and db > 0:
        return abs(da - db) / max(da, db) < 0.20
    return False


# ------------------------------------------------------------------ #
#  辅助：括号缩写 / 可接受替代答案
# ------------------------------------------------------------------ #

def _extract_parenthesized(text: str) -> str | None:
    m = re.search(r'\(([^)]+)\)', text)
    return m.group(1) if m else None


def _split_alternatives(ground_truth: str) -> list[str]:
    """拆分含 'also acceptable' 或 '(or X)' 的参考答案。"""
    alternatives = [ground_truth]
    m = re.match(r'^(.+?)\.\s+(.+?)\s*(?:\(.*?\)\s*)?(?:is\s+)?also\s+acceptable', ground_truth, re.I)
    if m:
        alternatives = [m.group(1).strip(), m.group(2).strip()]
        alternatives = [re.sub(r'\s*\(.*?\)', '', a).strip() for a in alternatives]
    m_or = re.search(r'\(or\s+(.+?)\)', ground_truth, re.I)
    if m_or:
        base = re.sub(r'\s*\(or\s+.+?\)', '', ground_truth).strip()
        alternatives = [base, m_or.group(1).strip()]
    return alternatives


# ------------------------------------------------------------------ #
#  Token-F1 & Exact Match
# ------------------------------------------------------------------ #

def _extract_primary_number(text: str) -> str | None:
    """Extract the primary number from text (digit or word form)."""
    norm = normalize_answer(text)
    # Find the first number in normalized text
    m = re.search(r'\b(\d+(?:\.\d+)?)\b', norm)
    if m:
        return m.group(1)
    return None


def _token_f1_single(prediction: str, ground_truth: str) -> float:
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    if temporal_equivalent(prediction, ground_truth):
        return 1.0
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
    """计算 token 级 F1，支持 IDK 匹配和多答案替代。"""
    prediction, ground_truth = str(prediction), str(ground_truth)
    if is_idk(prediction) and is_idk(ground_truth):
        return 1.0
    alternatives = _split_alternatives(ground_truth)
    return max(_token_f1_single(prediction, alt) for alt in alternatives)


def exact_match(prediction: str, ground_truth: str) -> float:
    """精确匹配，支持 IDK 匹配和括号缩写。"""
    prediction, ground_truth = str(prediction), str(ground_truth)
    if is_idk(prediction) and is_idk(ground_truth):
        return 1.0
    alternatives = _split_alternatives(ground_truth)
    for alt in alternatives:
        norm_pred = normalize_answer(prediction)
        norm_gt = normalize_answer(alt)
        if norm_pred == norm_gt:
            return 1.0
        paren_gt = _extract_parenthesized(alt)
        paren_pred = _extract_parenthesized(prediction)
        if paren_gt and normalize_answer(paren_gt) == norm_pred:
            return 1.0
        if paren_pred and normalize_answer(paren_pred) == norm_gt:
            return 1.0
    return 0.0


# ------------------------------------------------------------------ #
#  答案后处理
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
    re.compile(r'^\w+\s+\w+\s+at\s+', re.I),
    re.compile(r'^it\s+(took|was|is|cost|costs|lasted|takes)\s+', re.I),
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

_ARITHMETIC_PATTERN = re.compile(r'^[\d$,.\s+\-×÷*/=()]+?=\s*(.+?)$')


def clean_answer(text: str) -> str:
    """去除 LLM 常见废话前缀和尾部噪声。"""
    s = text.strip()
    # Yes/No 早期提取
    yn_early = re.match(r'^(yes|no)\b[,.\s!]', s, re.I)
    if yn_early and len(s) >= 10:
        return yn_early.group(1).capitalize()
    # 否定检测
    if (len(s) > 30 and not re.match(r'^(yes|no)\b', s, re.I)
        and re.search(r'\b(did not|didn\'t|was not|wasn\'t|is not|isn\'t|cannot|can\'t|never|not true|incorrect)\b', s, re.I)
        and not re.search(r'\d', s[:20])):
        return "No"
    # 算术提取
    m = _ARITHMETIC_PATTERN.match(s)
    if m:
        s = m.group(1).strip()
    # 去前缀
    for _ in range(3):
        for pat in _PREAMBLE_PATTERNS:
            cleaned = pat.sub('', s).strip()
            if cleaned:
                s = cleaned
    # 去尾部噪声
    for pat in _TRAILING_PATTERNS:
        s = pat.sub('', s).strip()
    # 去末尾句号
    if len(s) < 100 and s.endswith('.'):
        s = s[:-1].strip()
    # 去引号
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'", '\u201c', '\u201d'):
        s = s[1:-1].strip()
    if s.startswith(('\u201c', '"')) and s.endswith(('\u201d', '"')):
        s = s[1:-1].strip()
    # 去 markdown bold
    s = re.sub(r'\*\*([^*]+)\*\*', r'\1', s)
    return s


def extract_counting_answer(text: str, question: str) -> str:
    """对 'how many/much/long' 问题，提取数字+单位。"""
    q_lower = question.lower()
    if not re.search(r'\bhow\s+(many|much|long|often|far)\b', q_lower):
        return text
    _NUM_UNIT = r'\$?\d[\d,]*(?:\.\d+)?\s*(?:-?\s*)?(?:days?|weeks?|hours?|minutes?|months?|years?|miles?|km|dollars?|times?)?'
    m_final = re.search(r'(?:total|answer|=|is)\s*[:=]?\s*(' + _NUM_UNIT + r')\s*\.?\s*$', text, re.I)
    if m_final:
        return m_final.group(1).strip()
    m = re.search(r'(' + _NUM_UNIT + r')', text)
    if m:
        num_part = m.group(0).strip()
        if len(text) > len(num_part) * 3 and len(num_part) < 20:
            return num_part
    return text
