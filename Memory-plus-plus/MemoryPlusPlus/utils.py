"""
Memory++ Utility Functions

Core utilities for entity extraction, text normalization, scoring, and date parsing.
These are extracted from the main benchmark_eval_kg.py for modularity.
"""

import re
from datetime import datetime

# ------------------------------------------------------------------ #
#  Number word → digit mapping
# ------------------------------------------------------------------ #

_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}

# ------------------------------------------------------------------ #
#  Stop entities (common words to exclude from KG matching)
# ------------------------------------------------------------------ #

_STOP_ENTITIES = {
    "i", "me", "my", "you", "your", "we", "our", "they", "the",
    "a", "an", "it", "this", "that", "is", "was", "are", "were",
    "been", "be", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can",
    "and", "but", "or", "not", "no", "yes", "so", "if", "then",
    "what", "which", "who", "where", "when", "how", "why",
    "about", "with", "from", "for", "of", "in", "on", "at", "to",
    "by", "as", "up", "out", "into", "than", "also", "just",
    "very", "really", "actually", "recently", "usually", "always",
    "some", "any", "all", "each", "every", "both", "few", "many",
    "much", "more", "most", "other", "another", "same",
    "new", "old", "good", "great", "big", "long", "little",
    "first", "last", "next", "right", "well", "still", "even",
}

# ------------------------------------------------------------------ #
#  Entity extraction (regex-based NER)
# ------------------------------------------------------------------ #

def extract_entities(text: str) -> list[str]:
    """Extract named entities from text using regex patterns.

    Extracts: proper nouns, capitalized multi-word phrases, numbers with units,
    dates, email-like patterns, quoted strings.
    """
    entities = []
    # Capitalized words / multi-word names (2+ words starting with uppercase)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
        ent = m.group(1).strip()
        if ent.lower() not in _STOP_ENTITIES and len(ent) > 2:
            entities.append(ent)
    # Single capitalized words (not at sentence start)
    for m in re.finditer(r'(?<=[.!?]\s)\s*([A-Z][a-z]+)\b|(?<=\s)([A-Z][a-z]{2,})\b', text):
        ent = (m.group(1) or m.group(2) or "").strip()
        if ent and ent.lower() not in _STOP_ENTITIES:
            entities.append(ent)
    # Numbers with units
    for m in re.finditer(r'\b(\d+(?:\.\d+)?\s*(?:years?|months?|weeks?|days?|hours?|minutes?|miles?|km|dollars?|times?))\b', text, re.I):
        entities.append(m.group(1).strip())
    # Quoted strings
    for m in re.finditer(r'"([^"]{2,50})"', text):
        entities.append(m.group(1).strip())
    return entities


def extract_relation_triples(text: str) -> list[tuple[str, str, str]]:
    """Extract (subject, relation, object) triples from text using rule-based patterns.

    Zero LLM calls — pure regex extraction.

    Patterns:
    1. SVO: "I started learning guitar" → (i, started, learning guitar)
    2. Possessive: "My dog is a Golden Retriever" → (user, has_dog, golden retriever)
    3. Location: "I went to Serenity Yoga" → (user, location, serenity yoga)
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
        m = re.search(r"\b(?:my|our)\s+(\w+(?:\s+\w+)?)\s+(?:is|was|are|were)\s+(.+?)(?:\.|,|$)", sent, re.I)
        if m:
            subj = m.group(1).strip().lower()
            obj = m.group(2).strip().rstrip('.').lower()
            if 1 < len(obj) < 80:
                triples.append(("user", f"has_{subj}", obj))
        # Pattern 3: Location "at/in [Place]"
        m = re.search(r"\b(?:at|in|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", sent)
        if m:
            place = m.group(1).lower()
            if place not in _STOP_ENTITIES and len(place) > 2:
                triples.append(("user", "location", place))
    return triples

# ------------------------------------------------------------------ #
#  Date parsing
# ------------------------------------------------------------------ #

_DATE_PATTERNS = [
    (r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', '%Y-%m-%d'),
    (r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', '%m/%d/%Y'),
    (r'\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b', '%B %d %Y'),
]

def parse_date(date_str: str) -> datetime | None:
    """Parse a date string into a datetime object."""
    if not date_str:
        return None
    for pattern, fmt in _DATE_PATTERNS:
        m = re.search(pattern, date_str)
        if m:
            try:
                return datetime.strptime(m.group(0).replace(',', ''), fmt)
            except ValueError:
                continue
    return None

# ------------------------------------------------------------------ #
#  Answer normalization and scoring
# ------------------------------------------------------------------ #

def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison."""
    s = s.strip().lower()
    # Number words to digits
    for word, digit in _NUM_WORDS.items():
        s = re.sub(rf'\b{word}\b', digit, s)
    # Normalize floats: "2.0" → "2"
    s = re.sub(r'\b(\d+)\.0\b', r'\1', s)
    # Strip temporal directional words
    s = re.sub(r'\bago\b', '', s)
    s = re.sub(r'\b(approximately|about|around|nearly|roughly)\b', '', s)
    # Remove articles and punctuation
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = ' '.join(s.split())
    return s.strip()


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


from collections import Counter  # noqa: E402


def _is_idk(text: str) -> bool:
    """Check if text is an 'I don't know' response."""
    t = text.strip().lower()
    idk_phrases = [
        "i don't know", "i do not know", "i'm not sure",
        "i am not sure", "not mentioned", "no information",
        "cannot determine", "can't determine", "unknown",
        "not enough information", "insufficient information",
    ]
    return any(p in t for p in idk_phrases)
