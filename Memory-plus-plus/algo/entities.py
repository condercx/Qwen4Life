"""
零 LLM 开销的实体提取与关系三元组抽取。
"""

import re

# 非实体停用词
STOP_ENTITIES = {
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
    """从文本中提取命名实体（正则启发式，无 LLM 调用）。

    提取模式:
      1. 多词专有名词 (连续大写词)
      2. 非句首单个大写词
      3. 引号内容
      4. 数字+单位
      5. 日期
      6. "my X" 所有格后名词短语
    """
    entities = set()

    # 1. 多词专有名词
    for m in re.finditer(
        r'\b([A-Z][a-z]+(?:\s+(?:of|the|and|in|at|for|on|de|la|le))?\s+'
        r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text
    ):
        entities.add(m.group(0))

    # 2. 非句首单个大写词
    for m in re.finditer(r'(?<=[a-z,;]\s)([A-Z][a-z]{2,})\b', text):
        w = m.group(1)
        if w.lower() not in STOP_ENTITIES:
            entities.add(w)

    # 3. 引号内容
    for m in re.finditer(r'["\u201c]([^"\u201d]{2,60})["\u201d]', text):
        entities.add(m.group(1))
    for m in re.finditer(r"'([A-Z][^']{1,60})'", text):
        entities.add(m.group(1))

    # 4. 数字+单位
    for m in re.finditer(
        r'(\d+(?:\.\d+)?)\s*'
        r'(minutes?|hours?|days?|weeks?|months?|years?|'
        r'miles?|km|meters?|feet|'
        r'dollars?|\$|pounds?|euros?|'
        r'kg|lbs?|grams?|'
        r'times?|sessions?|classes?|lessons?)', text, re.IGNORECASE
    ):
        entities.add(m.group(0).strip())

    # 5. 日期
    for m in re.finditer(
        r'\b(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?'
        r'(?:,?\s*\d{4})?\b', text
    ):
        entities.add(m.group(0))

    # 6. "my X" 所有格
    for m in re.finditer(r'\bmy\s+([a-z]+(?:\s+[a-z]+)?)\b', text, re.IGNORECASE):
        val = m.group(1)
        if val.lower() not in STOP_ENTITIES and len(val) > 2:
            entities.add(val)

    return list(entities)


def extract_relation_triples(text: str) -> list[tuple[str, str, str]]:
    """抽取 (主语, 关系, 宾语) 三元组，纯规则，零 LLM 调用。

    模式:
      1. SVO: "I started learning guitar"
      2. 所有格: "My dog is a Golden Retriever"
      3. 地点: "I went to Serenity Yoga"
    """
    triples = []
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 10:
            continue

        # Pattern 1: SVO
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

        # Pattern 2: 所有格 "my X is/was Y"
        m = re.search(
            r"\b(?:my|our)\s+(\w+(?:\s+\w+)?)\s+(?:is|was|are|were)\s+(.+?)(?:\.|,|$)",
            sent, re.I
        )
        if m:
            subj = m.group(1).strip().lower()
            obj = m.group(2).strip().rstrip('.').lower()
            if 1 < len(obj) < 80:
                triples.append(("user", f"has_{subj}", obj))

        # Pattern 3: 地点 "at/in/to [Place]"
        m = re.search(
            r"\b(?:at|in|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b",
            sent
        )
        if m:
            place = m.group(1).lower()
            if place not in STOP_ENTITIES and len(place) > 2:
                triples.append(("user", "location", place))

    return triples
