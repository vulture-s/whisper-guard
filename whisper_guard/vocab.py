import re
from typing import List, Optional


def build_hotwords_prompt(words: List[str], language: str = "zh") -> str:
    cleaned = [word.strip() for word in words if word and word.strip()]
    return " ".join(cleaned)


def filter_filler_words(text: str, dictionary: Optional[List[str]] = None) -> str:
    fillers = dictionary or ["嗯嗯", "啊啊", "呃呃", "喔喔"]
    cleaned = text
    for filler in fillers:
        cleaned = cleaned.replace(filler, " ")
    return re.sub(r"\s+", " ", cleaned).strip()
