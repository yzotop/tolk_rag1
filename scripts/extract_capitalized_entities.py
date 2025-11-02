#!/usr/bin/env python3
"""
Extract capitalized tokens from posts and provide a coarse classification.

Classes:
 - person: detected via Natasha PER spans, dictionary matching, pattern detection, or cross-referencing
 - title: token appears inside quotation marks ("…" or «…»)
 - country: token matches a known country list (RU/EN)
 - unknown: fallback

Usage example:
    python scripts/extract_capitalized_entities.py \
        --input data/raw/tolk_posts_5000.csv \
        --output data/processed/capitalized_entities.csv \
        --authors data/aux/authors_politics_economics.txt \
        --min-count 2
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsNERTagger, Segmenter


# --- NLP pipeline -------------------------------------------------------------------

SEGMENTER = Segmenter()
EMBED = NewsEmbedding()
MORPH_TAGGER = NewsMorphTagger(EMBED)
NER_TAGGER = NewsNERTagger(EMBED)
MORPH_VOCAB = MorphVocab()


# --- Helpers ------------------------------------------------------------------------

CAP_WORD_RE = re.compile(r"\b[А-ЯA-Z][\w\-’']*", re.UNICODE)
FIRST_WORD_RE = re.compile(r"\S+")

COUNTRY_BASE_FORMS = {
    "Россия": ["Россия", "России", "Россию", "Россией", "Россие"],
    "СССР": ["СССР"],
    "Украина": ["Украина", "Украины", "Украине", "Украину"],
    "Беларусь": ["Беларусь", "Белоруссия", "Беларуси"],
    "Казахстан": ["Казахстан", "Казахстана", "Казахстане"],
    "Грузия": ["Грузия", "Грузии", "Грузию"],
    "Армения": ["Армения", "Армении", "Армению"],
    "Франция": ["Франция", "Франции", "Францию"],
    "Германия": ["Германия", "Германии", "Германию"],
    "США": ["США", "Соединенные", "Штаты"],
    "Китай": ["Китай", "Китая", "Китае"],
    "Италия": ["Италия", "Италии", "Италию"],
    "Испания": ["Испания", "Испании", "Испанию"],
    "Польша": ["Польша", "Польши", "Польшу"],
    "Англия": ["Англия", "Англии", "Англию"],
    "Великобритания": ["Великобритания", "Великобритании"],
    "Япония": ["Япония", "Японии", "Японию"],
    "Индия": ["Индия", "Индии", "Индию"],
    "Бразилия": ["Бразилия", "Бразилии", "Бразилию"],
    "Канада": ["Канада", "Канады", "Канаду"],
    "Австралия": ["Австралия", "Австралии", "Австралию"],
    "Russia": ["Russia"],
    "Ukraine": ["Ukraine"],
    "Belarus": ["Belarus"],
    "Kazakhstan": ["Kazakhstan"],
    "Georgia": ["Georgia"],
    "Armenia": ["Armenia"],
    "France": ["France"],
    "Germany": ["Germany"],
    "USA": ["USA", "United", "States", "America"],
    "China": ["China"],
    "Italy": ["Italy"],
    "Spain": ["Spain"],
    "Poland": ["Poland"],
    "England": ["England"],
    "Great Britain": ["Great", "Britain"],
    "Japan": ["Japan"],
    "India": ["India"],
    "Brazil": ["Brazil"],
    "Canada": ["Canada"],
    "Australia": ["Australia"],
}

COUNTRY_NAMES_LOWER = {form.lower() for forms in COUNTRY_BASE_FORMS.values() for form in forms}
COUNTRY_FORM_TO_BASE = {form.lower(): base for base, forms in COUNTRY_BASE_FORMS.items() for form in forms}

# Russian surname endings (comprehensive list)
RUSSIAN_SURNAME_ENDINGS = [
    "ов", "ова", "ово", "овы", "овым", "овыми", "овых",
    "ев", "ева", "ево", "евы", "евым", "евыми", "евых",
    "ин", "ина", "ино", "ины", "иным", "иными", "иных",
    "ский", "ская", "ское", "ские", "ским", "скими", "ских",
    "евский", "евская", "евское", "евские", "евским", "евскими", "евских",
    "овский", "овская", "овское", "овские", "овским", "овскими", "овских",
    "ской", "ской", "ская", "ско", "ские", "скими", "ских",
    "ич", "ович", "евич", "ыч",
    "ец", "ец", "ца", "це", "цы", "цами", "цов",
    "ок", "ка", "ко", "ки", "ком", "ками", "ков",
    "енко", "енко", "енко", "енко", "енко",
    "ук", "ука", "уки", "уком", "уками", "уков",
    "юк", "юка", "юки", "юком", "юками", "юков",
    "енко", "енко",
    "ко", "ко",
]


def normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def is_sentence_start(index: int, text: str) -> bool:
    if index <= 0:
        return False
    i = index - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    if i < 0:
        return False
    return text[i] in ".!?"


def get_first_cap_span(text: str) -> Tuple[int, int]:
    match = CAP_WORD_RE.search(text)
    if match:
        return match.span()
    return (-1, -1)


def get_first_word_span(text: str) -> Tuple[int, int]:
    match = FIRST_WORD_RE.search(text)
    if match:
        return match.span()
    return (-1, -1)


def tokens_in_quotes(text: str) -> set[str]:
    results: set[str] = set()
    for quote in re.findall(r"«([^»]+)»", text):
        for token in CAP_WORD_RE.findall(quote):
            results.add(token.strip())
    for quote in re.findall(r'"([^\"]+)"', text):
        for token in CAP_WORD_RE.findall(quote):
            results.add(token.strip())
    return results


def extract_person_spans(text: str) -> Tuple[List[Dict[str, Any]], set[str]]:
    doc = Doc(text)
    doc.segment(SEGMENTER)
    doc.tag_morph(MORPH_TAGGER)
    doc.tag_ner(NER_TAGGER)
    spans: List[Dict[str, Any]] = []
    tokens_lower: set[str] = set()
    for span in doc.spans:
        if span.type == "PER":
            span.normalize(MORPH_VOCAB)
            if not doc.tokens:
                continue
            if span.start >= len(doc.tokens):
                continue
            stop_index = min(span.stop - 1, len(doc.tokens) - 1)
            start_token = doc.tokens[span.start]
            end_token = doc.tokens[stop_index]
            spans.append({
                "start": start_token.start,
                "end": end_token.stop,
                "normal": span.normal,
            })
            for token in span.normal.split():
                tokens_lower.add(token.lower())
    return spans, tokens_lower


def find_person_span(person_spans: List[Dict[str, Any]], start: int, end: int) -> Optional[Dict[str, Any]]:
    for span in person_spans:
        if start >= span["start"] and end <= span["end"]:
            return span
    return None


def normalize_word(token: str) -> str:
    doc = Doc(token)
    doc.segment(SEGMENTER)
    doc.tag_morph(MORPH_TAGGER)
    if doc.tokens:
        tok = doc.tokens[0]
        tok.lemmatize(MORPH_VOCAB)
        if tok.lemma:
            return tok.lemma.capitalize() if token[0].isupper() else tok.lemma
    return token


def load_author_dictionary(path: Path) -> Tuple[set[str], set[str]]:
    """Load author dictionary and extract first names and surnames.
    
    Returns:
        (first_names_set, surnames_set) - both normalized and lowercase for matching
    """
    if not path.exists():
        print(f"Warning: Author dictionary not found at {path}")
        return set(), set()
    
    first_names: set[str] = set()
    surnames: set[str] = set()
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Skip entries that are clearly not names
            if any(x in line.lower() for x in ["film", "theorists", "century", "(film)", "(author)", "(economist)"]):
                continue
            
            # Extract name parts
            parts = line.split()
            if not parts:
                continue
            
            # Remove parenthetical suffixes and trailing punctuation
            clean_parts = []
            for part in parts:
                if "(" in part:
                    break
                clean_parts.append(part.rstrip(".,;"))
            
            if not clean_parts:
                continue
            
            # First name is first word (normalize it)
            if len(clean_parts) >= 1:
                first_name = clean_parts[0]
                # Skip single initials
                if len(first_name) > 1 and first_name.replace(".", "").replace("-", "").isalpha():
                    normalized_first = normalize_word(first_name).lower()
                    first_names.add(normalized_first)
            
            # Surname is last word (normalize it)
            if len(clean_parts) >= 2:
                surname = clean_parts[-1]
                # Skip very short surnames
                if len(surname) > 2 and surname.replace("-", "").isalpha():
                    normalized_surname = normalize_word(surname).lower()
                    surnames.add(normalized_surname)
    
    print(f"Loaded {len(first_names)} first names and {len(surnames)} surnames from {path}")
    return first_names, surnames


def has_surname_ending(token: str, doc: Doc) -> bool:
    """Check if token has Russian surname-like ending."""
    if not doc.tokens:
        return False
    
    # Get normalized form via lemmatization
    tok = doc.tokens[0]
    tok.lemmatize(MORPH_VOCAB)
    normalized = tok.lemma.lower() if tok.lemma else token.lower()
    
    # Check against surname endings
    for ending in RUSSIAN_SURNAME_ENDINGS:
        if normalized.endswith(ending):
            # Ensure minimum length
            if len(normalized) > len(ending) + 1:
                return True
    
    return False


def is_potential_surname(token: str, doc: Doc) -> bool:
    """Check if token could be a surname based on morphological patterns."""
    # Check for surname ending
    if not has_surname_ending(token, doc):
        return False
    
    # Exclude if it's a country name
    token_lower = token.lower()
    if token_lower in COUNTRY_FORM_TO_BASE:
        return False
    
    # Ensure reasonable length
    if len(token) < 3:
        return False
    
    return True


def get_sentence_boundaries(text: str) -> List[Tuple[int, int]]:
    """Find sentence boundaries in text.
    
    Returns:
        List of (start, end) positions for each sentence
    """
    if not text:
        return [(0, 0)]
    
    boundaries = []
    
    # Pattern to find sentence endings followed by space and capital
    pattern = re.compile(r'([.!?])\s+([А-ЯA-Z])', re.UNICODE)
    
    last_end = 0
    for match in pattern.finditer(text):
        end_pos = match.end() - len(match.group(2))  # Position before the capital letter
        if end_pos > last_end:
            boundaries.append((last_end, end_pos))
            last_end = end_pos
    
    # Add final sentence
    if last_end < len(text):
        boundaries.append((last_end, len(text)))
    
    # If no sentence boundaries found, entire text is one sentence
    if not boundaries:
        boundaries = [(0, len(text))]
    
    return boundaries


def get_token_sentence_index(token_start: int, token_end: int, sentence_boundaries: List[Tuple[int, int]]) -> int:
    """Find which sentence a token belongs to."""
    for idx, (sent_start, sent_end) in enumerate(sentence_boundaries):
        if sent_start <= token_start < sent_end:
            return idx
    return -1


def classify_token(
    token: str,
    start: int,
    end: int,
    person_spans: List[Dict[str, Any]],
    person_tokens: set[str],
    quoted_tokens: set[str],
    quoted_tokens_lower: set[str],
    author_first_names: set[str],
    author_surnames: set[str],
) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """Classify token and return classification, normalized form, and metadata.
    
    Returns:
        (classification, normalized, metadata_dict)
        metadata_dict contains:
            - detection_method: how this was detected
            - is_first_name: bool
            - is_dictionary_surname: bool
            - is_pattern_surname: bool
    """
    metadata: Dict[str, Any] = {
        "detection_method": None,
        "is_first_name": False,
        "is_dictionary_surname": False,
        "is_pattern_surname": False,
    }
    
    token_lower = token.lower()
    normalized_lower = normalize_word(token).lower()
    
    # 1. Natasha NER detection
    span = find_person_span(person_spans, start, end)
    if span:
        if start != span["start"]:
            return None, None, metadata
        metadata["detection_method"] = "natasha_ner"
        return "person", span["normal"], metadata
    
    # 2. Dictionary surname matching
    if normalized_lower in author_surnames:
        metadata["detection_method"] = "dictionary_surname"
        metadata["is_dictionary_surname"] = True
        return "person", normalize_word(token), metadata
    
    # 3. Dictionary first name matching
    if normalized_lower in author_first_names:
        metadata["detection_method"] = "dictionary_firstname"
        metadata["is_first_name"] = True
        return "person", normalize_word(token), metadata
    
    # 4. Morphological pattern detection for surnames
    doc = Doc(token)
    doc.segment(SEGMENTER)
    doc.tag_morph(MORPH_TAGGER)
    if is_potential_surname(token, doc):
        metadata["detection_method"] = "pattern_surname"
        metadata["is_pattern_surname"] = True
        return "person", normalize_word(token), metadata
    
    # 5. Natasha person tokens (from spans)
    if token_lower in person_tokens:
        metadata["detection_method"] = "natasha_token"
        return "person", normalize_word(token), metadata
    
    # 6. Title detection
    if token in quoted_tokens or token_lower in quoted_tokens_lower:
        metadata["detection_method"] = "quoted_title"
        return "title", token, metadata
    
    # 7. Country detection
    if token_lower in COUNTRY_FORM_TO_BASE:
        metadata["detection_method"] = "country_match"
        return "country", COUNTRY_FORM_TO_BASE[token_lower], metadata
    
    # 8. Unknown
    metadata["detection_method"] = "unknown"
    return "unknown", normalize_word(token), metadata


PUNCT_STRIP = ".,!?;:\"'()[]{}«»“”‟"


def extract_tokens_from_text(text: str) -> List[Tuple[str, int, int]]:
    tokens: List[Tuple[str, int, int]] = []
    seen_spans = set()
    for match in CAP_WORD_RE.finditer(text):
        start, end = match.span()
        if (start, end) in seen_spans:
            continue
        seen_spans.add((start, end))
        token = match.group().strip(PUNCT_STRIP)
        if not token or len(token) == 0:
            continue
        tokens.append((token, start, end))
    return tokens


def collect_entities(df: pd.DataFrame, author_first_names: set[str], author_surnames: set[str]) -> pd.DataFrame:
    aggregated: Dict[Tuple[str, str, str], Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "contexts": [], "original_counts": defaultdict(int)}
    )

    for row in df.itertuples(index=True):
        text_raw = getattr(row, "text", "")
        text = normalize_text(text_raw)
        if not text:
            continue

        first_cap_span = get_first_cap_span(text)
        first_word_span = get_first_word_span(text)

        quoted_tokens = tokens_in_quotes(text)
        quoted_tokens_lower = {qt.lower() for qt in quoted_tokens}
        person_spans, person_tokens = extract_person_spans(text)
        
        # Get sentence boundaries for cross-referencing
        sentence_boundaries = get_sentence_boundaries(text)

        tokens = extract_tokens_from_text(text)
        
        # Store token classifications with metadata
        token_classifications: List[Tuple[str, int, int, str, str, Dict[str, Any]]] = []
        # Format: (token, start, end, classification, normalized, metadata)

        # First pass: classify all tokens
        for idx, (token, start, end) in enumerate(tokens):
            if (start, end) == first_cap_span:
                continue
            if first_word_span != (-1, -1) and start < first_word_span[1] and end <= first_word_span[1]:
                continue
            if len(token) <= 1:
                continue

            if is_sentence_start(start, text):
                continue

            classification, normalized, metadata = classify_token(
                token,
                start,
                end,
                person_spans,
                person_tokens,
                quoted_tokens,
                quoted_tokens_lower,
                author_first_names,
                author_surnames,
            )
            if not classification:
                continue

            token_classifications.append((token, start, end, classification, normalized, metadata))

            # Store individual token
            detection_method = metadata.get("detection_method", "unknown")
            key = (normalized, classification, detection_method)
            entry = aggregated[key]
            entry["count"] += 1
            entry["original_counts"][token] += 1
            if len(entry["contexts"]) < 3:
                snippet_start = max(0, start - 40)
                snippet_end = min(len(text), end + 40)
                entry["contexts"].append(text[snippet_start:snippet_end].strip())

        # Second pass: expanded person pairing - merge person tokens with next capitalized token in same sentence
        # First, try adjacent pairs (immediately next to each other)
        for i in range(len(tokens) - 1):
            token_a, start_a, end_a = tokens[i]
            token_b, start_b, end_b = tokens[i + 1]

            if first_word_span != (-1, -1) and start_a < first_word_span[1] and end_b <= first_word_span[1]:
                continue

            if is_sentence_start(start_a, text):
                continue

            gap = text[end_a:start_b]
            if not gap or not gap.isspace():
                continue

            # Check if token_a is classified as person in token_classifications
            is_person_a = False
            for tok, st, en, cls, norm, meta in token_classifications:
                if st == start_a and en == end_a and cls == "person":
                    is_person_a = True
                    break

            # Skip if already part of a Natasha span
            span_combined = find_person_span(person_spans, start_a, end_b)
            if span_combined:
                continue

            # If token_a is person, try merging with token_b (even if token_b isn't classified yet)
            if is_person_a:
                normalized_pair = f"{normalize_word(token_a)} {normalize_word(token_b)}"
                key = (normalized_pair, "person", "adjacent_pair")
                entry = aggregated[key]
                entry["count"] += 1
                entry["original_counts"][f"{token_a} {token_b}"] += 1
                if len(entry["contexts"]) < 3:
                    snippet_start = max(0, start_a - 40)
                    snippet_end = min(len(text), end_b + 40)
                    entry["contexts"].append(text[snippet_start:snippet_end].strip())
        
        # Also merge any person token with next capitalized token in same sentence (non-adjacent)
        for i, (token_a, start_a, end_a, cls_a, norm_a, meta_a) in enumerate(token_classifications):
            if cls_a != "person":
                continue
            
            # Get sentence index
            sent_idx_a = get_token_sentence_index(start_a, end_a, sentence_boundaries)
            if sent_idx_a < 0:
                continue
            
            # Find token index in original tokens list
            token_idx_a = -1
            for idx, (tok, st, en) in enumerate(tokens):
                if st == start_a and en == end_a:
                    token_idx_a = idx
                    break
            
            if token_idx_a < 0:
                continue
            
            # Look for next capitalized token in same sentence
            for j in range(token_idx_a + 1, len(tokens)):
                token_b, start_b, end_b = tokens[j]
                
                # Must be in same sentence
                sent_idx_b = get_token_sentence_index(start_b, end_b, sentence_boundaries)
                if sent_idx_b != sent_idx_a:
                    break  # Left sentence, stop searching
                
                # Skip if token_b is sentence-initial
                if is_sentence_start(start_b, text):
                    continue
                
                # Check if already merged as adjacent pair
                if j == token_idx_a + 1:
                    gap = text[end_a:start_b]
                    if gap and gap.isspace():
                        continue  # Already handled as adjacent pair
                
                # Check if token_b is already part of a person span with token_a
                span_combined = find_person_span(person_spans, start_a, end_b)
                if span_combined:
                    break
                
                # Merge person token with next capitalized token
                normalized_pair = f"{norm_a} {normalize_word(token_b)}"
                key = (normalized_pair, "person", "expanded_pair")
                entry = aggregated[key]
                entry["count"] += 1
                entry["original_counts"][f"{token_a} {token_b}"] += 1
                if len(entry["contexts"]) < 3:
                    snippet_start = max(0, min(start_a, start_b) - 40)
                    snippet_end = min(len(text), max(end_a, end_b) + 40)
                    entry["contexts"].append(text[snippet_start:snippet_end].strip())
                
                # Only merge with first valid next token
                break

        # Third pass: cross-reference first names with surnames
        for i, (token_a, start_a, end_a, cls_a, norm_a, meta_a) in enumerate(token_classifications):
            if not meta_a.get("is_first_name", False):
                continue
            
            # Find sentence index for this token
            sent_idx_a = get_token_sentence_index(start_a, end_a, sentence_boundaries)
            if sent_idx_a < 0:
                continue
            
            # Search for potential surnames in the same sentence
            for j, (token_b, start_b, end_b, cls_b, norm_b, meta_b) in enumerate(token_classifications):
                if i == j:
                    continue
                
                # Must be in same sentence
                sent_idx_b = get_token_sentence_index(start_b, end_b, sentence_boundaries)
                if sent_idx_b != sent_idx_a:
                    continue
                
                # Check if token_b could be a surname
                is_surname_candidate = (
                    meta_b.get("is_dictionary_surname", False) or
                    meta_b.get("is_pattern_surname", False) or
                    (cls_b == "person" and not meta_b.get("is_first_name", False))
                )
                
                if not is_surname_candidate:
                    continue
                
                # Check distance (max 5 words apart)
                distance = abs(j - i)
                if distance > 5:
                    continue
                
                # Combine first name + surname
                normalized_combined = f"{norm_a} {norm_b}"
                key = (normalized_combined, "person", "cross_referenced")
                entry = aggregated[key]
                entry["count"] += 1
                entry["original_counts"][f"{token_a} {token_b}"] += 1
                if len(entry["contexts"]) < 3:
                    snippet_start = max(0, min(start_a, start_b) - 40)
                    snippet_end = min(len(text), max(end_a, end_b) + 40)
                    entry["contexts"].append(text[snippet_start:snippet_end].strip())

    rows = []
    for (normalized_token, cls, detection_method), data in aggregated.items():
        contexts_unique = list(dict.fromkeys(data["contexts"]))
        variants_sorted = sorted(data["original_counts"].items(), key=lambda x: (-x[1], x[0]))
        variant_examples = "; ".join(f"{tok} ({cnt})" for tok, cnt in variants_sorted[:5])
        rows.append({
            "normalized_token": normalized_token,
            "classification": cls,
            "detection_method": detection_method,
            "count": data["count"],
            "variant_examples": variant_examples,
            "sample_contexts": " | ".join(contexts_unique),
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract capitalized tokens and classify them")
    parser.add_argument("--input", default="data/raw/tolk_posts_5000.csv")
    parser.add_argument("--output", default="data/processed/capitalized_entities.csv")
    parser.add_argument("--authors", default="data/aux/authors_politics_economics.txt")
    parser.add_argument("--min-count", type=int, default=1)
    args = parser.parse_args()

    # Load author dictionary
    author_path = Path(args.authors)
    author_first_names, author_surnames = load_author_dictionary(author_path)

    df = pd.read_csv(args.input, encoding="utf-8")
    result_df = collect_entities(df, author_first_names, author_surnames)

    if args.min_count > 1:
        result_df = result_df[result_df["count"] >= args.min_count]

    result_df.sort_values(by="count", ascending=False, inplace=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved {len(result_df)} entities to {args.output}")


if __name__ == "__main__":
    main()


