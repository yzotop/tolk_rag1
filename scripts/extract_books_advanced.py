#!/usr/bin/env python3
"""
Advanced extraction of book titles and authors from Russian Telegram posts.

Features:
- Uses Natasha (NER + morphology) to detect person names
- Uses context-aware pattern matching for book titles (quotes + keywords)
- Supports optional dictionary of known authors (politics/economics)
- Provides confidence score and rationale for each extraction

Usage:
    python scripts/extract_books_advanced.py \
        --input data/raw/tolk_posts.csv \
        --output data/processed/books_advanced.csv \
        --authors data/aux/authors_politics_economics.txt \
        --json

Author dictionary format: one author per line ("Имя Фамилия").
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from natasha import (
    Doc,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    Segmenter,
)


# --- Configuration ------------------------------------------------------------------

# Keywords that increase confidence when found near title/author
BOOK_KEYWORDS = {
    "книга",
    "книгу",
    "книги",
    "роман",
    "издание",
    "прочитать",
    "читали",
    "читает",
    "прочтите",
    "издательство",
    "новинка",
    "литература",
    "автор",
    "написал",
    "написана",
    "написала",
    "монография",
    "исследование",
}

# Keywords indicating politics/economics domain
DOMAIN_KEYWORDS = {
    "политика",
    "политический",
    "экономика",
    "экономический",
    "экономика",
    "финансы",
    "финансовый",
    "рынок",
    "государство",
    "общество",
    "социология",
    "история",
    "капитал",
    "реформа",
    "кризис",
    "власть",
    "идеология",
    "геополитика",
    "стратегия",
    "социальный",
}

TITLE_MAX_LEN = 120
TITLE_MIN_LEN = 5
CONTEXT_WINDOW = 80  # characters left/right


@dataclass
class Extraction:
    message_id: int
    date_utc: Optional[str]
    book_title: Optional[str]
    author: Optional[str]
    confidence: float
    reasons: List[str]
    source_text: str


# --- Utilities ----------------------------------------------------------------------

def load_author_dictionary(path: Optional[Path]) -> set[str]:
    if not path:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"Author dictionary not found: {path}")
    authors = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                authors.add(name.lower())
    return authors


def regex_candidates(text: str) -> List[str]:
    """Extract potential titles from quotes and keyword patterns."""
    patterns = [
        r"«([^»]{3,})»",
        r'"([^\"]{3,})"',
        r"'([^']{3,})'",
        r"(?:книга|роман|монография|исследование)\s+[«\"]([^»\"]{3,})[»\"]",
        r"[«\"]([^»\"]{3,})[»\"]\s*[—–-]\s*(?:книга|роман|монография)",
    ]

    candidates = []
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for match in matches:
            cleaned = match.strip()
            if TITLE_MIN_LEN <= len(cleaned) <= TITLE_MAX_LEN:
                candidates.append(cleaned)
    return candidates


def find_keyword_context(text: str, substring: str, keywords: Iterable[str]) -> list[str]:
    reasons = []
    lower = text.lower()
    idx = lower.find(substring.lower())
    if idx == -1:
        return reasons
    start = max(0, idx - CONTEXT_WINDOW)
    end = min(len(text), idx + len(substring) + CONTEXT_WINDOW)
    context = lower[start:end]
    for kw in keywords:
        if kw in context:
            reasons.append(f"context:{kw}")
    return reasons


def score_title(text: str, candidate: str) -> tuple[float, list[str]]:
    reasons = []
    score = 0.0

    # Basic structure check
    if any(ch.isalpha() for ch in candidate):
        score += 0.2
        reasons.append("alpha")

    # Title case heuristic (not all-caps, not all lowercase)
    if not candidate.isupper() and not candidate.islower():
        score += 0.1
        reasons.append("capitalized")

    # Context keywords
    reasons.extend(find_keyword_context(text, candidate, BOOK_KEYWORDS))
    if any(r.startswith("context:") for r in reasons):
        score += 0.3

    # Domain keywords in context
    domain_hits = find_keyword_context(text, candidate, DOMAIN_KEYWORDS)
    reasons.extend(domain_hits)
    if domain_hits:
        score += 0.2

    # Surrounding quotation marks boost
    if f'«{candidate}»' in text or f'"{candidate}"' in text:
        score += 0.2
        reasons.append("quotes")

    return score, reasons


def normalize_whitespace(text) -> str:
    if text is None:
        return ""
    if isinstance(text, float):
        if pd.isna(text):
            return ""
        text = str(text)
    elif not isinstance(text, str):
        text = str(text)
    return re.sub(r"\s+", " ", text).strip()


# --- Natasha pipeline ---------------------------------------------------------------

_segmenter = Segmenter()
_emb = NewsEmbedding()
_morph_tagger = NewsMorphTagger(_emb)
_ner_tagger = NewsNERTagger(_emb)
_morph_vocab = MorphVocab()


def extract_persons(text: str) -> list[str]:
    doc = Doc(text)
    doc.segment(_segmenter)
    doc.tag_morph(_morph_tagger)
    doc.tag_ner(_ner_tagger)

    persons = []
    for span in doc.spans:
        if span.type == "PER":
            span.normalize(_morph_vocab)
            persons.append(span.normal)
    return persons


# --- Main extraction ----------------------------------------------------------------

def extract_from_post(
    row: pd.Series,
    author_dict: set[str],
) -> list[Extraction]:
    text = normalize_whitespace(row.get("text", ""))
    if not text:
        return []

    message_id = int(row.get("message_id", -1))
    date_utc = row.get("date_utc")
    preview = text[:300]

    extractions: list[Extraction] = []

    # Candidate titles
    candidates = regex_candidates(text)
    title_scores = {}
    for cand in candidates:
        score, reasons = score_title(text, cand)
        if score >= 0.4:  # minimum threshold
            title_scores[cand] = (score, reasons)

    # Persons via NER
    authors_raw = extract_persons(text)
    authors_scored: dict[str, tuple[float, list[str]]] = {}
    for person in authors_raw:
        person_lower = person.lower()
        reasons = []
        score = 0.2  # base

        # Dictionary boost
        if person_lower in author_dict:
            score += 0.4
            reasons.append("dict")

        # Keyword context boost
        context_reasons = find_keyword_context(text, person, BOOK_KEYWORDS)
        reasons.extend(context_reasons)
        if context_reasons:
            score += 0.2

        domain_reasons = find_keyword_context(text, person, DOMAIN_KEYWORDS)
        reasons.extend(domain_reasons)
        if domain_reasons:
            score += 0.1

        if score >= 0.4:
            authors_scored[person] = (score, reasons)

    def valid_author(name: Optional[str]) -> bool:
        if not name:
            return False
        name = name.strip()
        if len(name) < 3:
            return False
        # reject single-letter tokens or initials-only strings
        if len(name.split()) == 1:
            return len(name) > 1 and name[0].isalpha() and any(ch.isalpha() for ch in name[1:])
        return True

    # Combine titles and authors
    if title_scores:
        for title, (t_score, t_reasons) in title_scores.items():
            if authors_scored:
                for author, (a_score, a_reasons) in authors_scored.items():
                    if not valid_author(author):
                        continue
                    extraction = Extraction(
                        message_id=message_id,
                        date_utc=date_utc,
                        book_title=title,
                        author=author,
                        confidence=min(1.0, t_score + a_score),
                        reasons=t_reasons + a_reasons,
                        source_text=preview,
                    )
                    extractions.append(extraction)
            else:
                extraction = Extraction(
                    message_id=message_id,
                    date_utc=date_utc,
                    book_title=title,
                    author=None,
                    confidence=min(1.0, t_score + 0.1),
                    reasons=t_reasons,
                    source_text=preview,
                )
                extractions.append(extraction)

    # Authors without titles (less confident)
    elif authors_scored:
        for author, (a_score, a_reasons) in authors_scored.items():
            if not valid_author(author):
                continue
            extraction = Extraction(
                message_id=message_id,
                date_utc=date_utc,
                book_title=None,
                author=author,
                confidence=min(1.0, a_score),
                reasons=a_reasons,
                source_text=preview,
            )
            extractions.append(extraction)

    return extractions


def process_dataframe(
    df: pd.DataFrame,
    author_dict: set[str],
) -> pd.DataFrame:
    results: list[Extraction] = []

    for idx, row in df.iterrows():
        extracted = extract_from_post(row, author_dict)
        results.extend(extracted)
        if (idx + 1) % 200 == 0:
            print(f"   Processed {idx + 1:,} posts, found {len(results):,} mentions...")

    if not results:
        return pd.DataFrame()

    data = [
        {
            "message_id": ext.message_id,
            "date_utc": ext.date_utc,
            "book_title": ext.book_title,
            "author": ext.author,
            "confidence": round(ext.confidence, 3),
            "reasons": "; ".join(ext.reasons),
            "source_text": ext.source_text,
        }
        for ext in results
    ]

    df_out = pd.DataFrame(data)
    df_out = df_out.drop_duplicates(
        subset=["message_id", "book_title", "author"], keep="last"
    )
    df_out = df_out.sort_values(by=["confidence", "date_utc"], ascending=[False, False])
    return df_out


# --- CLI ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Advanced book extraction")
    ap.add_argument("--input", default="data/raw/tolk_posts.csv")
    ap.add_argument("--output", default="data/processed/books_advanced.csv")
    ap.add_argument(
        "--authors",
        type=str,
        default=None,
        help="Optional path to author dictionary (one name per line)",
    )
    ap.add_argument("--json", action="store_true", help="Also save JSON output")
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence threshold (default: 0.6)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    json_path = output_path.with_suffix('.json')

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        print("   Run scripts/fetch_tolk_posts.py or copy existing dataset")
        return

    author_dict = load_author_dictionary(Path(args.authors) if args.authors else None)
    if author_dict:
        print(f"✅ Loaded {len(author_dict):,} authors from dictionary")
    else:
        print("ℹ️  No author dictionary provided")

    print(f"📥 Reading posts from {input_path}...")
    df = pd.read_csv(input_path, encoding='utf-8')
    print(f"   Loaded {len(df):,} posts\n")

    df_results = process_dataframe(df, author_dict)

    if df_results.empty:
        print("⚠️  No book mentions detected. Consider lowering thresholds or updating patterns.")
        return

    # Apply confidence threshold
    before = len(df_results)
    df_results = df_results[df_results['confidence'] >= args.min_confidence]
    after = len(df_results)

    print(f"\n📊 Extracted {before:,} mentions; {after:,} >= confidence {args.min_confidence}")
    unique_titles = df_results['book_title'].dropna().nunique()
    unique_authors = df_results['author'].dropna().nunique()
    print(f"   Unique titles: {unique_titles:,}")
    print(f"   Unique authors: {unique_authors:,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n💾 Saved CSV to {output_path}")

    if args.json:
        with json_path.open("w", encoding="utf-8") as jf:
            json.dump(
                df_results.to_dict(orient='records'),
                jf,
                ensure_ascii=False,
                indent=2,
            )
        print(f"   JSON saved to {json_path}")

    print("\n📚 Top examples:")
    for _, row in df_results.head(5).iterrows():
        title = row['book_title'] or '(нет названия)'
        author = row['author'] or '(нет автора)'
        print(f"   • {title} — {author} | conf={row['confidence']:.2f} | {row['reasons']}")


if __name__ == "__main__":
    main()


