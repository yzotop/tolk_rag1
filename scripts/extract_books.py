#!/usr/bin/env python3
"""
Extract book names and authors from Russian text posts.
Uses pattern matching and NLP for Russian language.
"""

import re
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd

# Russian-specific patterns for book extraction
# Common patterns: "книга", "автор", "написано", quotation marks, etc.

# Patterns for book titles (in Russian)
BOOK_TITLE_PATTERNS = [
    # Quoted titles: «книга», "книга", 'книга'
    r'«([^»]{3,80})»',  # Russian quotes
    r'"([^"]{3,80})"',  # English quotes
    r"'([^']{3,80})'",  # Single quotes
    
    # Patterns with keywords
    r'(?:книга|роман|повесть|рассказ|произведение)\s+[«"]([^»"]{3,80})[»"]',
    r'[«"]([^»"]{3,80})[»"]\s*(?:—|–|-)\s*(?:книга|роман|повесть)',
    
    # Title: Author format (Russian)
    r'([А-ЯЁ][А-Яа-яё\s\-]{2,50})\s*[—–:]\s*(?:автор|написал|написала)',
    
    # After "название", "называется"
    r'(?:название|называется|называли)\s+[«"]?([^«»"\.\,\n]{3,60})[«»"]?',
]

# Patterns for author names (Russian names)
AUTHOR_PATTERNS = [
    # Author name after keywords
    r'(?:автор|написал|написала|писатель|писательница)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2})',
    r'(?:автор|написал|написала|писатель|писательница):\s*([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2})',
    
    # Name before "написал", "автор"
    r'([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)\s+(?:написал|написала|автор)',
    
    # Format: "Имя Фамилия — автор книги"
    r'([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)\s*[—–-]\s*(?:автор|писатель)',
    
    # Full name pattern (2-3 words starting with capital)
    r'\b([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)\s+(?:написал|написала|автор)',
]

# Common Russian words to filter out false positives
STOP_WORDS = {
    'книга', 'роман', 'повесть', 'рассказ', 'автор', 'написал', 'написала',
    'писатель', 'писательница', 'произведение', 'публикация', 'издание',
    'глава', 'страница', 'страниц', 'том', 'часть'
}

def extract_quoted_titles(text: str) -> List[str]:
    """Extract titles from quotation marks."""
    titles = []
    
    # Russian quotes «»
    matches = re.findall(r'«([^»]{3,80})»', text)
    titles.extend(matches)
    
    # English quotes ""
    matches = re.findall(r'"([^"]{3,80})"', text)
    titles.extend(matches)
    
    # Filter out common false positives
    filtered = []
    for title in titles:
        title_lower = title.lower().strip()
        # Skip if it's a stop word or too short/long
        if len(title) < 5 or len(title) > 80:
            continue
        if title_lower in STOP_WORDS:
            continue
        # Skip if it's mostly punctuation or numbers
        if sum(c.isalnum() or c in ' -' for c in title) < len(title) * 0.6:
            continue
        filtered.append(title.strip())
    
    return filtered

def extract_pattern_titles(text: str) -> List[str]:
    """Extract titles using keyword patterns."""
    titles = []
    
    for pattern in BOOK_TITLE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match else ''
            if match and 5 <= len(match) <= 80:
                titles.append(match.strip())
    
    return titles

def extract_authors(text: str) -> List[str]:
    """Extract author names from text."""
    authors = []
    
    for pattern in AUTHOR_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match else ''
            # Validate Russian name (2-3 words, starts with capital)
            if match:
                parts = match.split()
                if 2 <= len(parts) <= 3:
                    # Check all parts start with capital
                    if all(p and p[0].isupper() and p[0].isalpha() for p in parts):
                        authors.append(match.strip())
    
    # Deduplicate
    seen = set()
    unique_authors = []
    for author in authors:
        author_lower = author.lower()
        if author_lower not in seen:
            seen.add(author_lower)
            unique_authors.append(author)
    
    return unique_authors

def extract_books_from_text(text: str) -> Dict[str, List[str]]:
    """Extract books and authors from a single text."""
    if not text or not isinstance(text, str):
        return {'titles': [], 'authors': []}
    
    # Extract quoted titles (most reliable)
    quoted_titles = extract_quoted_titles(text)
    
    # Extract pattern-based titles
    pattern_titles = extract_pattern_titles(text)
    
    # Combine and deduplicate titles
    all_titles = quoted_titles + pattern_titles
    seen = set()
    unique_titles = []
    for title in all_titles:
        title_lower = title.lower()
        if title_lower not in seen:
            seen.add(title_lower)
            unique_titles.append(title)
    
    # Extract authors
    authors = extract_authors(text)
    
    return {
        'titles': unique_titles,
        'authors': authors
    }

def process_posts(df: pd.DataFrame) -> pd.DataFrame:
    """Process all posts and extract books."""
    results = []
    
    print(f"📚 Processing {len(df)} posts...")
    
    for idx, row in df.iterrows():
        text = str(row.get('text', ''))
        
        if not text or len(text) < 10:
            continue
        
        extracted = extract_books_from_text(text)
        
        if extracted['titles'] or extracted['authors']:
            for title in extracted['titles']:
                results.append({
                    'message_id': row.get('message_id'),
                    'date_utc': row.get('date_utc'),
                    'book_title': title,
                    'author': None,
                    'source_text': text[:200],  # Preview
                })
            
            for author in extracted['authors']:
                # If we have titles but no author for them, create entries
                if extracted['titles']:
                    for title in extracted['titles']:
                        results.append({
                            'message_id': row.get('message_id'),
                            'date_utc': row.get('date_utc'),
                            'book_title': title,
                            'author': author,
                            'source_text': text[:200],
                        })
                else:
                    # Author mentioned but no title found
                    results.append({
                        'message_id': row.get('message_id'),
                        'date_utc': row.get('date_utc'),
                        'book_title': None,
                        'author': author,
                        'source_text': text[:200],
                    })
        
        if (idx + 1) % 100 == 0:
            print(f"   Processed {idx + 1}/{len(df)} posts, found {len(results)} book mentions...")
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Extract books and authors from posts")
    parser.add_argument("--input", default="data/raw/tolk_posts.csv", help="Input CSV file")
    parser.add_argument("--output", default="data/processed/books.csv", help="Output CSV file")
    parser.add_argument("--json", action="store_true", help="Also save as JSON")
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("   Run 'python scripts/fetch_tolk_posts.py' first")
        sys.exit(1)
    
    print(f"📥 Loading posts from {input_file}...")
    df = pd.read_csv(input_file, encoding='utf-8')
    print(f"   Loaded {len(df):,} posts\n")
    
    # Process posts
    books_df = process_posts(df)
    
    if len(books_df) == 0:
        print("\n⚠️  No books found in posts")
        sys.exit(0)
    
    # Save CSV
    print(f"\n💾 Saving to {output_file}...")
    books_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✅ Saved {len(books_df):,} book mentions")
    
    # Statistics
    unique_titles = books_df['book_title'].dropna().nunique()
    unique_authors = books_df['author'].dropna().nunique()
    posts_with_books = books_df['message_id'].nunique()
    
    print(f"\n📊 Statistics:")
    print(f"   Unique book titles: {unique_titles}")
    print(f"   Unique authors: {unique_authors}")
    print(f"   Posts mentioning books: {posts_with_books}")
    
    # Save JSON if requested
    if args.json:
        json_file = output_file.with_suffix('.json')
        books_df.to_json(json_file, orient='records', force_ascii=False, indent=2)
        print(f"   Also saved: {json_file}")
    
    # Show sample
    print("\n📖 Sample books found:")
    for _, row in books_df.head(5).iterrows():
        title = row['book_title'] or '(no title)'
        author = row['author'] or '(no author)'
        print(f"   • {title} — {author}")

if __name__ == "__main__":
    main()


