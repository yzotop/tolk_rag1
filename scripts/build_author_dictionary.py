#!/usr/bin/env python3
"""
Build a list of politics/economics authors using the Wikipedia API.

Usage:
    python scripts/build_author_dictionary.py \
        --output data/aux/authors_politics_economics.txt \
        --target-count 2000

The script queries a curated set of Wikipedia categories (both English and
Russian) and collects page titles for living/deceased economists, political
writers, sociologists, historians, etc. Results are merged, deduplicated,
sorted, and written to the output file (one name per line).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, List, Set

import requests


WIKIPEDIA_API_URL = "https://{lang}.wikipedia.org/w/api.php"

USER_AGENT = (
    "TelegramBookExtractor/1.0 (+https://t.me/tolk_tolk) "
    f"requests/{requests.__version__}"
)

# Relevant categories to crawl. Each entry is a (language, category_title)
CATEGORY_SEEDS = [
    ("en", "Category:Political writers"),
    ("en", "Category:Political scientists"),
    ("en", "Category:Economists"),
    ("en", "Category:Economic historians"),
    ("en", "Category:Sociologists"),
    ("en", "Category:Political philosophers"),
    ("ru", "Категория:Политологи"),
    ("ru", "Категория:Политические писатели"),
    ("ru", "Категория:Экономисты"),
    ("ru", "Категория:Экономисты России"),
    ("ru", "Категория:Экономисты по алфавиту"),
    ("ru", "Категория:Социологи"),
    ("ru", "Категория:Политические философы"),
]


def fetch_category_members(lang: str, category: str, max_depth: int = 1) -> Set[str]:
    """Fetch page titles from a category, optionally exploring subcategories."""

    titles: Set[str] = set()
    visited: Set[str] = set()

    def _fetch(cat: str, depth: int) -> None:
        if (cat, depth) in visited:
            return
        visited.add((cat, depth))

        cmcontinue = None
        while True:
            params = {
                "action": "query",
                "format": "json",
                "list": "categorymembers",
                "cmtitle": cat,
                "cmlimit": "500",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue

            response = requests.get(
                WIKIPEDIA_API_URL.format(lang=lang),
                params=params,
                headers={"Userent": USER_AGENT},
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()

            for member in data.get("query", {}).get("categorymembers", []):
                title = member.get("title", "").strip()
                if not title:
                    continue
                if title.startswith("Category:") or title.startswith("Категория:"):
                    if depth < max_depth:
                        _fetch(title, depth + 1)
                else:
                    titles.add(title)

            cmcontinue = data.get("continue", {}).get("cmcontinue")
            if not cmcontinue:
                break
            time.sleep(0.5)  # be polite

    _fetch(category, depth=0)
    return titles


def normalize_title(title: str) -> str:
    return title.replace("_", " ").strip()


def collect_authors(target_count: int) -> List[str]:
    authors: Set[str] = set()

    for lang, category in CATEGORY_SEEDS:
        print(f"📚 Fetching {category} ({lang})")
        try:
            members = fetch_category_members(lang, category, max_depth=1)
        except Exception as exc:
            print(f"   ⚠️ Failed to fetch {category}: {exc}")
            continue

        normalized = {normalize_title(title) for title in members}
        print(f"   → {len(normalized)} names")
        authors.update(normalized)

        if len(authors) >= target_count:
            break

    print(f"\n✅ Collected {len(authors)} unique names")
    # Sort by language-agnostic alphabetical order
    return sorted(authors)


def write_output(authors: Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for author in authors:
            f.write(author + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build author dictionary from Wikipedia")
    parser.add_argument("--output", default="data/aux/authors_politics_economics.txt")
    parser.add_argument("--target-count", type=int, default=2000)
    args = parser.parse_args()

    output_path = Path(args.output)
    authors = collect_authors(args.target_count)

    if not authors:
        print("⚠️ No authors collected. Abort.")
        return

    write_output(authors[: args.target_count], output_path)
    print(f"\n💾 Saved {min(len(authors), args.target_count)} authors to {output_path}")


if __name__ == "__main__":
    main()


