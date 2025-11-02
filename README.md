# Tolk RAG1 - Book Extraction from Telegram Channel

Extract book names and authors from posts in the Telegram channel [@tolk_tolk](https://t.me/tolk_tolk).

## Features

- Fetch all posts from `@tolk_tolk` channel
- Extract book titles (русские книги)
- Extract author names
- Export to structured CSV/JSON format

## Setup

1. Create virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Telegram API (same credentials as other projects):
- API ID: 25604239
- API Hash: be0d223a25ba43ac03f64d658b577d2c

## Usage

### 1. Fetch Posts
```bash
python scripts/fetch_tolk_posts.py
```

### 2a. Basic Extraction
```bash
python scripts/extract_books.py \
  --input data/raw/tolk_posts.csv \
  --output data/processed/books.csv
```

### 2b. Advanced Extraction (Natasha + author dictionary)
```bash
python scripts/extract_books_advanced.py \
  --input data/raw/tolk_posts.csv \
  --output data/processed/books_advanced.csv \
  --authors data/aux/authors_politics_economics.txt \
  --json
```

To regenerate the author dictionary (~2k names from Wikipedia):
```bash
python scripts/build_author_dictionary.py \
  --output data/aux/authors_politics_economics.txt \
  --target-count 2000
```

## Project Structure

```
tolk_rag1/
├── scripts/
│   ├── fetch_tolk_posts.py          # Fetch posts from @tolk_tolk
│   ├── extract_books.py             # Basic extraction
│   └── extract_books_advanced.py    # Advanced extraction (Natasha + heuristics)
├── data/
│   ├── raw/                         # Raw posts CSV
│   ├── processed/                   # Extracted books CSV/JSON
│   └── aux/                         # Supporting resources (author dictionaries, etc.)
├── scripts/build_author_dictionary.py # Wikipedia-based author harvesting
└── models/                          # NLP models (if needed)
```


