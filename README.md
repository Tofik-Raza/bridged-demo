
# 🧠 Natural Language to Filtered Semantic Search with Pinecone

This project provides an API that enables **natural language querying** over a collection of news article metadata, using **Pinecone vector search** and **metadata filtering**.

It uses a public Google Sheet as a dataset source and supports intelligent filtering based on:
- Author names (e.g., “by Mary Poppins”)
- Tags or topics (e.g., “about Rohit Sharma”)
- Publication date (e.g., “last month”, “in 2024”)

---

## 🔧 Features

- 💬 Natural language query input
- 🧠 Semantic search with `sentence-transformers`
- 🧪 Metadata filtering (author, tags, date)
- 🌐 Article preview via web scraping
- ⚡ FastAPI backend
- 🌲 Pinecone vector index integration

---

## 🗂 Dataset

The data is sourced from a **public Google Sheet**, which contains metadata like:
- `title`, `author`, `publishedDate`, `pageURL`, `tags`

📌 **To change the dataset**:
> Simply **replace the CSV export link** in the `process_csv_data()` function and then hit the `/load` API to reload the index.

```python
csv_url = "https://docs.google.com/spreadsheets/d/.../export?format=csv"
```

---

## 🚀 Getting Started

### 1. Clone and install

```bash
git clone https://github.com/your-username/semantic-search-pinecone.git
cd semantic-search-pinecone
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### 2. Set your Pinecone API key

Edit the `PINECONE_API_KEY` in `app.py`:

```python
PINECONE_API_KEY = "your-pinecone-key-here"
```

---

### 3. Run the app

```bash
uvicorn app:app --reload
```

---

## 📦 API Endpoints

### `GET /load`
→ Loads data from Google Sheet and pushes it to Pinecone

```bash
curl http://localhost:8000/load
```

### `POST /result`
→ Accepts a natural language query and returns semantically matched articles with metadata + scraped text.

#### Request Body:

```json
{
  "query": "Anything by Mary Poppins on Rohit Sharma?"
}
```

#### Response:

```json
{
  "query": "...",
  "filter": {
    "author": "mary poppins",
    "tags": { "$in": ["rohitsharma", "rohit", "sharma"] }
  },
  "results": [
    {
      "title": "...",
      "pageURL": "...",
      "author": "...",
      "tags": [...],
      "score": 0.93,
      "article_text": "First few paragraphs of the article..."
    }
  ]
}
```

---

## 🧱 Architecture

### Components:

- **Google Sheet CSV** — acts as the live public data source.
- **FastAPI backend** — handles data processing, embeddings, search, and filtering.
- **SentenceTransformers** — `all-MiniLM-L6-v2` model converts text to 384-dimensional embeddings.
- **Pinecone** — vector similarity search + filtering using indexed metadata.

### Flow:

```
Natural Query
    ↓
Extracted Filter (author/tags/date)
    ↓
Query Embedded → Vector
    ↓
Search Pinecone (top_k=5, filter=...)
    ↓
Article Preview Text Scraped
    ↓
Return Full Results
```

---
# Date Extraction:

This function intelligently extracts date-related filters from natural language by recognizing phrases like “last year,” “last week,” “yesterday,” or even specific formats like “15 June 2025.” It parses both relative and absolute time references to build an accurate published_day, published_month, and published_year filter—improving result relevance by covering nearly all possible user intents around dates.
---

---

## 🔍 Metadata Filter Extraction

| Intent            | Trigger Phrase / Example                        |
|------------------|--------------------------------------------------|
| Author           | "by Mary Poppins"                                |
| Tags             | "about Rohit Sharma", "on the topic of cricket"  |
| Date             | "last year", "last month", "in 2024"             |

---

## ⚠ Notes

- Only first 5000 characters of article text are fetched to keep responses light.
- Uses NLTK for tokenization and stopword removal.
- Duplicate tags are automatically removed during extraction.
- Test Cases and Result are shared in Tests.Json.

---

## 📄 Requirements

```
fastapi
uvicorn
pandas
nltk
re
bs4
requests
sentence-transformers
pinecone-client
```

Install with:

```bash
pip install -r requirements.txt
```

---



## 🧪 Testing Example

```bash
curl -X POST http://localhost:8000/result \
 -H "Content-Type: application/json" \
 -d '{"query": "Anything by Mary Poppins about Rohit Sharma"}'
```
