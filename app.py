from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import datetime
import re
import requests
from bs4 import BeautifulSoup
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import uuid
import spacy
nlp = spacy.load("en_core_web_md")

PINECONE_API_KEY = "pcsk_t6Cys_UkpYujWCKxaUr5DSuv7fTdQEanncAcQhvYHhWJZsP1wUk5R2JPWRnRHPaJ3cneX"
INDEX_NAME = "bridget-index"

pc = Pinecone(
    api_key=(PINECONE_API_KEY)
    )

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
    name=INDEX_NAME,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index(name=INDEX_NAME)

model = SentenceTransformer("all-MiniLM-L6-v2")
app = FastAPI()

def get_embedding(text):
    return model.encode(text).tolist()

def process_csv_data():
    csv_url = "https://docs.google.com/spreadsheets/d/1yky4n9AtCms7cniQ3CahdaaBOpt0gEWcl2VcJHMvMQ8/export?format=csv&gid=2005119392"
    df = pd.read_csv(csv_url)
    df['author'] = df['author'].apply(lambda x: x.strip().lower())
    df['publishedDate'] = pd.to_datetime(df['publishedDate'])
    df['published_year'] = df['publishedDate'].dt.year
    df['published_month'] = df['publishedDate'].dt.month
    df['published_day'] = df['publishedDate'].dt.day
    df['tags'] = df['tags'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    return df

def push_to_pinecone():
    df = process_csv_data()
    vectors = []
    for i, row in df.iterrows():
        text = row["title"] + " " + " ".join(row["tags"])
        embedding = get_embedding(text)
        metadata = {
            "pageURL": row["pageURL"],
            "title": row["title"],
            "author": row["author"],
            "published_year": row["published_year"],
            "published_month": row["published_month"],
            "published_day": row["published_day"],
            "tags": [re.sub(r'\W+', '', tag).lower() for tag in row["tags"]],
        }
        vectors.append((str(uuid.uuid4()), embedding, metadata))
    index.upsert(vectors)

def extract_date_info(text):

    now = datetime.datetime.now()
    if "last year" in text:
        return {"published_year": {"$eq": now.year - 1}}
    elif "last month" in text:
        if now.month > 1:
            return {"published_month": {"$eq": now.month - 1}}
        else:
            return {"published_month": {"$eq": 12}
                    , "published_year": {"$eq": now.year - 1}}
    elif "last week" in text:
        if now.day > 7:
            return {"published_day": {"$gte": now.day - 7}}
        elif now.month > 1:
            return {"published_day": {"$gte": now.day -7},
                    "published_month": {"$eq": now.month - 1}}
        else:
            return {"published_day": {"$gte": now.day - 7},
                    "published_month": {"$eq": 12},
                    "published_year": {"$eq": now.year - 1}}
    elif "last day" in text or "yesterday" in text:
        if now.day > 1:
            return {"published_day": {"$gte": now.day - 1}}
        elif now.month > 1:
            return {"published_day": {"$gte": now.day -1},
                    "published_month": {"$eq": now.month - 1}}
        else:
            return {"published_day": {"$gte": now.day - 1},
                    "published_month": {"$eq": 12},
                    "published_year": {"$eq": now.year - 1}}

    match = re.search(r"(?:in|from)?\s*(?:(\d{1,2})\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?", text, re.IGNORECASE)
    if match:
        day_str = match.group(1)
        month_str = match.group(2)
        year_str = match.group(3)

        month = datetime.datetime.strptime(month_str, "%B").month

        if day_str and year_str:
            return {
                "published_day": {"$eq": int(day_str)},
                "published_month": {"$eq": month},
                "published_year": {"$eq": int(year_str)}
            }

        elif day_str and not year_str:
            return {
                "published_day": {"$eq": int(day_str)},
                "published_month": {"$eq": month},
                "published_year": {"$eq": now.year}
            }

        elif not day_str and year_str:
            return {
                "published_month": {"$eq": month},
                "published_year": {"$eq": int(year_str)}
            }

        elif not day_str and not year_str:
            if month < now.month or (month == now.month and now.day >= 1):
                return {
                    "published_month": {"$eq": month},
                    "published_year": {"$eq": now.year}
                }
            else:
                return {
                    "published_month": {"$eq": month},
                    "published_year": {"$eq": now.year - 1}
                }

    match_year = re.search(r"(\d{4})", text)
    if match_year:
        return {"published_year": {"$eq": int(match_year.group(1))}}
    return {}

def extract_author(text):
    author_patterns = [
        r"by ([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", r"\bby (prof\.?|dr\.?) ([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", r"\bwritten by ([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)", r"\bauthored by ([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)",
        r"\bwork of ([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)",r"\bfrom ([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)",r"\barticle[s]* by ([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)",r"\bposted by ([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)",
        r"\bpublication[s]* from ([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)",r"\bresearcher ([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)",r"\bprof\.? ([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)",r"\bdr\.? ([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)",
        r"\bexpert ([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)",r"\bauthor[s]* (?:named|called)? ([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)"
    ]

    for pattern in author_patterns:
        match = re.search(pattern, text)
        if match:
            name = match.group(match.lastindex or 1)
            return {"author": name.strip().lower()}
    
    return {}

def extract_tags(text):
    tags = set()
    lowered = text.lower()

    topic_patterns = [
        r"\babout ([\w\s\-']+)", r"\bon ([\w\s\-']+)", r"\btagged with ([\w\s\-']+)", r"\brelated to ([\w\s\-']+)", r"\bregarding ([\w\s\-']+)",
        r"\bcovering ([\w\s\-']+)", r"\bfeaturing ([\w\s\-']+)", r"\bdiscussing ([\w\s\-']+)", r"\bfocused on ([\w\s\-']+)", r"\bdealing with ([\w\s\-']+)",
        r"\bmentioning ([\w\s\-']+)", r"\bconcerning ([\w\s\-']+)", r"\bwith topic[s]* (?:of|about|in|like|include)? ([\w\s\-']+)", r"\binvolving ([\w\s\-']+)", r"\breferring to ([\w\s\-']+)",
        r"\brelating to ([\w\s\-']+)", r"\bsubject of ([\w\s\-']+)", r"\bposts on ([\w\s\-',&]+)", r"\bdiscussions on ([\w\s\-',&]+)", r"\bresources on ([\w\s\-',&]+)",
        r"\bdata about ([\w\s\-',&]+)", r"\bexamples of ([\w\s\-',&]+)", r"\bexplaining ([\w\s\-',&]+)", r"\bcentered on ([\w\s\-',&]+)", r"\bstudy on ([\w\s\-',&]+)",
        r"\bnews about ([\w\s\-',&]+)", r"\bwork on ([\w\s\-',&]+)", r"\bin the field of ([\w\s\-',&]+)", r"\bpapers on ([\w\s\-',&]+)", r"\binfo on ([\w\s\-',&]+)",
        r"\bfrom the domain of ([\w\s\-',&]+)", r"\bdiscuss(?:es|ed|ing)? ([\w\s\-']+)", r"\btalking about ([\w\s\-']+)", r"\bexploring ([\w\s\-']+)", r"\baddressing ([\w\s\-']+)",
        r"\bexplain(?:s|ed|ing)? ([\w\s\-']+)", r"\bdescribing ([\w\s\-']+)", r"\bshow(?:s|ing)? ([\w\s\-']+)", r"\bmention(?:s|ed)? ([\w\s\-']+)"
    ]

    for pattern in topic_patterns:
        matches = re.findall(pattern, lowered)
        for match in matches:
            doc = nlp(match)
            topic_tokens = []

            for token in doc:
                if token.pos_ in ("VERB", "AUX") or token.is_stop:
                    break
                cleaned = re.sub(r"[^\w\-]", "", token.text.lower().strip())
                if 2 < len(cleaned) < 20:
                    topic_tokens.append(cleaned)

            if topic_tokens:
                tags.add("".join(topic_tokens))  # e.g. "deep learning" → "deeplearning"

    return {"tags": sorted(tags)} if tags else {"tags": []}

def build_filter(text):
    text = text.replace("‘", "'").replace("’", "'").replace(",", " ").replace(".", "")
    info = {}
    info.update(extract_author(text))
    info.update(extract_date_info(text.lower()))
    info.update(extract_tags(text))
    return info

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        article_div = soup.find("div", class_="article_content") or soup.find("div", class_="content-area")
        if not article_div:
            paragraphs = soup.find_all("p")
        else:
            paragraphs = article_div.find_all("p")
        return "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))[:5000]
    except Exception as e:
        return f"[Error fetching article: {str(e)}]"

class QueryInput(BaseModel):
    query: str

@app.get("/load")
def load_data():
    push_to_pinecone()
    return {"status": "Uploaded to Pinecone"}

@app.post("/result")
async def search_articles(data: QueryInput):
    filter_dict = build_filter(data.query)
    query_embedding = get_embedding(data.query)

    pinecone_filter = {}

    author = filter_dict.get("author")
    if author:
        pinecone_filter["author"] = author

    for key in ("published_year", "published_month", "published_day"):
        if key in filter_dict:
            pinecone_filter[key] = filter_dict[key]

    if "tags" in filter_dict and filter_dict["tags"]:
        pinecone_filter["tags"] = {"$in": filter_dict["tags"]}

    result = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True,
        filter=pinecone_filter
    )

    results = []
    for match in result.get("matches", []):
        metadata = match["metadata"]
        metadata["score"] = match["score"]
        metadata["article_text"] = extract_text_from_url(metadata["pageURL"])
        results.append(metadata)

    return {
        "query": data.query,
        "filter": pinecone_filter,
        "results": results
    }
