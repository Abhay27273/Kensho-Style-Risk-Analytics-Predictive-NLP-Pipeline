import streamlit as st
import torch
from transformers import pipeline
import trafilatura
import requests
from bs4 import BeautifulSoup
from datetime import timedelta
import yfinance as yf

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

def get_article_text(url):
    # 1) Try trafilatura
    try:
        downloaded = trafilatura.fetch_url(url, headers=HEADERS)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text and len(text) > 200:
                return text
    except:
        pass

    # 2) Fallback: requests + BeautifulSoup
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        paras = soup.find_all("p")
        content = "\n\n".join(
            p.get_text(strip=True) for p in paras if len(p.get_text(strip=True)) > 50
        )
        if content:
            return content
    except:
        pass

    # 3) Fallback: newspaper3k
    try:
        art = Article(url, language="en")
        art.download()
        art.parse()
        if art.text and len(art.text) > 200:
            return art.text
    except:
        pass

    return None

@st.cache_resource
def load_pipelines():
    # Sentiment & NER
    sent_pipe = pipeline(
        "sentiment-analysis",
        model="yiyanghkust/finbert-tone",
        tokenizer="yiyanghkust/finbert-tone",
        device=0 if torch.cuda.is_available() else -1
    )
    ner_pipe = pipeline(
        "ner",
        model="abhay2727/Bert-NER-Finance",
        tokenizer="abhay2727/Bert-NER-Finance",
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )
    # Market-impact classifier (your fine-tuned model)
    market_model_id = "abhay2727/Final_finance_model"
    market_pipe = pipeline(
        "text-classification",
        model=market_model_id,
        tokenizer=market_model_id,
        device=0 if torch.cuda.is_available() else -1,
        return_all_scores=False
    )
    return sent_pipe, ner_pipe, market_pipe

sentiment_pipe, ner_pipe, market_pipe = load_pipelines()

st.title("🔍 Financial News Impact Predictor (v2)")

url_input   = st.text_input("Enter Article URL:")
symbol      = st.text_input("Market Ticker:", "^GSPC")
hours       = st.number_input("Window Hours:", min_value=1, max_value=168, value=24, step=1)
threshold   = st.number_input("Threshold (%):", min_value=0.1, max_value=10.0, value=0.5, step=0.1) / 100

if st.button("Fetch & Analyze"):
    raw = get_article_text(url_input)
    if not raw:
        st.error("Extraction failed—check URL or network.")
        st.stop()

    snippet = raw[:512]
    # 1. Sentiment
    sent = sentiment_pipe(snippet)[0]
    # 2. NER
    ents = ner_pipe(snippet)
    # 3. Market prediction
    pred = market_pipe(snippet)[0]

    # Display results
    st.subheader("🧠 Sentiment")
    st.write(f"{sent['label']} (conf: {sent['score']:.2f})")

    st.subheader("🏷️ Named Entities")
    for e in ents:
        st.write(f"- **{e['word']}**: {e['entity_group']}")

    st.subheader("📈 Market Impact Prediction")
    label = pred["label"]
    score = pred["score"]
    if label == "UP":
        st.success(f"Likely UP (score: {score:.2f})")
    elif label == "DOWN":
        st.error(f"Likely DOWN (score: {score:.2f})")
    else:
        st.info(f"Likely NEUTRAL (score: {score:.2f})")
