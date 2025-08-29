# app.py
import os
from functools import lru_cache
from typing import Dict, List, Tuple

import torch
import spacy
import requests
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download, HfFileSystem
from wikidata.client import Client
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fake News Classifier with Knowledge Graph", layout="wide")
st.title("📰 Fake News Detector with Knowledge Graph Insight")

# ── Config ───────────────────────────────────────────────────────────────────
REPO_ID = "FastBuilder1000/fake-news-model"
FILENAME = "best_model.pth"   # the file you pushed to the repo
MODEL_LOCAL_PATH = FILENAME   # keep in CWD
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)

# polite UA for public APIs (helps with throttling)
USER_AGENT = "FakeNewsDetector/1.0 (+https://huggingface.co) StreamlitApp"

# Wikidata image property
WD_PROP_IMAGE = "P18"

# ── HTTP session with retries ────────────────────────────────────────────────
def _make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.headers.update({"User-Agent": USER_AGENT})
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

@st.cache_resource
def get_session():
    return _make_session()

session = get_session()

# ── Download + load all resources (cached) ───────────────────────────────────
@st.cache_resource(show_spinner=True)
def download_and_load_everything():
    # 1) Download model weights via huggingface_hub (respects auth & caching)
    if not os.path.exists(MODEL_LOCAL_PATH):
        st.info("Downloading model from Hugging Face…")
        _ = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=".",
            local_dir_use_symlinks=False,
            token=HF_TOKEN,
            force_download=False,
            resume_download=True,
        )

    # 2) Tokenizer & model shell
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # We’ll instantiate a compatible model head and load weights robustly.
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )

    # 3) Load weights (works for state_dict or whole-model checkpts)
    loaded = torch.load(MODEL_LOCAL_PATH, map_location="cpu")
    if isinstance(loaded, dict):
        # likely a state_dict
        # Some training scripts save under keys like 'model_state_dict'
        state_dict = (
            loaded.get("state_dict")
            or loaded.get("model_state_dict")
            or loaded
        )
        model.load_state_dict(state_dict, strict=False)
    else:
        # Rare case: a whole model was torch.saved
        model = loaded

    model.eval()

    # 4) spaCy & Wikidata client
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # If the model isn't present in the environment for some reason
        from spacy.cli import download as spacy_download
        spacy_download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    client = Client()  # Wikidata client

    return tokenizer, model, nlp, client

tokenizer, model, nlp, client = download_and_load_everything()

# ── Prediction ───────────────────────────────────────────────────────────────
def predict_article(article: str) -> str:
    inputs = tokenizer(
        article,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        output = model(**inputs)
        logits = output.logits
        pred_id = torch.argmax(logits, dim=1).item()
        # NOTE: flip to match how your dataset labeled "real"/"fake" if needed
        return "Real" if pred_id == 1 else "Fake"

# ── Knowledge Graph helpers ─────────────────────────────────────────────────
def extract_named_entities(text: str) -> List[str]:
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE"}]
    # uniq while preserving order
    seen, out = set(), []
    for e in entities:
        if e not in seen:
            seen.add(e)
            out.append(e)
    # keep it manageable to avoid API storms
    return out[:8]

WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"

@lru_cache(maxsize=512)
def search_entity(name: str):
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "search": name,
        "language": "en",
        "limit": 1,
    }
    r = session.get(WIKIDATA_SEARCH_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("search"):
        return data["search"][0]["id"]
    return None

@lru_cache(maxsize=512)
def get_entity_info(qid: str) -> Dict:
    # Use wikidata client for label/description
    entity = client.get(qid, load=True)
    label = getattr(entity, "label", qid)
    description = getattr(entity, "description", "")

    # Try to fetch image filename from claims (P18)
    image_url = None
    try:
        claims = entity.data.get("claims", {})
        images = claims.get(WD_PROP_IMAGE, [])
        if images:
            # snak/datavalue/value content is the filename
            fname = images[0]["mainsnak"]["datavalue"]["value"]
            # Direct image path: faster & embeddable
            image_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{fname.replace(' ', '_')}?width=300"
    except Exception:
        pass

    return {"label": label, "description": description, "image": image_url}

@lru_cache(maxsize=2048)
def check_relationship(qid1: str, qid2: str) -> bool:
    # Simple heuristic: do the entities share any property keys?
    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": f"{qid1}|{qid2}",
        "props": "claims",
    }
    r = session.get(WIKIDATA_SEARCH_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    claims1 = data.get("entities", {}).get(qid1, {}).get("claims", {})
    claims2 = data.get("entities", {}).get(qid2, {}).get("claims", {})
    return bool(set(claims1.keys()) & set(claims2.keys()))

def analyze_article_entities(text: str) -> Tuple[Dict, List[Dict]]:
    entities = extract_named_entities(text)
    found_entities: Dict[str, Dict] = {}

    for ent_text in entities:
        try:
            qid = search_entity(ent_text)
            if qid:
                info = get_entity_info(qid)
                found_entities[ent_text] = {"qid": qid, **info}
        except Exception as e:
            st.debug(f"Wikidata lookup failed for {ent_text}: {e}")

    # Pairwise relationship checks (cap to avoid N^2 blowups)
    names = list(found_entities.keys())[:10]
    relations = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            e1, e2 = names[i], names[j]
            qid1, qid2 = found_entities[e1]["qid"], found_entities[e2]["qid"]
            try:
                related = check_relationship(qid1, qid2)
            except Exception:
                related = False
            relations.append({"entity_1": e1, "entity_2": e2, "related": related})

    return found_entities, relations

def generate_explanation(relations: List[Dict]) -> str:
    if not relations:
        return "No cross-entity relationships were found."
    lines = []
    for rel in relations:
        e1, e2, ok = rel["entity_1"], rel["entity_2"], rel["related"]
        if ok:
            lines.append(f"🔗 '{e1}' and '{e2}' share at least one Wikidata property.")
        else:
            lines.append(f"❌ No direct shared properties found for '{e1}' and '{e2}'.")
    return "\n".join(lines)

# ── UI ────────────────────────────────────────────────────────────────────────
article_input = st.text_area("📝 Paste a news article or statement here:", height=300)

if st.button("🔍 Analyze"):
    if not article_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing…"):
            try:
                prediction = predict_article(article_input)
            except Exception as e:
                st.error(f"Prediction failed. Check model weights/label mapping. Details: {e}")
                st.stop()

            entities_info, relations = analyze_article_entities(article_input)
            explanation = generate_explanation(relations)

        st.subheader("🧠 BERT Prediction")
        st.markdown(f"### This article is classified as **{prediction.upper()}**")

        st.subheader("🧾 Named Entities Found")
        if entities_info:
            for ent_name, ent_data in entities_info.items():
                desc = ent_data.get("description") or "—"
                st.markdown(f"**{ent_name}** — {desc}")
                if ent_data.get("image"):
                    st.image(ent_data["image"], caption=ent_name, use_container_width=False)
        else:
            st.info("No named entities found.")

        st.subheader("📘 Knowledge Graph Explanation")
        st.markdown(explanation)
