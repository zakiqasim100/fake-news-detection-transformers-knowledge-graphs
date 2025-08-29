
# app.py

import os


import torch
import spacy
import requests
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
from wikidata.client import Client

# SETUP PAGE 
st.set_page_config(page_title="Fake News Classifier with Knowledge Graph", layout="wide")
st.title("📰 Fake News Detector with Knowledge Graph Insight")

# CONFIG 
MODEL_PATH = "best_model.pth"
HUGGINGFACE_MODEL_URL = "https://huggingface.co/zakiqasim100/fake-news-model/resolve/main/best_model.pth"


# LOAD MODELS AND CLIENTS 
@st.cache_resource
def download_and_load_everything():
    # Download model if not already present
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Hugging Face...")
        with requests.get(HUGGINGFACE_MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    # tokenizer & model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)

    model.eval()

    # spaCy & Wikidata
    nlp = spacy.load("en_core_web_sm")
    client = Client()

    return tokenizer, model, nlp, client

tokenizer, model, nlp, client = download_and_load_everything()

# BERT PREDICTION FUNCTION 
def predict_article(article):
    inputs = tokenizer(article, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**inputs)
        logits = output.logits
        prediction = torch.argmax(logits, dim=1).item()
        return "Real" if prediction == 1 else "Fake"

#  KNOWLEDGE GRAPH FUNCTIONS 
def extract_named_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]
    return list(set([ent[0] for ent in entities]))

def search_entity(name):
    url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&search={name}&language=en"
    response = requests.get(url)
    data = response.json()
    if data['search']:
        return data['search'][0]['id']
    return None

def get_entity_info(qid):
    entity = client.get(qid, load=True)
    image_entity = entity.get('image')
    image_url = None
    if image_entity:
        image_url = 'https://commons.wikimedia.org/wiki/File:' + image_entity[0].replace(' ', '_')
    return {
        "label": entity.label,
        "description": entity.description,
        "image": image_url
    }

def check_relationship(qid1, qid2):
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&ids={qid1}|{qid2}&props=claims"
    response = requests.get(url)
    data = response.json()
    claims1 = data['entities'].get(qid1, {}).get('claims', {})
    claims2 = data['entities'].get(qid2, {}).get('claims', {})
    return bool(set(claims1.keys()) & set(claims2.keys()))

def analyze_article_entities(text):
    entities = extract_named_entities(text)
    found_entities = {}

    for ent_text in entities:
        qid = search_entity(ent_text)
        if qid:
            info = get_entity_info(qid)
            found_entities[ent_text] = {"qid": qid, **info}

    relations = []
    entity_names = list(found_entities.keys())
    for i in range(len(entity_names)):
        for j in range(i+1, len(entity_names)):
            e1, e2 = entity_names[i], entity_names[j]
            qid1, qid2 = found_entities[e1]["qid"], found_entities[e2]["qid"]
            are_related = check_relationship(qid1, qid2)
            relations.append({
                "entity_1": e1,
                "entity_2": e2,
                "related": are_related
            })

    return found_entities, relations

def generate_explanation(relations):
    explanations = []
    for rel in relations:
        e1, e2 = rel['entity_1'], rel['entity_2']
        if rel['related']:
            explanations.append(f"🔗 The entities '{e1}' and '{e2}' are related based on Wikidata.")
        else:
            explanations.append(f"❌ The entities '{e1}' and '{e2}' are not directly related according to Wikidata.")
    return "\n".join(explanations)

# STREAMLIT INTERFACE 
article_input = st.text_area("📝 Paste a news article or statement here:", height=300)

if st.button("🔍 Analyze"):
    if not article_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            prediction = predict_article(article_input)
            entities_info, relations = analyze_article_entities(article_input)
            explanation = generate_explanation(relations)

        st.subheader("🧠 BERT Prediction")
        st.markdown(f"### This article is classified as **{prediction.upper()}**")

        st.subheader("🧾 Named Entities Found")
        if entities_info:
            for ent_name, ent_data in entities_info.items():
                st.markdown(f"**{ent_name}** — {ent_data['description']}")
                if ent_data.get("image"):
                    st.markdown(f"[Image]({ent_data['image']})")
        else:
            st.info("No named entities found.")

        st.subheader("📘 Knowledge Graph Explanation")
        st.markdown(explanation)


that was app.py, also the requirements.txt is 
torch==2.2.0
transformers==4.51.1
spacy==3.7.2
requests==2.32.3
streamlit==1.44.1
wikidata==0.8.1

# force compatible numpy version
numpy<2

# spaCy model
en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz


