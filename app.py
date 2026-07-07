import streamlit as st
import pandas as pd
import joblib
import re

# ===============================
# Load Model
# ===============================

absa_model = joblib.load("absa_model.pkl")
ner_model = joblib.load("ner_model.pkl")

# ===============================
# Preprocessing
# ===============================

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return text.split()

# ===============================
# Feature Extraction CRF
# ===============================

def word2features(sent, i):

    word = str(sent[i][0])

    features = {
        "bias":1.0,
        "word.lower":word.lower(),
        "word[-3:]":word[-3:],
        "word[-2:]":word[-2:],
        "word[:2]":word[:2],
        "word[:3]":word[:3],
        "word.isupper":word.isupper(),
        "word.istitle":word.istitle(),
        "word.isdigit":word.isdigit(),
        "word.length":len(word)
    }

    if i>0:
        prev=str(sent[i-1][0])
        features["-1:word.lower"]=prev.lower()
    else:
        features["BOS"]=True

    if i<len(sent)-1:
        nxt=str(sent[i+1][0])
        features["+1:word.lower"]=nxt.lower()
    else:
        features["EOS"]=True

    return features


def sentence2features(words):

    sent=[(w,"O") for w in words]

    return [
        word2features(sent,i)
        for i in range(len(sent))
    ]


# ===============================
# Predict NER
# ===============================

def predict_ner(review):

    review=preprocess(review)

    words=tokenize(review)

    features=sentence2features(words)

    tags=ner_model.predict([features])[0]

    return words,tags


# ===============================
# Extract Aspect
# ===============================

def extract_aspects(words,tags):

    aspects=[]

    current=[]

    label=None

    for word,tag in zip(words,tags):

        if tag.startswith("B-"):

            if current:

                aspects.append({
                    "aspect":label,
                    "context":" ".join(current)
                })

            label=tag.replace("B-","")

            current=[word]

        elif tag.startswith("I-"):

            current.append(word)

        else:

            if current:

                aspects.append({
                    "aspect":label,
                    "context":" ".join(current)
                })

                current=[]

                label=None

    if current:

        aspects.append({
            "aspect":label,
            "context":" ".join(current)
        })

    return aspects


# ===============================
# Predict Review
# ===============================

def predict_review(review):

    words,tags=predict_ner(review)

    aspects=extract_aspects(words,tags)

    hasil=[]

    for item in aspects:

        sentiment=absa_model.predict([item["context"]])[0]

        hasil.append({

            "Aspect":item["aspect"],

            "Context":item["context"],

            "Sentiment":sentiment

        })

    return pd.DataFrame(hasil)


# ===============================
# UI
# ===============================

st.set_page_config(
    page_title="ABSA Restaurant",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ Aspect Based Sentiment Analysis")

st.write("Analisis Sentimen Review Restoran Bahasa Indonesia")

review=st.text_area(

    "Masukkan Review",

    height=200

)

if st.button("Analisis"):

    if review.strip()=="":

        st.warning("Masukkan review terlebih dahulu")

    else:

        hasil=predict_review(review)

        st.dataframe(hasil,use_container_width=True)