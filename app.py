import streamlit as st
import pandas as pd
import joblib
import re
from datetime import datetime

# ===============================
# Konfigurasi Halaman
# ===============================

st.set_page_config(
    page_title="ABSA Restaurant Analyzer",
    page_icon="🍽️",
    layout="wide"
)

# ===============================
# Load Model
# ===============================

@st.cache_resource
def load_models():
    absa_model = joblib.load("absa_model.pkl")
    ner_model = joblib.load("ner_model.pkl")
    return absa_model, ner_model

try:
    absa_model, ner_model = load_models()
    st.success("✅ Model berhasil dimuat!")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

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
        "bias": 1.0,
        "word.lower": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word[:2]": word[:2],
        "word[:3]": word[:3],
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.length": len(word)
    }
    if i > 0:
        prev = str(sent[i-1][0])
        features["-1:word.lower"] = prev.lower()
    else:
        features["BOS"] = True
    if i < len(sent)-1:
        nxt = str(sent[i+1][0])
        features["+1:word.lower"] = nxt.lower()
    else:
        features["EOS"] = True
    return features

def sentence2features(words):
    sent = [(w, "O") for w in words]
    return [word2features(sent, i) for i in range(len(sent))]

# ===============================
# Predict NER
# ===============================

def predict_ner(review):
    review = preprocess(review)
    words = tokenize(review)
    features = sentence2features(words)
    tags = ner_model.predict([features])[0]
    return words, tags

# ===============================
# Extract Aspect
# ===============================

def extract_aspects(words, tags):
    aspects = []
    current = []
    label = None
    for word, tag in zip(words, tags):
        if tag.startswith("B-"):
            if current:
                aspects.append({
                    "aspect": label,
                    "context": " ".join(current)
                })
            label = tag.replace("B-", "")
            current = [word]
        elif tag.startswith("I-"):
            current.append(word)
        else:
            if current:
                aspects.append({
                    "aspect": label,
                    "context": " ".join(current)
                })
                current = []
                label = None
    if current:
        aspects.append({
            "aspect": label,
            "context": " ".join(current)
        })
    return aspects

# ===============================
# Predict Review
# ===============================

def predict_review(review):
    words, tags = predict_ner(review)
    aspects = extract_aspects(words, tags)
    hasil = []
    for item in aspects:
        sentiment = absa_model.predict([item["context"]])[0]
        hasil.append({
            "Aspect": item["aspect"],
            "Context": item["context"],
            "Sentiment": sentiment
        })
    return pd.DataFrame(hasil)

# ===============================
# UI - Header
# ===============================

st.title("🍽️ Aspect Based Sentiment Analysis")
st.markdown("Analisis Sentimen Review Restoran Berbasis Aspek")
st.divider()

# ===============================
# UI - Sidebar
# ===============================

with st.sidebar:
    st.header("📊 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini menggunakan **Named Entity Recognition (NER)** 
    dan **Aspect Based Sentiment Analysis (ABSA)** untuk 
    menganalisis sentimen review restoran.
    """)
    
    st.divider()
    
    st.subheader("🎯 Aspek yang Dianalisis")
    st.markdown("""
    - 🍔 **Food** - Makanan
    - 🛎️ **Service** - Pelayanan
    - 🏠 **Ambience** - Suasana
    - 💰 **Price** - Harga
    - 📌 **Miscellaneous** - Lainnya
    """)
    
    st.divider()
    
    st.subheader("📝 Sentimen")
    st.markdown("""
    - 🟢 **Positive** - Positif
    - 🔴 **Negative** - Negatif
    - ⚪ **Neutral** - Netral
    """)
    
    st.divider()
    
    st.subheader("💡 Tips")
    st.markdown("""
    Masukkan review restoran dalam bahasa Indonesia 
    untuk mendapatkan hasil analisis yang optimal.
    """)

# ===============================
# UI - Main Content
# ===============================

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("✍️ Masukkan Review")
    review = st.text_area(
        "Review Restoran",
        height=150,
        placeholder="Contoh: Makanan enak, pelayanan lambat, tempatnya nyaman...",
        label_visibility="collapsed"
    )

with col2:
    st.subheader("⚙️ Analisis")
    analyze_button = st.button("🔍 Analisis Sekarang", use_container_width=True)
    st.caption("Klik tombol untuk menganalisis review")
    
    # Contoh review button
    if st.button("📋 Contoh Review", use_container_width=True):
        st.session_state['contoh'] = "Makanan enak banget, pelayanannya ramah, tapi harganya mahal"

# Gunakan contoh dari session state
if 'contoh' in st.session_state and not review:
    review = st.session_state['contoh']

# ===============================
# UI - Results
# ===============================

if analyze_button:
    if review.strip() == "":
        st.warning("⚠️ Masukkan review terlebih dahulu!")
    else:
        with st.spinner("🔄 Menganalisis review..."):
            hasil = predict_review(review)
            
            # Show original review
            st.divider()
            st.subheader("📝 Review yang Dianalisis")
            st.info(f'"{review}"')
            
            # Show results
            if len(hasil) > 0:
                # Metrics row
                st.subheader("📊 Ringkasan Sentimen")
                
                sentiment_counts = hasil['Sentiment'].value_counts()
                total = len(hasil)
                
                col_pos, col_neg, col_neu, col_total = st.columns(4)
                
                pos_count = sentiment_counts.get('Positive', 0)
                neg_count = sentiment_counts.get('Negative', 0)
                neu_count = sentiment_counts.get('Neutral', 0)
                
                with col_pos:
                    st.metric("🟢 Positif", pos_count)
                with col_neg:
                    st.metric("🔴 Negatif", neg_count)
                with col_neu:
                    st.metric("⚪ Netral", neu_count)
                with col_total:
                    st.metric("📊 Total Aspek", total)
                
                # Display results as table
                st.subheader("📋 Hasil Analisis per Aspek")
                
                # Create dataframe with colored sentiment
                def color_sentiment(val):
                    if val.lower() == 'positive':
                        return 'background-color: #d4edda; color: #155724;'
                    elif val.lower() == 'negative':
                        return 'background-color: #f8d7da; color: #721c24;'
                    else:
                        return 'background-color: #e2e3e5; color: #383d41;'
                
                styled_df = hasil.style.applymap(color_sentiment, subset=['Sentiment'])
                st.dataframe(styled_df, use_container_width=True, height=300)
                
                # Display NER visualization
                st.divider()
                st.subheader("🏷️ Named Entity Recognition (NER)")
                
                words, tags = predict_ner(review)
                
                # Create a horizontal display of tokens with colored tags using columns
                tag_colors = {
                    'B-FOOD': '#fce4ec',
                    'I-FOOD': '#f8bbd0',
                    'B-SERVICE': '#e3f2fd',
                    'I-SERVICE': '#bbdefb',
                    'B-AMBIENCE': '#e8f5e9',
                    'I-AMBIENCE': '#c8e6c9',
                    'B-PRICE': '#fff3e0',
                    'I-PRICE': '#ffe0b2',
                    'B-MISCELLANEOUS': '#f3e5f5',
                    'I-MISCELLANEOUS': '#e1bee7',
                    'O': '#f5f5f5'
                }
                
                # Display tokens in a nice format
                token_html = '<div style="display: flex; flex-wrap: wrap; gap: 0.3rem; padding: 1rem; background: #f8f9fa; border-radius: 10px; border: 1px solid #e0e0e0;">'
                for word, tag in zip(words, tags):
                    color = tag_colors.get(tag, '#f5f5f5')
                    tag_display = tag if tag != 'O' else ''
                    
                    token_html += f'''
                    <span style="
                        background: {color};
                        padding: 0.2rem 0.5rem;
                        border-radius: 5px;
                        font-size: 0.9rem;
                        border: 1px solid #ddd;
                        display: inline-block;
                    ">
                        {word}
                        <span style="font-size: 0.6rem; color: #666; margin-left: 0.2rem;">{tag_display}</span>
                    </span>
                    '''
                token_html += '</div>'
                st.markdown(token_html, unsafe_allow_html=True)
                
                # Legend
                st.caption("🎨 **Legend:** Food 🍔 | Service 🛎️ | Ambience 🏠 | Price 💰 | Misc 📌")
                
                # Download results
                st.divider()
                csv = hasil.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil (CSV)",
                    data=csv,
                    file_name=f"absa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            else:
                st.info("ℹ️ Tidak ditemukan aspek yang dapat dianalisis dalam review.")

# ===============================
# UI - Footer
# ===============================

st.divider()
st.caption("""
Dibangun dengan ❤️ menggunakan **Streamlit** · **CRF** untuk NER · **Multinomial Naive Bayes** untuk ABSA
""")