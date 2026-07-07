import streamlit as st
import pandas as pd
import joblib
import re
import matplotlib.pyplot as plt
from datetime import datetime

# ===============================
# Konfigurasi Halaman
# ===============================

st.set_page_config(
    page_title="ABSA Restaurant Analyzer",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# Custom CSS
# ===============================

st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Title styling */
    .main-title {
        font-size: 3rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    /* Subtitle styling */
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border: 1px solid #f0f0f0;
        margin-bottom: 1.5rem;
    }
    
    .card h3 {
        color: #333;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Sentiment badge styling */
    .badge-positive {
        background: #d4edda;
        color: #155724;
        padding: 0.25rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-negative {
        background: #f8d7da;
        color: #721c24;
        padding: 0.25rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-neutral {
        background: #e2e3e5;
        color: #383d41;
        padding: 0.25rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Info box */
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.5rem 2rem !important;
        border-radius: 25px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
        font-size: 1rem !important;
        padding: 1rem !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #999;
        font-size: 0.85rem;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }
    
    /* Aspect tags */
    .aspect-tag {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .aspect-FOOD { background: #fce4ec; color: #c62828; }
    .aspect-SERVICE { background: #e3f2fd; color: #0d47a1; }
    .aspect-AMBIENCE { background: #e8f5e9; color: #1b5e20; }
    .aspect-PRICE { background: #fff3e0; color: #e65100; }
    .aspect-MISCELLANEOUS { background: #f3e5f5; color: #4a148c; }
    
    /* Sentiment count boxes */
    .sentiment-box {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .sentiment-positive { background: #d4edda; border: 2px solid #28a745; }
    .sentiment-negative { background: #f8d7da; border: 2px solid #dc3545; }
    .sentiment-neutral { background: #e2e3e5; border: 2px solid #6c757d; }
    
    .sentiment-box .count {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .sentiment-box .label {
        font-size: 0.9rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

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

st.markdown('<div class="main-title">🍽️ Aspect Based Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Analisis Sentimen Review Restoran Berbasis Aspek</div>', unsafe_allow_html=True)

# ===============================
# UI - Sidebar
# ===============================

with st.sidebar:
    st.markdown("### 📊 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini menggunakan **Named Entity Recognition (NER)** 
    dan **Aspect Based Sentiment Analysis (ABSA)** untuk 
    menganalisis sentimen review restoran.
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Aspek yang Dianalisis")
    st.markdown("""
    - 🍔 **Food** - Makanan
    - 🛎️ **Service** - Pelayanan
    - 🏠 **Ambience** - Suasana
    - 💰 **Price** - Harga
    - 📌 **Miscellaneous** - Lainnya
    """)
    
    st.markdown("---")
    st.markdown("### 📝 Sentimen")
    st.markdown("""
    - 🟢 **Positive** - Positif
    - 🔴 **Negative** - Negatif
    - ⚪ **Neutral** - Netral
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    Masukkan review restoran dalam bahasa Indonesia 
    untuk mendapatkan hasil analisis yang optimal.
    """)
    
    st.markdown("---")
    st.markdown("### 📌 Contoh Review")
    contoh_review = """Makanan enak banget, pelayanannya ramah, tapi harganya mahal"""
    if st.button("📋 Gunakan Contoh", use_container_width=True):
        st.session_state['contoh'] = contoh_review

# ===============================
# UI - Main Content
# ===============================

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### ✍️ Masukkan Review")
    
    default_text = st.session_state.get('contoh', '')
    
    review = st.text_area(
        "Review Restoran",
        value=default_text,
        height=150,
        placeholder="Contoh: Makanan enak, pelayanan lambat, tempatnya nyaman...",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("### ⚙️ Analisis")
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button("🔍 Analisis Sekarang", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Klik tombol untuk menganalisis review")

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
            st.markdown("---")
            st.markdown("### 📝 Review yang Dianalisis")
            st.markdown(f'<div class="info-box">"{review}"</div>', unsafe_allow_html=True)
            
            # Show results
            if len(hasil) > 0:
                # Create two columns for results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### 📊 Hasil Analisis")
                    
                    # Display results as beautiful cards
                    for idx, row in hasil.iterrows():
                        aspect = row['Aspect']
                        context = row['Context']
                        sentiment = row['Sentiment']
                        
                        # Determine sentiment badge
                        if sentiment.lower() == 'positive':
                            badge = 'badge-positive'
                            emoji = '🟢'
                        elif sentiment.lower() == 'negative':
                            badge = 'badge-negative'
                            emoji = '🔴'
                        else:
                            badge = 'badge-neutral'
                            emoji = '⚪'
                        
                        # Aspect tag class
                        aspect_class = f"aspect-{aspect.upper()}" if aspect else "aspect-MISCELLANEOUS"
                        aspect_display = aspect if aspect else "Miscellaneous"
                        
                        st.markdown(f"""
                        <div class="card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span class="aspect-tag {aspect_class}">{aspect_display}</span>
                                    <span style="margin-left: 0.5rem; font-weight: 500;">"{context}"</span>
                                </div>
                                <div>
                                    <span class="{badge}">{emoji} {sentiment}</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 📈 Ringkasan Sentimen")
                    
                    # Sentiment count
                    sentiment_counts = hasil['Sentiment'].value_counts()
                    total = len(hasil)
                    
                    # Display sentiment counts as boxes
                    col_pos, col_neg, col_neu = st.columns(3)
                    
                    pos_count = sentiment_counts.get('Positive', 0)
                    neg_count = sentiment_counts.get('Negative', 0)
                    neu_count = sentiment_counts.get('Neutral', 0)
                    
                    with col_pos:
                        st.markdown(f"""
                        <div class="sentiment-box sentiment-positive">
                            <div class="count">{pos_count}</div>
                            <div class="label">🟢 Positive</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_neg:
                        st.markdown(f"""
                        <div class="sentiment-box sentiment-negative">
                            <div class="count">{neg_count}</div>
                            <div class="label">🔴 Negative</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_neu:
                        st.markdown(f"""
                        <div class="sentiment-box sentiment-neutral">
                            <div class="count">{neu_count}</div>
                            <div class="label">⚪ Neutral</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Total aspects
                    st.markdown(f"**Total Aspek**: {total}")
                    
                    # Create bar chart using matplotlib
                    if len(sentiment_counts) > 0:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        
                        colors = []
                        for s in sentiment_counts.index:
                            if s.lower() == 'positive':
                                colors.append('#28a745')
                            elif s.lower() == 'negative':
                                colors.append('#dc3545')
                            else:
                                colors.append('#6c757d')
                        
                        ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors, edgecolor='white', linewidth=2)
                        ax.set_ylabel('Jumlah')
                        ax.set_title('Distribusi Sentimen')
                        
                        # Add value labels on top of bars
                        for i, v in enumerate(sentiment_counts.values):
                            ax.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
                        
                        st.pyplot(fig)
                
                # Display NER visualization
                st.markdown("---")
                st.markdown("### 🏷️ Named Entity Recognition (NER)")
                
                words, tags = predict_ner(review)
                
                # Create a horizontal display of tokens with colored tags
                ner_html = '<div style="display: flex; flex-wrap: wrap; gap: 0.3rem; padding: 1rem; background: #f8f9fa; border-radius: 10px;">'
                for word, tag in zip(words, tags):
                    tag_color = {
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
                    }.get(tag, '#f5f5f5')
                    
                    tag_display = tag if tag != 'O' else ''
                    
                    ner_html += f'''
                    <span style="
                        background: {tag_color};
                        padding: 0.2rem 0.5rem;
                        border-radius: 5px;
                        font-size: 0.9rem;
                        border: 1px solid #e0e0e0;
                    ">
                        {word}
                        <span style="
                            font-size: 0.6rem;
                            color: #666;
                            margin-left: 0.2rem;
                        ">{tag_display}</span>
                    </span>
                    '''
                ner_html += '</div>'
                st.markdown(ner_html, unsafe_allow_html=True)
                
                # Download results
                st.markdown("---")
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

st.markdown("""
<div class="footer">
    <p>
        Dibangun dengan ❤️ menggunakan 
        <strong>Streamlit</strong> · 
        <strong>CRF</strong> untuk NER · 
        <strong>Multinomial Naive Bayes</strong> untuk ABSA
    </p>
    <p style="font-size: 0.75rem; color: #bbb;">
        © 2024 ABSA Restaurant Analyzer | All Rights Reserved
    </p>
</div>
""", unsafe_allow_html=True)