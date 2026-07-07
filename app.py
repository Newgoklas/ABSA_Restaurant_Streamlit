import streamlit as st
import pandas as pd
import joblib
import re
from datetime import datetime
import os

# ===============================
# Konfigurasi Halaman
# ===============================

st.set_page_config(
    page_title="ABSA Restaurant Analyzer",
    page_icon="🍽️",
    layout="wide"
)

# ===============================
# Cek dan Load Model dengan Error Handling
# ===============================

@st.cache_resource
def load_models():
    """Load model dengan error handling yang baik"""
    try:
        # Cek apakah file model ada
        if not os.path.exists("absa_model.pkl"):
            st.error("❌ File 'absa_model.pkl' tidak ditemukan!")
            return None, None
        
        if not os.path.exists("ner_model.pkl"):
            st.error("❌ File 'ner_model.pkl' tidak ditemukan!")
            return None, None
        
        # Load model
        absa_model = joblib.load("absa_model.pkl")
        ner_model = joblib.load("ner_model.pkl")
        
        return absa_model, ner_model
    
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None

# Load models
absa_model, ner_model = load_models()

if absa_model is None or ner_model is None:
    st.stop()

st.success("✅ Model berhasil dimuat!")

# ===============================
# Preprocessing Functions
# ===============================

def preprocess(text):
    """Preprocessing teks"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    """Tokenisasi teks"""
    if not text:
        return []
    return text.split()

# ===============================
# Feature Extraction CRF
# ===============================

def word2features(sent, i):
    """Ekstraksi fitur untuk CRF"""
    try:
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
    except Exception as e:
        return {"bias": 1.0, "error": str(e)}

def sentence2features(words):
    """Konversi kalimat ke fitur"""
    try:
        sent = [(w, "O") for w in words]
        return [word2features(sent, i) for i in range(len(sent))]
    except Exception as e:
        st.error(f"Error in sentence2features: {str(e)}")
        return []

# ===============================
# Predict Functions
# ===============================

def predict_ner(review):
    """Prediksi NER"""
    try:
        review = preprocess(review)
        words = tokenize(review)
        
        if not words:
            return [], []
            
        features = sentence2features(words)
        
        if not features:
            return words, ["O"] * len(words)
            
        tags = ner_model.predict([features])[0]
        return words, tags
        
    except Exception as e:
        st.error(f"Error in NER prediction: {str(e)}")
        return [], []

def extract_aspects(words, tags):
    """Ekstrak aspek dari hasil NER"""
    aspects = []
    current = []
    label = None
    
    try:
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
        
    except Exception as e:
        st.error(f"Error extracting aspects: {str(e)}")
        return []

def predict_review(review):
    """Prediksi sentimen per aspek"""
    try:
        words, tags = predict_ner(review)
        
        if not words:
            return pd.DataFrame()
            
        aspects = extract_aspects(words, tags)
        hasil = []
        
        for item in aspects:
            try:
                sentiment = absa_model.predict([item["context"]])[0]
                hasil.append({
                    "Aspect": item["aspect"] if item["aspect"] else "Miscellaneous",
                    "Context": item["context"],
                    "Sentiment": sentiment
                })
            except Exception as e:
                hasil.append({
                    "Aspect": item["aspect"] if item["aspect"] else "Miscellaneous",
                    "Context": item["context"],
                    "Sentiment": "Error"
                })
        
        return pd.DataFrame(hasil)
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return pd.DataFrame()

# ===============================
# UI Functions
# ===============================

def display_sentiment_badge(sentiment):
    """Tampilkan badge sentimen"""
    if sentiment.lower() == 'positive':
        return "🟢 Positive"
    elif sentiment.lower() == 'negative':
        return "🔴 Negative"
    else:
        return "⚪ Neutral"

def display_ner_tokens(words, tags):
    """Tampilkan hasil NER dengan warna"""
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
    return token_html

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
    
    st.divider()
    
    # Contoh review
    if st.button("📋 Contoh Review", use_container_width=True):
        st.session_state['contoh'] = "Makanan enak banget, pelayanannya ramah, tapi harganya mahal"

# ===============================
# UI - Main Content
# ===============================

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("✍️ Masukkan Review")
    
    default_text = st.session_state.get('contoh', '')
    
    review = st.text_area(
        "Review Restoran",
        value=default_text,
        height=150,
        placeholder="Contoh: Makanan enak, pelayanan lambat, tempatnya nyaman...",
        label_visibility="collapsed"
    )

with col2:
    st.subheader("⚙️ Analisis")
    analyze_button = st.button("🔍 Analisis Sekarang", use_container_width=True, type="primary")
    st.caption("Klik tombol untuk menganalisis review")

# ===============================
# UI - Results
# ===============================

if analyze_button:
    if not review or review.strip() == "":
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
                
                # Add sentiment badge column
                hasil['Sentiment Badge'] = hasil['Sentiment'].apply(display_sentiment_badge)
                
                # Display table
                st.dataframe(
                    hasil[['Aspect', 'Context', 'Sentiment Badge']],
                    use_container_width=True,
                    height=300,
                    column_config={
                        "Aspect": "Aspek",
                        "Context": "Konteks",
                        "Sentiment Badge": "Sentimen"
                    }
                )
                
                # Display NER visualization
                st.divider()
                st.subheader("🏷️ Named Entity Recognition (NER)")
                
                words, tags = predict_ner(review)
                
                if words and tags:
                    token_html = display_ner_tokens(words, tags)
                    st.markdown(token_html, unsafe_allow_html=True)
                    
                    # Legend
                    st.caption("🎨 **Legend:** B-FOOD/I-FOOD 🍔 | B-SERVICE/I-SERVICE 🛎️ | B-AMBIENCE/I-AMBIENCE 🏠 | B-PRICE/I-PRICE 💰 | B-MISCELLANEOUS/I-MISCELLANEOUS 📌")
                else:
                    st.info("Tidak ada token yang dapat divisualisasikan")
                
                # Download results
                st.divider()
                
                col_download1, col_download2, col_download3 = st.columns([1, 2, 1])
                with col_download2:
                    csv = hasil.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Hasil (CSV)",
                        data=csv,
                        file_name=f"absa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
            else:
                st.warning("⚠️ Tidak ditemukan aspek yang dapat dianalisis dalam review.")
                st.info("Pastikan review mengandung kata-kata yang terkait dengan makanan, pelayanan, suasana, atau harga.")

# ===============================
# UI - Footer
# ===============================

st.divider()
st.caption("""
Dibangun dengan ❤️ menggunakan **Streamlit** · **CRF** untuk NER · **Multinomial Naive Bayes** untuk ABSA
""")