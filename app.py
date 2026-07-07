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
# Initialize Session State
# ===============================

if 'history' not in st.session_state:
    st.session_state.history = []

if 'contoh' not in st.session_state:
    st.session_state.contoh = ""

# ===============================
# Load Model
# ===============================

@st.cache_resource
def load_models():
    try:
        absa_model = joblib.load("absa_model.pkl")
        return absa_model
    except Exception as e:
        return None

absa_model = load_models()

if absa_model is None:
    st.warning("⚠️ Model ABSA tidak ditemukan. Gunakan rule-based sentiment analysis.")

# ===============================
# Rule-Based Sentiment
# ===============================

POSITIVE_WORDS = {
    'enak', 'lezat', 'nikmat', 'sedap', 'mantap', 'juara', 'recommended',
    'cepat', 'ramah', 'baik', 'memuaskan', 'profesional', 'luar biasa',
    'bersih', 'nyaman', 'cozy', 'tenang', 'sejuk', 'indah', 'asri',
    'murah', 'terjangkau', 'worth', 'sesuai', 'hemat',
    'bagus', 'sangat', 'sekali', 'banget', 'suka', 'senang',
    'oke', 'ok', 'good', 'great', 'awesome', 'perfect',
    'nyaman', 'segar', 'rapi', 'tertata', 'luas', 'terang'
}

NEGATIVE_WORDS = {
    'lambat', 'buruk', 'jelek', 'tidak', 'kurang', 'kecewa', 'parah',
    'menyebalkan', 'hambar', 'basi', 'tidak fresh',
    'mahal', 'terlalu', 'kotor', 'berisik', 'pengap',
    'tidak ramah', 'tidak profesional', 'tidak memuaskan',
    'rusak', 'pecah', 'bocor', 'hilang', 'salah',
    'aneh', 'payah', 'gagal', 'nyesal', 'menyesal',
    'tidak nyaman', 'ga nyaman', 'gak nyaman', 'sumpek', 'panas',
    'gelap', 'sempit', 'kumuh', 'bau', 'tidak bersih'
}

def get_sentiment_rule_based(context):
    """Rule-based sentiment classification"""
    context_lower = context.lower()
    
    pos_score = 0
    neg_score = 0
    
    # Check negative phrases first
    negative_phrases = [
        'tidak enak', 'ga enak', 'gak enak', 'tidak nyaman', 'ga nyaman', 'gak nyaman',
        'tidak bersih', 'ga bersih', 'tidak ramah', 'ga ramah',
        'tidak cepat', 'ga cepat', 'tidak puas', 'ga puas'
    ]
    
    for phrase in negative_phrases:
        if phrase in context_lower:
            neg_score += 5
    
    # Check positive phrases
    positive_phrases = [
        'sangat enak', 'enak sekali', 'enak banget', 'sangat nyaman',
        'sangat baik', 'sangat memuaskan', 'sangat cepat'
    ]
    
    for phrase in positive_phrases:
        if phrase in context_lower:
            pos_score += 5
    
    # Check individual words
    words = context_lower.split()
    
    for i, word in enumerate(words):
        # Check negation
        if word in ['tidak', 'ga', 'gak', 'kurang', 'bukan', 'belum']:
            if i + 1 < len(words):
                next_word = words[i + 1]
                for pw in POSITIVE_WORDS:
                    if pw in next_word:
                        neg_score += 3
                        break
                for nw in NEGATIVE_WORDS:
                    if nw in next_word:
                        pos_score += 3
                        break
            continue
        
        # Check positive words
        for pw in POSITIVE_WORDS:
            if pw in word:
                pos_score += 1
                break
        
        # Check negative words
        for nw in NEGATIVE_WORDS:
            if nw in word:
                neg_score += 1
                break
    
    # Special cases
    if 'nyaman' in context_lower and ('tidak' in context_lower or 'ga' in context_lower or 'gak' in context_lower):
        neg_score += 5
    
    if 'enak' in context_lower and ('tidak' in context_lower or 'ga' in context_lower or 'gak' in context_lower):
        neg_score += 5
    
    # Determine sentiment
    if pos_score > neg_score:
        return 'Positive'
    elif neg_score > pos_score:
        return 'Negative'
    else:
        return 'Neutral'

def predict_sentiment_hybrid(text):
    """Hybrid prediction: model + rule-based"""
    try:
        if absa_model is not None:
            try:
                pred = absa_model.predict([text])[0]
                if pred == 'Neutral':
                    rule_pred = get_sentiment_rule_based(text)
                    if rule_pred != 'Neutral':
                        return rule_pred
                return pred
            except:
                return get_sentiment_rule_based(text)
        else:
            return get_sentiment_rule_based(text)
    except:
        return get_sentiment_rule_based(text)

# ===============================
# Aspect Extraction (Simple Rule-Based)
# ===============================

def extract_aspects_simple(review):
    """Extract aspects using simple keyword matching"""
    review_lower = review.lower()
    aspects = []
    
    # Define aspect keywords
    aspect_keywords = {
        'FOOD': ['makanan', 'makan', 'rasa', 'enak', 'lezat', 'hambar', 'basi', 'menu', 'porsi'],
        'SERVICE': ['pelayanan', 'service', 'ramah', 'cepat', 'lambat', 'staff', 'pelayan'],
        'AMBIENCE': ['tempat', 'suasana', 'nyaman', 'bersih', 'kotor', 'dekorasi', 'ruangan'],
        'PRICE': ['harga', 'mahal', 'murah', 'worth', 'terjangkau', 'biaya']
    }
    
    # Check each aspect
    for aspect, keywords in aspect_keywords.items():
        if any(keyword in review_lower for keyword in keywords):
            aspects.append({
                "aspect": aspect,
                "context": review
            })
    
    # If no aspects found, use entire review as miscellaneous
    if not aspects:
        aspects.append({
            "aspect": "MISCELLANEOUS",
            "context": review
        })
    
    return aspects

def predict_review(review):
    """Predict sentiment for each aspect in review"""
    try:
        # Extract aspects using simple rules
        aspects = extract_aspects_simple(review)
        
        hasil = []
        for item in aspects:
            sentiment = predict_sentiment_hybrid(item["context"])
            hasil.append({
                "Aspect": item["aspect"] if item["aspect"] else "Miscellaneous",
                "Context": item["context"],
                "Sentiment": sentiment
            })
        
        return pd.DataFrame(hasil)
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return pd.DataFrame()

# ===============================
# UI Functions
# ===============================

def color_sentiment(val):
    if val is None:
        return ''
    val_str = str(val)
    if 'Positive' in val_str or 'positive' in val_str:
        return 'background-color: #d4edda; color: #155724; font-weight: bold;'
    elif 'Negative' in val_str or 'negative' in val_str:
        return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
    else:
        return 'background-color: #e2e3e5; color: #383d41; font-weight: bold;'

def get_sentiment_emoji(sentiment):
    if sentiment is None:
        return '⚪'
    sent_str = str(sentiment).lower()
    if 'positive' in sent_str:
        return '🟢'
    elif 'negative' in sent_str:
        return '🔴'
    else:
        return '⚪'

def get_aspect_emoji(aspect):
    """Get emoji for aspect"""
    aspect_emoji = {
        'FOOD': '🍔',
        'SERVICE': '🛎️',
        'AMBIENCE': '🏠',
        'PRICE': '💰',
        'MISCELLANEOUS': '📌'
    }
    return aspect_emoji.get(aspect.upper(), '📌')

# ===============================
# UI - Main
# ===============================

st.title("🍽️ Aspect Based Sentiment Analysis")
st.markdown("Analisis Sentimen Review Restoran Berbasis Aspek")
st.divider()

# ===============================
# Sidebar
# ===============================

with st.sidebar:
    st.header("📊 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini menggunakan **Aspect Based Sentiment Analysis (ABSA)** 
    untuk menganalisis sentimen review restoran.
    
    **Fitur Hybrid**: Model + Rule-Based untuk akurasi lebih baik.
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
    
    st.subheader("📌 Contoh Review")
    contoh = st.selectbox(
        "Pilih contoh review:",
        [
            "pelayanan lambat",
            "makanannya tidak enak",
            "tempatnya tidak nyaman",
            "makanannya enak",
            "makanan enak pelayanan ramah",
            "harga mahal tempat nyaman",
            "pelayanan cepat makanan enak"
        ]
    )
    if st.button("📋 Gunakan Contoh", use_container_width=True):
        st.session_state.contoh = contoh
    
    st.divider()
    
    # History Section
    st.subheader("📜 Riwayat Analisis")
    
    if st.session_state.history:
        st.caption(f"Total: {len(st.session_state.history)} analisis")
        
        if st.button("🗑️ Hapus Riwayat", use_container_width=True):
            st.session_state.history = []
            st.rerun()
        
        for i, entry in enumerate(reversed(st.session_state.history[-10:])):
            with st.expander(f"#{len(st.session_state.history) - i}: {entry['review'][:50]}..."):
                st.write(f"**Review:** {entry['review']}")
                st.write(f"**Waktu:** {entry['timestamp']}")
                st.write("**Hasil:**")
                st.dataframe(entry['result'], use_container_width=True, height=150)
    else:
        st.info("Belum ada riwayat analisis")

# ===============================
# Main Content
# ===============================

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("✍️ Masukkan Review")
    
    default_text = st.session_state.contoh
    
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
# Results
# ===============================

if analyze_button:
    if not review or review.strip() == "":
        st.warning("⚠️ Masukkan review terlebih dahulu!")
    else:
        with st.spinner("🔄 Menganalisis review..."):
            hasil = predict_review(review)
            
            st.divider()
            st.subheader("📝 Review yang Dianalisis")
            st.info(f'"{review}"')
            
            if len(hasil) > 0:
                # Metrics
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
                
                # Results table with aspect emojis
                st.subheader("📋 Hasil Analisis per Aspek")
                
                # Add emoji to aspect
                display_df = hasil.copy()
                display_df['Aspect'] = display_df['Aspect'].apply(
                    lambda x: f"{get_aspect_emoji(x)} {x}"
                )
                display_df['Sentiment'] = display_df['Sentiment'].apply(
                    lambda x: f"{get_sentiment_emoji(x)} {x}"
                )
                
                styled_df = display_df.style.map(
                    color_sentiment, 
                    subset=['Sentiment']
                )
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=300,
                    column_config={
                        "Aspect": "Aspek",
                        "Context": "Konteks",
                        "Sentiment": "Sentimen"
                    }
                )
                
                # Save to history
                history_entry = {
                    "review": review,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "result": hasil.copy()
                }
                st.session_state.history.append(history_entry)
                
                # Download
                st.divider()
                csv = hasil.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil (CSV)",
                    data=csv,
                    file_name=f"absa_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            else:
                st.warning("⚠️ Tidak ditemukan aspek yang dapat dianalisis dalam review.")
                st.info("💡 Pastikan review mengandung kata-kata seperti 'makanan', 'pelayanan', 'tempat', atau 'harga'.")

st.divider()
st.caption("""
Dibangun dengan ❤️ menggunakan **Streamlit** · **Multinomial Naive Bayes** untuk ABSA · **Hybrid Rule-Based** untuk akurasi lebih baik
""")