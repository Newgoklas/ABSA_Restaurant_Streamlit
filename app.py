import streamlit as st
import pandas as pd
import joblib
import re
from datetime import datetime
import plotly.graph_objects as go

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
        ner_model = joblib.load("ner_model.pkl")
        return absa_model, ner_model
    except Exception as e:
        return None, None

absa_model, ner_model = load_models()

if absa_model is None or ner_model is None:
    st.warning("⚠️ Model tidak ditemukan. Gunakan rule-based sentiment analysis.")
else:
    st.success("✅ Model berhasil dimuat!")

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
        'tidak cepat', 'ga cepat', 'tidak puas', 'ga puas',
        'tidak enak', 'kurang enak', 'tidak enak'
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

def predict_sentiment_hybrid(context):
    """Hybrid prediction: model + rule-based"""
    try:
        if absa_model is not None:
            try:
                pred = absa_model.predict([context])[0]
                if pred == 'Neutral':
                    rule_pred = get_sentiment_rule_based(context)
                    if rule_pred != 'Neutral':
                        return rule_pred
                return pred
            except:
                return get_sentiment_rule_based(context)
        else:
            return get_sentiment_rule_based(context)
    except:
        return get_sentiment_rule_based(context)

# ===============================
# Preprocessing
# ===============================

def preprocess(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    if not text:
        return []
    return text.split()

# ===============================
# Feature Extraction CRF
# ===============================

def word2features(sent, i):
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
    except:
        return {"bias": 1.0}

def sentence2features(words):
    try:
        sent = [(w, "O") for w in words]
        return [word2features(sent, i) for i in range(len(sent))]
    except:
        return []

# ===============================
# Predict Functions
# ===============================

def predict_ner(review):
    """Predict NER with safe fallback"""
    try:
        review = preprocess(review)
        words = tokenize(review)
        
        if not words:
            return [], []
        
        if ner_model is None:
            return words, ["O"] * len(words)
        
        features = sentence2features(words)
        if not features:
            return words, ["O"] * len(words)
        
        tags = ner_model.predict([features])[0]
        
        if not isinstance(tags, list):
            tags = list(tags)
        
        if len(tags) != len(words):
            return words, ["O"] * len(words)
            
        return words, tags
        
    except Exception as e:
        words = tokenize(preprocess(review))
        return words, ["O"] * len(words) if words else [], []

def extract_aspects(words, tags):
    aspects = []
    current = []
    label = None
    
    try:
        if not words or not tags or len(words) != len(tags):
            return []
            
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
    except:
        return []

def extract_aspects_fallback(review):
    """Extract aspects using simple rules if NER fails"""
    review_lower = review.lower()
    aspects = []
    
    food_keywords = ['makanan', 'makan', 'rasa', 'enak', 'lezat', 'hambar', 'basi']
    if any(word in review_lower for word in food_keywords):
        aspects.append({
            "aspect": "FOOD",
            "context": review
        })
    
    service_keywords = ['pelayanan', 'service', 'ramah', 'cepat', 'lambat', 'staff']
    if any(word in review_lower for word in service_keywords):
        aspects.append({
            "aspect": "SERVICE",
            "context": review
        })
    
    ambience_keywords = ['tempat', 'suasana', 'nyaman', 'bersih', 'kotor', 'dekorasi']
    if any(word in review_lower for word in ambience_keywords):
        aspects.append({
            "aspect": "AMBIENCE",
            "context": review
        })
    
    price_keywords = ['harga', 'mahal', 'murah', 'worth', 'terjangkau']
    if any(word in review_lower for word in price_keywords):
        aspects.append({
            "aspect": "PRICE",
            "context": review
        })
    
    if not aspects:
        aspects.append({
            "aspect": "MISCELLANEOUS",
            "context": review
        })
    
    return aspects

def predict_review(review):
    try:
        words, tags = predict_ner(review)
        
        if not words:
            return pd.DataFrame()
        
        aspects = extract_aspects(words, tags)
        
        if not aspects:
            aspects = extract_aspects_fallback(review)
        
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
# UI Functions - Enhanced NER Display
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

def display_ner_tokens_enhanced(words, tags):
    """
    Enhanced NER visualization with better styling
    """
    try:
        if not words:
            return """
            <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; border: 1px solid #e0e0e0; text-align: center; color: #666;">
                Tidak ada kata untuk divisualisasikan
            </div>
            """
        
        if not tags:
            return """
            <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; border: 1px solid #e0e0e0; text-align: center; color: #666;">
                Tidak ada tag NER yang tersedia
            </div>
            """
        
        if not isinstance(tags, list):
            tags = list(tags)
        
        if len(words) != len(tags):
            return f"""
            <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; border: 1px solid #e0e0e0; text-align: center; color: #666;">
                Jumlah kata ({len(words)}) tidak sama dengan jumlah tag ({len(tags)})
            </div>
            """
        
        # Tag color mapping with better colors
        tag_config = {
            'B-FOOD': {'color': '#fce4ec', 'border': '#e57373', 'label': 'Food'},
            'I-FOOD': {'color': '#f8bbd0', 'border': '#f06292', 'label': 'Food'},
            'B-SERVICE': {'color': '#e3f2fd', 'border': '#64b5f6', 'label': 'Service'},
            'I-SERVICE': {'color': '#bbdefb', 'border': '#42a5f5', 'label': 'Service'},
            'B-AMBIENCE': {'color': '#e8f5e9', 'border': '#81c784', 'label': 'Ambience'},
            'I-AMBIENCE': {'color': '#c8e6c9', 'border': '#66bb6a', 'label': 'Ambience'},
            'B-PRICE': {'color': '#fff3e0', 'border': '#ffb74d', 'label': 'Price'},
            'I-PRICE': {'color': '#ffe0b2', 'border': '#ffa726', 'label': 'Price'},
            'B-MISCELLANEOUS': {'color': '#f3e5f5', 'border': '#ce93d8', 'label': 'Misc'},
            'I-MISCELLANEOUS': {'color': '#e1bee7', 'border': '#ab47bc', 'label': 'Misc'},
            'O': {'color': '#f5f5f5', 'border': '#bdbdbd', 'label': 'Other'}
        }
        
        token_html = """
        <div style="
            display: flex; 
            flex-wrap: wrap; 
            gap: 0.5rem; 
            padding: 1.5rem; 
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px; 
            border: 2px solid #e0e0e0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
        """
        
        for word, tag in zip(words, tags):
            config = tag_config.get(tag, tag_config['O'])
            tag_display = config['label'] if tag != 'O' else ''
            
            token_html += f"""
            <span style="
                background: {config['color']};
                padding: 0.4rem 0.8rem;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 500;
                border: 2px solid {config['border']};
                display: inline-block;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                transition: all 0.3s ease;
            ">
                {word}
                <span style="
                    font-size: 0.65rem;
                    color: #666;
                    margin-left: 0.4rem;
                    background: rgba(255,255,255,0.7);
                    padding: 0.1rem 0.4rem;
                    border-radius: 10px;
                    font-weight: 600;
                ">{tag_display}</span>
            </span>
            """
        token_html += '</div>'
        return token_html
        
    except Exception as e:
        return f"""
        <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; border: 1px solid #e0e0e0; text-align: center; color: #666;">
            Error visualizing NER tokens: {str(e)}
        </div>
        """

def display_ner_legend():
    """Display legend for NER tags"""
    legend_html = """
    <div style="
        display: flex; 
        flex-wrap: wrap; 
        gap: 0.5rem; 
        padding: 0.5rem;
        background: white;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-top: 0.5rem;
    ">
        <span style="background: #fce4ec; padding: 0.2rem 0.6rem; border-radius: 4px; border: 2px solid #e57373; font-size: 0.8rem;">🍔 Food</span>
        <span style="background: #e3f2fd; padding: 0.2rem 0.6rem; border-radius: 4px; border: 2px solid #64b5f6; font-size: 0.8rem;">🛎️ Service</span>
        <span style="background: #e8f5e9; padding: 0.2rem 0.6rem; border-radius: 4px; border: 2px solid #81c784; font-size: 0.8rem;">🏠 Ambience</span>
        <span style="background: #fff3e0; padding: 0.2rem 0.6rem; border-radius: 4px; border: 2px solid #ffb74d; font-size: 0.8rem;">💰 Price</span>
        <span style="background: #f3e5f5; padding: 0.2rem 0.6rem; border-radius: 4px; border: 2px solid #ce93d8; font-size: 0.8rem;">📌 Misc</span>
        <span style="background: #f5f5f5; padding: 0.2rem 0.6rem; border-radius: 4px; border: 2px solid #bdbdbd; font-size: 0.8rem;">⚪ Other</span>
    </div>
    """
    return legend_html

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
    Aplikasi ini menggunakan **Named Entity Recognition (NER)** 
    dan **Aspect Based Sentiment Analysis (ABSA)** untuk 
    menganalisis sentimen review restoran.
    
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
                
                # Results table
                st.subheader("📋 Hasil Analisis per Aspek")
                
                display_df = hasil.copy()
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
                
                # Enhanced NER Visualization
                st.divider()
                st.subheader("🏷️ Named Entity Recognition (NER)")
                
                words, tags = predict_ner(review)
                
                # Display enhanced NER tokens
                ner_html = display_ner_tokens_enhanced(words, tags)
                st.markdown(ner_html, unsafe_allow_html=True)
                
                # Display legend
                if words and tags and len(words) == len(tags):
                    st.markdown(display_ner_legend(), unsafe_allow_html=True)
                
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
Dibangun dengan ❤️ menggunakan **Streamlit** · **CRF** untuk NER · **Multinomial Naive Bayes** untuk ABSA · **Hybrid Rule-Based** untuk akurasi lebih baik
""")