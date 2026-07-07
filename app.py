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
    negation_found = False
    
    for i, word in enumerate(words):
        # Check negation
        if word in ['tidak', 'ga', 'gak', 'kurang', 'bukan', 'belum']:
            negation_found = True
            if i + 1 < len(words):
                next_word = words[i + 1]
                # Check if next word is positive -> makes it negative
                for pw in POSITIVE_WORDS:
                    if pw in next_word:
                        neg_score += 3
                        break
                # Check if next word is negative -> makes it positive
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
        
        # Ensure tags length matches words length
        if len(tags) != len(words):
            return words, ["O"] * len(words)
            
        return words, tags
        
    except Exception as e:
        # Fallback: return words with O tags
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
    
    # Check for food-related keywords
    food_keywords = ['makanan', 'makan', 'rasa', 'enak', 'lezat', 'hambar', 'basi']
    if any(word in review_lower for word in food_keywords):
        aspects.append({
            "aspect": "FOOD",
            "context": review
        })
    
    # Check for service-related keywords
    service_keywords = ['pelayanan', 'service', 'ramah', 'cepat', 'lambat', 'staff']
    if any(word in review_lower for word in service_keywords):
        aspects.append({
            "aspect": "SERVICE",
            "context": review
        })
    
    # Check for ambience-related keywords
    ambience_keywords = ['tempat', 'suasana', 'nyaman', 'bersih', 'kotor', 'dekorasi']
    if any(word in review_lower for word in ambience_keywords):
        aspects.append({
            "aspect": "AMBIENCE",
            "context": review
        })
    
    # Check for price-related keywords
    price_keywords = ['harga', 'mahal', 'murah', 'worth', 'terjangkau']
    if any(word in review_lower for word in price_keywords):
        aspects.append({
            "aspect": "PRICE",
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
    try:
        words, tags = predict_ner(review)
        
        # If no words, return empty dataframe
        if not words:
            return pd.DataFrame()
        
        # Try to extract aspects from NER
        aspects = extract_aspects(words, tags)
        
        # If NER fails to extract aspects, use fallback
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

def display_ner_tokens(words, tags):
    """Display NER tokens with colors - safe version"""
    try:
        # Validate inputs
        if not words:
            return '<div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; border: 1px solid #e0e0e0; text-align: center; color: #666;">Tidak ada kata untuk divisualisasikan</div>'
        
        if not tags:
            return '<div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; border: 1px solid #e0e0e0; text-align: center; color: #666;">Tidak ada tag NER yang tersedia</div>'
        
        if len(words) != len(tags):
            return f'<div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; border: 1px solid #e0e0e0; text-align: center; color: #666;">Jumlah kata ({len(words)}) tidak sama dengan jumlah tag ({len(tags)})</div>'
        
        tag_colors = {
            'B-FOOD': '#fce4ec', 'I-FOOD': '#f8bbd0',
            'B-SERVICE': '#e3f2fd', 'I-SERVICE': '#bbdefb',
            'B-AMBIENCE': '#e8f5e9', 'I-AMBIENCE': '#c8e6c9',
            'B-PRICE': '#fff3e0', 'I-PRICE': '#ffe0b2',
            'B-MISCELLANEOUS': '#f3e5f5', 'I-MISCELLANEOUS': '#e1bee7',
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
    except Exception as e:
        return f'<div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; border: 1px solid #e0e0e0; text-align: center; color: #666;">Error visualizing NER tokens: {str(e)}</div>'

# ===============================
# UI - Main
# ===============================

st.title("🍽️ Aspect Based Sentiment Analysis")
st.markdown("Analisis Sentimen Review Restoran Berbasis Aspek")
st.divider()

# Sidebar
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
        st.session_state['contoh'] = contoh

# Main content
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

# Results
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
                
                # Create a copy for display
                display_df = hasil.copy()
                display_df['Sentiment'] = display_df['Sentiment'].apply(
                    lambda x: f"{get_sentiment_emoji(x)} {x}"
                )
                
                # Apply styling using map
                styled_df = display_df.style.map(
                    color_sentiment, 
                    subset=['Sentiment']
                )
                
                # Display styled dataframe
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
                
                # NER Visualization - FIXED
                st.divider()
                st.subheader("🏷️ Named Entity Recognition (NER)")
                
                words, tags = predict_ner(review)
                
                # Display NER tokens using safe function
                ner_html = display_ner_tokens(words, tags)
                st.markdown(ner_html, unsafe_allow_html=True)
                
                # Show legend only if we have valid tokens
                if words and tags and len(words) == len(tags):
                    st.caption("🎨 **Legend:** B-FOOD/I-FOOD 🍔 | B-SERVICE/I-SERVICE 🛎️ | B-AMBIENCE/I-AMBIENCE 🏠 | B-PRICE/I-PRICE 💰 | B-MISCELLANEOUS/I-MISCELLANEOUS 📌")
                
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