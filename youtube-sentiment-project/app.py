import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from googleapiclient.discovery import build
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
import html
import string

# --- FUNGSI PEMBERSIH ---
def clean_text(text):
    text = html.unescape(text)
    text = text.lower() 
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    text = re.sub(r'@\w+', '', text) 
    text = re.sub(r'#', '', text)
    text = re.sub(r'\d+', '', text)
    # text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- CONFIG & UI ---
st.set_page_config(page_title="IndoBERT Analyzer", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    .stTextInput label { color: white !important; }
    .main-title {
        color: white; font-size: 3rem; font-weight: 700; margin: 0;
        display: flex; align-items: center; gap: 20px;
    }
    div.stButton > button:first-child {
        background-color: #28A745; color: white; border-radius: 8px;
        border: none; padding: 10px 24px; font-weight: bold;
    }
    .sentiment-card {
        background-color: #1E1E1E; padding: 20px; border-radius: 12px;
        text-align: center; border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

# HEADER
st.markdown(f"""
    <div class="main-title">
        <img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Youtube_logo.png" width="60">
        <span>YouTube Sentiment Analysis</span>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# API KEY
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]

# LOAD MODEL
@st.cache_resource
def load_local_model():
    model_path = "w11wo/indonesian-roberta-base-sentiment-classifier" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

# Inisialisasi model
analyzer = load_local_model()

# FUNGSI YOUTUBE 
def get_comments_data(video_url):
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', video_url)
    if not video_id_match: return None
    video_id = video_id_match.group(1)
    
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    request = youtube.commentThreads().list(
        part="snippet", 
        videoId=video_id, 
        maxResults=100, 
        order="relevance"  
    )
    response = request.execute()
    
    authors, comments = [], []
    for item in response['items']:
        snippet = item['snippet']['topLevelComment']['snippet']
        authors.append(snippet['authorDisplayName'])
        comments.append(snippet['textDisplay'])
    return authors, comments

# ALUR APLIKASI
input_url = st.text_input("Salin Link YouTube di sini:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Analisis Sekarang"):
    if input_url:
        with st.spinner("Menganalisis 100 komentar..."):
            try:
                # 1. Ambil Data
                authors, raw_comments = get_comments_data(input_url)
                
                # 2. Preprocessing 
                cleaned_comments = [clean_text(c) for c in raw_comments]
                
                # 3. Prediksi AI 
                results = analyzer(cleaned_comments, truncation=True, max_length=128)
                print(results[:5])

                # 4. Buat Dataframe
                df = pd.DataFrame({
                    'Nama Akun': authors,
                    'Komentar': cleaned_comments, 
                    'Label_Murni': [res['label'] for res in results] 
                })

                # 5. MAPPING LABEL 
                mapping = {
                    'LABEL_0': 'Positif', 
                    'LABEL_1': 'Netral',
                    'LABEL_2': 'Negatif',
                    'positive': 'Positif',
                    'neutral': 'Netral',
                    'negative': 'Negatif'
                }
                
                # Terapkan mapping ke kolom Label
                df['Label'] = df['Label_Murni'].map(mapping)
                
                # 6. Statistik untuk Grafik
                counts = df['Label'].value_counts()
                total = len(df)
                dominant_label = counts.idxmax()
                dominant_percent = (counts.max() / total * 100).round(1)

                # --- TAMPILAN DASHBOARD ---
                st.markdown("---")
                st.markdown(f"""
                    <div class="sentiment-card">
                        <h3 style='margin:0; color:#6C757D;'>Sentimen Dominan</h3>
                        <h1 style='margin:0; color:#28A745;'>{dominant_label} ({dominant_percent}%)</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                col_chart, col_table = st.columns([1, 1])
                with col_chart:
                    st.subheader("📊 Distribusi Sentimen")
                    fig = px.pie(
                        values=counts.values, 
                        names=counts.index, 
                        hole=0.5,
                        template="plotly_dark",
                        color=counts.index,
                        color_discrete_map={
                            'Positif': '#28A745', # Hijau
                            'Netral': '#007BFF',  # Biru
                            'Negatif': '#DC3545'  # Merah
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_table:
                    st.subheader("📝 Hasil Analisis")
                    st.dataframe(df[['Nama Akun', 'Komentar', 'Label']], use_container_width=True, height=400)
            
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
    else:
        st.error("Masukkan link YouTube terlebih dahulu!")