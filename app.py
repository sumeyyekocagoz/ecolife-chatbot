import streamlit as st
import os
import pandas as pd
import numpy as np
import faiss
import time
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# .env dosyasındaki API anahtarını yükle (Bu, lokal çalışma içindir. 
# Streamlit Cloud'da secrets kullanılır)
load_dotenv()

# --- API Anahtarlarını Yükleme ve Kontrol Etme ---

# Google API anahtarını al (Önce Streamlit Secrets'tan dene, sonra lokal .env'den)
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

# Hugging Face token'ını al (Sadece Streamlit Secrets'tan)
HF_TOKEN = st.secrets.get("HUGGING_FACE_HUB_TOKEN")

# Anahtar kontrolleri
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY bulunamadı. Lütfen Streamlit Secrets'ı kontrol edin.")
    st.stop()

if not HF_TOKEN:
    st.error("HUGGING_FACE_HUB_TOKEN bulunamadı. Lütfen Streamlit Secrets'ı kontrol edin.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# --- Yardımcı Fonksiyonlar ---

def get_gemini_response(question, chat_history):
    # DÜZELTME: En modern 'gemini-1.5-flash' modelini kullanıyoruz.
    # requirements.txt'deki sürüm güncellemesi bu modelin bulunmasını sağlayacak.
    model = genai.GenerativeModel('gemini-1.5-flash') 
    
    chat = model.start_chat(history=chat_history) 
    
    try:
        response = chat.send_message(question, stream=False)
        return response.text
    except Exception as e:
        st.error(f"Gemini API hatası: {e}")
        return "Üzgünüm, bir hata oluştu."

def get_faiss_index(texts, model, index_path="faiss_index_v2"):
    if os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
            print("Varolan FAISS index'i yüklendi.")
            return index
        except Exception as e:
            print(f"Index yüklenirken hata: {e}. Index yeniden oluşturulacak.")

    print("Yeni FAISS index'i oluşturuluyor...")
    try:
        embeddings = model.encode(texts, convert_to_tensor=False)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        faiss.write_index(index, index_path)
        print("Yeni index oluşturuldu ve kaydedildi.")
        return index
    except Exception as e:
        st.error(f"FAISS index oluşturulurken hata: {e}")
        return None

def get_context_from_faiss(index, query, model, k=5):
    try:
        query_embedding = model.encode([query], convert_to_tensor=False)
        distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
        return indices[0] 
    except Exception as e:
        st.error(f"FAISS araması sırasında hata: {e}")
        return []

def safe_text_extraction(row):
    try:
        # JSONL dosyanızdaki anahtarların 'Soru' ve 'Cevap' olduğunu varsayıyorum.
        return f"Soru: {row['Soru']} Cevap: {row['Cevap']}"
    except KeyError:
        if 'text' in row:
            return row['text']
        return "" 
    except TypeError:
        return "" 

# --- Veri Yükleme ve Önbelleğe Alma ---

@st.cache_resource():
def load_resources():
    st.info("Kaynaklar yükleniyor (Bu işlem birkaç dakika sürebilir)...")
    
    try:
        embedding_model = SentenceTransformer(
            'all-MiniLM-L6-v2',
            token=HF_TOKEN
        )
    except Exception as e:
        st.error(f"SentenceTransformer yüklenirken hata oluştu. Hata: {e}")
        st.stop()
        
    DATA_FILE = 'llama.jsonl'
    try:
        df = pd.read_json(DATA_FILE, lines=True)
    except FileNotFoundError:
        st.error(f"HATA: '{DATA_FILE}' dosyası proje klasöründe bulunamadı.")
        st.stop()
    except Exception as e:
        st.error(f"'{DATA_FILE}' dosyası okunurken hata: {e}")
        st.stop()
        
    df['text'] = df.apply(safe_text_extraction, axis=1)
    texts = df['text'].dropna().tolist()
    
    if not texts:
        st.error("Veritabanından metin okunamadı.")
        st.stop()
        
    faiss_index = get_faiss_index(texts, embedding_model)
    
    if faiss_index is None:
        st.error("FAISS index'i yüklenemedi veya oluşturulamadı.")
        st.stop()
        
    st.success("Kaynaklar başarıyla yüklendi! Chatbot hazır.")
    
    return embedding_model, faiss_index, texts

# --- Streamlit Uygulaması ---

st.set_page_config(page_title="EcoLife Chatbot", page_icon="🌱")
st.title("🌱 EcoLife - Vegan & Ekolojik Yaşam Asistanı")
st.caption("Akbank Generative-AI Bootcamp Projesi")

try:
    embedding_model, faiss_index, texts = load_resources()
except Exception as e:
    st.error(f"Kaynaklar yüklenirken kritik bir hata oluştu: {e}")
    st.stop()

# Oturum durumunu (chat geçmişi) başlat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append({
        "role": "model", 
        "parts": [{"text": "Merhaba! Ben EcoLife. Veganlık ve ekolojik yaşam hakkında sorularınızı yanıtlamak için buradayım."}] 
    })

# Chat geçmişini ekrana yazdır
for message in st.session_state.chat_history:
    display_role = "assistant" if message["role"] == "model" else message["role"]
    with st.cache_resource:
        st.markdown(message["parts"][0]["text"])

# Kullanıcıdan yeni giriş al
if prompt := st.chat_input("Veganlık veya ekolojik yaşam hakkında bir soru sorun..."):
    
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "parts": [{"text": prompt}]})

    context_indices = get_context_from_faiss(faiss_index, prompt, embedding_model, k=5)
    context_texts = [texts[i] for i in context_indices]
    
    combined_prompt = f"""
    Kullanıcı Sorusu: {prompt}
    Bilgi Tabanından Alınan İlgili Bağlam (Lütfen cevabını bu bağlama dayandır):
    {"---".join(context_texts)}
    Lütfen YALNIZCA sağlanan bağlamı kullanarak kullanıcı sorusunu yanıtla. Eğer cevap bağlamda yoksa, 'Bu konuda bilgim bulunmuyor.' de.
    """

    with st.spinner("EcoLife düşünüyor..."):
        response_text = get_gemini_response(combined_prompt, st.session_state.chat_history)
    
    with st.chat_message("assistant"):
        st.markdown(response_text)
    st.session_state.chat_history.append({"role": "model", "parts": [{"text": response_text}]})


