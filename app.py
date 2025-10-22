import streamlit as st
import os
import pandas as pd
import numpy as np
import faiss
import time
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# .env dosyasındaki API anahtarını yükle
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Hata kontrolü
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY bulunamadı. Lütfen .env dosyanızı kontrol edin.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# --- Yardımcı Fonksiyonlar (Colab'dan alındı) ---

def get_gemini_response(question, chat_history):
    """
    Gemini modelinden yanıt alır.
    """
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=chat_history)
    
    try:
        response = chat.send_message(question, stream=False)
        return response.text
    except Exception as e:
        st.error(f"Gemini API hatası: {e}")
        return "Üzgünüm, bir hata oluştu."

def get_faiss_index(texts, model, index_path="faiss_index_v2"):
    """
    Metinlerden bir FAISS index'i oluşturur veya yükler.
    """
    if os.path.exists(index_path):
        try:
            # Varolan index'i yükle
            index = faiss.read_index(index_path)
            print("Varolan FAISS index'i yüklendi.")
            return index
        except Exception as e:
            print(f"Index yüklenirken hata: {e}. Index yeniden oluşturulacak.")

    # Yeni index oluştur
    print("Yeni FAISS index'i oluşturuluyor...")
    try:
        embeddings = model.encode(texts, convert_to_tensor=False)
        
        # FAISS index'ini oluştur
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) mesafesi
        index.add(np.array(embeddings).astype('float32'))
        
        # Index'i diske kaydet
        faiss.write_index(index, index_path)
        print("Yeni index oluşturuldu ve kaydedildi.")
        return index
    except Exception as e:
        st.error(f"FAISS index oluşturulurken hata: {e}")
        return None

def get_context_from_faiss(index, query, model, k=5):
    """
    FAISS index'inden ilgili bağlamı alır.
    """
    try:
        query_embedding = model.encode([query], convert_to_tensor=False)
        distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
        
        # 'texts' listesine ihtiyacımız olacak. Bu listeyi global'de veya
        # cache'lenmiş fonksiyondan almamız gerekiyor.
        # Bu fonksiyonun çalışması için 'texts' listesine erişim varsayılıyor.
        # Daha iyi bir yapı için 'texts' listesini de parametre olarak alabilir.
        # Şimdilik, 'texts'in bu fonksiyonun çağrıldığı yerde mevcut olduğunu varsayıyoruz.
        
        # 'texts' listesi 'load_resources' fonksiyonunda tanımlı,
        # bu yüzden bu fonksiyonu 'load_resources' içinde kullanmak
        # veya 'texts'i de cache'lemek daha iyi olur.
        # Ancak Colab'daki yapıya sadık kalmak için, 'texts' listesini
        # ana uygulamada yüklüyoruz ve bu fonksiyonu orada çağırıyoruz.
        
        # Düzeltme: 'texts' listesini de döndürelim.
        return indices[0] # Sadece index'leri döndür, metinleri ana fonksiyonda al
    
    except Exception as e:
        st.error(f"FAISS araması sırasında hata: {e}")
        return []

def safe_text_extraction(row):
    """
    Veri çerçevesi satırından metin çıkarır.
    """
    try:
        return f"Soru: {row['Soru']} Cevap: {row['Cevap']}"
    except TypeError:
        return "" # Hatalı veya eksik veri varsa boş döndür

# --- Veri Yükleme ve Önbelleğe Alma ---

@st.cache_resource
def load_resources():
    """
    Ağır kaynakları (model, veri, index) yükler ve cache'ler.
    Bu fonksiyon uygulama başlarken sadece bir kez çalışır.
    """
    st.info("Kaynaklar yükleniyor (Bu işlem birkaç dakika sürebilir)...")
    
    # 1. Gömme (Embedding) Modelini Yükle
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Veritabanını Yükle
    try:
        df = pd.read_excel('combined_data_v2.xlsx')
    except FileNotFoundError:
        st.error("HATA: 'combined_data_v2.xlsx' dosyası proje klasöründe bulunamadı.")
        st.stop()
        
    # 3. Metinleri Hazırla
    df['text'] = df.apply(safe_text_extraction, axis=1)
    texts = df['text'].dropna().tolist()
    
    if not texts:
        st.error("Veritabanından metin okunamadı.")
        st.stop()
        
    # 4. FAISS Index'ini Yükle veya Oluştur
    faiss_index = get_faiss_index(texts, embedding_model)
    
    if faiss_index is None:
        st.error("FAISS index'i yüklenemedi veya oluşturulamadı.")
        st.stop()
        
    st.success("Kaynaklar başarıyla yüklendi! Chatbot hazır.")
    
    # 'texts' listesini de döndürüyoruz ki RAG aramasında kullanabilelim
    return embedding_model, faiss_index, texts

# --- Streamlit Uygulaması ---

st.set_page_config(page_title="EcoLife Chatbot", page_icon="🌱")
st.title("🌱 EcoLife - Vegan & Ekolojik Yaşam Asistanı")
st.caption("Akbank Generative-AI Bootcamp Projesi")

# Kaynakları yükle (cache'den gelir)
try:
    embedding_model, faiss_index, texts = load_resources()
except Exception as e:
    st.error(f"Kaynaklar yüklenirken kritik bir hata oluştu: {e}")
    st.stop()


# Oturum durumunu (chat geçmişi) başlat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": "Merhaba! Ben EcoLife. Veganlık ve ekolojik yaşam hakkında sorularınızı yanıtlamak için buradayım."
    })


# Chat geçmişini ekrana yazdır
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan yeni giriş al
if prompt := st.chat_input("Veganlık veya ekolojik yaşam hakkında bir soru sorun..."):
    
    # Kullanıcı mesajını göster ve geçmişe ekle
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # RAG: İlgili bağlamı FAISS'den al
    context_indices = get_context_from_faiss(faiss_index, prompt, embedding_model, k=5)
    
    # İndex'lere göre metinleri al
    context_texts = [texts[i] for i in context_indices]
    
    # Gemini için birleştirilmiş prompt hazırla
    combined_prompt = f"""
    Kullanıcı Sorusu: {prompt}

    Bilgi Tabanından Alınan İlgili Bağlam (Lütfen cevabını bu bağlama dayandır):
    {"---".join(context_texts)}

    Lütfen YALNIZCA sağlanan bağlamı kullanarak kullanıcı sorusunu yanıtla. Eğer cevap bağlamda yoksa, 'Bu konuda bilgim bulunmuyor, ancak farklı bir şekilde sorabilir misiniz?' de.
    """

    # Modeli çağır ve yanıtı al
    with st.spinner("EcoLife düşünüyor..."):
        response_text = get_gemini_response(combined_prompt, st.session_state.chat_history)
    
    # Yanıtı göster ve geçmişe ekle
    with st.chat_message("assistant"):
        st.markdown(response_text)
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})

