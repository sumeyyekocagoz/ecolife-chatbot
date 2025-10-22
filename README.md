# EcoLife Chatbot 🌱

---

## 🔴 CANLI DEMO (LIVE DEMO) 🔴

**Chatbot'u canlı olarak denemek için aşağıdaki linke tıklayın:**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ecolife-chatbot-38ovjw6gbpwbglf9rhgcse.streamlit.app)

---

## About This Project

EcoLife is a chatbot based on the **RAG (Retrieval-Augmented Generation)** architecture, developed as a final project for the **Akbank Generative-AI Bootcamp**. 

This project aims to provide accurate and context-aware answers to user queries on **veganism and ecological living**. It utilizes a specialized `llama.jsonl` dataset, which is converted into a FAISS vector index to find the most relevant context for user questions.

## 🛠️ Key Features

* **Architecture:** Retrieval-Augmented Generation (RAG)
* **Domain:** Veganism & Ecological Living
* **Knowledge Base:** `llama.jsonl` dataset
* **Vector Search:** FAISS index for efficient similarity search.
* **Context:** Akbank Generative-AI Bootcamp Project.

## 💻 Tech Stack

* **Language:** Python
* **LLM:** Google Gemini API (`gemini-1.5-flash`)
* **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
* **Vector Database:** FAISS (faiss-cpu)
* **Frontend:** Streamlit
* **Deployment:** Streamlit Community Cloud

## 🚀 Getting Started (Run Locally)

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

* Python 3.8 or higher
* A Google Gemini API Key
* A Hugging Face User Access Token (for downloading the embedding model)

### Installation

1.  **Clone the repository (Replace `YOUR_USERNAME` with your GitHub username):**
    ```sh
    git clone [https://github.com/YOUR_USERNAME/ecolife-chatbot.git](https://github.com/YOUR_USERNAME/ecolife-chatbot.git)
    ```
2.  **Navigate to the project directory:**
    ```sh
    cd ecolife-chatbot
    ```
3.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```
4.  **Set up your environment variables:**
    * Create a `.env` file in the root directory.
    * Add your API keys to the `.env` file (this file is listed in `.gitignore` and will not be pushed to GitHub):
        ```
        GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
        HUGGING_FACE_HUB_TOKEN="YOUR_HUGGING_FACE_TOKEN_HERE"
        ```
5.  **Run the application:**
    ```sh
    streamlit run app.py
    ```

