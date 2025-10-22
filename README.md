# ecolife-chatbot
EcoLife: A RAG-based chatbot on veganism &amp; ecological living with a 5500-entry knowledge base. An Akbank Generative-AI Bootcamp project.
# EcoLife Chatbot 🌱

# EcoLife Chatbot 🌱

## About This Project

EcoLife is a chatbot based on the **RAG (Retrieval-Augmented Generation)** architecture, developed as a final project for the **Akbank Generative-AI Bootcamp**. This project aims to provide accurate and context-aware answers to user queries on **veganism and ecological living**, utilizing a specialized knowledge base of over 5500 entries.

## 🚀 Project Goals

* **Provide Accurate Information:** To serve as a reliable and quick source of information on complex topics like veganism and sustainable lifestyles.
* **Implement RAG Architecture:** To enhance the factual accuracy of a generative AI model by leveraging a large, external knowledge base.
* **Showcase Bootcamp Skills:** To demonstrate the practical application of the skills and knowledge acquired during the Akbank Generative-AI Bootcamp.

## 🛠️ Key Features

* **Architecture:** Retrieval-Augmented Generation (RAG)
* **Domain:** Veganism & Ecological Living
* **Knowledge Base:** A custom-built dataset with 5500+ entries.
* **Context:** Developed for the Akbank Generative-AI Bootcamp.

## 💻 Tech Stack

* **Language:** Python
* **LLM:** Google Gemini API 
* **Vector Database:** FAISS
* **Embeddings:** Sentence-Transformers

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

* Python 3.8 or higher
* An API key for the generative model you are using (e.g., Google AI Studio).

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    ```
2.  **Navigate to the project directory:**
    ```sh
    cd YOUR_REPOSITORY_NAME
    ```
3.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```
4.  **Set up your environment variables:**
    * Create a `.env` file in the root directory.
    * Add your API key to the `.env` file:
        ```
        API_KEY="YOUR_API_KEY_HERE"
        ```

5.  **Run the application:**
    ```sh
    streamlit run app.py
    ```

---
