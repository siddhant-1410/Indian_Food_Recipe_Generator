# Indian Food Recipe Generator from Images 🇮🇳🍲

A deep learning-powered application that generates **Indian food recipes** from food images — with English to Hindi translation and top-5 dish recommendations.

## Demo Video :-


https://github.com/user-attachments/assets/dfc6a854-b182-45ee-9e40-d7f8566d0d38


## 🚀 Project Overview

This project aims to simplify recipe generation for Indian cuisine enthusiasts by allowing users to upload a food image and receive:

- The **generated recipe** in English
- An **optional Hindi translation** of the recipe
- **Top 5 similar dish recommendations** based on the uploaded food image

## ✨ Key Features

- 🖼️ **Image Classification** of 25 popular Indian dishes
- 📝 **Recipe Generation** using Retrieval-Augmented Generation (RAG)
- 🌐 **Recipe Translation** from English to Hindi (BLEU score: 83.567)
- 🍛 **Dish Recommendation System** (Top 5 similar dishes)
- ⚙️ **Flask backend** for smooth integration

---

## 📊 Dataset Details

### 1️⃣ Image Classification Dataset  
- **Classes**: 25 Indian dishes  
- **Images per class**: 250–300  
- **Total images**: 6,850  
- **Image resolution**: 224x224 px (resized for CNN)  
- **Examples of classes**: Biryani, Dosa, Paneer Butter Masala, Chole Bhature, Idli, etc.

| Metric | Value |
|--------|-------|
| Classes | 25 |
| Avg images/class | ~274 |
| Total images | 6,850 |
| Image size | 224x224 |

---

### 2️⃣ Recipe Dataset (For Retrieval & Generation)

- **Recipes per class**: ~80–100  
- **Total recipes**: ~2,300  
- **Fields**: Dish Name, Ingredients, Preparation Steps  
- **Storage format**: JSON  
- **Embedding method**: Sentence embeddings using Google Gemini API  
- **Vector DB**: FAISS

| Metric | Value |
|--------|-------|
| Recipes | ~2,300 |
| Avg recipes/class | ~92 |
| Fields | Name, Ingredients, Steps |
| Vector Store | FAISS |

---

### 3️⃣ English-Hindi Translation Dataset

- **Pairs**: 10,000 English-Hindi sentence pairs  
- **Format**: JSONL (`{"source": "...", "target": "..."}`)  
- **Domain**: Recipe-specific sentences (ingredients & instructions)

| Metric | Value |
|--------|-------|
| Sentence pairs | 10,000 |
| Format | JSONL |
| BLEU Score (after fine-tuning) | **83.567** |

---

## 🛠️ Model Architecture & Training

### 📷 Image Classifier
- **Model**: CNN (Custom Sequential model)
- **Accuracy**: **83.0%**
- **Training time**: 50 epochs (~1.5 hrs on GPU)

### ✏️ Recipe Generator (RAG)
- **Embedding**: Google Gemini embeddings
- **Retriever**: FAISS with cosine similarity
- **Generator**: Retrieved recipe returned as output

### 🌐 Translation Model
- **Base model**: Helsinki-NLP Opus-MT (English-Hindi)
- **Fine-tuning**: 10 epochs on recipe dataset
- **Final BLEU Score**: **83.567**

### 🍛 Recommender System
- **Method**: Cosine similarity on image embeddings
- **Top-N recommendations**: 5 dishes

---

## ⚙️ Tech Stack

- Python
- TensorFlow / Keras
- Hugging Face Transformers
- Google Gemini API
- FAISS (Facebook AI Similarity Search)
- Flask
- OpenCV
- LangChain

---

## 📈 Results Summary

| Component | Metric | Value |
|-----------|--------|-------|
| Image Classifier | Accuracy | **83.0%** |
| Recipe Generator | Retrieval Accuracy | ~89% |
| Translator | BLEU Score | **83.567** |
| Recommender | Top-5 Precision | ~85% |

---
