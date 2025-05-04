import torch
import faiss
import numpy as np
import pdfplumber
import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import os

#API Key loading -
load_dotenv()
api_key = os.getenv("API_KEY")


# Gemini API Key - replace with your actual key
GEMINI_API_KEY = api_key  # This will be overridden by the key passed from app.py

# Embedding model cache
_embedding_model = None
_translation_model = None
_translation_tokenizer = None

def get_embedding_model():
    """Lazy loading of embedding model"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _embedding_model

def initialize_gemini(api_key):
    """Initialize Gemini API"""
    if not api_key:
        raise ValueError("Gemini API key is required")
    genai.configure(api_key=api_key)

def load_translation_model(model_path):
    """Load the translation model and tokenizer"""
    global _translation_model, _translation_tokenizer
    
    if _translation_model is None or _translation_tokenizer is None:
        try:
            from transformers import MarianMTModel, MarianTokenizer
            
            _translation_tokenizer = MarianTokenizer.from_pretrained(model_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _translation_model = MarianMTModel.from_pretrained(model_path).to(device)
            
            return _translation_model, _translation_tokenizer, device
        except Exception as e:
            raise Exception(f"Error loading translation model: {str(e)}")
    
    return _translation_model, _translation_tokenizer, torch.device("cuda" if torch.cuda.is_available() else "cpu")

def translate_text(text, model_path="models/best_model"):
    """Translate text from English to Hindi after cleaning it by removing '#' and '*' characters."""
    try:
        # Clean the text
        cleaned_text = text.replace("#", "").replace("*", "").lower()

        model, tokenizer, device = load_translation_model(model_path)

        print("\n")
        print("LENGTH OF THE TEXT:   ")
        print(len(cleaned_text))
        print("\n")
        print(cleaned_text)
        print("\n")
        print("\n")

        if len(cleaned_text) > 150 or "\n" in cleaned_text:
            lines = cleaned_text.split("\n")
            print("\n")
            print("Lines")
            print("\n")
            print(lines)
            print("\n")

            translated_lines = []

            for line in lines:
                if line.strip():  # Skip empty lines
                    translated_line = translate_line(line, model, tokenizer, device)
                    translated_lines.append(translated_line)
                else:
                    translated_lines.append("")  # Keep empty lines as is

            return "\n".join(translated_lines)
        else:
            return translate_line(cleaned_text, model, tokenizer, device)
    except Exception as e:
        raise Exception(f"Translation error: {str(e)}")

def translate_line(line, model, tokenizer, device):
    """Translate a single line of text"""
    if not line.strip():
        return line
        
    inputs = tokenizer(line, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF file"""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def find_matching_pdf(pdf_folder, dish_name):
    """Find the most relevant PDF for a dish"""
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    
    if not pdf_files:
        return None
    
    # Try exact match first
    for pdf_file in pdf_files:
        if dish_name.lower() in pdf_file.lower():
            return pdf_file
    
    # If fuzzy matching is needed, you could implement it here
    # For simplicity, we're just returning None if no exact match
    return None

def chunk_text(text, chunk_size=2048, chunk_overlap=400):
    """Split text into chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

def create_vector_store(text):
    """Create FAISS vector store from text chunks"""
    embedding_model = get_embedding_model()
    dimension = embedding_model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dimension)

    chunks = chunk_text(text)
    if not chunks:
        return None, embedding_model, []
    
    embeddings = []
    for chunk in chunks:
        embedding = embedding_model.encode(
            chunk, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        embeddings.append(embedding)

    if embeddings:
        index.add(np.array(embeddings, dtype=np.float32))
        return index, embedding_model, chunks
    
    return None, embedding_model, []

def load_vector_store(pdf_path):
    """
    Load vector store from pre-generated cache with improved path handling and debugging
    """
    # Get absolute paths to ensure consistency
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    pdf_filename = os.path.basename(pdf_path)
    
    # Define possible cache paths to check
    possible_paths = [
        # Path relative to the current file in modules directory
        os.path.join(os.path.dirname(__file__), '..', 'vector_cache', f"{pdf_filename}.vectorstore"),
        # Direct path from base directory
        os.path.join(base_dir, 'vector_cache', f"{pdf_filename}.vectorstore"),
        # In case it's stored with the PDFs
        os.path.join(os.path.dirname(pdf_path), f"{pdf_filename}.vectorstore"),
        # Directly in the base directory as a fallback
        os.path.join(base_dir, f"{pdf_filename}.vectorstore")
    ]
    
    print(f"Looking for vector store for: {pdf_filename}")
    
    # Try each possible path
    for cache_path in possible_paths:
        if os.path.exists(cache_path):
            print(f"Found vector store at: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    index, embedding_model, chunks = pickle.load(f)
                print(f"Successfully loaded vector store with {len(chunks)} chunks")
                return index, embedding_model, chunks
            except Exception as e:
                print(f"Error loading vector store from {cache_path}: {str(e)}")
    
    print(f"No valid vector store found for {pdf_filename}. Vector store will be created on-the-fly.")
    
    # If no valid vector store is found, create one on-the-fly
    try:
        from recipe_generator import extract_text_from_pdf, create_vector_store
        
        print(f"Extracting text from {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        
        if text:
            print(f"Creating vector store for {pdf_filename}")
            index, embedding_model, chunks = create_vector_store(text)
            
            # Save the newly created vector store
            os.makedirs(os.path.join(base_dir, 'vector_cache'), exist_ok=True)
            cache_path = os.path.join(base_dir, 'vector_cache', f"{pdf_filename}.vectorstore")
            
            with open(cache_path, 'wb') as f:
                pickle.dump((index, embedding_model, chunks), f)
            print(f"Saved new vector store to {cache_path}")
            
            return index, embedding_model, chunks
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
    
    # Return empty results if all methods fail
    print("Returning empty vector store")
    embedding_model = get_embedding_model()
    empty_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
    return empty_index, embedding_model, []

def retrieve_relevant_chunks(query, index, embedding_model, chunks, top_k=3):
    """Retrieve most relevant text chunks for a query"""
    if not index or index.ntotal == 0 or not chunks:
        return []
    
    query_embedding = embedding_model.encode(
        query, 
        convert_to_numpy=True, 
        normalize_embeddings=True
    ).astype(np.float32)
    
    _, indices = index.search(np.array([query_embedding]), min(top_k, len(chunks)))
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def generate_response(prompt, api_key):
    """Generate response using Gemini API"""
    initialize_gemini(api_key)
    
    system_prompt = (
        "You are a professional chef providing complete and well-structured recipes."
        "Ensure the response includes all ingredients and steps in a clear, logical order."
        "Format the recipe with clear sections for ingredients and preparation steps."
        "Write short instructions preferably one line sentences."
    )
    final_prompt = f"{system_prompt}\n\n{prompt}"

    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(final_prompt)
        return response.text.strip()
    except Exception as e:
        raise Exception(f"Error generating recipe: {str(e)}")

def generate_recipe(dish_name, pdf_folder, api_key):
    """Generate recipe for a dish using pre-generated vector stores"""
    # Find matching PDF
    matching_pdf = find_matching_pdf(pdf_folder, dish_name)
    
    # If no matching PDF, generate recipe directly
    if not matching_pdf:
        prompt = (
            f"Create a detailed recipe for {dish_name} with ingredients and "
            f"step-by-step instructions. Include cooking time, serving size, "
            f"and difficulty level."
        )
        return generate_response(prompt, api_key)
    
    # Process PDF and generate recipe using RAG
    pdf_path = os.path.join(pdf_folder, matching_pdf)
    
    # Load pre-generated vector store
    index, embedding_model, chunks = load_vector_store(pdf_path)
    
    if not index:
        # Fallback if vector store not found
        prompt = f"Create a detailed recipe for {dish_name} based on traditional methods."
        return generate_response(prompt, api_key)
    
    retrieved_texts = retrieve_relevant_chunks(
        f"Recipe for {dish_name}", 
        index, 
        embedding_model, 
        chunks
    )
    
    context = "\n".join(retrieved_texts) if retrieved_texts else ""
    prompt = (
        f"Recipe for {dish_name}:\n\n{context}\n\n"
        f"Based on the above information, provide a well-structured recipe "
        f"with ingredients list and step-by-step preparation instructions. "
        f"If the information is incomplete, use your knowledge to fill in gaps."
    )
    
    return generate_response(prompt, api_key)