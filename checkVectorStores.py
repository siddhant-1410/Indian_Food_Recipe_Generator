"""
Vector Store Access Diagnostic and Fix

This script checks if vector stores are being properly accessed and created.
It will:
1. Check the current directory structure
2. Generate or regenerate vector stores for PDFs
3. Verify if vector stores can be loaded correctly
"""

import os
import sys
import pickle
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import shutil

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, 'pdfs')
VECTOR_CACHE_FOLDER = os.path.join(BASE_DIR, 'vector_cache')

print(f"Current working directory: {os.getcwd()}")
print(f"Base directory: {BASE_DIR}")
print(f"PDF folder path: {PDF_FOLDER}")
print(f"Vector cache folder path: {VECTOR_CACHE_FOLDER}")

# Ensure cache directory exists
os.makedirs(VECTOR_CACHE_FOLDER, exist_ok=True)

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
        print(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        raise

def chunk_text(text, chunk_size=2048, chunk_overlap=400):
    """Split text into chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

def create_vector_store(text, pdf_filename):
    """Create FAISS vector store from text chunks"""
    print(f"Creating vector store for: {pdf_filename}")
    
    # Initialize embedding model
    embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    dimension = embedding_model.get_sentence_embedding_dimension()
    
    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)

    # Create text chunks
    chunks = chunk_text(text)
    if not chunks:
        print(f"Warning: No text chunks extracted from {pdf_filename}")
        return None, embedding_model, []
    
    print(f"Number of chunks created: {len(chunks)}")
    
    # Generate embeddings
    embeddings = []
    for chunk in chunks:
        embedding = embedding_model.encode(
            chunk, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        embeddings.append(embedding)

    if embeddings:
        embeddings_array = np.array(embeddings, dtype=np.float32)
        index.add(embeddings_array)
        
        # Save vector store
        cache_path = os.path.join(VECTOR_CACHE_FOLDER, f"{pdf_filename}.vectorstore")
        with open(cache_path, 'wb') as f:
            pickle.dump((index, embedding_model, chunks), f)
        print(f"Vector store saved to: {cache_path}")
        
        # Verify if it was saved correctly
        if os.path.exists(cache_path):
            print(f"✅ Verified: Vector store file exists at {cache_path}")
            print(f"   File size: {os.path.getsize(cache_path)} bytes")
        else:
            print(f"❌ Error: Vector store file was not created at {cache_path}")
        
        return index, embedding_model, chunks
    
    print(f"Warning: No embeddings created for {pdf_filename}")
    return None, embedding_model, []

def load_vector_store(pdf_path):
    """Test loading vector store from cache"""
    pdf_filename = os.path.basename(pdf_path)
    cache_path = os.path.join(VECTOR_CACHE_FOLDER, f"{pdf_filename}.vectorstore")
    
    print(f"Attempting to load vector store from: {cache_path}")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                index, embedding_model, chunks = pickle.load(f)
            print(f"✅ Successfully loaded vector store for {pdf_filename}")
            print(f"   Number of chunks loaded: {len(chunks)}")
            print(f"   Number of vectors in index: {index.ntotal}")
            return True, index, embedding_model, chunks
        except Exception as e:
            print(f"❌ Error loading vector store: {str(e)}")
            return False, None, None, []
    else:
        print(f"❌ Vector store file doesn't exist at: {cache_path}")
        return False, None, None, []

def fix_modules_access():
    """Fix potential import paths for modules"""
    # Create an empty __init__.py in modules directory if it doesn't exist
    modules_dir = os.path.join(BASE_DIR, 'modules')
    os.makedirs(modules_dir, exist_ok=True)
    
    init_file = os.path.join(modules_dir, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('# Module initialization file')
        print(f"Created {init_file}")

def check_if_vectorstore_in_wrong_location():
    """Check if vector stores are in incorrect locations and fix them"""
    # Look for .vectorstore files in the base directory
    for filename in os.listdir(BASE_DIR):
        if filename.endswith('.vectorstore'):
            source_path = os.path.join(BASE_DIR, filename)
            target_path = os.path.join(VECTOR_CACHE_FOLDER, filename)
            
            print(f"Found vector store in wrong location: {source_path}")
            print(f"Moving to correct location: {target_path}")
            
            # Move the file to the proper directory
            shutil.move(source_path, target_path)
            print(f"✅ Moved vector store file to correct location")

def process_all_pdfs():
    """Process all PDFs in the folder and create vector stores"""
    if not os.path.exists(PDF_FOLDER):
        print(f"❌ PDF folder doesn't exist: {PDF_FOLDER}")
        return
    
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the PDF folder")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\nProcessing {i}/{len(pdf_files)}: {pdf_file}")
        
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        
        # Check if vector store already exists
        success, index, model, chunks = load_vector_store(pdf_path)
        
        if not success:
            # Try to create vector store
            try:
                print(f"Creating vector store for: {pdf_file}")
                text = extract_text_from_pdf(pdf_path)
                if text:
                    create_vector_store(text, pdf_file)
                else:
                    print(f"❌ No text extracted from {pdf_file}")
            except Exception as e:
                print(f"❌ Error processing {pdf_file}: {str(e)}")

def main():
    """Main function to run diagnostics and fixes"""
    print("=" * 50)
    print("VECTOR STORE ACCESS DIAGNOSTIC")
    print("=" * 50)
    
    # Fix modules access if needed
    fix_modules_access()
    
    # Check if vector stores are in wrong location
    check_if_vectorstore_in_wrong_location()
    
    # Process all PDFs
    process_all_pdfs()
    
    print("\n" + "=" * 50)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()