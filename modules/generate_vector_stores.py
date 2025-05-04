import os
import pickle
from recipe_generator import extract_text_from_pdf, create_vector_store, get_embedding_model

def pregenerate_vector_stores(pdf_folder, vector_cache_folder):
    """Generate vector stores for all PDFs in the folder"""
    # Ensure cache directory exists
    os.makedirs(vector_cache_folder, exist_ok=True)
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for i, pdf_file in enumerate(pdf_files):
        print(f"Processing {i+1}/{len(pdf_files)}: {pdf_file}")
        
        pdf_path = os.path.join(pdf_folder, pdf_file)
        cache_path = os.path.join(vector_cache_folder, f"{pdf_file}.vectorstore")
        
        # Skip if already exists
        if os.path.exists(cache_path):
            print(f"Vector store already exists for {pdf_file}, skipping")
            continue
        
        try:
            # Extract text and create vector store
            text = extract_text_from_pdf(pdf_path)
            index, embedding_model, chunks = create_vector_store(text)
            
            if index and chunks:
                # Save vector store
                with open(cache_path, 'wb') as f:
                    pickle.dump((index, embedding_model, chunks), f)
                print(f"Successfully created vector store for {pdf_file}")
            else:
                print(f"Failed to create vector store for {pdf_file}")
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
    
    print("Vector store generation complete!")

if __name__ == "__main__":
    # Use the same paths as in your app
    pdf_folder = "./pdfs"
    vector_cache_folder = "vector_cache"
    pregenerate_vector_stores(pdf_folder, vector_cache_folder)