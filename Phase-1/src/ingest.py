import os
import glob
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

# --- CONFIGURATION ---
DOCS_DIRECTORY = "../devdocs-rag/raw_docs/docs/en/docs"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "fastapi_docs"

# Chunking specifications based on Phase 1 recommendations:
# ~500-800 tokens with ~100 token overlap
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def load_documents(source_dir: str):
    """
    Recursively scans the directory and loads all .md files.
    Extracts metadata for source tracing and citations.
    """
    print(f" Scanning directory: {source_dir}...")
    markdown_files = glob.glob(f"{source_dir}/**/*.md", recursive=True)
    
    if not markdown_files:
        raise FileNotFoundError(
            f"No .md files found in {source_dir}. Check your path!"
        )

    print(f"Found {len(markdown_files)} Markdown files. Loading...")
    
    documents = []
    for file_path in markdown_files:
        try:
            # TextLoader loads file content and adds source metadata
            loader = TextLoader(file_path, encoding="utf-8")
            loaded_docs = loader.load()
            
            # Clean source path relative to the workspace for clean citations
            for doc in loaded_docs:
                doc.metadata["source"] = str(Path(file_path).relative_to(source_dir))
                documents.append(doc)
        except Exception as e:
            print(f"⚠️ Warning: Failed to load {file_path}: {e}")
            
    return documents


def chunk_documents(documents):
    """
    Splits documents into overlapping chunks using Markdown-aware separators.
    Preserves context across code blocks and headings.
    """
    print(" Chunking documents with sliding window overlap...")
    
    # Separators prioritized: Headers -> Code Blocks -> Paragraphs -> Sentences -> Words
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n# ",
            "\n## ",
            "\n### ",
            "\n```",
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f" Split {len(documents)} source files into {len(chunks)} chunks.")
    return chunks


def store_in_chromadb(chunks):
    """
    Pushes chunks and their metadata directly into persistent ChromaDB.
    """
    print(f" Initializing ChromaDB persistent storage at '{CHROMA_DB_DIR}'...")
    
    # Initialize Chroma native persistent client
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    # Get or create vector collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity for vector search
    )
    
    print(" Upserting document chunks into ChromaDB...")
    
    # Batch process insertion to handle large docsets efficiently
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        
        ids = [f"doc_{i + idx}" for idx in range(len(batch))]
        documents = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        
        # Note: ChromaDB automatically handles default sentence-transformer embeddings 
        # when adding raw documents if no custom embedder is passed.
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"  Processed batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}")

    print(f" Ingestion complete! Total items in collection '{COLLECTION_NAME}': {collection.count()}")


if __name__ == "__main__":
    # 1. Load files
    docs = load_documents(DOCS_DIRECTORY)
    
    # 2. Chunk text
    chunks = chunk_documents(docs)
    
    # 3. Store in Vector DB
    store_in_chromadb(chunks)
