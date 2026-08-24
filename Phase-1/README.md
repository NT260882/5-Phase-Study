# Project Overview

This project is designed to process and store Markdown documents in a vector database using ChromaDB. It includes three main components: `ingest.py`, `retriever.py`, and `rag_pipeline.py`.

## Components

### `ingest.py`

- **Purpose**: Loads Markdown documents from a specified directory, chunks them into smaller parts, and stores them in a ChromaDB collection.
- **Key Features**:
  - Recursively scans the directory for `.md` files.
  - Splits documents into overlapping chunks using Markdown-aware separators.
  - Stores chunks and their metadata in ChromaDB.

### `retriever.py`

- **Purpose**: Retrieves and re-ranks document chunks based on a user query using a hybrid approach combining BM25 and a Cross-Encoder model.
- **Key Features**:
  - Initializes a ChromaDB client and loads all ingested chunks into a BM25 index.
  - Performs sparse keyword search using BM25 and dense semantic search using ChromaDB.
  - Deduplicates and re-scores candidates using a Cross-Encoder model.

### `rag_pipeline.py`

- **Purpose**: Generates answers to user queries by retrieving relevant document chunks and sending them to a language model (LLM) for response generation.
- **Key Features**:
  - Uses the `HybridReRankRetriever` to fetch and re-rank relevant chunks.
  - Formats the retrieved chunks into a structured context string.
  - Sends the context and user query to an LLM and generates a response.
  - Ensures that all answers are strictly based on the provided documentation and includes source citations.

## Usage

1. **Install Dependencies**:
   ```bash
   pip install langchain_community langchain_text_splitters chromadb rank_bm25 sentence_transformers ollama
   ```

2. **Run the Ingestion Script**:
   ```bash
   python3 src/ingest.py
   ```

3. **Test the Retrieval and RAG Pipeline**:
   ```bash
   python3 src/rag_pipeline.py
   ```

## Configuration

- **`ingest.py`**:
  - `DOCS_DIRECTORY`: Directory containing the Markdown files to be processed.
  - `CHROMA_DB_DIR`: Directory where the ChromaDB database will be stored.
  - `COLLECTION_NAME`: Name of the collection in ChromaDB where the documents will be stored.
  - `CHUNK_SIZE` and `CHUNK_OVERLAP`: Size and overlap of the text chunks created from the documents.

- **`retriever.py`**:
  - `CHROMA_DB_DIR`: Directory where the ChromaDB database is stored.
  - `COLLECTION_NAME`: Name of the collection in ChromaDB where the documents are stored.
  - `RERANKER_MODEL`: Name of the Cross-Encoder model used for re-ranking.

- **`rag_pipeline.py`**:
  - `MODEL_NAME`: Name of the language model to be used for generating answers.
  - `SYSTEM_PROMPT`: System prompt for the LLM to follow strict rules for generating answers.

## Contributing

Feel free to contribute to this project by submitting pull requests or opening issues. Ensure that any changes are well-documented and tested.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.