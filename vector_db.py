import uuid
import warnings

# Suppress warnings that might clutter the UI, especially HuggingFace symlink warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import chromadb
from chromadb.config import Settings

try:
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    pass

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RELEVANCE_THRESHOLD = float('inf')

class VectorDBError(Exception):
    pass

class SessionVectorDB:
    def __init__(self, session_id: str = None, run_health_check: bool = True):
        # run_health_check is ignored as HuggingFace models run completely in-process 
        self.session_id = session_id or str(uuid.uuid4())
        
        self.client = chromadb.PersistentClient(path="./chroma_db", settings=Settings(anonymized_telemetry=False))
        self.collection_name = f"session_{self.session_id.replace('-', '_')}"
        
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        except Exception as e:
             raise VectorDBError(f"Failed to initialize HuggingFace embeddings: {e}")
            
        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )

    def add_documents(self, chunks: list[str], progress_callback=None):
        """Vectorize and store text chunks."""
        if not chunks:
            return
            
        batch_size = 20
        total = len(chunks)
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            self.vector_store.add_texts(texts=batch)
            if progress_callback:
                progress_callback(min(1.0, (i + len(batch)) / total))

    def query_database(self, user_query: str, k: int = 3, threshold: float = RELEVANCE_THRESHOLD) -> list[str]:
        """
        Retrieve top k most relevant chunks.
        Discards chunks if their distance score is greater than the threshold.
        """
        if not user_query.strip():
            return []
            
        results = self.vector_store.similarity_search_with_score(user_query, k=k)
        
        valid_chunks = []
        for doc, score in results:
            if score <= threshold:
                valid_chunks.append(doc.page_content)
        
        return valid_chunks

    def clear_session(self):
        """Cleanup session collection to save space."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
