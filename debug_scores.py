import sys
from vector_db import SessionVectorDB
from langchain_community.vectorstores import Chroma

def test():
    db = SessionVectorDB(run_health_check=False)
    collections = db.client.list_collections()
    if not collections:
        print("No collections found.")
        return
    
    # Grab the most recent session collection
    col = collections[-1]
    db.collection_name = col.name
    db.vector_store = Chroma(
            client=db.client,
            collection_name=db.collection_name,
            embedding_function=db.embeddings
        )
    
    query = "what is dsa"
    print(f"Querying collection: {col.name} for '{query}'")
    results = db.vector_store.similarity_search_with_score(query, k=3)
    
    for doc, score in results:
        print(f"Score: {score:.4f} | Content: {doc.page_content[:60]}...")

if __name__ == "__main__":
    test()
