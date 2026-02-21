import pytest
from unittest.mock import patch, MagicMock
from vector_db import check_ollama_health, SessionVectorDB, VectorDBError

@patch('requests.get')
def test_health_check_pass(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": [{"name": "nomic-embed-text:latest"}]}
    mock_get.return_value = mock_response
    
    check_ollama_health()

@patch('requests.get')
def test_health_check_fail_no_model(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": [{"name": "llama3:latest"}]}
    mock_get.return_value = mock_response
    
    with pytest.raises(VectorDBError, match="not found"):
        check_ollama_health()

@patch('requests.get')
def test_health_check_unreachable(mock_get):
    import requests
    mock_get.side_effect = requests.RequestException("Timeout")
    
    with pytest.raises(VectorDBError, match="unreachable"):
        check_ollama_health()

@patch('vector_db.check_ollama_health')
@patch('vector_db.OllamaEmbeddings')
@patch('vector_db.Chroma')
def test_session_vector_db(mock_chroma, mock_embeddings, mock_health):
    db = SessionVectorDB(session_id="test_session", run_health_check=False)
    assert db.session_id == "test_session"
    assert db.collection_name == "session_test_session"
    
    db.add_documents(["chunk1", "chunk2"])
    mock_chroma.return_value.add_texts.assert_called_once_with(texts=["chunk1", "chunk2"])
    
    mock_doc = MagicMock()
    mock_doc.page_content = "relevant chunk"
    mock_chroma.return_value.similarity_search_with_score.return_value = [(mock_doc, 1.0), (mock_doc, 2.0)]
    
    results = db.query_database("test query", threshold=1.5)
    assert len(results) == 1
    assert results[0] == "relevant chunk"
