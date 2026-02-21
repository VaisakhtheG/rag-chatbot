import pytest
from unittest.mock import patch, MagicMock
from llm_integration import truncate_context, generate_response, LLMIntegrationError

def test_truncate_context():
    # 50 words is slightly over 50 tokens
    chunks = ["Word " * 50] * 10 
    truncated = truncate_context(chunks, max_tokens=150) # Should allow ~2-3 chunks depending on tiktoken
    assert 1 <= len(truncated.split("\n\n")) <= 3

@patch('requests.post')
def test_generate_response(mock_post):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    
    import json
    chunks = [
        {"message": {"content": "Hello "}},
        {"message": {"content": "World"}}
    ]
    mock_response.iter_lines.return_value = [json.dumps(c).encode('utf-8') for c in chunks]
    
    mock_post.return_value = mock_response
    
    generator = generate_response("query?", ["context"])
    result = "".join(list(generator))
    
    assert result == "Hello World"

@patch('requests.post')
def test_generate_response_error(mock_post):
    import requests
    mock_post.side_effect = requests.RequestException("API down")
    
    with pytest.raises(LLMIntegrationError, match="Failed to communicate with Ollama"):
        list(generate_response("query", ["context"]))
