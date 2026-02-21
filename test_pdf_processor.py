import os
import pytest
from reportlab.pdfgen import canvas
from pdf_processor import process_pdf, PDFProcessingError

MOCK_PDF_PATH = "mock_test_doc.pdf"

@pytest.fixture(autouse=True)
def setup_and_teardown():
    # Setup: Create a mock PDF
    c = canvas.Canvas(MOCK_PDF_PATH)
    c.drawString(100, 750, "This is a test PDF document.")
    c.drawString(100, 730, "It contains some text to extract and chunk.")
    # Add enough text to ensure chunking
    long_text = "Word word word " * 200
    c.drawString(100, 710, long_text)
    c.save()
    
    yield # Run tests
    
    # Teardown: Remove the mock PDF
    if os.path.exists(MOCK_PDF_PATH):
        os.remove(MOCK_PDF_PATH)

def test_process_pdf():
    # The default chunk_size is 1000
    chunks = process_pdf(MOCK_PDF_PATH)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert "This is a test PDF document." in chunks[0]

def test_missing_file():
    with pytest.raises(PDFProcessingError, match="File not found"):
        process_pdf("nonexistent_file.pdf")

def test_invalid_type():
    with open("fake.txt", "w") as f:
        f.write("test")
    
    # Rename to pdf to trick extension, but mimetypes uses extension so mimetypes might say application/pdf.
    # Actually wait, mimetypes.guess_type on 'fake_pdf.pdf' will return 'application/pdf'. 
    # Let's see if PDFProcessingError is raised during extraction because it's not a valid PDF!
    os.rename("fake.txt", "fake_pdf.pdf")
    with pytest.raises(PDFProcessingError, match="Failed to read PDF"):
        process_pdf("fake_pdf.pdf")
    os.remove("fake_pdf.pdf")
