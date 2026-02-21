import os
import mimetypes
import pdfplumber
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_FILE_SIZE_MB = 50
MAX_PAGE_COUNT = 500

class PDFProcessingError(Exception):
    pass

def validate_pdf(filepath: str):
    if not os.path.exists(filepath):
        raise PDFProcessingError(f"File not found: {filepath}")
    
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise PDFProcessingError(f"File exceeds maximum size limit of {MAX_FILE_SIZE_MB}MB.")
    
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type != 'application/pdf':
        raise PDFProcessingError(f"Invalid file type: {mime_type}. Expected application/pdf.")

def extract_text_from_pdf(filepath: str) -> str:
    validate_pdf(filepath)
    
    text_content = []
    try:
        with pdfplumber.open(filepath) as pdf:
            if len(pdf.pages) > MAX_PAGE_COUNT:
                raise PDFProcessingError(f"Document exceeds maximum page limit of {MAX_PAGE_COUNT}.")
            
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_content.append(text)
    except Exception as e:
        try:
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                if reader.is_encrypted:
                    raise PDFProcessingError("PDF is encrypted or password-protected.")
                if len(reader.pages) > MAX_PAGE_COUNT:
                    raise PDFProcessingError(f"Document exceeds maximum page limit of {MAX_PAGE_COUNT}.")
                
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_content.append(text)
        except PDFProcessingError:
            raise
        except Exception as fallback_e:
            raise PDFProcessingError(f"Failed to read PDF. error: {str(e)} | fallback error: {str(fallback_e)}")

    full_text = "\n".join(text_content)
    if not full_text.strip():
        raise PDFProcessingError("No selectable text found in PDF. It might be a scanned image or empty.")
    
    return full_text

def clean_text(text: str) -> str:
    import re
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

def process_pdf(filepath: str) -> list[str]:
    """Main pipeline function linking extraction, cleaning, and chunking."""
    raw_text = extract_text_from_pdf(filepath)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)
    return chunks
