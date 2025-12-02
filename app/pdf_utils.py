"""
Document text extraction and chunking utilities
Supports multiple file formats using Docling
"""
import os
from typing import List
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from pypdf import PdfReader  # Fallback for PDF


# Supported file extensions
SUPPORTED_EXTENSIONS = {
    '.pdf': 'PDF',
    '.docx': 'Word Document',
    '.pptx': 'PowerPoint',
    '.xlsx': 'Excel',
    '.md': 'Markdown',
    '.html': 'HTML',
    '.htm': 'HTML',
    '.csv': 'CSV',
    '.png': 'Image',
    '.jpg': 'Image',
    '.jpeg': 'Image',
    '.tiff': 'Image',
    '.bmp': 'Image',
    '.webp': 'Image'
}


def get_file_type(filename: str) -> str:
    """Get file type from extension"""
    ext = os.path.splitext(filename.lower())[1]
    return SUPPORTED_EXTENSIONS.get(ext, 'Unknown')


def is_supported_file(filename: str) -> bool:
    """Check if file format is supported"""
    ext = os.path.splitext(filename.lower())[1]
    return ext in SUPPORTED_EXTENSIONS


def extract_text_from_document(file_path: str, filename: str) -> str:
    """
    Extract text from various document formats using Docling
    
    Supports: PDF, DOCX, PPTX, XLSX, HTML, Markdown, CSV, Images (PNG, JPEG, etc.)
    """
    import os
    ext = os.path.splitext(filename.lower())[1]
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"  📋 File Details:")
    print(f"     - Format: {ext.upper()}")
    print(f"     - Size: {file_size_mb:.2f} MB ({file_size:,} bytes)")
    print(f"     - Path: {file_path}")
    
    try:
        # Use Docling for most formats
        if ext in ['.pdf', '.docx', '.pptx', '.xlsx', '.html', '.htm', '.md', '.csv']:
            print(f"  🔄 Step 1: Initializing Docling converter...")
            # Configure Docling converter
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True  # Enable OCR for scanned documents
            pipeline_options.do_table_structure = False  # Disable table structure for faster processing
            
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: pipeline_options
                }
            )
            print(f"     ✓ Converter initialized (OCR: enabled)")
            
            print(f"  🔄 Step 2: Converting document to Markdown format...")
            # Convert document
            result = converter.convert(file_path)
            print(f"     ✓ Document converted successfully")
            
            print(f"  🔄 Step 3: Extracting text from converted document...")
            # Extract text from the document
            text = result.document.export_to_markdown()
            text_length = len(text)
            text_words = len(text.split())
            
            if not text or not text.strip():
                # Fallback to pypdf for PDF if Docling fails
                if ext == '.pdf':
                    print(f"     ⚠ No text extracted, trying fallback method...")
                    return extract_text_from_pdf_fallback(file_path)
                raise ValueError(f"No text extracted from {filename}")
            
            print(f"  ✓ Text Extraction Complete:")
            print(f"     - Characters extracted: {text_length:,}")
            print(f"     - Words extracted: {text_words:,}")
            print(f"     - Average word length: {text_length/text_words:.1f} chars" if text_words > 0 else "")
            
            return text
        
        # For images, Docling will handle OCR
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp']:
            print(f"  🔄 Step 1: Initializing OCR pipeline for image...")
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            
            converter = DocumentConverter()
            print(f"     ✓ OCR converter initialized")
            
            print(f"  🔄 Step 2: Running OCR on image...")
            print(f"     (This may take a moment for large images)")
            result = converter.convert(file_path)
            print(f"     ✓ OCR processing complete")
            
            print(f"  🔄 Step 3: Extracting text from OCR results...")
            text = result.document.export_to_markdown()
            text_length = len(text)
            text_words = len(text.split())
            
            if not text or not text.strip():
                raise ValueError(f"No text extracted from image {filename}")
            
            print(f"  ✓ OCR Text Extraction Complete:")
            print(f"     - Characters extracted: {text_length:,}")
            print(f"     - Words extracted: {text_words:,}")
            
            return text
        
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    except Exception as e:
        # Fallback to pypdf for PDF files
        if ext == '.pdf':
            print(f"⚠ Docling failed, using fallback for PDF: {e}")
            return extract_text_from_pdf_fallback(file_path)
        raise Exception(f"Failed to extract text from {filename}: {str(e)}")


def extract_text_from_pdf_fallback(file_path: str) -> str:
    """Fallback PDF text extraction using pypdf"""
    print(f"     🔄 Using pypdf fallback method...")
    reader = PdfReader(file_path)
    num_pages = len(reader.pages)
    print(f"     - Total pages: {num_pages}")
    
    text = ""
    for i, page in enumerate(reader.pages, 1):
        page_text = page.extract_text()
        text += page_text + "\n"
        if i % 10 == 0 or i == num_pages:
            print(f"     - Processed page {i}/{num_pages}")
    
    text_length = len(text)
    text_words = len(text.split())
    print(f"     ✓ Fallback extraction complete:")
    print(f"       - Characters: {text_length:,}")
    print(f"       - Words: {text_words:,}")
    return text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks with overlap
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    print(f"  🔄 Step 4: Chunking text...")
    print(f"     - Chunk size: {chunk_size} characters")
    print(f"     - Overlap: {overlap} characters")
    print(f"     - Input text length: {len(text):,} characters")
    
    if not text.strip():
        print(f"     ⚠ Empty text, no chunks created")
        return []
    
    chunks = []
    start = 0
    sentence_breaks = 0
    hard_breaks = 0
    
    while start < len(text):
        end = start + chunk_size
        found_sentence_break = False
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings
            for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                last_punct = text.rfind(punct, start, end)
                if last_punct != -1:
                    end = last_punct + 1
                    sentence_breaks += 1
                    found_sentence_break = True
                    break
        
        if not found_sentence_break:
            hard_breaks += 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    avg_chunk_size = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
    print(f"  ✓ Chunking Complete:")
    print(f"     - Total chunks created: {len(chunks)}")
    print(f"     - Average chunk size: {avg_chunk_size:.0f} characters")
    print(f"     - Sentence boundary breaks: {sentence_breaks}")
    print(f"     - Hard breaks (no sentence found): {hard_breaks}")
    print(f"     - Chunk size range: {min(len(c) for c in chunks) if chunks else 0} - {max(len(c) for c in chunks) if chunks else 0} characters")
    
    return chunks

