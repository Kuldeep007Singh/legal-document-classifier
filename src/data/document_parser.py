import os
from typing import Optional
from pypdf import PdfReader

class PDFTextExtractor:
    """
    Extracts text content from PDF files.
    """
    
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from a single PDF file.
        
        :param pdf_path: Path to the PDF file
        :return: Extracted text as string, or None if extraction fails
        """
        try:
            if not os.path.exists(pdf_path):
                print(f"File not found: {pdf_path}")
                return None
            
            reader = PdfReader(pdf_path)
            text = ""
            
            # Extract text from all pages
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    text += page_text + "\n"
                except Exception as e:
                    print(f"Error extracting text from page {page_num} of {pdf_path}: {e}")
                    continue
            
            return text.strip() if text.strip() else None
            
        except Exception as e:
            print(f"Error processing PDF file {pdf_path}: {e}")
            return None

    def extract_text_from_multiple_pdfs(self, pdf_paths: list) -> dict:
        """
        Extract text from multiple PDF files.
        
        :param pdf_paths: List of PDF file paths
        :return: Dictionary with filepath as key and extracted text as value
        """
        results = {}
        
        for pdf_path in pdf_paths:
            print(f"Processing: {os.path.basename(pdf_path)}")
            text = self.extract_text_from_pdf(pdf_path)
            results[pdf_path] = text
            
        return results

if __name__ == "__main__":
    # Test the extractor
    extractor = PDFTextExtractor()
    
    # Test with a single PDF file (update path as needed)
    test_pdf = "path/to/your/test.pdf"  # Replace with actual PDF path for testing
    if os.path.exists(test_pdf):
        text = extractor.extract_text_from_pdf(test_pdf)
        print(f"Extracted text length: {len(text) if text else 0}")
        if text:
            print("First 500 characters:")
            print(text[:500])
