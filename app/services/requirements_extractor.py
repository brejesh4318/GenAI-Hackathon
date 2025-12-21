import json
from pathlib import Path
from typing import List, Dict, Tuple
from app.utilities.helper import Helper
from app.services.llm_helpers import extract_requirements_llm_with_context
from app.utilities import dc_logger

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})


class RequirementsExtractor:
    """
    Extract structured requirements from specification documents.
    
    This class handles:
    - Document parsing (PDF, DOCX, TXT, MD)
    - Page-level text extraction
    - Chunking large documents for LLM processing
    - LLM-based requirement identification
    - JSON export for requirements and raw pages
    """
    
    def __init__(self, pages_per_chunk: int = 5, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize the requirements extractor.
        
        Args:
            pages_per_chunk (int): Number of pages to process in each LLM call (default: 5)
            model_name (str): Gemini model to use for extraction (default: gemini-2.0-flash-exp)
        """
        self.pages_per_chunk = pages_per_chunk
        self.model_name = model_name
        self.helper = Helper()
        logger.info(f"Initialized RequirementsExtractor with {pages_per_chunk} pages/chunk, model: {model_name}")
    
    def extract_from_file(self, file_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract requirements from a specification document file.
        
        Args:
            file_path (str): Path to document file (PDF, DOCX, TXT, MD)
        
        Returns:
            Tuple[List[Dict], List[Dict]]: 
                - requirements: List of requirement dicts with requirement_id, text, source_page
                - raw_pages: List of page dicts with page_no and raw_text
        
        Example:
            >>> extractor = RequirementsExtractor()
            >>> reqs, pages = extractor.extract_from_file("specification.pdf")
            >>> print(f"Found {len(reqs)} requirements across {len(pages)} pages")
        """
        logger.info(f"Starting requirements extraction from: {file_path}")
        
        # Extract raw page texts using Docling helper
        pages = self.helper.extract_doc_pages(file_path)
        logger.info(f"Extracted {len(pages)} pages from document")
        
        # Extract requirements from pages
        requirements, raw_pages = self.extract_from_pages(pages)
        
        logger.info(f"Extraction complete: {len(requirements)} requirements found across {len(raw_pages)} pages")
        return requirements, raw_pages
    
    def extract_from_pages(self, pages: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract requirements from a list of page texts.
        
        Args:
            pages (List[str]): List of page texts (one string per page)
        
        Returns:
            Tuple[List[Dict], List[Dict]]:
                - requirements: List of requirement dicts
                - raw_pages: List of page dicts with page_no and raw_text
        """
        logger.info(f"Processing {len(pages)} pages for requirement extraction")
        
        # Create raw pages structure (preserve original page texts)
        raw_pages = [
            {"page_no": idx + 1, "raw_text": page_text}
            for idx, page_text in enumerate(pages)
        ]
        
        # Chunk pages for LLM processing
        chunks = self._chunk_pages(pages, self.pages_per_chunk)
        logger.info(f"Created {len(chunks)} chunks ({self.pages_per_chunk} pages/chunk)")
        
        # Extract requirements from each chunk
        all_requirements = []
        current_req_id = 1
        
        for chunk_idx, (chunk_text, page_numbers) in enumerate(chunks):
            logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunks)} (pages {page_numbers[0]}-{page_numbers[-1]})")
            
            # Call LLM to extract requirements from this chunk
            chunk_requirements = self._extract_requirements_via_llm(
                chunk_text, 
                page_numbers,
                current_req_id
            )
            
            # Accumulate requirements
            all_requirements.extend(chunk_requirements)
            current_req_id += len(chunk_requirements)
            
            logger.info(f"Extracted {len(chunk_requirements)} requirements from chunk {chunk_idx + 1}")
        
        logger.info(f"Total requirements extracted: {len(all_requirements)}")
        return all_requirements, raw_pages
    
    def _chunk_pages(self, pages: List[str], pages_per_chunk: int = 10) -> List[Tuple[str, List[int]]]:
        """
        Split pages into chunks for LLM processing with page annotations.
        
        Args:
            pages (List[str]): List of page texts
            pages_per_chunk (int): Number of pages per chunk (default: 5)
        
        Returns:
            List[Tuple[str, List[int]]]: List of (chunk_text, page_numbers)
                - chunk_text: Annotated text with page markers
                - page_numbers: List of page numbers in this chunk (1-indexed)
        
        Example:
            >>> pages = ["Page 1 content", "Page 2 content", "Page 3 content"]
            >>> chunks = extractor._chunk_pages(pages, pages_per_chunk=2)
            >>> print(len(chunks))  # 2 chunks
            >>> print(chunks[0][1])  # [1, 2]
        """
        chunks = []
        
        for i in range(0, len(pages), pages_per_chunk):
            # Get slice of pages for this chunk
            chunk_pages = pages[i:i + pages_per_chunk]
            
            # Calculate page numbers (1-indexed)
            page_numbers = list(range(i + 1, i + len(chunk_pages) + 1))
            
            # Prepare annotated chunk text
            chunk_text = self._prepare_chunk_text(chunk_pages, page_numbers)
            
            chunks.append((chunk_text, page_numbers))
        
        return chunks
    
    def _prepare_chunk_text(self, pages: List[str], page_numbers: List[int]) -> str:
        """
        Prepare chunk text with page annotations.
        
        Args:
            pages (List[str]): List of page texts for this chunk
            page_numbers (List[int]): Corresponding page numbers (1-indexed)
        
        Returns:
            str: Annotated chunk text with page markers
        
        Format:
            === PAGE 1 ===
            <page 1 content>
            === PAGE 2 ===
            <page 2 content>
        """
        annotated_sections = []
        
        for page_text, page_no in zip(pages, page_numbers):
            # Add page marker
            section = f"=== PAGE {page_no} ===\n{page_text}"
            annotated_sections.append(section)
        
        # Join all sections with double newline separator
        chunk_text = "\n\n".join(annotated_sections)
        return chunk_text
    
    def _extract_requirements_via_llm(
        self, 
        chunk_text: str, 
        page_numbers: List[int],
        start_req_id: int = 1
    ) -> List[Dict]:
        """
        Extract requirements from a chunk using LLM.
        
        Args:
            chunk_text (str): Annotated chunk text with page markers
            page_numbers (List[int]): Page numbers in this chunk
            start_req_id (int): Starting requirement ID for sequential numbering
        
        Returns:
            List[Dict]: List of requirement dicts with:
                - requirement_id (str): e.g., "REQ-001"
                - text (str): Exact requirement sentence
                - source_page (int): Page number where requirement was found
        """
        try:
            # Call LLM helper function
            requirements = extract_requirements_llm_with_context(
                chunk_text=chunk_text,
                start_req_id=start_req_id,
                model_name=self.model_name
            )
            
            # Validate that source_page is within expected range
            for req in requirements:
                if req['source_page'] not in page_numbers:
                    logger.warning(
                        f"Requirement {req['requirement_id']} references page {req['source_page']} "
                        f"which is outside chunk range {page_numbers}. Adjusting to first page."
                    )
                    req['source_page'] = page_numbers[0]
            
            return requirements
            
        except Exception as e:
            logger.error(f"Error extracting requirements from chunk: {str(e)}")
            # Return empty list to allow processing to continue
            return []
    
    def save_requirements_to_json(self, requirements: List[Dict], path: str) -> None:
        """
        Save requirements to JSON file.
        
        Args:
            requirements (List[Dict]): List of requirement dicts
            path (str): Output JSON file path
        
        Format:
            [
                {
                    "requirement_id": "REQ-001",
                    "text": "<exact requirement sentence>",
                    "source_page": 12
                }
            ]
        """
        try:
            # Ensure parent directory exists
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write JSON with pretty formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(requirements, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(requirements)} requirements to {path}")
            
        except Exception as e:
            logger.error(f"Error saving requirements to {path}: {str(e)}")
            raise
    
    def save_raw_pages_to_json(self, raw_pages: List[Dict], path: str) -> None:
        """
        Save raw page texts to JSON file.
        
        Args:
            raw_pages (List[Dict]): List of page dicts with page_no and raw_text
            path (str): Output JSON file path
        
        Format:
            [
                {
                    "page_no": 1,
                    "raw_text": "<original page text>"
                }
            ]
        """
        try:
            # Ensure parent directory exists
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write JSON with pretty formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(raw_pages, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(raw_pages)} pages to {path}")
            
        except Exception as e:
            logger.error(f"Error saving raw pages to {path}: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python requirements_extractor.py <file_path> [output_dir]")
        print("Example: python requirements_extractor.py specification.pdf ./output")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor
    extractor = RequirementsExtractor(pages_per_chunk=10)
    
    # Extract requirements and raw pages
    print(f"\n{'='*60}")
    print(f"Extracting requirements from: {file_path}")
    print(f"{'='*60}\n")
    
    requirements, raw_pages = extractor.extract_from_file(file_path)
    
    # Save to JSON files
    requirements_path = Path(output_dir) / "requirements.json"
    raw_pages_path = Path(output_dir) / "raw_pages.json"
    
    extractor.save_requirements_to_json(requirements, str(requirements_path))
    extractor.save_raw_pages_to_json(raw_pages, str(raw_pages_path))
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Extraction Complete!")
    print(f"{'='*60}")
    print(f"Total Pages: {len(raw_pages)}")
    print(f"Total Requirements: {len(requirements)}")
    print(f"\nOutput Files:")
    print(f"  - Requirements: {requirements_path}")
    print(f"  - Raw Pages: {raw_pages_path}")
    print(f"{'='*60}\n")
    
    # Display sample requirements
    if requirements:
        print(f"Sample Requirements (first 3):")
        for req in requirements[:3]:
            print(f"  {req['requirement_id']}: {req['text'][:80]}... (Page {req['source_page']})")
