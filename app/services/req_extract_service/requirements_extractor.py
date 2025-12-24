import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from app.utilities.env_util import EnvironmentVariableRetriever
from app.utilities.helper import Helper
from app.services.req_extract_service.llm_helpers import extract_requirements_llm_with_context
from app.utilities import dc_logger
from app.utilities.db_utilities.mongo_implementation import MongoImplement
from app.utilities.constants import Constants
from app.utilities.singletons_factory import DcSingleton

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})


class RequirementsExtractor(metaclass=DcSingleton):
    """
    Production-ready requirements extractor with MongoDB persistence.
    
    Features:
    - Document parsing (PDF, DOCX, TXT, MD)
    - Page-level text extraction with SHA-256 hashing
    - LLM-based requirement identification
    - MongoDB storage for requirements and raw pages
    - Version-aware storage linked to SQLite projects/versions
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
        
        # Initialize MongoDB client
        mongo_config = Constants.fetch_constant("mongo_db")
        self.mongo_client = MongoImplement(
            connection_string=EnvironmentVariableRetriever.get_env_variable("MONGO_DB_URI"),
            db_name=mongo_config["db_name"],
            max_pool=mongo_config["max_pool_size"],
            server_selection_timeout=mongo_config["server_selection_timeout"]
        )
        
        collections = Constants.fetch_constant("mongo_collections")
        self.requirements_collection = collections["requirements"]
        self.document_pages_collection = collections["document_pages"]
        
        logger.info(f"Initialized RequirementsExtractor with {pages_per_chunk} pages/chunk, model: {model_name}")
    
    def extract_and_store(
        self, 
        file_path: str,
        project_id: str,
        version_id: str,
        document_name: Optional[str] = None
    ) -> Dict:
        """
        Extract requirements from document and store in MongoDB.
        
        Args:
            file_path (str): Path to document file (PDF, DOCX, TXT, MD)
            project_id (str): UUID from SQLite projects table
            version_id (str): UUID from SQLite versions table
            document_name (str): Optional document name (defaults to filename)
        
        Returns:
            Dict: {
                "success": bool,
                "document_pages_id": str,  # MongoDB ObjectId
                "requirements_ids": List[str],  # List of MongoDB ObjectIds
                "total_pages": int,
                "total_requirements": int,
                "message": str
            }
        """
        try:
            logger.info(f"Starting extraction and storage for project_id={project_id}, version_id={version_id}")
            
            # Default document name to filename
            if not document_name:
                document_name = Path(file_path).name
            
            # Extract raw page texts
            pages = self.helper.extract_doc_pages(file_path)
            logger.info(f"Extracted {len(pages)} pages from document: {document_name}")
            
            # Create raw pages with per-page hashing
            raw_pages_with_hash = self._create_hashed_pages(pages)
            
            # Store raw pages in MongoDB
            doc_pages_id = self._store_document_pages(
                project_id=project_id,
                version_id=version_id,
                document_name=document_name,
                pages=raw_pages_with_hash
            )
            logger.info(f"Stored document pages with ID: {doc_pages_id}")
            
            # Extract requirements from pages
            requirements = self._extract_requirements_from_pages(pages)
            logger.info(f"Extracted {len(requirements)} requirements")
            
            # Store requirements in MongoDB
            requirement_ids = self._store_requirements(
                project_id=project_id,
                version_id=version_id,
                document_name=document_name,
                requirements=requirements
            )
            logger.info(f"Stored {len(requirement_ids)} requirements in MongoDB")
            
            return {
                "success": True,
                "document_pages_id": doc_pages_id,
                "requirements_ids": requirement_ids,
                "total_pages": len(pages),
                "total_requirements": len(requirements),
                "message": f"Successfully extracted {len(requirements)} requirements from {len(pages)} pages"
            }
            
        except Exception as e:
            logger.error(f"Error in extract_and_store: {str(e)}", exc_info=True)
            return {
                "success": False,
                "document_pages_id": None,
                "requirements_ids": [],
                "total_pages": 0,
                "total_requirements": 0,
                "message": f"Extraction failed: {str(e)}"
            }
    
    def _create_hashed_pages(self, pages: List[str]) -> List[Dict]:
        """
        Create page dictionaries with SHA-256 hashes.
        
        Args:
            pages (List[str]): List of page texts
        
        Returns:
            List[Dict]: List of page dicts with page_no, raw_text, and page_hash
        """
        hashed_pages = []
        
        for idx, page_text in enumerate(pages):
            # Calculate SHA-256 hash for this page
            page_hash = hashlib.sha256(page_text.encode('utf-8')).hexdigest()
            
            hashed_pages.append({
                "page_no": idx + 1,
                "raw_text": page_text,
                "page_hash": page_hash
            })
        
        logger.debug(f"Created hashes for {len(hashed_pages)} pages")
        return hashed_pages
    
    def _store_document_pages(
        self,
        project_id: str,
        version_id: str,
        document_name: str,
        pages: List[Dict]
    ) -> str:
        """
        Store raw document pages with hashes in MongoDB.
        
        Args:
            project_id (str): SQLite project UUID
            version_id (str): SQLite version UUID
            document_name (str): Name of the document
            pages (List[Dict]): List of page dicts with page_no, raw_text, page_hash
        
        Returns:
            str: MongoDB ObjectId of inserted document
        """
        document_data = {
            "project_id": project_id,
            "version_id": version_id,
            "document_name": document_name,
            "pages": pages,
            "total_pages": len(pages),
            "created_at": datetime.now()
        }
        
        doc_id = self.mongo_client.insert_one(self.document_pages_collection, document_data)
        logger.info(f"Stored document pages: {document_name} with {len(pages)} pages")
        return doc_id
    
    def _store_requirements(
        self,
        project_id: str,
        version_id: str,
        document_name: str,
        requirements: List[Dict]
    ) -> List[str]:
        """
        Store extracted requirements in MongoDB.
        
        Args:
            project_id (str): SQLite project UUID
            version_id (str): SQLite version UUID
            document_name (str): Name of the document
            requirements (List[Dict]): List of requirement dicts
        
        Returns:
            List[str]: List of MongoDB ObjectIds
        """
        if not requirements:
            logger.warning("No requirements to store")
            return []
        
        # Add metadata to each requirement
        enriched_requirements = []
        for req in requirements:
            enriched_req = {
                "project_id": project_id,
                "version_id": version_id,
                "document_name": document_name,
                "requirement_id": req.get("requirement_id"),
                "text": req.get("text"),
                "source_page": req.get("source_page"),
                "created_at": datetime.now()
            }
            enriched_requirements.append(enriched_req)
        
        # Bulk insert
        inserted_ids = self.mongo_client.insert_many(self.requirements_collection, enriched_requirements)
        logger.info(f"Bulk inserted {len(inserted_ids)} requirements")
        
        return [str(obj_id) for obj_id in inserted_ids]
    
  
    def _extract_requirements_from_pages(self, pages: List[str]) -> List[Dict]:
        """
        Extract requirements from a list of page texts.
        
        Args:
            pages (List[str]): List of page texts (one string per page)
        
        Returns:
            List[Dict]: List of requirement dicts with requirement_id, text, source_page
        """
        logger.info(f"Processing {len(pages)} pages for requirement extraction")
        
        # Chunk pages for LLM processing
        chunks = self._chunk_pages(pages, self.pages_per_chunk)
        logger.info(f"Created {len(chunks)} chunks ({self.pages_per_chunk} pages/chunk)")
        
        # Extract requirements from each chunk
        all_requirements = []
        current_req_id = 1
        
        for chunk_idx, (chunk_text, page_numbers) in enumerate(chunks):
            logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunks)} (pages {page_numbers[0]}-{page_numbers[-1]})")
            
            try:
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
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx + 1}: {str(e)}", exc_info=True)
                # Continue processing remaining chunks
                continue
        
        logger.info(f"Total requirements extracted: {len(all_requirements)}")
        return all_requirements
    
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
        



    
    # --- Utility Methods for Backward Compatibility ----------------
    
    def extract_from_file(self, file_path: str) -> Tuple[str, None]:
        """
        DEPRECATED: Legacy method for backward compatibility with graph_pipeline.
        Returns markdown text and None for images (to match DocumentParser interface).
        
        For new code, use extract_and_store() instead.
        
        Args:
            file_path (str): Path to document file
        
        Returns:
            Tuple[str, None]: (markdown_text, None)
        """
        logger.warning("extract_from_file() is deprecated. Use extract_and_store() for MongoDB persistence.")
        
        try:
            pages = self.helper.extract_doc_pages(file_path)
            # Join all pages with separator
            markdown = "\n\n".join([f"=== PAGE {i+1} ===\n{page}" for i, page in enumerate(pages)])
            return markdown, None
        except Exception as e:
            logger.error(f"Error in extract_from_file: {str(e)}", exc_info=True)
            return "", None
    
    def get_document_pages_by_version(self, version_id: str) -> List[Dict]:
        """
        Retrieve document pages for a specific version.
        
        Args:
            version_id (str): SQLite version UUID
        
        Returns:
            List[Dict]: List of document page records
        """
        try:
            results = self.mongo_client.read(
                self.document_pages_collection,
                {"version_id": version_id}
            )
            return results
        except Exception as e:
            logger.error(f"Error retrieving document pages: {str(e)}", exc_info=True)
            return []
    
    def get_requirements_by_version(self, version_id: str) -> List[Dict]:
        """
        Retrieve requirements for a specific version.
        
        Args:
            version_id (str): SQLite version UUID
        
        Returns:
            List[Dict]: List of requirement records
        """
        try:
            results = self.mongo_client.read(
                self.requirements_collection,
                {"version_id": version_id}
            )
            return results
        except Exception as e:
            logger.error(f"Error retrieving requirements: {str(e)}", exc_info=True)
            return []
    
    def compare_document_versions(self, old_version_id: str, new_version_id: str) -> Dict:
        """
        Compare document pages between two versions (placeholder for future implementation).
        
        Args:
            old_version_id (str): Previous version UUID
            new_version_id (str): New version UUID
        
        Returns:
            Dict: Comparison results with added/removed/modified pages
        """
        logger.info(f"Version comparison placeholder - to be implemented later")
        # TODO: Implement version comparison logic using page hashes
        return {
            "status": "not_implemented",
            "message": "Version comparison will be implemented in future release"
        }


# CLI usage for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python requirements_extractor.py <file_path> <project_id> <version_id> [document_name]")
        print("Example: python requirements_extractor.py spec.pdf proj-uuid-123 ver-uuid-456")
        sys.exit(1)
    
    file_path = sys.argv[1]
    project_id = sys.argv[2]
    version_id = sys.argv[3]
    document_name = sys.argv[4] if len(sys.argv) > 4 else None
    
    extractor = RequirementsExtractor(pages_per_chunk=10)
    
    print(f"\nExtracting from: {file_path}")
    print(f"Project: {project_id}, Version: {version_id}\n")
    
    result = extractor.extract_and_store(
        file_path=file_path,
        project_id=project_id,
        version_id=version_id,
        document_name=document_name
    )
    
    print(f"\n{'='*50}")
    print(f"Status: {'✓ Success' if result['success'] else '✗ Failed'}")
    print(f"Pages: {result['total_pages']} | Requirements: {result['total_requirements']}")
    print(f"Document ID: {result['document_pages_id']}")
    print(f"{result['message']}")
    print(f"{'='*50}\n")

