from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from app.utilities.env_util import EnvironmentVariableRetriever
from app.utilities.helper import Helper
from app.services.req_extract_service import utils as req_utils
from app.services.req_extract_service.storage import RequirementsStorage
from app.services.req_extract_service import page_processor
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
        
        # Initialize storage handler
        self.storage = RequirementsStorage(
            mongo_client=self.mongo_client,
            requirements_collection=self.requirements_collection,
            document_pages_collection=self.document_pages_collection
        )
        
        logger.info(f"Initialized RequirementsExtractor with {pages_per_chunk} pages/chunk, model: {model_name}")
    
    def extract_and_store(
        self, 
        file_path: str,
        project_id: str,
        version_id: str,
        document_name: Optional[str] = None,
        previous_version_id: Optional[str] = None
    ) -> Dict:
        """
        Extract requirements from document and store in MongoDB with version-aware diffing.
        
        Args:
            file_path (str): Path to document file (PDF, DOCX, TXT, MD)
            project_id (str): UUID from SQLite projects table
            version_id (str): UUID from SQLite versions table
            document_name (str): Optional document name (defaults to filename)
            previous_version_id (str): UUID of previous version for diffing (None for V1)
        
        Returns:
            Dict: {
                "success": bool,
                "document_pages_id": str,  # MongoDB ObjectId
                "requirements_ids": List[str],  # List of MongoDB ObjectIds
                "total_pages": int,
                "total_requirements": int,
                "diff": Dict,  # Diff summary if previous_version_id provided
                "generate_tests_for": List[Dict],  # Requirements needing test generation
                "message": str
            }
        """
        try:
            logger.info(f"Starting extraction for project_id={project_id}, version_id={version_id}, "
                       f"previous_version_id={previous_version_id}")
            
            # Default document name to filename
            if not document_name:
                document_name = Path(file_path).name
            
            # Extract raw page texts
            pages = self.helper.extract_doc_pages(file_path)
            logger.info(f"Extracted {len(pages)} pages from document: {document_name}")
            
            # Create raw pages with per-page hashing
            raw_pages_with_hash = self._create_hashed_pages(pages)
            
            # Store raw pages in MongoDB
            doc_pages_id = self.storage.store_document_pages(
                project_id=project_id,
                version_id=version_id,
                document_name=document_name,
                pages=raw_pages_with_hash
            )
            logger.info(f"Stored document pages with ID: {doc_pages_id}")
            
            # Extract requirements from pages (with page diffing if previous version exists)
            requirements = page_processor.process_pages_for_requirements(
                project_id=project_id,
                version_id=version_id,
                pages=pages,
                hashed_pages=raw_pages_with_hash,
                previous_version_id=previous_version_id,
                storage_handler=self.storage,
                helper=self.helper,
                model_name=self.model_name,
                pages_per_chunk=self.pages_per_chunk,
                page_filter_func=lambda page_dict, prev_ver_id, project_id, version_id: self._should_process_page(
                    project_id,
                    version_id,
                    page_dict["page_hash"],
                    page_dict["page_no"],
                    prev_ver_id
                )
            )
            logger.info(f"Extracted {len(requirements)} requirements")
            
            # Compare with previous version (if exists)
            if previous_version_id:
                diff = self._compare_requirements_with_previous(
                    requirements, project_id, previous_version_id
                )
                
                # Store only new/modified requirements
                requirements_to_store = diff["new"] + diff["modified"]
                requirement_ids = self.storage.store_requirements(
                    project_id=project_id,
                    version_id=version_id,
                    document_name=document_name,
                    requirements=requirements_to_store
                )
                
                # Mark removed requirements as obsolete
                self.storage.mark_requirements_obsolete(diff["removed"], version_id)
                
                logger.info(f"Stored {len(requirement_ids)} new/modified requirements")
                
                return {
                    "success": True,
                    "document_pages_id": doc_pages_id,
                    "requirements_ids": requirement_ids,
                    "total_pages": len(pages),
                    "total_requirements": len(requirements),
                    "diff": diff,
                    "generate_tests_for": requirements_to_store,
                    "message": f"Successfully extracted {len(requirements)} requirements: "
                              f"{len(diff['new'])} new, {len(diff['modified'])} modified, "
                              f"{len(diff['unchanged'])} unchanged, {len(diff['removed'])} removed"
                }
            else:
                # V1: Store all requirements
                requirement_ids = self.storage.store_requirements(
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
                    "generate_tests_for": requirements,
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
        Create page dictionaries with normalized text and SHA-256 hashes.
        
        Args:
            pages (List[str]): List of raw page texts
        
        Returns:
            List[Dict]: List of page dicts with page_no, raw_text, normalized_text, page_hash
        """
        return req_utils.create_hashed_pages(pages, self.helper.normalize_text)
    
    def _should_process_page(
        self, 
        project_id: str, 
        version_id: str,
        page_hash: str, 
        page_no: int, 
        previous_version_id: Optional[str]
    ) -> bool:
        """
        Determine if a page needs LLM processing based on hash comparison.
        
        Args:
            page_hash (str): SHA-256 hash of current page
            page_no (int): Page number
            previous_version_id (str): Previous version UUID (None for V1)
        
        Returns:
            bool: True if page should be sent to LLM, False if unchanged
        """
        # For V1 (no previous version), process all pages
        if not previous_version_id:
            return True
        
        try:
            # Get previous version's document pages
            prev_docs = self.mongo_client.read(
                self.document_pages_collection,
                {"project_id": project_id, "version_id": previous_version_id}
            )
            
            if not prev_docs:
                logger.warning(f"No previous document pages found for version {previous_version_id}")
                return True
            
            # Use utility function for comparison
            prev_pages = prev_docs[0].get("pages", [])
            return req_utils.should_process_page(page_hash, page_no, prev_pages)
                
        except Exception as e:
            logger.error(f"Error checking page hash: {str(e)}")
            # On error, process the page to be safe
            return True
    
    def _compare_requirements_with_previous(
        self,
        current_requirements: List[Dict],
        project_id: str,
        previous_version_id: str
    ) -> Dict[str, List[Dict]]:
        """
        Compare current requirements with previous version using hashes.
        
        Args:
            current_requirements (List[Dict]): New requirements with hashes
            project_id (str): SQLite project UUID
            previous_version_id (str): Previous version UUID
            previous_version_id (str): Previous version UUID
        
        Returns:
            Dict: {
                "unchanged": List[Dict],  # Reuse existing tests
                "new": List[Dict],        # Generate tests
                "modified": List[Dict],   # Regenerate tests
                "removed": List[Dict]     # Mark tests obsolete
            }
        """
        logger.info(f"Comparing requirements with version {previous_version_id}")
        
        try:
            # Get previous requirements
            prev_reqs = self.storage.get_requirements_by_version(project_id, previous_version_id)
            
            # Use utility function for comparison
            return req_utils.compare_requirements_by_hash(current_requirements, prev_reqs)
            
        except Exception as e:
            logger.error(f"Error comparing requirements: {str(e)}", exc_info=True)
            # On error, treat all as new
            return {
                "unchanged": [],
                "new": current_requirements,
                "modified": [],
                "removed": []
            }
    
    # --- Utility Methods for Backward Compatibility ----------------
    
    # def extract_from_file(self, file_path: str) -> Tuple[str, None]:
    #     """
    #     DEPRECATED: Legacy method for backward compatibility with graph_pipeline.
    #     Returns markdown text and None for images (to match DocumentParser interface).
        
    #     For new code, use extract_and_store() instead.
        
    #     Args:
    #         file_path (str): Path to document file
        
    #     Returns:
    #         Tuple[str, None]: (markdown_text, None)
    #     """
    #     logger.warning("extract_from_file() is deprecated. Use extract_and_store() for MongoDB persistence.")
        
    #     try:
    #         pages = self.helper.extract_doc_pages(file_path)
    #         # Join all pages with separator
    #         markdown = "\n\n".join([f"=== PAGE {i+1} ===\n{page}" for i, page in enumerate(pages)])
    #         return markdown, None
    #     except Exception as e:
    #         logger.error(f"Error in extract_from_file: {str(e)}", exc_info=True)
    #         return "", None
    
    # def get_document_pages_by_version(self, version_id: str) -> List[Dict]:
    #     """Retrieve document pages for a specific version."""
    #     return self.storage.get_document_pages_by_version(version_id)
    
    # def get_requirements_by_version(self, version_id: str) -> List[Dict]:
    #     """Retrieve requirements for a specific version."""
    #     return self.storage.get_requirements_by_version(version_id)
    
    # def compare_document_versions(self, old_version_id: str, new_version_id: str) -> Dict:
    #     """
    #     Compare document pages between two versions (placeholder for future implementation).
        
    #     Args:
    #         old_version_id (str): Previous version UUID
    #         new_version_id (str): New version UUID
        
    #     Returns:
    #         Dict: Comparison results with added/removed/modified pages
    #     """
    #     logger.info(f"Version comparison placeholder - to be implemented later")
    #     # TODO: Implement version comparison logic using page hashes
    #     return {
    #         "status": "not_implemented",
    #         "message": "Version comparison will be implemented in future release"
    #     }


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
        version_id=str(version_id)  ,
        document_name=document_name
    )
    
    print(f"\n{'='*50}")
    print(f"Status: {'✓ Success' if result['success'] else '✗ Failed'}")
    print(f"Pages: {result['total_pages']} | Requirements: {result['total_requirements']}")
    print(f"Document ID: {result['document_pages_id']}")
    print(f"{result['message']}")
    print(f"{'='*50}\n")

