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
        project_id: int,
        version_id: int,
        document_name: Optional[str] = None,
        previous_version_id: Optional[int] = None
    ) -> Dict:
        """
        Extract requirements from document and store in MongoDB with version-aware diffing.
        
        Args:
            file_path (str): Path to document file (PDF, DOCX, TXT, MD)
            project_id (int): ID from SQLite projects table
            version_id (int): ID from SQLite versions table
            document_name (str): Optional document name (defaults to filename)
            previous_version_id (int): ID of previous version for diffing (None for V1)
        
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
            # pages = self.helper.extract_doc_pages(file_path)#TODO:uncomment this line
            
            with open(r"D:\projects\GenAI-Hackathon\test-doc.json", "r") as f:
                import json
                pages = json.load(f)
                pages = pages["test-doc"]
            
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
            
            # Extract requirements from changed pages only
            requirements_from_changed_pages, unchanged_page_numbers = page_processor.process_pages_for_requirements(
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
            logger.info(f"Extracted {len(requirements_from_changed_pages)} requirements from changed pages")
            
            # Compare with previous version (if exists)
            if previous_version_id:
                # Get changed page numbers
                changed_page_numbers = [p["page_no"] for p in raw_pages_with_hash if p["page_no"] not in unchanged_page_numbers]
                
                diff = self._compare_requirements_with_previous(
                    requirements_from_changed_pages,
                    project_id,
                    previous_version_id,
                    changed_page_numbers
                )
                
                # Store only new/modified requirements from changed pages
                requirements_to_store = diff["new"] + diff["modified"]
                requirement_ids = self.storage.store_requirements(
                    project_id=project_id,
                    version_id=version_id,
                    document_name=document_name,
                    requirements=requirements_to_store
                )
                
                # Mark removed requirements as obsolete (only from changed pages)
                self.storage.mark_requirements_obsolete(diff["removed"], version_id)
                
                logger.info(f"Stored {len(requirement_ids)} new/modified requirements from changed pages")
                
                # Total requirements = unchanged + new + modified
                total_requirements = len(diff["unchanged"]) + len(diff["new"]) + len(diff["modified"])
                
                return {
                    "success": True,
                    "document_pages_id": doc_pages_id,
                    "requirements_ids": requirement_ids,
                    "total_pages": len(pages),
                    "total_requirements": total_requirements,
                    "diff": diff,
                    "generate_tests_for": requirements_to_store,
                    "message": f"Successfully processed {len(changed_page_numbers)} changed pages: "
                              f"{len(diff['new'])} new requirements, {len(diff['modified'])} modified, "
                              f"{len(diff['unchanged'])} unchanged (from {len(unchanged_page_numbers)} unchanged pages), "
                              f"{len(diff['removed'])} removed"
                }
            else:
                # V1: Store all requirements
                requirement_ids = self.storage.store_requirements(
                    project_id=project_id,
                    version_id=version_id,
                    document_name=document_name,
                    requirements=requirements_from_changed_pages
                )
                logger.info(f"Stored {len(requirement_ids)} requirements in MongoDB")
                
                return {
                    "success": True,
                    "document_pages_id": doc_pages_id,
                    "requirements_ids": requirement_ids,
                    "total_pages": len(pages),
                    "total_requirements": len(requirements_from_changed_pages),
                    "generate_tests_for": requirements_from_changed_pages,
                    "message": f"Successfully extracted {len(requirements_from_changed_pages)} requirements from {len(pages)} pages"
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
        project_id: int, 
        version_id: int,
        page_hash: str, 
        page_no: int, 
        previous_version_id: Optional[int]
    ) -> bool:
        """
        Determine if a page needs LLM processing based on hash comparison.
        
        Args:
            page_hash (str): SHA-256 hash of current page
            page_no (int): Page number
            previous_version_id (int): Previous version ID (None for V1)
        
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
        project_id: int,
        previous_version_id: int,
        changed_page_numbers: List[int]
    ) -> Dict[str, List[Dict]]:
        """
        Compare requirements from changed pages only with previous version.
        
        Args:
            current_requirements (List[Dict]): Requirements from changed pages
            project_id (int): SQLite project ID
            previous_version_id (int): Previous version ID
            changed_page_numbers (List[int]): Page numbers that changed
        
        Returns:
            Dict: {
                "unchanged": List[Dict],  # Requirements from unchanged pages (not stored again)
                "new": List[Dict],        # New requirements on changed pages
                "modified": List[Dict],   # Modified requirements on changed pages
                "removed": List[Dict]     # Removed requirements from changed pages
            }
        """
        logger.info(f"Comparing requirements on {len(changed_page_numbers)} changed pages with version {previous_version_id}")
        
        try:
            # Get ALL previous requirements
            all_prev_reqs = self.storage.get_requirements_by_version(project_id, previous_version_id)
            
            # Split previous requirements by page
            prev_reqs_on_changed_pages = [
                req for req in all_prev_reqs 
                if req.get("source_page") in changed_page_numbers
            ]
            prev_reqs_on_unchanged_pages = [
                req for req in all_prev_reqs 
                if req.get("source_page") not in changed_page_numbers
            ]
            
            logger.info(f"Previous version had {len(prev_reqs_on_changed_pages)} requirements on changed pages, "
                       f"{len(prev_reqs_on_unchanged_pages)} on unchanged pages")
            
            # Compare only requirements from changed pages
            diff = req_utils.compare_requirements_by_hash(current_requirements, prev_reqs_on_changed_pages)
            
            # Unchanged = requirements from unchanged pages (reference only, not stored)
            diff["unchanged"] = prev_reqs_on_unchanged_pages
            
            return diff
            
        except Exception as e:
            logger.error(f"Error comparing requirements: {str(e)}", exc_info=True)
            # On error, treat all as new
            return {
                "unchanged": [],
                "new": current_requirements,
                "modified": [],
                "removed": []
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
        version_id=str(version_id)  ,
        document_name=document_name
    )
    
    print(f"\n{'='*50}")
    print(f"Status: {'✓ Success' if result['success'] else '✗ Failed'}")
    print(f"Pages: {result['total_pages']} | Requirements: {result['total_requirements']}")
    print(f"Document ID: {result['document_pages_id']}")
    print(f"{result['message']}")
    print(f"{'='*50}\n")

