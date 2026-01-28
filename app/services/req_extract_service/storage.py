"""
Storage operations for requirements extraction service.
Handles all MongoDB interactions for documents, pages, and requirements.
"""
from typing import List, Dict
from bson import ObjectId

from datetime import datetime
from app.utilities import dc_logger

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})


class RequirementsStorage:
    """Handles MongoDB storage operations for requirements and document pages."""
    
    def __init__(self, mongo_client, requirements_collection: str, document_pages_collection: str):
        """
        Initialize storage handler.
        
        Args:
            mongo_client: MongoDB client instance
            requirements_collection: Name of requirements collection
            document_pages_collection: Name of document pages collection
        """
        self.mongo_client = mongo_client
        self.requirements_collection = requirements_collection
        self.document_pages_collection = document_pages_collection
    
    def store_document_pages(
        self,
        project_id: int,
        version_id: int,
        document_name: str,
        pages: List[Dict]
    ) -> str:
        """
        Store raw document pages with hashes in MongoDB.
        
        Args:
            project_id (int): SQLite project ID
            version_id (int): SQLite version ID
            document_name (str): Name of the document
            pages (List[Dict]): List of page dicts with page_no, raw_text, normalized_text, page_hash
        
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
    
    def store_requirements(
        self,
        project_id: int,
        version_id: int,
        document_name: str,
        requirements: List[Dict]
    ) -> List[str]:
        """
        Store extracted requirements in MongoDB with hash-based identity.
        
        Args:
            project_id (int): SQLite project ID
            version_id (int): SQLite version ID
            document_name (str): Name of the document
            requirements (List[Dict]): List of requirement dicts with hashes
        
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
                "requirement_id": req.get("requirement_id"),  # Document ID (e.g., SFSYST1.1, REQ-001)
                "req_id": req.get("req_id"),  # Backend hash-based ID (REQ-a1b2c3d4)
                "text": req.get("text"),
                "normalized_text": req.get("normalized_text"),
                "requirement_hash": req.get("requirement_hash"),
                "source_page": req.get("source_page"),
                "status": req.get("status", "active"),
                "test_cases": [],  # Track linked test case IDs
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            enriched_requirements.append(enriched_req)
        
        # Bulk insert
        inserted_ids = self.mongo_client.insert_many(self.requirements_collection, enriched_requirements)
        logger.info(f"Bulk inserted {len(inserted_ids)} requirements")
        
        return [str(obj_id) for obj_id in inserted_ids]
    
    def get_document_pages_by_version(self, version_id: int) -> List[Dict]:
        """
        Retrieve document pages for a specific version.
        
        Args:
            version_id (int): SQLite version ID
        
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
    
    def get_requirements_by_version(self, project_id: int, version_id: int) -> List[Dict]:
        """
        Retrieve requirements for a specific project and version.
        
        Args:
            project_id (int): SQLite project ID
            version_id (int): SQLite version ID
        
        Returns:
            List[Dict]: List of requirement records
        """
        try:
            results = self.mongo_client.read(
                self.requirements_collection,
                {"project_id": project_id, "version_id": version_id}
            )
            return results
        except Exception as e:
            logger.error(f"Error retrieving requirements: {str(e)}", exc_info=True)
            return []
    
    def mark_requirements_obsolete(self, removed_requirements: List[Dict], version_id: int):
        """
        Mark removed requirements as obsolete.
        
        Args:
            removed_requirements (List[Dict]): Requirements that no longer exist
            version_id (int): Current version ID
        """
        if not removed_requirements:
            return
        
        logger.info(f"Marking {len(removed_requirements)} requirements as obsolete")
        
        for req in removed_requirements:
            try:
                # Update status to obsolete
                self.mongo_client.update_one(
                    self.requirements_collection,
                    {"_id": req["_id"]},
                    {"status": "obsolete", "updated_at": datetime.now()}
                )
            except Exception as e:
                logger.error(f"Error marking requirement obsolete: {str(e)}")
                continue
    
    def link_testcase_to_requirement(self, requirement_id: str, testcase_id: str):
        """
        Link a test case to a requirement by adding testcase_id to requirement's test_cases array.
        
        Note: Using direct collection access for $addToSet operation since update_one wraps with $set.
        
        Args:
            requirement_id (str): MongoDB ObjectId of requirement
            testcase_id (str): MongoDB ObjectId of test case
        """
        try:
            # Use direct collection access to handle $addToSet + $set together
            collection = self.mongo_client.database[self.requirements_collection]
            collection.update_one(
                {"_id": ObjectId(requirement_id)},
                {
                    "$addToSet": {"test_cases": testcase_id},
                    "$set": {"updated_at": datetime.now()}
                }
            )
            logger.debug(f"Linked testcase {testcase_id} to requirement {requirement_id}")
        except Exception as e:
            logger.error(f"Error linking testcase to requirement: {str(e)}")
    
    def get_requirements_without_testcases(self, project_id: int, version_id: int) -> List[Dict]:
        """
        Get requirements that don't have test cases yet.
        
        Args:
            project_id (int): SQLite project ID
            version_id (int): SQLite version ID
        
        Returns:
            List[Dict]: Requirements without test cases
        """
        try:
            results = self.mongo_client.read(
                self.requirements_collection,
                {
                    "project_id": project_id,
                    "version_id": version_id,
                    "status": "active",
                    "$or": [
                        {"test_cases": {"$exists": False}},
                        {"test_cases": {"$size": 0}}
                    ]
                }
            )
            return results
        except Exception as e:
            logger.error(f"Error retrieving requirements without testcases: {str(e)}", exc_info=True)
            return []
