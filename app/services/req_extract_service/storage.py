"""
Storage operations for requirements extraction service.
Handles all MongoDB interactions for documents, pages, and requirements.
"""
from typing import List, Dict
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
        project_id: str,
        version_id: str,
        document_name: str,
        requirements: List[Dict]
    ) -> List[str]:
        """
        Store extracted requirements in MongoDB with hash-based identity.
        
        Args:
            project_id (str): SQLite project UUID
            version_id (str): SQLite version UUID
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
                "req_id": req.get("req_id"),
                "text": req.get("text"),
                "normalized_text": req.get("normalized_text"),
                "requirement_hash": req.get("requirement_hash"),
                "source_page": req.get("source_page"),
                "status": req.get("status", "active"),
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            enriched_requirements.append(enriched_req)
        
        # Bulk insert
        inserted_ids = self.mongo_client.insert_many(self.requirements_collection, enriched_requirements)
        logger.info(f"Bulk inserted {len(inserted_ids)} requirements")
        
        return [str(obj_id) for obj_id in inserted_ids]
    
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
    
    def get_requirements_by_version(self, project_id: str, version_id: str) -> List[Dict]:
        """
        Retrieve requirements for a specific project and version.
        
        Args:
            project_id (str): SQLite project UUID
            version_id (str): SQLite version UUID
        
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
    
    def mark_requirements_obsolete(self, removed_requirements: List[Dict], version_id: str):
        """
        Mark removed requirements as obsolete.
        
        Args:
            removed_requirements (List[Dict]): Requirements that no longer exist
            version_id (str): Current version ID
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
                    {"$set": {"status": "obsolete", "updated_at": datetime.now()}}
                )
            except Exception as e:
                logger.error(f"Error marking requirement obsolete: {str(e)}")
                continue
