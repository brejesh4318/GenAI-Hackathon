"""
Requirements Extraction Router
API endpoints for standalone requirement extraction testing
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from sqlalchemy import and_
from typing import Optional
from app.services.req_extract_service.requirements_extractor import RequirementsExtractor
from datetime import datetime
from app.utilities.helper import Helper
from app.utilities.db_utilities.sqlite_implementation import SQLiteImplement
from app.utilities.db_utilities.models import Project, Version, User
from app.services.auth_service import AuthService
from app.utilities import dc_logger
import os

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})
router = APIRouter(prefix="/requirements", tags=["Requirements"])

# Initialize services
sqlite_client = SQLiteImplement()
auth_service = AuthService()
requirements_extractor = RequirementsExtractor(pages_per_chunk=10)


@router.post("/extract")
async def extract_requirements(
    project_id: str,
    version_id: Optional[str] = None,
    version_name: Optional[str] = None,
    previous_version_id: Optional[str] = None,
    file: UploadFile = File(...),
    current_user: User = Depends(auth_service.get_current_user)
):
    """
    Extract requirements from document and store in MongoDB.
    
    Automatically creates a new version if version_id is not provided.
    Automatically detects previous version for intelligent diffing (unless specified).
    
    Args:
        project_id: SQLite project ID (integer)
        version_id: SQLite version ID (optional - will auto-create if not provided)
        version_name: Name for new version (optional - defaults to auto-generated)
        previous_version_id: Previous version ID for diffing (optional - auto-detects if not provided)
        file: Document file (PDF, DOCX, TXT, MD)
    
    Returns:
        Extraction result with requirements and statistics
    """
    try:
        logger.info(f"Extracting requirements for project={project_id}, version={version_id}")
        
        # Verify project exists
        project = sqlite_client.get_by_id(Project, int(project_id))
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"status": "Failed", "message": f"Project {project_id} not found"}
            )
        
        # Auto-create version if not provided
        if not version_id:
            
            # Get existing versions count for this project
            with sqlite_client.get_session() as session:
                version_count = session.query(Version).filter(
                    Version.project_id == int(project_id)
                ).count()
            
            # Generate version name if not provided
            if not version_name:
                version_name = f"v{version_count + 1}.0"
            
            # Create new version
            new_version = Version(
                project_id=int(project_id),
                version_name=version_name,
                description=f"Auto-created from document: {file.filename}",
                is_active=True,
                no_documents=0,
                no_test_cases=0
            )
            
            version_id = sqlite_client.create(new_version)
            logger.info(f"Auto-created new version: {version_id} ({version_name}) for project {project_id}")
        else:
            # Verify provided version exists
            version = sqlite_client.get_by_id(Version, int(version_id))
            if not version or version.project_id != int(project_id):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"status": "Failed", "message": f"Version {version_id} not found for project {project_id}"}
                )
            logger.info(f"Using existing version: {version_id} ({version.version_name})")
        
        # Use provided previous_version_id or auto-detect if not provided
        if previous_version_id:
            # Verify provided previous version exists
            prev_version = sqlite_client.get_by_id(Version, int(previous_version_id))
            if not prev_version or prev_version.project_id != int(project_id):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"status": "Failed", "message": f"Previous version {previous_version_id} not found for project {project_id}"}
                )
            logger.info(f"Using provided previous version: {previous_version_id} ({prev_version.version_name})")
        else:
            # Auto-detect previous version by finding most recent version before current one
            with sqlite_client.get_session() as session:
                previous_version = session.query(Version).filter(
                    and_(
                        Version.project_id == int(project_id),
                        Version.id < int(version_id),
                        Version.id != int(version_id)
                    )
                ).order_by(Version.created_at.desc()).first()
                
                if previous_version:
                    previous_version_id = str(previous_version.id)
                    logger.info(f"Auto-detected previous version: {previous_version_id} ({previous_version.version_name})")
                else:
                    logger.info("No previous version found - this is the first version")
        
        # Save uploaded file temporarily
        file_content = await file.read()
        temp_path = Helper.save_file("/tmp", content=file_content, filename=file.filename)
        
        try:
            # Extract and store requirements
            result = requirements_extractor.extract_and_store(
                file_path=temp_path,
                project_id=project_id,
                version_id=version_id,
                previous_version_id=previous_version_id,
                document_name=file.filename
            )
            
            if result["success"]:
                # Retrieve stored requirements for response
                requirements = requirements_extractor.storage.get_requirements_by_version(
                    project_id, version_id
                )
                
                # Format requirements for display
                formatted_requirements = []
                for req in requirements:
                    formatted_requirements.append({
                        "internal_id": req.get("req_id"),
                        "document_id": req.get("document_requirement_id", "N/A"),
                        "text": req.get("text"),
                        "source_page": req.get("source_page"),
                        "status": req.get("status", "active"),
                        "requirement_hash": req.get("requirement_hash")
                    })
                
                response_data = {
                    "extraction_result": result,
                    "requirements": formatted_requirements,
                    "version_info": {
                        "version_id": version_id,
                        "version_name": sqlite_client.get_by_id(Version, int(version_id)).version_name,
                        "is_new_version": version_id and not version_id  # Will be updated below
                    },
                    "statistics": {
                        "total_pages": result["total_pages"],
                        "total_requirements": result["total_requirements"],
                        "has_diff": "diff" in result
                    }
                }
                
                if "diff" in result:
                    response_data["diff_summary"] = {
                        "unchanged": len(result["diff"]["unchanged"]),
                        "new": len(result["diff"]["new"]),
                        "modified": len(result["diff"]["modified"]),
                        "removed": len(result["diff"]["removed"])
                    }
                
                return {
                    "status": "Success",
                    "message": f"Extracted {result['total_requirements']} requirements from {result['total_pages']} pages",
                    "data": response_data
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={"status": "Failed", "message": result["message"]}
                )
        
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"Cleaned up temp file: {temp_path}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting requirements: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": f"Error extracting requirements: {str(e)}"}
        )


@router.get("/list")
async def list_requirements(
    project_id: str,
    version_id: str,
    current_user: User = Depends(auth_service.get_current_user)
):
    """
    List all requirements for a specific project and version.
    
    Args:
        project_id: SQLite project UUID
        version_id: SQLite version UUID
    
    Returns:
        List of requirements with metadata
    """
    try:
        logger.info(f"Listing requirements for project={project_id}, version={version_id}")
        
        # Verify project exists
        project = sqlite_client.get_by_id(Project, int(project_id))
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"status": "Failed", "message": f"Project {project_id} not found"}
            )
        
        # Get requirements
        requirements = requirements_extractor.storage.get_requirements_by_version(
            project_id, version_id
        )
        
        # Format response
        formatted_requirements = []
        for req in requirements:
            formatted_requirements.append({
                "internal_id": req.get("req_id"),
                "document_id": req.get("document_requirement_id", "N/A"),
                "text": req.get("text"),
                "source_page": req.get("source_page"),
                "status": req.get("status", "active"),
                "created_at": req.get("created_at"),
                "updated_at": req.get("updated_at")
            })
        
        return {
            "status": "Success",
            "message": f"Found {len(formatted_requirements)} requirements",
            "data": {
                "project_id": project_id,
                "version_id": version_id,
                "total_requirements": len(formatted_requirements),
                "requirements": formatted_requirements
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing requirements: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": f"Error listing requirements: {str(e)}"}
        )


@router.get("/compare")
async def compare_versions(
    project_id: str,
    version_id_1: str,
    version_id_2: str,
    current_user: User = Depends(auth_service.get_current_user)
):
    """
    Compare requirements between two versions.
    
    Args:
        project_id: SQLite project UUID
        version_id_1: First version UUID
        version_id_2: Second version UUID
    
    Returns:
        Comparison results showing differences
    """
    try:
        logger.info(f"Comparing requirements: {version_id_1} vs {version_id_2}")
        
        # Get requirements for both versions
        reqs_v1 = requirements_extractor.storage.get_requirements_by_version(
            project_id, version_id_1
        )
        reqs_v2 = requirements_extractor.storage.get_requirements_by_version(
            project_id, version_id_2
        )
        
        # Create hash maps
        v1_by_hash = {req["requirement_hash"]: req for req in reqs_v1}
        v2_by_hash = {req["requirement_hash"]: req for req in reqs_v2}
        
        # Find differences
        unchanged = []
        modified = []
        new = []
        removed = []
        
        for hash_val, req in v2_by_hash.items():
            if hash_val in v1_by_hash:
                unchanged.append({
                    "internal_id": req.get("req_id"),
                    "document_id": req.get("document_requirement_id", "N/A"),
                    "text": req.get("text")
                })
            else:
                new.append({
                    "internal_id": req.get("req_id"),
                    "document_id": req.get("document_requirement_id", "N/A"),
                    "text": req.get("text")
                })
        
        for hash_val, req in v1_by_hash.items():
            if hash_val not in v2_by_hash:
                removed.append({
                    "internal_id": req.get("req_id"),
                    "document_id": req.get("document_requirement_id", "N/A"),
                    "text": req.get("text")
                })
        
        return {
            "status": "Success",
            "message": "Version comparison completed",
            "data": {
                "project_id": project_id,
                "version_1": {
                    "version_id": version_id_1,
                    "total_requirements": len(reqs_v1)
                },
                "version_2": {
                    "version_id": version_id_2,
                    "total_requirements": len(reqs_v2)
                },
                "comparison": {
                    "unchanged": len(unchanged),
                    "new": len(new),
                    "removed": len(removed),
                    "unchanged_list": unchanged,
                    "new_list": new,
                    "removed_list": removed
                }
            }
        }
    
    except Exception as e:
        logger.error(f"Error comparing versions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": f"Error comparing versions: {str(e)}"}
        )
