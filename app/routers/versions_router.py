"""
Version management endpoints
"""
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse

from app.routers.datamodel import VersionCreateRequest
from app.utilities import dc_logger
from app.utilities.constants import Constants
from app.utilities.env_util import EnvironmentVariableRetriever
from app.utilities.db_utilities.sqlite_implementation import SQLiteImplement
from app.utilities.db_utilities.models import Project, Version, ProjectPermission, User
from app.services.auth_service import AuthService

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})

router = APIRouter(
    prefix="/versions",
    tags=["Versions"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "not found"}}
)

# Initialize SQLite client
sqlite_config = Constants.fetch_constant("sqlite_db")
sqlite_client = SQLiteImplement(
    db_path=sqlite_config["db_path"],
    max_pool_size=sqlite_config["max_pool_size"]
)


@router.post("/")
async def create_version(
    request: VersionCreateRequest,
    current_user: User = Depends(AuthService.get_current_user)
):
    """Create a new version under a project (requires owner/editor permission)"""
    logger.info(f"Creating version: {request.version_name} for project: {request.project_id}")
    try:
        user_id = current_user.id
        
        # Check if project exists using ORM
        project = sqlite_client.get_by_id(Project, request.project_id)
        
        if not project:
            logger.warning(f"Project with id '{request.project_id}' not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "status": "Failed",
                    "message": f"Project with id '{request.project_id}' not found"
                }
            )
        
        # Check user has permission (owner or editor)
        user_perms = sqlite_client.get_all(
            ProjectPermission,
            filters={"project_id": request.project_id, "user_id": user_id}
        )
        
        if not user_perms or user_perms[0].permission_level not in ['owner', 'editor']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "status": "Failed",
                    "message": "You don't have permission to create versions for this project"
                }
            )
        
        # Check if version name already exists for this project
        existing_versions = sqlite_client.get_all(
            Version, 
            filters={"project_id": request.project_id, "version_name": request.version_name}
        )
        
        if existing_versions:
            logger.warning(f"Version '{request.version_name}' already exists for project '{request.project_id}'")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "status": "Failed",
                    "message": f"Version '{request.version_name}' already exists for this project"
                }
            )
        
        # Create new version using ORM model
        new_version = Version(
            project_id=request.project_id,
            version_name=request.version_name,
            description=request.description,
            is_active=request.is_active if request.is_active is not None else True,
            no_documents=0,
            no_test_cases=0
        )
        
        version_id = sqlite_client.create(new_version)
        
        logger.info(f"Created version in SQLite with UUID: {version_id}")
        return {
            "status": "Success",
            "version_id": version_id,
            "message": f"Version '{request.version_name}' created successfully"
        }
    
    except HTTPException:
        raise
    except Exception as exe:
        logger.exception("Failed to create version")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(exe)}
        )


@router.get("/project/{project_id}")
async def get_versions(project_id: int, current_user: User = Depends(AuthService.get_current_user)):
    """Get all versions for a project (requires permission)"""
    user_id = current_user.id
    logger.info(f"Fetching versions for project: {project_id}")
    try:
        # Check if project exists using ORM
        project = sqlite_client.get_by_id(Project, project_id)
        
        if not project:
            logger.warning(f"Project with id '{project_id}' not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "status": "Failed",
                    "message": f"Project with id '{project_id}' not found"
                }
            )
        
        # Check user has permission
        user_perms = sqlite_client.get_all(
            ProjectPermission,
            filters={"project_id": project_id, "user_id": user_id}
        )
        if not user_perms:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "status": "Failed",
                    "message": "You don't have permission to view versions for this project"
                }
            )
        
        # Get all versions for this project using ORM
        versions = sqlite_client.get_all(Version, filters={"project_id": project_id}, order_by=Version.created_at.desc())
        
        response_versions = []
        for version in versions:
            version_data = {
                "versionId": version.id,
                "versionName": version.version_name,
                "description": version.description or "",
                "isActive": version.is_active,
                "noDocuments": version.no_documents,
                "noTestCases": version.no_test_cases,
                "createdAt": version.created_at.isoformat() if version.created_at else None,
                "updatedAt": version.updated_at.isoformat() if version.updated_at else None
            }
            response_versions.append(version_data)
        
        logger.info(f"Fetched {len(response_versions)} versions for project {project_id}")
        return JSONResponse(
            content={
                "status": "Success",
                "projectId": project_id,
                "projectName": project.project_name,
                "versions": response_versions
            },
            status_code=status.HTTP_200_OK
        )
    
    except HTTPException:
        raise
    except Exception as exe:
        logger.exception("Failed to fetch versions")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(exe)}
        )


@router.get("/{version_id}")
async def get_version_detail(version_id: int, current_user: User = Depends(AuthService.get_current_user)):
    """Get detailed information about a specific version (requires authentication)"""
    user_id = current_user.id
    logger.info(f"Fetching details for version: {version_id}")
    try:
        # Get version using ORM
        version = sqlite_client.get_by_id(Version, version_id)
        
        if not version:
            logger.warning(f"Version with id '{version_id}' not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "status": "Failed",
                    "message": f"Version with id '{version_id}' not found"
                }
            )
        
        # Check user has permission to view project
        user_perms = sqlite_client.get_all(
            ProjectPermission,
            filters={"project_id": version.project_id, "user_id": user_id}
        )
        if not user_perms:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "status": "Failed",
                    "message": "You don't have permission to view this version"
                }
            )
        
        # Get project details
        project = sqlite_client.get_by_id(Project, version.project_id)
        
        version_detail = {
            "versionId": version.id,
            "projectId": version.project_id,
            "projectName": project.project_name if project else "",
            "versionName": version.version_name,
            "description": version.description or "",
            "isActive": version.is_active,
            "noDocuments": version.no_documents,
            "noTestCases": version.no_test_cases,
            "createdAt": version.created_at.isoformat() if version.created_at else None,
            "updatedAt": version.updated_at.isoformat() if version.updated_at else None
        }
        
        return JSONResponse(
            content={"status": "Success", "version": version_detail},
            status_code=status.HTTP_200_OK
        )
    
    except HTTPException:
        raise
    except Exception as exe:
        logger.exception("Failed to fetch version details")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(exe)}
        )


@router.put("/{version_id}")
async def update_version(version_id: int, request: VersionCreateRequest):
    """Update version details"""
    logger.info(f"Updating version: {version_id}")
    try:
        # Check if version exists using ORM
        version = sqlite_client.get_by_id(Version, version_id)
        
        if not version:
            logger.warning(f"Version with id '{version_id}' not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "status": "Failed",
                    "message": f"Version with id '{version_id}' not found"
                }
            )
        
        # Update version using ORM
        update_data = {
            "version_name": request.version_name,
            "description": request.description,
            "is_active": request.is_active if request.is_active is not None else True
        }
        sqlite_client.update(Version, version_id, update_data)
        
        logger.info(f"Updated version {version_id}")
        return {
            "status": "Success",
            "message": f"Version updated successfully"
        }
    
    except HTTPException:
        raise
    except Exception as exe:
        logger.exception("Failed to update version")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(exe)}
        )

