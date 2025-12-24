"""
Project management endpoints
"""
import asyncio
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse

from app.routers.datamodel import ProjectCreateRequest
from app.utilities import dc_logger
from app.utilities.constants import Constants
from app.utilities.env_util import EnvironmentVariableRetriever
from app.utilities.db_utilities.mongo_implementation import MongoImplement
from app.utilities.db_utilities.sqlite_implementation import SQLiteImplement
from app.utilities.db_utilities.models import Project, Version, ProjectPermission, User
from app.utilities.helper import Helper
from app.services.auth_service import AuthService

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})

router = APIRouter(
    prefix="/projects",
    tags=["Projects"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "not found"}}
)

# Initialize database clients
mongo_client = MongoImplement(
    connection_string=EnvironmentVariableRetriever.get_env_variable("MONGO_DB_URI"),
    db_name=Constants.fetch_constant("mongo_db")["db_name"],
    max_pool=Constants.fetch_constant("mongo_db")["max_pool_size"],
    server_selection_timeout=Constants.fetch_constant("mongo_db")["server_selection_timeout"]
)

sqlite_config = Constants.fetch_constant("sqlite_db")
sqlite_client = SQLiteImplement(
    db_path=sqlite_config["db_path"],
    max_pool_size=sqlite_config["max_pool_size"]
)


@router.post("/")
async def create_project(
    request: ProjectCreateRequest,
    current_user: User = Depends(AuthService.get_current_user)
):
    """Create a new project (requires authentication)"""
    logger.info(f"Creating project: {request.project_name} by user {current_user.email}")
    try:
        user_id = current_user.id
        
        # Check if project name already exists using ORM
        existing = sqlite_client.get_all(Project, filters={"project_name": request.project_name})
        
        if existing:
            logger.warning(f"Project with name '{request.project_name}' already exists")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "status": "Failed",
                    "message": f"Project with name '{request.project_name}' already exists"
                }
            )
        
        # Create new project using ORM model
        new_project = Project(
            project_name=request.project_name,
            description=request.description,
            organization_id=request.organization_id,
            owner_id=user_id,  # Set owner to current user
            no_test_cases=0,
            no_documents=0
        )
        
        project_id = sqlite_client.create(new_project)
        
        # Grant owner permission to creator
        owner_perm = ProjectPermission(
            project_id=project_id,
            user_id=user_id,
            permission_level="owner",
            granted_by=user_id
        )
        sqlite_client.create(owner_perm)
        
        logger.info(f"Created project in SQLite with UUID: {project_id}")
        return {"status": "Success", "project_id": project_id}

    except HTTPException:
        raise
    except Exception as exe:
        logger.exception("Failed to create project")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(exe)}
        )


@router.get("/")
async def get_projects(current_user: User = Depends(AuthService.get_current_user)):
    """Get all projects (requires authentication)"""
    try:
        user_id = current_user.id
        # Get projects user has access to
        user_perms = sqlite_client.get_all(ProjectPermission, filters={"user_id": user_id})
        project_ids = [perm.project_id for perm in user_perms]
        
        # Filter projects by permission
        all_projects = sqlite_client.get_all(Project, order_by=Project.created_at.desc())
        projects = [p for p in all_projects if p.id in project_ids]
        logger.info(f"Fetching projects for authenticated user {user_id}")
        
        response_projects = []
        for project in projects:
            compliances = []
            # Get test cases from MongoDB using SQL project_id (UUID)
            test_cases = mongo_client.read("test_cases", {"project_id": project.id})
            for test_case in test_cases:
                if test_case.get("compliance_reference_standard") and test_case.get("compliance_reference_standard") not in compliances:
                    compliances.append(test_case.get("compliance_reference_standard"))
            
            # Get version count using ORM
            version_count = sqlite_client.count(Version, filters={"project_id": project.id})
            
            data = {
                "projectName": project.project_name,
                "projectId": project.id,
                "description": project.description or "",
                "TestCasesGenerated": project.no_test_cases,
                "documents": project.no_documents,
                "versions": version_count,
                "ComplianceReferenceStandards": compliances,
                "status": "active" if project.no_test_cases > 0 else "review" if project.no_documents > 0 else "completed",
                "UpdatedTime": Helper.time_saved_format(project.updated_at if project.updated_at else datetime.now())
            }
            response_projects.append(data)
        
        logger.info(f"Fetched {len(response_projects)} projects")
        return JSONResponse(content=response_projects, status_code=status.HTTP_200_OK)
    
    except Exception as exe:
        logger.error(f"Failed to fetch projects: {exe}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(exe)}
        )


@router.get("/{project_id}")
async def get_project(project_id: str, current_user: User = Depends(AuthService.get_current_user)):
    """Get project details by ID (requires authentication)"""
    user_id = current_user.id
    try:
        project = sqlite_client.get_by_id(Project, project_id)
        
        if not project:
            logger.warning(f"Project with id '{project_id}' not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"status": "Failed", "message": f"Project with id '{project_id}' not found"}
            )
        
        # Check user has permission
        user_perms = sqlite_client.get_all(
            ProjectPermission,
            filters={"project_id": project_id, "user_id": user_id}
        )
        if not user_perms:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"status": "Failed", "message": "You don't have permission to view this project"}
            )
        
        # Get compliance standards from test cases
        compliances = []
        test_cases = mongo_client.read("test_cases", {"project_id": project.id})
        for test_case in test_cases:
            if test_case.get("compliance_reference_standard") and test_case.get("compliance_reference_standard") not in compliances:
                compliances.append(test_case.get("compliance_reference_standard"))
        
        # Get version count
        version_count = sqlite_client.count(Version, filters={"project_id": project.id})
        
        project_data = {
            "projectId": project.id,
            "projectName": project.project_name,
            "description": project.description or "",
            "organizationId": project.organization_id,
            "testCasesGenerated": project.no_test_cases,
            "documents": project.no_documents,
            "versions": version_count,
            "complianceStandards": compliances,
            "createdAt": project.created_at.isoformat() if project.created_at else None,
            "updatedAt": project.updated_at.isoformat() if project.updated_at else None
        }
        
        return JSONResponse(content={"status": "Success", "project": project_data}, status_code=status.HTTP_200_OK)
    
    except HTTPException:
        raise
    except Exception as exe:
        logger.exception("Failed to fetch project")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(exe)}
        )

