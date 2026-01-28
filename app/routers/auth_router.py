"""
Authentication and User Management Routes
Handles user registration, login, and permission management
Uses SQLite ORM for user data and permissions
"""
from fastapi import APIRouter, HTTPException, status, Depends
from typing import Optional
from datetime import timedelta
from app.routers.datamodel import UserRegisterRequest, UserLoginRequest, ProjectPermissionRequest
from app.utilities import dc_logger
from app.services.auth_service import AuthService
from app.utilities.constants import Constants
from app.utilities.db_utilities.sqlite_implementation import SQLiteImplement
from app.utilities.db_utilities.models import User, ProjectPermission
from fastapi.responses import JSONResponse

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "notfound"}}
)

# Initialize SQLite client
sqlite_config = Constants.fetch_constant("sqlite_db")
sqlite_client = SQLiteImplement(
    db_path=sqlite_config["db_path"],
    max_pool_size=sqlite_config["max_pool_size"]
)
logger.info("SQLite client initialized for auth")

# Initialize AuthService
auth_service = AuthService()


@router.post("/register")
async def register_user(request: UserRegisterRequest):
    """Register a new user"""
    logger.info(f"Registering new user: {request.email}")
    
    try:
        # Check if user already exists
        existing = sqlite_client.get_all(User, filters={"email": request.email})
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "status": "Failed",
                    "message": f"User with email '{request.email}' already exists"
                }
            )
        
        # Hash password and create user
        hashed_pw = auth_service.get_password_hash(request.password)
        new_user = User(
            email=request.email,
            password_hash=hashed_pw,
            full_name=request.full_name
        )
        
        user_id = sqlite_client.create(new_user)
        
        if user_id:
            logger.info(f"User registered successfully: {user_id}")
            
            # Create access token
            access_token = auth_service.create_access_token(
                data={"sub": str(user_id), "email": request.email}
            )
            
            return JSONResponse(
                content={
                    "status": "Success",
                    "message": "User registered successfully",
                    "user": {
                        "id": user_id,
                        "email": request.email,
                        "full_name": request.full_name
                    },
                    "access_token": access_token,
                    "token_type": "bearer"
                },
                status_code=status.HTTP_201_CREATED
            )
        
    except HTTPException:
        raise
    except Exception as exe:
        logger.exception("Failed to register user")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(exe)}
        )


@router.post("/login")
async def login_user(request: UserLoginRequest):
    """Login user and return JWT token"""
    logger.info(f"Login attempt for user: {request.email}")
    
    try:
        # Get user from database
        users = sqlite_client.get_all(User, filters={"email": request.email})
        
        if not users:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"status": "Failed", "message": "Invalid email or password"}
            )
        
        user = users[0]
        
        # Check if user is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"status": "Failed", "message": "User account is inactive"}
            )
        
        # Verify password
        if not auth_service.verify_password(request.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"status": "Failed", "message": "Incorrect email or password"}
            )
        
        # Create access token
        access_token = auth_service.create_access_token(
            data={"sub": str(user.id), "email": user.email}
        )
        
        logger.info(f"User logged in successfully: {user.id}")
        
        return JSONResponse(
            content={
                "status": "Success",
                "message": "Login successful",
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "full_name": user.full_name
                },
                "access_token": access_token,
                "token_type": "bearer"
            }
        )
        
    except HTTPException:
        raise
    except Exception as exe:
        logger.exception("Failed to login user")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(exe)}
        )


@router.get("/me")
async def get_current_user_info(current_user: User = Depends(AuthService.get_current_user)):
    """Get current user information from token"""
    try:
        user_id = current_user.id
        user = sqlite_client.get_by_id(User, user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"status": "Failed", "message": "User not found"}
            )
        
        return JSONResponse(
            content={
                "status": "Success",
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "created_at": user.created_at.isoformat() if user.created_at else None
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as exe:
        logger.exception("Failed to get user info")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(exe)}
        )


@router.post("/project-permission")
async def grant_project_permission(
    request: ProjectPermissionRequest,
    current_user: User = Depends(AuthService.get_current_user)
):
    """
    Grant project access to another user
    Only project owners can grant permissions
    """
    logger.info(f"Granting permission for project {request.project_id}")
    
    try:
        granter_id = current_user.id
        
        # Check if granter has owner permission
        granter_perms = sqlite_client.get_all(
            ProjectPermission,
            filters={"project_id": request.project_id, "user_id": granter_id}
        )
        
        if not granter_perms or granter_perms[0].permission_level != 'owner':
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"status": "Failed", "message": "Only project owners can grant permissions"}
            )
        
        # Get target user by email
        target_users = sqlite_client.get_all(User, filters={"email": request.user_email})
        
        if not target_users:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"status": "Failed", "message": f"User {request.user_email} not found"}
            )
        
        target_user_id = target_users[0].id
        
        # Check if permission already exists
        existing = sqlite_client.get_all(
            ProjectPermission,
            filters={"project_id": request.project_id, "user_id": target_user_id}
        )
        
        if existing:
            # Update existing permission
            sqlite_client.update(
                ProjectPermission,
                existing[0].id,
                {"permission_level": request.permission_level, "granted_by": granter_id}
            )
        else:
            # Create new permission
            new_perm = ProjectPermission(
                project_id=request.project_id,
                user_id=target_user_id,
                permission_level=request.permission_level,
                granted_by=granter_id
            )
            sqlite_client.create(new_perm)
        
        logger.info(f"Permission granted successfully")
        return JSONResponse(
            content={
                "status": "Success",
                "message": f"Permission '{request.permission_level}' granted to {request.user_email}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as exe:
        logger.exception("Failed to grant permission")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(exe)}
        )


@router.get("/project-permissions/{project_id}")
async def get_project_permissions(
    project_id: int,
    current_user: User = Depends(AuthService.get_current_user)
):
    """Get all users with access to a project"""
    try:
        # Check if current user has access
        user_id = current_user.id
        access = sqlite_client.get_all(
            ProjectPermission,
            filters={"project_id": project_id, "user_id": user_id}
        )
        
        if not access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"status": "Failed", "message": "You don't have access to this project"}
            )
        
        # Get all permissions with user info
        all_perms = sqlite_client.get_all(ProjectPermission, filters={"project_id": project_id})
        
        permissions_data = []
        for perm in all_perms:
            user = sqlite_client.get_by_id(User, perm.user_id)
            if user:
                permissions_data.append({
                    "email": user.email,
                    "full_name": user.full_name,
                    "permission_level": perm.permission_level,
                    "granted_at": perm.granted_at.isoformat() if perm.granted_at else None
                })
        
        return JSONResponse(
            content={
                "status": "Success",
                "project_id": project_id,
                "permissions": permissions_data
            }
        )
        
    except HTTPException:
        raise
    except Exception as exe:
        logger.exception("Failed to get project permissions")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "Failed", "message": str(exe)}
        )

