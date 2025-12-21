"""
Authentication and User Management Routes
Handles user registration, login, and permission management
Uses PostgreSQL for user data and permissions
"""
from fastapi import APIRouter, HTTPException, status, Depends, Header
from typing import Optional
from datetime import timedelta
from app.routers.datamodel import UserRegisterRequest, UserLoginRequest, ProjectPermissionRequest
from app.utilities import dc_logger
from app.utilities.auth_utils import hash_password, verify_password, create_access_token, decode_access_token
from app.utilities.constants import Constants
from app.utilities.env_util import EnvironmentVariableRetriever
from app.utilities.db_utilities.postgres_implementation import DbUtil
from fastapi.responses import JSONResponse

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})

auth_router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
    responses={status.HTTP_404_NOT_FOUND: {"description": "notfound"}}
)

# Initialize PostgreSQL client
pg_config = Constants.fetch_constant("postgres_db")
pg_client = DbUtil(
    dbname=pg_config["db_name"],
    dbuser=EnvironmentVariableRetriever.get_env_variable("POSTGRES_USER"),
    dbhost=pg_config["host"],
    dbpassword=EnvironmentVariableRetriever.get_env_variable("POSTGRES_PASSWORD"),
    dbport=pg_config["port"],
    min_pool_size=1,
    max_pool_size=pg_config["max_pool_size"]
)
logger.info("PostgreSQL client initialized for auth")


def get_current_user(authorization: Optional[str] = Header(None)):
    """
    Dependency to get current user from JWT token
    Usage: user = Depends(get_current_user)
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"status": "Failed", "message": "Authorization header missing"}
        )
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"status": "Failed", "message": "Invalid authentication scheme"}
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"status": "Failed", "message": "Invalid authorization header format"}
        )
    
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"status": "Failed", "message": "Invalid or expired token"}
        )
    
    return payload


@auth_router.post("/register")
async def register_user(request: UserRegisterRequest):
    """Register a new user"""
    logger.info(f"Registering new user: {request.email}")
    
    try:
        # Check if user already exists
        check_query = "SELECT id FROM users WHERE email = %s"
        existing = pg_client.select_query_v3(check_query, (request.email,))
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "status": "Failed",
                    "message": f"User with email '{request.email}' already exists"
                }
            )
        
        # Hash password and insert user
        hashed_pw = hash_password(request.password)
        insert_query = """
            INSERT INTO users (email, password_hash, full_name)
            VALUES (%s, %s, %s)
            RETURNING id, email, full_name, created_at
        """
        result = pg_client.select_query_v3(
            insert_query,
            (request.email, hashed_pw, request.full_name)
        )
        
        if result:
            user = result[0]
            logger.info(f"User registered successfully: {user['id']}")
            
            # Create access token
            access_token = create_access_token(
                data={"sub": str(user['id']), "email": user['email']}
            )
            
            return JSONResponse(
                content={
                    "status": "Success",
                    "message": "User registered successfully",
                    "user": {
                        "id": str(user['id']),
                        "email": user['email'],
                        "full_name": user['full_name']
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


@auth_router.post("/login")
async def login_user(request: UserLoginRequest):
    """Login user and return JWT token"""
    logger.info(f"Login attempt for user: {request.email}")
    
    try:
        # Get user from database
        query = """
            SELECT id, email, password_hash, full_name, is_active
            FROM users WHERE email = %s
        """
        result = pg_client.select_query_v3(query, (request.email,))
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"status": "Failed", "message": "Invalid email or password"}
            )
        
        user = result[0]
        
        # Check if user is active
        if not user['is_active']:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"status": "Failed", "message": "User account is inactive"}
            )
        
        # Verify password
        if not verify_password(request.password, user['password_hash']):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"status": "Failed", "message": "Invalid email or password"}
            )
        
        # Create access token
        access_token = create_access_token(
            data={"sub": str(user['id']), "email": user['email']}
        )
        
        logger.info(f"User logged in successfully: {user['id']}")
        
        return JSONResponse(
            content={
                "status": "Success",
                "message": "Login successful",
                "user": {
                    "id": str(user['id']),
                    "email": user['email'],
                    "full_name": user['full_name']
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


@auth_router.get("/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information from token"""
    try:
        user_id = current_user.get("sub")
        query = """
            SELECT id, email, full_name, created_at
            FROM users WHERE id = %s
        """
        result = pg_client.select_query_v3(query, (user_id,))
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"status": "Failed", "message": "User not found"}
            )
        
        user = result[0]
        return JSONResponse(
            content={
                "status": "Success",
                "user": {
                    "id": str(user['id']),
                    "email": user['email'],
                    "full_name": user['full_name'],
                    "created_at": user['created_at'].isoformat() if user['created_at'] else None
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


@auth_router.post("/project-permission")
async def grant_project_permission(
    request: ProjectPermissionRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Grant project access to another user
    Only project owners can grant permissions
    """
    logger.info(f"Granting permission for project {request.project_id}")
    
    try:
        granter_id = current_user.get("sub")
        
        # Check if granter has owner permission
        check_query = """
            SELECT permission_level FROM project_permissions
            WHERE project_id = %s AND user_id = %s
        """
        granter_perm = pg_client.select_query_v3(check_query, (request.project_id, granter_id))
        
        if not granter_perm or granter_perm[0]['permission_level'] != 'owner':
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"status": "Failed", "message": "Only project owners can grant permissions"}
            )
        
        # Get target user ID by email
        user_query = "SELECT id FROM users WHERE email = %s"
        target_user = pg_client.select_query_v3(user_query, (request.user_email,))
        
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"status": "Failed", "message": f"User {request.user_email} not found"}
            )
        
        target_user_id = target_user[0]['id']
        
        # Insert or update permission
        upsert_query = """
            INSERT INTO project_permissions (project_id, user_id, permission_level, granted_by)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (project_id, user_id)
            DO UPDATE SET permission_level = EXCLUDED.permission_level,
                         granted_at = CURRENT_TIMESTAMP,
                         granted_by = EXCLUDED.granted_by
            RETURNING id
        """
        result = pg_client.select_query_v3(
            upsert_query,
            (request.project_id, str(target_user_id), request.permission_level, granter_id)
        )
        
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


@auth_router.get("/project-permissions/{project_id}")
async def get_project_permissions(
    project_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get all users with access to a project"""
    try:
        # Check if current user has access
        user_id = current_user.get("sub")
        access_check = """
            SELECT permission_level FROM project_permissions
            WHERE project_id = %s AND user_id = %s
        """
        access = pg_client.select_query_v3(access_check, (project_id, user_id))
        
        if not access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"status": "Failed", "message": "You don't have access to this project"}
            )
        
        # Get all permissions
        query = """
            SELECT u.email, u.full_name, pp.permission_level, pp.granted_at
            FROM project_permissions pp
            JOIN users u ON pp.user_id = u.id
            WHERE pp.project_id = %s
            ORDER BY pp.granted_at DESC
        """
        permissions = pg_client.select_query_v3(query, (project_id,))
        
        return JSONResponse(
            content={
                "status": "Success",
                "project_id": project_id,
                "permissions": [
                    {
                        "email": p['email'],
                        "full_name": p['full_name'],
                        "permission_level": p['permission_level'],
                        "granted_at": p['granted_at'].isoformat() if p['granted_at'] else None
                    }
                    for p in permissions
                ]
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
