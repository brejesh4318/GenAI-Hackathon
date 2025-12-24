"""
Authentication Service - Centralized authentication and authorization management

Consolidates authentication utilities, token management, password hashing,
and user validation. Provides FastAPI dependencies for route protection.
"""
from datetime import datetime, timedelta, UTC
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.utilities.db_utilities.sqlite_implementation import SQLiteImplement
from app.utilities.db_utilities.models import User
from app.utilities.singletons_factory import DcSingleton
from app.utilities.constants import Constants
from app.utilities import dc_logger

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})


class AuthService(metaclass=DcSingleton):
    """
    Centralized authentication service handling:
    - Password hashing and verification
    - JWT token creation and validation
    - User authentication and authorization
    - FastAPI dependency injection for protected routes
    
    Singleton pattern ensures consistent auth configuration across the app.
    """

    def __init__(self):
        """Initialize authentication service with JWT config and password hashing"""
        # Load JWT settings from constants
        jwt_config = Constants.fetch_constant("jwt")
        self.secret_key = jwt_config.get("secret_key", "your-secret-key-change-in-production")
        self.algorithm = jwt_config.get("algorithm", "HS256")
        self.access_token_expire_minutes = jwt_config.get("access_token_expire_minutes", 30)
        
        # Password hashing context
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # HTTP Bearer security scheme for Swagger UI
        self.security = HTTPBearer()
        
        # Initialize SQLite database
        sqlite_config = Constants.fetch_constant("sqlite_db")
        self.sqlite_db = SQLiteImplement(
            db_path=sqlite_config["db_path"],
            max_pool_size=sqlite_config["max_pool_size"]
        )
        
        # Store singleton instance for static method access
        AuthService._instance = self
        logger.info(f"AuthService initialized with algorithm={self.algorithm}, expire={self.access_token_expire_minutes}min")

    # ==================== Password Management ====================

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify plain password against its hash.
        
        Args:
            plain_password: Plain text password from user
            hashed_password: Bcrypt hashed password from database
            
        Returns:
            True if password matches, False otherwise
        """
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """
        Hash a password using bcrypt.
        
        Args:
            password: Plain text password to hash
            
        Returns:
            Bcrypt hashed password string
        """
        return self.pwd_context.hash(password)

    def hash_password(self, password: str) -> str:
        """Alias for get_password_hash() for backward compatibility"""
        return self.get_password_hash(password)

    # ==================== Token Management ====================

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """
        Create a JWT access token.
        
        Args:
            data: Dictionary with user data (e.g., {"sub": user_id, "email": email})
            expires_delta: Custom token expiration time
            
        Returns:
            Encoded JWT token string
            
        Example:
            token = auth_service.create_access_token({"sub": user.id, "email": user.email})
        """
        to_encode = data.copy()
        expire = datetime.now(UTC) + (expires_delta or timedelta(minutes=self.access_token_expire_minutes))
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        logger.debug(f"Created token for user: {data.get('sub')}")
        return encoded_jwt

    @staticmethod
    def decode_access_token(token: str) -> dict:
        """
        Decode and validate a JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded payload dictionary with user data
            
        Raises:
            JWTError: If token is invalid or expired
        """
        try:
            instance = AuthService._instance
            payload = jwt.decode(token, instance.secret_key, algorithms=[instance.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise JWTError("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            raise JWTError(f"Invalid token: {str(e)}")

    # ==================== User Authentication ====================

    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """
        Authenticate a user by email and password.
        
        Args:
            email: User email address
            password: Plain text password
            
        Returns:
            User object if authentication successful, None otherwise
        """
        with self.sqlite_db.get_session() as session:
            user = session.query(User).filter(User.email == email).first()
            
            if not user:
                logger.warning(f"Authentication failed: User not found for email {email}")
                return None
            
            if not self.verify_password(password, user.password_hash):
                logger.warning(f"Authentication failed: Invalid password for email {email}")
                return None
            
            if not user.is_active:
                logger.warning(f"Authentication failed: User account inactive for email {email}")
                return None
            
            logger.info(f"User authenticated successfully: {email}")
            return user

    # ==================== FastAPI Dependencies ====================

    @staticmethod
    def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ) -> User:
        """
        FastAPI dependency that validates JWT and returns the current user object.
        Works like a JWT filter - validates token and retrieves user from database.
        
        Usage in routes:
            @router.get("/protected")
            async def protected_route(current_user: User = Depends(AuthService.get_current_user)):
                return {"user": current_user.email}
        
        Args:
            credentials: HTTP Bearer token from Authorization header
            
        Returns:
            User object from database
            
        Raises:
            HTTPException: 401 if token is invalid or user not found
        """
        token = credentials.credentials
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"status": "Failed", "message": "Could not validate credentials"},
            headers={"WWW-Authenticate": "Bearer"},
        )

        try:
            payload = AuthService.decode_access_token(token)
            user_id: str = payload.get("sub")
            if user_id is None:
                raise credentials_exception
        except JWTError:
            raise credentials_exception

        # Query user from database
        instance = AuthService._instance
        with instance.sqlite_db.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user is None:
                logger.warning(f"User not found for id: {user_id}")
                raise credentials_exception
            
            if not user.is_active:
                logger.warning(f"Inactive user attempted access: {user_id}")
                raise credentials_exception

        return user

    @staticmethod
    def get_current_user_id(current_user: User = Depends(get_current_user)) -> str:
        """
        Dependency to get only the current user's ID.
        
        Usage in routes:
            user_id: str = Depends(AuthService.get_current_user_id)
        
        Args:
            current_user: User object from get_current_user dependency
            
        Returns:
            User ID string
        """
        return current_user.id

    @staticmethod
    def get_current_user_optional(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
    ) -> Optional[User]:
        """
        Optional authentication - returns user if token is provided and valid, None otherwise.
        Useful for endpoints that work differently for authenticated vs anonymous users.
        
        Usage in routes:
            @router.get("/optional")
            async def optional_route(current_user: Optional[User] = Depends(AuthService.get_current_user_optional)):
                if current_user:
                    return {"message": f"Hello {current_user.email}"}
                return {"message": "Hello guest"}
        
        Args:
            credentials: Optional HTTP Bearer token
            
        Returns:
            User object if authenticated, None otherwise
        """
        if not credentials:
            return None

        try:
            token = credentials.credentials
            payload = AuthService.decode_access_token(token)
            user_id: str = payload.get("sub")
            
            if user_id is None:
                return None

            # Query user from database
            instance = AuthService._instance
            with instance.sqlite_db.get_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if user and user.is_active:
                    return user
                return None
                
        except (JWTError, Exception) as e:
            logger.debug(f"Optional auth failed: {str(e)}")
            return None

    # ==================== Utility Methods ====================

    @staticmethod
    def get_instance() -> 'AuthService':
        """
        Get the singleton instance of AuthService.
        
        Returns:
            AuthService instance
        """
        if not hasattr(AuthService, '_instance'):
            AuthService()
        return AuthService._instance
