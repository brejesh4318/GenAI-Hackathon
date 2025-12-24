"""
Authentication Helper - Centralized authentication management
"""
from datetime import datetime, timedelta
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

# Get SQLite database instance
sqlite_db = SQLiteImplement(
    db_path=Constants.fetch_constant("sqlite_db")["db_path"],
    max_pool_size=Constants.fetch_constant("sqlite_db")["max_pool_size"]
)


class AuthManager(metaclass=DcSingleton):
    """
    Handles authentication, token creation, validation, and user resolution.
    Singleton pattern ensures consistent auth configuration across the app.
    """

    def __init__(self):
        # Load JWT settings from constants
        jwt_config = Constants.fetch_constant("jwt")
        self.secret_key = jwt_config.get("secret_key", "your-secret-key-change-in-production")
        self.algorithm = jwt_config.get("algorithm", "HS256")
        self.access_token_expire_minutes = jwt_config.get("access_token_expire_minutes", 30)
        
        # Password hashing context
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # HTTP Bearer security scheme for Swagger UI
        self.security = HTTPBearer()
        
        # Store singleton instance for static method access
        AuthManager._instance = self
        logger.info(f"AuthManager initialized with algorithm={self.algorithm}")

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify plain password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash the password using bcrypt."""
        return self.pwd_context.hash(password)

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """
        Create a JWT access token.
        
        Args:
            data: Dictionary with user data (e.g., {"sub": user_id, "email": email})
            expires_delta: Custom token expiration time
            
        Returns:
            Encoded JWT token string
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=self.access_token_expire_minutes))
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    @staticmethod
    def decode_access_token(token: str) -> dict:
        """
        Decode and validate a JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded payload dictionary
            
        Raises:
            JWTError: If token is invalid or expired
        """
        try:
            instance = AuthManager._instance
            payload = jwt.decode(token, instance.secret_key, algorithms=[instance.algorithm])
            return payload
        except JWTError as e:
            logger.warning(f"Token decode failed: {str(e)}")
            raise JWTError(f"Invalid token: {str(e)}")

    @staticmethod
    def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ) -> User:
        """
        FastAPI dependency that validates JWT and returns the current user object.
        Works like a JWT filter - validates token and retrieves user from database.
        
        Usage in routes:
            current_user: User = Depends(AuthManager.get_current_user)
        
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
            payload = AuthManager.decode_access_token(token)
            user_id: str = payload.get("sub")
            if user_id is None:
                raise credentials_exception
        except JWTError:
            raise credentials_exception

        # Query user from database
        with sqlite_db.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user is None:
                logger.warning(f"User not found for id: {user_id}")
                raise credentials_exception

        return user

    @staticmethod
    def get_current_user_id(current_user: User = Depends(get_current_user)) -> str:
        """
        Dependency to get only the current user's ID.
        
        Usage in routes:
            user_id: str = Depends(AuthManager.get_current_user_id)
        
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
        
        Usage in routes:
            current_user: Optional[User] = Depends(AuthManager.get_current_user_optional)
        
        Args:
            credentials: Optional HTTP Bearer token
            
        Returns:
            User object if authenticated, None otherwise
        """
        if not credentials:
            return None

        try:
            token = credentials.credentials
            payload = AuthManager.decode_access_token(token)
            user_id: str = payload.get("sub")
            
            if user_id is None:
                return None

            # Query user from database
            with sqlite_db.get_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                return user
                
        except (JWTError, Exception) as e:
            logger.debug(f"Optional auth failed: {str(e)}")
            return None


# Create singleton instance
auth_manager = AuthManager()
