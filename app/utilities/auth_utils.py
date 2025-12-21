"""
Authentication and authorization utilities
"""
from datetime import datetime, timedelta
from typing import Optional
import jwt
from passlib.context import CryptContext
from app.utilities import dc_logger
from app.utilities.constants import Constants

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings - Add to constants.yaml
SECRET_KEY = Constants.fetch_constant("jwt", {}).get("secret_key", "your-secret-key-change-in-production")
ALGORITHM = Constants.fetch_constant("jwt", {}).get("algorithm", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = Constants.fetch_constant("jwt", {}).get("access_token_expire_minutes", 30)


def hash_password(password: str) -> str:
    """Hash a password for storing"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a stored password against one provided by user"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token
    :param data: Dictionary with user data (e.g., {"sub": user_id, "email": email})
    :param expires_delta: Token expiration time
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decode and verify JWT token
    :param token: JWT token string
    :return: Token payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        return None
    except jwt.InvalidTokenError:
        logger.warning("Invalid token")
        return None
