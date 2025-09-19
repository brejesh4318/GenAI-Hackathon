from pydantic import BaseModel, Field, validate_call
from typing import List, Optional
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime


class ProjectCreateRequest(BaseModel):
    project_name: str = Field(..., description="Name of the project")
    description: Optional[str] = Field(None, description="Description of the project")

