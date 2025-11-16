from pydantic import BaseModel, Field, validate_call
from typing import List, Optional
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime


class ProjectCreateRequest(BaseModel):
    project_name: str = Field(..., description="Name of the project")
    description: Optional[str] = Field(None, description="Description of the project")


class JiraPushRequest(BaseModel):
    project_id: str = Field(..., example= "id")
    jira_project_key: str = Field(..., example= "HS")
    jira_mail_id: str = Field(..., example= "username@mail.com")
    jira_api: str = Field(..., example="api_key")
    domain_name: str = Field(..., example="domain_name")
    selected_ids: List[str] = Field(...,example=["ids"])
