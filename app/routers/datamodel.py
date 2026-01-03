from pydantic import BaseModel, Field, validate_call, EmailStr
from typing import List, Optional
from datetime import datetime


class UserRegisterRequest(BaseModel):
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    full_name: str = Field(..., description="User full name")


class UserLoginRequest(BaseModel):
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., description="User password")


class ProjectCreateRequest(BaseModel):
    project_name: str = Field(..., description="Name of the project")
    description: Optional[str] = Field(None, description="Description of the project")
    organization_id: Optional[str] = Field(None, description="Organization ID (optional)")


class VersionCreateRequest(BaseModel):
    project_id: int = Field(..., description="ID of the parent project")
    version_name: str = Field(..., description="Version name (e.g., v1.0, v2.0)")
    description: Optional[str] = Field(None, description="Description of the version")
    is_active: Optional[bool] = Field(True, description="Whether this version is active")


class ProjectPermissionRequest(BaseModel):
    project_id: str = Field(..., description="MongoDB project ID")
    user_email: EmailStr = Field(..., description="User email to grant permission")
    permission_level: str = Field(..., description="Permission level: owner, editor, viewer")


class JiraPushRequest(BaseModel):
    project_id: str = Field(..., example= "id")
    jira_project_key: str = Field(..., example= "HS")
    jira_mail_id: str = Field(..., example= "username@mail.com")
    jira_api: str = Field(..., example="api_key")
    domain_name: str = Field(..., example="domain_name")
    selected_ids: List[str] = Field(...,example=["ids"])
