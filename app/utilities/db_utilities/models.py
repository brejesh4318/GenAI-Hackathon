"""
SQLAlchemy Models for SQLite Database
Models automatically create tables and handle relationships
"""
from sqlalchemy import Column, String, Integer, Boolean, ForeignKey, Text, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


def generate_uuid():
    """Generate UUID as string"""
    return str(uuid.uuid4())


class Project(Base):
    """Project model - stores project metadata"""
    __tablename__ = 'projects'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    owner_id = Column(String(36), nullable=True)
    organization_id = Column(String(36), nullable=True)
    no_test_cases = Column(Integer, default=0)
    no_documents = Column(Integer, default=0)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    versions = relationship("Version", back_populates="project", cascade="all, delete-orphan")
    permissions = relationship("ProjectPermission", back_populates="project", cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'project_name': self.project_name,
            'description': self.description,
            'owner_id': self.owner_id,
            'organization_id': self.organization_id,
            'no_test_cases': self.no_test_cases,
            'no_documents': self.no_documents,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }


class Version(Base):
    """Version model - stores version information under projects"""
    __tablename__ = 'versions'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    version_name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    no_documents = Column(Integer, default=0)
    no_test_cases = Column(Integer, default=0)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    project = relationship("Project", back_populates="versions")
    
    # Unique constraint on project_id + version_name
    __table_args__ = (
        {'sqlite_autoincrement': True},
    )
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'project_id': self.project_id,
            'version_name': self.version_name,
            'description': self.description,
            'is_active': self.is_active,
            'no_documents': self.no_documents,
            'no_test_cases': self.no_test_cases,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }


class User(Base):
    """User model - stores user accounts"""
    __tablename__ = 'users'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    permissions = relationship("ProjectPermission", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user")
    
    def to_dict(self):
        """Convert model to dictionary (excluding password)"""
        return {
            'id': self.id,
            'email': self.email,
            'full_name': self.full_name,
            'is_active': self.is_active,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }


class ProjectPermission(Base):
    """Project permissions model - manages access control"""
    __tablename__ = 'project_permissions'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(String(36), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    permission_level = Column(String(50), nullable=False)  # owner, editor, viewer
    granted_at = Column(TIMESTAMP, server_default=func.now())
    granted_by = Column(String(36), nullable=True)
    
    # Relationships
    project = relationship("Project", back_populates="permissions")
    user = relationship("User", back_populates="permissions")
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'project_id': self.project_id,
            'user_id': self.user_id,
            'permission_level': self.permission_level,
            'granted_at': self.granted_at,
            'granted_by': self.granted_by
        }


class AuditLog(Base):
    """Audit log model - tracks user actions"""
    __tablename__ = 'audit_logs'
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(String(255), nullable=True)
    action_metadata = Column(Text, nullable=True)  # JSON string - renamed from 'metadata' (reserved by SQLAlchemy)
    ip_address = Column(String(45), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'action': self.action,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'action_metadata': self.action_metadata,
            'ip_address': self.ip_address,
            'created_at': self.created_at
        }
