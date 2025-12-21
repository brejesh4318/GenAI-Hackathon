"""
SQLite Database Implementation with SQLAlchemy ORM
Uses UUID for unique, non-reusable IDs
Models automatically create tables
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import StaticPool
from typing import List, Dict, Any, Optional, Type, TypeVar, Tuple
from contextlib import contextmanager
from app.utilities import dc_logger
from app.utilities.singletons_factory import DcSingleton
from app.utilities.db_utilities.models import Base, Project, Version, User, ProjectPermission, AuditLog

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})

T = TypeVar('T')


class SQLiteImplement(metaclass=DcSingleton):
    """SQLite database implementation using SQLAlchemy ORM"""
    
    def __init__(self, db_path: str, max_pool_size: int = 5):
        """
        Initialize SQLite with SQLAlchemy
        :param db_path: Path to SQLite database file
        :param max_pool_size: Maximum number of connections in pool
        """
        self.db_path = db_path
        
        # Create engine with connection pooling
        self.engine = create_engine(
            f'sqlite:///{db_path}',
            connect_args={'check_same_thread': False},
            poolclass=StaticPool,
            echo=False  # Set to True for SQL query logging
        )
        
        # Create session factory
        self.SessionLocal = scoped_session(sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        ))
        
        logger.info(f"SQLAlchemy engine initialized for {db_path}")
        
        # Create all tables from models
        self._create_tables()
    
    def _create_tables(self):
        """Create all tables from SQLAlchemy models"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("All tables created successfully from models")
        except Exception as e:
            logger.error(f"Error creating tables: {e}", exc_info=True)
            raise
    
    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions
        Usage:
            with sqlite_client.get_session() as session:
                project = session.query(Project).filter_by(id=project_id).first()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    # ORM-based methods using models
    
    def create(self, model_instance) -> Optional[str]:
        """
        Create a new record using model instance
        :param model_instance: SQLAlchemy model instance
        :return: ID of created record or None
        """
        with self.get_session() as session:
            try:
                session.add(model_instance)
                session.flush()  # Get the generated ID
                record_id = model_instance.id
                logger.debug(f"Created {type(model_instance).__name__} with ID: {record_id}")
                return record_id
            except Exception as e:
                logger.error(f"Error creating record: {e}", exc_info=True)
                return None
    
    def get_by_id(self, model_class: Type[T], record_id: str) -> Optional[T]:
        """
        Get a record by ID
        :param model_class: SQLAlchemy model class (Project, Version, etc.)
        :param record_id: Record UUID
        :return: Model instance or None
        """
        with self.get_session() as session:
            try:
                return session.query(model_class).filter_by(id=record_id).first()
            except Exception as e:
                logger.error(f"Error fetching {model_class.__name__}: {e}", exc_info=True)
                return None
    
    def get_all(self, model_class: Type[T], filters: Optional[Dict] = None, order_by=None) -> List[T]:
        """
        Get all records with optional filters
        :param model_class: SQLAlchemy model class
        :param filters: Dictionary of filter conditions
        :param order_by: Column to order by
        :return: List of model instances
        """
        with self.get_session() as session:
            try:
                query = session.query(model_class)
                if filters:
                    query = query.filter_by(**filters)
                if order_by is not None:
                    query = query.order_by(order_by)
                return query.all()
            except Exception as e:
                logger.error(f"Error fetching all {model_class.__name__}: {e}", exc_info=True)
                return []
    
    def update(self, model_class: Type[T], record_id: str, update_data: Dict) -> bool:
        """
        Update a record by ID
        :param model_class: SQLAlchemy model class
        :param record_id: Record UUID
        :param update_data: Dictionary of fields to update
        :return: True if successful, False otherwise
        """
        with self.get_session() as session:
            try:
                record = session.query(model_class).filter_by(id=record_id).first()
                if record:
                    for key, value in update_data.items():
                        setattr(record, key, value)
                    logger.debug(f"Updated {model_class.__name__} with ID: {record_id}")
                    return True
                return False
            except Exception as e:
                logger.error(f"Error updating record: {e}", exc_info=True)
                return False
    
    def delete(self, model_class: Type[T], record_id: str) -> bool:
        """
        Delete a record by ID
        :param model_class: SQLAlchemy model class
        :param record_id: Record UUID
        :return: True if successful, False otherwise
        """
        with self.get_session() as session:
            try:
                record = session.query(model_class).filter_by(id=record_id).first()
                if record:
                    session.delete(record)
                    logger.debug(f"Deleted {model_class.__name__} with ID: {record_id}")
                    return True
                return False
            except Exception as e:
                logger.error(f"Error deleting record: {e}", exc_info=True)
                return False
    
    def count(self, model_class: Type[T], filters: Optional[Dict] = None) -> int:
        """
        Count records with optional filters
        :param model_class: SQLAlchemy model class
        :param filters: Dictionary of filter conditions
        :return: Count of records
        """
        with self.get_session() as session:
            try:
                query = session.query(model_class)
                if filters:
                    query = query.filter_by(**filters)
                return query.count()
            except Exception as e:
                logger.error(f"Error counting {model_class.__name__}: {e}", exc_info=True)
                return 0
    
    def close(self):
        """Close all connections"""
        self.SessionLocal.remove()
        self.engine.dispose()
        logger.info("SQLite connections closed")