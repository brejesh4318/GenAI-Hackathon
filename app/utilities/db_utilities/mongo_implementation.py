from app.utilities import dc_logger 
from app.utilities.constants import Constants
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import sys
from typing import List, Tuple, Union
from app.utilities.singletons_factory import DcSingleton

logger = dc_logger.LoggerAdap(
    dc_logger.get_logger(__name__), {"mom-generation": "v3"}
)


class MongoImplement(metaclass=DcSingleton):
    """Mongodb implementation class for Robust db.
    Args:
        MongoInterface (Class): Interface for all the Mongodb implementation.
    """

    def __init__(self,connection_string,db_name,max_pool,server_selection_timeout) -> None:
        """Constructor to create a connection with the robust db.
        """
        self.connection_string = connection_string
        self.db_name = db_name
        self.max_pool = max_pool
        self.server_selection_timeout=server_selection_timeout
        logger.info("Initializing connection pool for database connection, with {}".format(self.db_name))
        try:
            self.client = MongoClient(
                self.connection_string,
                maxPoolSize=self.max_pool,
                serverSelectionTimeoutMS=self.server_selection_timeout,
                uuidRepresentation="standard"
            )
            self.client.server_info()
            self.database = self.client[self.db_name]
            logger.info(f"Made {self.max_pool} max_connections with {self.db_name}")
        except (ConnectionFailure, ServerSelectionTimeoutError):
            logger.error(f"Could not Connect with databse={self.db_name}. Please check the connection string",exc_info=True)
            sys.exit(0)

    def insert_one(self,collection_name: str, data):
        """
        Function to insert one item in a collection.

        Args:
            collection_name (str): Name of the collection.
            data (dict): Data to be inserted.

        Returns:
            _id: ID of the inserted document.
        """
        session = self.client.start_session(causal_consistency=True)
        session.start_transaction()
        try:
            _id = self.database[collection_name].insert_one(data)
            session.commit_transaction()
            logger.info(f"Inserted data to {collection_name}")
            return _id.inserted_id.__str__()
        except Exception as exe:
            logger.error(f"Could not perform insertion on {collection_name} collection || {exe}",exc_info=True)
            session.abort_transaction()
            raise exe
        finally:
            session.end_session()

    def insert_many(self,collection_name: str, data_list):
        """
        Function to insert multiple items in a collection.

        Args:
            collection_name (str): Name of the collection.
            data_list (list): List of dictionaries containing data to be inserted.

        Returns:
            inserted_ids: List of IDs of the inserted documents.
        """
        session = self.client.start_session(causal_consistency=True)
        session.start_transaction()
        try:
            inserted_ids = self.database[collection_name].insert_many(
                data_list
            )
            session.commit_transaction()
            logger.info(f"Inserted {len(data_list)} items to {collection_name}")
            return inserted_ids.inserted_ids
        except Exception as exe:
            logger.error(f"Could not perform insertion on {collection_name} collection || {exe}",exc_info=True)
            session.abort_transaction()
            raise exe
        finally:
            session.end_session()

    def update_one(self, collection_name: str, query: dict, data: dict):
        """
        Function to update one item in a collection.

        Args:
            collection_name (str): Name of the collection.
            query (dict): Query to find the document to update.
            data (dict): Data to update in the document.
        """
        session = self.client.start_session(causal_consistency=True)
        session.start_transaction()
        try:
            self.database[collection_name].update_one(
                query, {"$set": data}
            )
            session.commit_transaction()
        except Exception as exe:
            logger.error(
                f"Error in Updating the collection name="
                f" {collection_name} || query ={query} || {exe}, starting rollback.",
                exc_info=True,
            )
            session.abort_transaction()
            raise exe
        finally:
            session.end_session()
    
    def update_many(self, collection_name: str, query: dict, data: dict):
        """
        Function to update one item in a collection.

        Args:
            collection_name (str): Name of the collection.
            query (dict): Query to find the document to update.
            data (dict): Data to update in the document.
        """
        session = self.client.start_session(causal_consistency=True)
        session.start_transaction()
        try:
            self.database[collection_name].update_many(
                query, {"$set": data}
            )
            session.commit_transaction()
        except Exception as exe:
            logger.error(
                f"Error in Updating the collection name="
                f" {collection_name} || query ={query} || {exe}, starting rollback.",
                exc_info=True,
            )
            session.abort_transaction()
            raise exe
        finally:
            session.end_session()   


    def read(self, collection_name: str,query: dict, sort: Union[List[Tuple], None] = None,col_names: Union[List[str], None] = None,max_count: int = None):
        """
        Function to read data from the database.

        Args:
            collection_name (str): Name of the collection.
            query (dict): Query to filter the documents.
            sort (Union[List[Tuple], None], optional): List of tuples to sort the
                documents. Defaults to None.
            col_names (Union[List[str], None], optional): List of column names to
                project. Defaults to None.
            max_count (int, optional): Maximum number of documents to return.
                Defaults to None.

        Returns:
            list: List of documents matching the query.
        """
        data = []
        session = self.client.start_session(causal_consistency=True)
        session.start_transaction()
        try:
            # if '_id' in query.keys():
            #     if len(query['_id']['$in'])>10:
            #         logger.info(f"Fetching documents from {collection_name} || query = {query['_id']['$in'][:10]} ....")
            #     else:
            #         logger.info(f"Fetching documents from {collection_name} || query = {query}")
            # else:
            #     logger.info(f"Fetching documents from {collection_name} || query = {query}")
            if max_count:
                resp = self.database[collection_name].find(
                    filter=query, projection=col_names, sort=sort
                ).limit(max_count)
            else:
                resp = self.database[collection_name].find(
                    filter=query, projection=col_names, sort=sort
                )
            session.commit_transaction()
            for item in resp:
                data.append(item)
        except Exception as exe:
            logger.error(
                f"Error in executing the query = {query} for collection name ="
                f" {collection_name} || {exe}, starting rollback.",
                exc_info=True)
            session.abort_transaction()
            raise exe
        finally:
            session.end_session()
        return data


    def delete_one(self, collection_name, query):
        """
        Function to delete one item from a collection.

        Args:
            collection_name (str): Name of the collection.
            query (dict): Query to find the document to delete.
        """
        session = self.client.start_session(causal_consistency=True)
        session.start_transaction()
        try:
            self.database[collection_name].delete_one(query)
            session.commit_transaction()
            logger.info(f"Removed data from {collection_name} || query = {query}")
        except Exception as exe:
            logger.error(
                f"Could not perform deletion on {collection_name} collection || {exe}",
                exc_info=True)
            session.abort_transaction()
            raise exe
        finally:
            session.end_session()


    def delete_many(self, collection_name, query):
        """
        Function to delete multiple items from a collection.

        Args:
            collection_name (str): Name of the collection.
            query (dict): Query to find the documents to delete.
        """
        session = self.client.start_session(causal_consistency=True)
        session.start_transaction()
        try:
            result = self.database[collection_name].delete_many(query)
            session.commit_transaction()
            logger.info(
                f"Removed {result.deleted_count} documents from {collection_name} || query = {query}"
            )
        except Exception as exe:
            logger.error(
                f"Could not perform deletion on {collection_name} collection || {exe}",
                exc_info=True)
            session.abort_transaction()
            raise exe
        finally:
            session.end_session()