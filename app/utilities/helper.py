    

from datetime import datetime
from typing import Tuple, List
from app.utilities import dc_logger
from app.utilities.singletons_factory import DcSingleton
from app.utilities.document_parser_depreceated import DocumentParser
import os

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})

class Helper(metaclass = DcSingleton):
    @staticmethod
    def read_file(file_path: str) -> Tuple[str, List[str]]:
        """
        Reads and parses documents from a file using DocumentParser.
        Supports: .docx, .pdf, .txt, .md
        
        Args:
            file_path (str): Path to the file.
            
        Returns:
            Tuple[str, List[str]]: (markdown_content, list_of_base64_image_uris)
            - markdown_content: Document text as markdown string
            - list_of_base64_image_uris: Images in format "data:image/png;base64,..."
                                        Empty list for text files
        """
        parser = DocumentParser()
        return parser.parse_file(file_path)
    
    @staticmethod
    def extract_doc_pages(file_path: str) -> List[str]:
        """
        Extract raw text per page from PDF/DOCX/TXT/MD files.
        
        Args:
            file_path (str): Path to the document file.
            
        Returns:
            List[str]: List of page texts (one string per page).
                      For text/markdown files, returns single-item list.
        """
        parser = DocumentParser()
        return parser.extract_doc_pages(file_path)
            
    
    @staticmethod
    def save_file(tmp_path: str, content: bytes, filename: str) -> str:
        """
        Save a file from binary content to a specified temporary path.

        Args:
            tmp_path (str): The temporary path to save the file.
            content (bytes): The binary content of the file.
            filename (str): The name of the file.

        Returns:
            str: The path where the file is saved.

        Raises:
            Exception: If there is an error saving the file.
        """
        try:
            if not os.path.exists(tmp_path):
                    os.mkdir(tmp_path)
            path = os.path.join(tmp_path, filename)
            with open(path, 'wb') as f:
                f.write(content)
            return path
        except Exception as exe:
              logger.error("Error occcured in save pdf")
              raise exe 
    
    @staticmethod
    def time_saved_format(start_time: datetime, end_time: datetime = datetime.now()) -> str:
        """
        Returns a human-readable string representing the time difference between two datetimes.

        Args:
            start_time (datetime): The start time.
            end_time (datetime): The end time.

        Returns:
            str: Time difference in days, hours, minutes, or seconds.
        """
        diff = end_time - start_time
        total_seconds = int(diff.total_seconds())
        days = total_seconds // (3600 * 24)
        hours = (total_seconds % (3600 * 24)) // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if days >= 1:
            return f"{days} day{'s' if days != 1 else ''}"
        elif hours >= 1:
            return f"{hours} hour{'s' if hours != 1 else ''}"
        elif minutes >= 1:
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        else:
            return f"{seconds} second{'s' if seconds != 1 else ''}"



