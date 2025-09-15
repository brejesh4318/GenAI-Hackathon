    

from pathlib import Path
from app.utilities import dc_logger
from app.utilities.singletons_factory import DcSingleton
import PyPDF2
import docx  # for .docx
import markdown  # for .md
import os

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})


class Helper(metaclass = DcSingleton):
    @staticmethod
    def read_file(file_path: str) -> str:
            """
            Reads and parses text from a file.
            Supports: .docx, .pdf, .txt, .md
            
            Args:
                file_path (str): Path to the file.
                
            Returns:
                str: Extracted text content.
            """
            file_ext = Path(file_path).suffix.lower()

            if file_ext == ".docx":
                return _read_docx(file_path)
            elif file_ext == ".pdf":
                return _read_pdf(file_path)
            elif file_ext == ".txt":
                return _read_txt(file_path)
            elif file_ext == ".md":
                return _read_md(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
    
    @staticmethod
    def save_pdf(tmp_path: str, content: bytes, filename: str) -> str:
        """
        Save a PDF file from binary content to a specified temporary path.

        Args:
            tmp_path (str): The temporary path to save the PDF file.
            content (bytes): The binary content of the PDF file.
            filename (str): The name of the PDF file.

        Returns:
            str: The path where the PDF file is saved.

        Raises:
            Exception: If there is an error saving the PDF file.
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


def _read_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def _read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def _read_pdf(file_path: str):
    with open(file_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text())
    return "\n".join(pages_text)

def _read_md(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_md = f.read()
    # Convert markdown to plain text
    html = markdown.markdown(raw_md)
    # strip HTML tags (optional)
    return "".join(html.split("<")[0::2])  # quick strip, or use BeautifulSoup for better parsing
