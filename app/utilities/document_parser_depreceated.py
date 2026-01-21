from pathlib import Path
from typing import Tuple, List

from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from app.utilities.singletons_factory import DcSingleton
from app.utilities import dc_logger

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})


class DocumentParser(metaclass=DcSingleton):
    """Parse documents and extract markdown content with images."""
    
    SUPPORTED_DOCLING_FORMATS = {'.pdf', '.docx', ".json"}
    SUPPORTED_TEXT_FORMATS = {'.txt', '.md'}
    
    def __init__(self, image_scale: float = 2.0):
        """
        Initialize the document parser.
        
        Args:
            image_scale: Resolution scale for extracted images (default: 2.0)
        """
        self.image_scale = image_scale
        self._setup_docling_converter()
    
    def _setup_docling_converter(self):
        """Setup Docling converter with image extraction enabled."""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = self.image_scale
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = True
        
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    
    def parse_file(self, file_path: str) -> Tuple[str, List[str]]:
        """
        Parse a document file and extract markdown content with images.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (markdown_string, list_of_base64_images)
            - markdown_string: Document content as markdown
            - list_of_base64_images: List of images in format "data:image/png;base64,..."
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
            Exception: For other parsing errors
        """
        # Validate file existence
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file extension
        file_extension = path.suffix.lower()
        
        # Route to appropriate parser
        if file_extension in self.SUPPORTED_DOCLING_FORMATS:
            return self._parse_with_docling(file_path)
        elif file_extension in self.SUPPORTED_TEXT_FORMATS:
            return self._parse_text_file(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {self.SUPPORTED_DOCLING_FORMATS | self.SUPPORTED_TEXT_FORMATS}"
            )
    
    def _parse_with_docling(self, file_path: str) -> Tuple[str, List[str]]:
        """
        Parse PDF/DOCX files using Docling.
        
        Args:
            file_path: Path to PDF or DOCX file
            
        Returns:
            Tuple of (markdown_content, base64_images_list)
        """
        try:
            logger.info(f"Parsing {file_path} with Docling...")
            
            # Convert document
            result = self.doc_converter.convert(file_path)
            
            # Extract markdown content
            markdown_content = result.document.export_to_markdown()
            
            # Extract images as base64 data URIs
            images_base64 = []
            for picture in result.document.pictures:
                # Get the base64 data URI from docling
                # Format: data:image/png;base64,iVBORw0K...
                image_uri = picture.image.uri.__str__()
                images_base64.append(image_uri)
            
            logger.info(f"Successfully parsed {file_path}: {len(images_base64)} images extracted")
            return markdown_content, images_base64
            
        except Exception as e:
            logger.error(f"Error parsing {file_path} with Docling: {str(e)}")
            raise Exception(f"Failed to parse document with Docling: {str(e)}")
    
    def _parse_text_file(self, file_path: str) -> Tuple[str, List[str]]:
        """
        Parse plain text or markdown files.
        
        Args:
            file_path: Path to .txt or .md file
            
        Returns:
            Tuple of (file_content, empty_list)
        """
        try:
            logger.info(f"Reading text file: {file_path}")
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Return content as-is with empty image list
            logger.info(f"Successfully read {file_path}")
            return content, []
            
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                logger.warning(f"Read {file_path} with latin-1 encoding")
                return content, []
            except Exception as e:
                logger.error(f"Error reading {file_path}: {str(e)}")
                raise Exception(f"Failed to read text file: {str(e)}")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            raise Exception(f"Failed to read text file: {str(e)}")
    
    def extract_doc_pages(self, file_path: str) -> List[str]:
        """
        Extract raw text from each page of a document.
        
        Args:
            file_path: Path to the document file (PDF, DOCX, TXT, MD)
            
        Returns:
            List[str]: List of page texts (one string per page)
            For text/markdown files, returns a single-item list with full content.
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
            Exception: For other parsing errors
        """
        # Validate file existence
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = path.suffix.lower()
        
        # Handle text files (return as single page)
        if file_extension in self.SUPPORTED_TEXT_FORMATS:
            content, _ = self._parse_text_file(file_path)
            return [content]
        
        # Handle PDF/DOCX with Docling
        if file_extension in self.SUPPORTED_DOCLING_FORMATS:
            try:
                logger.info(f"Extracting pages from {file_path} with Docling...")
                
                # Convert document
                result = self.doc_converter.convert(file_path)
                
                # Extract text per page
                pages = []
                for no, pg in result.document.pages.items():
                    page_text = result.document.export_to_markdown(page_no=no)
                    if page_text:  
                        pages.append(page_text)
                    else:
                        pages.append("Empty page")  # Empty page                
                logger.info(f"Successfully extracted {len(pages)} pages from {file_path}")
                return pages
                
            except Exception as e:
                logger.error(f"Error extracting pages from {file_path}: {str(e)}")
                raise Exception(f"Failed to extract pages: {str(e)}")
        
        else:
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {self.SUPPORTED_DOCLING_FORMATS | self.SUPPORTED_TEXT_FORMATS}"
            )


# # Convenience function for single-use parsing
# def parse_document(file_path: str, image_scale: float = 2.0) -> Tuple[str, List[str]]:
#     """
#     Convenience function to parse a document file.
    
#     Args:
#         file_path: Path to the document file
#         image_scale: Resolution scale for images (default: 2.0)
        
#     Returns:
#         Tuple of (markdown_string, list_of_base64_images)
        
#     Example:
#         >>> markdown, images = parse_document("document.pdf")
#         >>> print(f"Content: {markdown[:100]}")
#         >>> print(f"Found {len(images)} images")
#         >>> if images:
#         >>>     print(f"First image: {images[0][:50]}...")
#     """
#     parser = DocumentParser(image_scale=image_scale)
#     return parser.parse_file(file_path)


# if __name__ == "__main__":
#     # Example usage
#     import sys
    
#     if len(sys.argv) < 2:
#         print("Usage: python document_parser.py <file_path>")
#         print("Example: python document_parser.py test-prd.pdf")
#         sys.exit(1)
    
#     file_path = sys.argv[1]
    
#     try:
#         markdown, images = parse_document(file_path)
        
#         print(f"\n{'='*60}")
#         print(f"Parsed: {file_path}")
#         print(f"{'='*60}")
#         print(f"\nMarkdown content length: {len(markdown)} characters")
#         print(f"Images extracted: {len(images)}")
        
#         if images:
#             print(f"\nFirst image preview: {images[0][:80]}...")
        
#         print(f"\nMarkdown preview (first 500 chars):")
#         print(f"{'-'*60}")
#         print(markdown[:500])
#         print(f"{'-'*60}")
        
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         sys.exit(1)
