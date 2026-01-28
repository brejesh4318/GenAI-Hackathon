"""
LLM Helper Functions for Requirements Extraction
"""

from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from app.services.llm_services.llm_implementation.gemini_models import Gemini25FlashLLM, Gemini25FlashLiteLLM
from app.services.prompt_service import PromptService
from app.utilities import dc_logger
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})

prompt_service = PromptService()
system_prompt = prompt_service.get("requirement-extractor")
class Requirement(BaseModel):
    """Single requirement extracted from document."""
    requirement_id: str = Field(description="Relevant Requirement Identifier given in the document")
    text: str = Field(description="EXACT requirement text as written in the document, copied verbatim without any modification")
    source_page: int = Field(description="The page number where this requirement was found")


class RequirementsList(BaseModel):
    """List of requirements extracted from a document chunk."""
    requirements: Optional[List[Requirement]] = Field([], description="give found requirements if requirements found else empty list")


@retry(
    retry=retry_if_exception_type((OutputParserException, ValueError)),
    stop=stop_after_attempt(3),
    wait=wait_fixed(2)
)
def extract_requirements_llm(chunk_text: str, model_name: str = "gemini-2.0-flash-exp") -> List[Dict]:
    """
    Extract requirements from a document chunk using Gemini LLM with structured output.
    
    Automatically retries up to 3 times on parsing failures with 2-second delays.
    
    Args:
        chunk_text (str): Text chunk containing multiple pages with page annotations.
                         Format: === PAGE X === followed by page content.
        model_name (str): Gemini model to use (default: gemini-2.0-flash-exp)
    
    Returns:
        List[Dict]: List of requirement dictionaries with keys:
            - requirement_id (str): Document-provided identifier or generated ID
            - text (str): Exact requirement sentence
            - source_page (int): Page number where requirement was found
    
    Raises:
        ValueError: If LLM returns no content or invalid response
        OutputParserException: If structured output parsing fails after retries
    
    Example:
        >>> chunk = "=== PAGE 1 ===\\nREQ-001: System must validate input\\n=== PAGE 2 ===\\nNo requirements"
        >>> requirements = extract_requirements_llm(chunk)
        >>> print(requirements)
        [{'requirement_id': 'REQ-001', 'text': 'System must validate input', 'source_page': 1}]
    """
    try:
        logger.info(f"Extracting requirements from chunk using {model_name}")
        
        # Initialize Gemini model
        llm = Gemini25FlashLiteLLM(temperature=0.5).get_llm()
        
        # Prepare messages
        messages = [
            ("system", system_prompt),
            ("human", f"Extract all requirements from this document chunk:\n\n{chunk_text}")
        ]
        
        # Invoke LLM with structured output
        result = llm.with_structured_output(RequirementsList, include_raw=True).invoke(messages)
        
        # Validate response
        if not result or "raw" not in result:
            raise ValueError("Invalid LLM response structure: missing 'raw' key")
        
        raw_content = result["raw"].content
        if not raw_content:
            raise ValueError("LLM returned empty content")
        
        # Handle different content types (string vs list)
        if isinstance(raw_content, list):
            # Extract text from list of content items
            text_parts = []
            for item in raw_content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            content_text = "\n".join(text_parts)
        else:
            content_text = str(raw_content)
        
        # Parse structured output
        parser = PydanticOutputParser(pydantic_object=RequirementsList)
        parsed_result = parser.parse(content_text)
        
        # Convert to dictionaries
        requirements_list = [req.model_dump() for req in parsed_result.requirements]
        
        logger.info(f"Successfully extracted {len(requirements_list)} requirements from chunk")
        return requirements_list
        
    except OutputParserException as e:
        logger.error(f"Failed to parse requirements output: {str(e)}", exc_info=True)
        raise
    except ValueError as e:
        logger.error(f"Invalid LLM response: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during requirement extraction: {str(e)}", exc_info=True)
        raise ValueError(f"Requirement extraction failed: {str(e)}") from e