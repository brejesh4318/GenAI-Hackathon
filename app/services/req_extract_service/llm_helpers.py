"""
LLM Helper Functions for Requirements Extraction
"""

from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from app.services.llm_services.llm_implementation.gemini_models import Gemini25FlashLLM, Gemini25FlashLiteLLM
from app.services.prompt_service import PromptService
from app.utilities import dc_logger

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})

prompt_service = PromptService()
system_prompt = prompt_service.get("requirement-extractor")
class Requirement(BaseModel):
    """Single requirement extracted from document."""
    text: str = Field(description="EXACT requirement text as written in the document, copied verbatim without any modification")
    source_page: int = Field(description="The page number where this requirement was found")


class RequirementsList(BaseModel):
    """List of requirements extracted from a document chunk."""
    requirements: List[Requirement] = Field(..., description="give found requirements")


def extract_requirements_llm(chunk_text: str, model_name: str = "gemini-2.0-flash-exp") -> List[Dict]:
    """
    Extract requirements from a document chunk using Gemini LLM with structured output.
    
    Args:
        chunk_text (str): Text chunk containing multiple pages with page annotations.
                         Format: === PAGE X === followed by page content.
        model_name (str): Gemini model to use (default: gemini-2.0-flash-exp)
    
    Returns:
        List[Dict]: List of requirement dictionaries with keys:
            - requirement_id (str): Unique identifier
            - text (str): Exact requirement sentence
            - source_page (int): Page number where requirement was found
    
    Example:
        >>> chunk = "=== PAGE 1 ===\\nREQ-001: System must validate input\\n=== PAGE 2 ===\\nNo requirements"
        >>> requirements = extract_requirements_llm(chunk)
        >>> print(requirements)
        [{'requirement_id': 'REQ-001', 'text': 'System must validate input', 'source_page': 1}]
    """
    try:
        logger.info(f"Extracting requirements from chunk using {model_name}")
        
        # Initialize Gemini model with structured output
        llm = Gemini25FlashLiteLLM(temperature=0.5).get_llm()
        

        # Prepare the message with context
        messages = [
            ("system", system_prompt),
            ("human", f"Extract all requirements from this document chunk:\n\n{chunk_text}")
        ]
        
        # Invoke LLM with structured output
        result: RequirementsList = llm.with_structured_output(RequirementsList, include_raw=True).invoke(messages)
        parser = PydanticOutputParser(pydantic_object=RequirementsList)
        parsed_result = parser.parse(result["raw"].content)
        
        # Convert Pydantic models to dictionaries
        requirements_list = [req.model_dump() for req in parsed_result.requirements]
        
        logger.info(f"Extracted {len(requirements_list)} requirements from chunk")
        return requirements_list
        
    except Exception as e:
          # Return empty list on error to allow processing to continue
        return []


def extract_requirements_llm_with_context(
    chunk_text: str, 
    start_req_id: int = 1,
    model_name: str = "gemini-2.0-flash-exp"
) -> List[Dict]:
    """
    Extract requirements with context-aware ID numbering.
    
    Args:
        chunk_text (str): Text chunk with page annotations
        start_req_id (int): Starting requirement ID number for sequential numbering
        model_name (str): Gemini model to use
    
    Returns:
        List[Dict]: List of requirement dictionaries
    """
    requirements = extract_requirements_llm(chunk_text, model_name)
    
    # Renumber requirements to ensure global uniqueness
    # for idx, req in enumerate(requirements):
    #     req_num = start_req_id + idx
    #     req['requirement_id'] = f"REQ-{req_num:03d}"


    return requirements