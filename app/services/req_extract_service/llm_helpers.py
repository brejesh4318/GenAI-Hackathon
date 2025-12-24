"""
LLM Helper Functions for Requirements Extraction
"""

from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from app.services.llm_services.llm_implementation.gemini_models import Gemini25FlashLLM
from app.utilities import dc_logger

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})


class Requirement(BaseModel):
    """Single requirement extracted from document."""
    requirement_id: str = Field(description="Unique identifier for the requirement (e.g., REQ-001)")
    text: str = Field(description="The exact requirement sentence from the document, not rewritten or modified")
    source_page: int = Field(description="The page number where this requirement was found")


class RequirementsList(BaseModel):
    """List of requirements extracted from a document chunk."""
    requirements: List[Requirement] = Field(
        default_factory=list,
        description="List of extracted requirements"
    )


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
        llm = Gemini25FlashLLM(temperature=0.5).get_llm()
        
        # Create structured output chain
        structured_llm = llm.with_structured_output(RequirementsList, method="json_schema")
        
        # Craft extraction prompt
        system_prompt = """You are a requirements extraction expert. Your task is to:
            1. Carefully read the provided document chunk (which may contain multiple pages)
            2. Identify ALL explicit requirements, specifications, or functional statements
            3. Extract the EXACT text of each requirement (do not rewrite or paraphrase)
            4. Extract requirement_id given in document chunk and asssign it corresponding to each requirement
            5. Note the source_page number where each requirement appears

            **What counts as a requirement:**
            - Functional requirements (The system shall/must/will...)
            - Functional Requirements comes with requirement id
            - Non-functional requirements (Performance, security, usability constraints)
            - Business rules and constraints
            - Compliance and regulatory requirements

            **What to IGNORE:**
            - General introductions or background information
            - Definitions or glossaries (unless they define specific requirements)
            - Examples or illustrations (unless they specify required behavior)
            - Boilerplate text or headers/footers

            **Page number detection:**
            - Pages are annotated in the format: === PAGE X ===
            - Extract the page number X for each requirement

            **Important:**
            - Return ONLY requirements explicitly stated in the 
            - Requirement Id should match exactly as given in the document chunk.
            - Do NOT infer, generate, or create requirements and requirement id
            - If a page has no requirements, that's fine - just skip it
            - Preserve the exact wording from the source document
            Return your extraction as a structured list. """

        # Prepare the message with context
        messages = [
            ("system", system_prompt),
            ("human", f"Extract all requirements from this document chunk:\n\n{chunk_text}")
        ]
        
        # Invoke LLM with structured output
        result: RequirementsList = structured_llm.invoke(messages)
        
        # Convert Pydantic models to dictionaries
        requirements_list = [req.model_dump() for req in result.requirements]
        
        logger.info(f"Extracted {len(requirements_list)} requirements from chunk")
        return requirements_list
        
    except Exception as e:
        logger.error(f"Error extracting requirements via LLM: {str(e)}")
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
