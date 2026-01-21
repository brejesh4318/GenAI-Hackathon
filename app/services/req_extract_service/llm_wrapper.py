"""
LLM interaction wrapper for requirements extraction.
Handles all LLM-related operations.
"""
from typing import List, Dict
from app.services.req_extract_service.llm_helpers import extract_requirements_llm_with_context
from app.utilities import dc_logger

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})


def extract_requirements_via_llm(
    chunk_text: str, 
    page_numbers: List[int],
    start_req_id: int,
    model_name: str
) -> List[Dict]:
    """
    Extract requirements from a chunk using LLM.
    
    Args:
        chunk_text (str): Annotated chunk text with page markers
        page_numbers (List[int]): Page numbers in this chunk
        start_req_id (int): Starting requirement ID for sequential numbering
        model_name (str): Model name to use for extraction
    
    Returns:
        List[Dict]: List of requirement dicts with:
            - text (str): Exact requirement sentence
            - source_page (int): Page number where requirement was found
    """
    try:
        # Call LLM helper function
        requirements = extract_requirements_llm_with_context(
            chunk_text=chunk_text,
            start_req_id=start_req_id,
            model_name=model_name
        )
        
        # Validate that source_page is within expected range
        for req in requirements:
            if req['source_page'] not in page_numbers:
                logger.warning(
                    f"Requirement references page {req['source_page']} "
                    f"which is outside chunk range {page_numbers}. Adjusting to first page."
                )
                req['source_page'] = page_numbers[0]
        req_dict = {}
        for req in requirements:
                page = req["source_page"]

                if page in req_dict:
                    req_dict[page]["text"] += "\n" + req["text"]
                else:
                    req_dict[page] = {
                        "source_page": page,
                        "text": req["text"]
                    }
        return requirements
        
    except Exception as e:
        logger.error(f"Error extracting requirements from chunk: {str(e)}")
        # Return empty list to allow processing to continue
        return []
