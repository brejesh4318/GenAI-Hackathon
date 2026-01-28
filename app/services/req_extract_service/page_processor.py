"""
Page processing module for requirements extraction.
Handles LLM extraction loop and chunking logic.
"""
from typing import List, Dict
from app.services.req_extract_service import utils as req_utils
from app.services.req_extract_service import llm_wrapper
from app.services.req_extract_service.storage import RequirementsStorage
from app.utilities import dc_logger

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})


def process_pages_for_requirements(
    project_id: int,
    version_id: int,
    pages: List[str],
    hashed_pages: List[Dict],
    previous_version_id: int,
    storage_handler: RequirementsStorage,
    helper,
    model_name: str,
    pages_per_chunk: int,
    page_filter_func
) -> tuple[List[Dict], List[int]]:
    """
    Extract requirements from changed pages only.
    
    Args:
        project_id: Project ID
        version_id: Current version ID
        pages (List[str]): List of page texts
        hashed_pages (List[Dict]): Pages with hashes for comparison
        previous_version_id (int): Previous version ID for diffing (None for V1)
        storage_handler: Storage instance for retrieving previous requirements
        helper: Helper instance for hash generation
        model_name (str): Model name for LLM
        pages_per_chunk (int): Pages to process per LLM call
        page_filter_func: Function to determine if page needs processing
    
    Returns:
        Tuple[List[Dict], List[int]]: 
            - List of requirements from changed pages
            - List of unchanged page numbers
    """
    logger.info(f"Processing {len(pages)} pages for requirement extraction")
    
    # Identify pages that need LLM processing
    pages_to_process = []
    unchanged_page_numbers = []
    
    for page_dict in hashed_pages:
        if page_filter_func(page_dict, previous_version_id, project_id, version_id):
            pages_to_process.append(page_dict)
        else:
            unchanged_page_numbers.append(page_dict["page_no"])
    
    logger.info(f"{len(pages_to_process)}/{len(pages)} pages changed, {len(unchanged_page_numbers)} unchanged")
    
    # If no pages changed, return empty
    if not pages_to_process:
        logger.info("No pages changed, no new requirements to extract")
        return [], unchanged_page_numbers
    
    # Get pages to process for LLM (use original page texts and maintain page numbers)
    pages_with_numbers = []
    for page_dict in pages_to_process:
        page_idx = page_dict["page_no"] - 1
        pages_with_numbers.append((pages[page_idx], page_dict["page_no"]))
    
    # Chunk pages for LLM processing with original page numbers
    chunks = req_utils.chunk_pages_with_numbers(pages_with_numbers, pages_per_chunk)
    logger.info(f"Created {len(chunks)} chunks ({pages_per_chunk} pages/chunk)")
    
    # Extract requirements from each chunk
    all_requirements = []
    current_req_id = 1
    
    for chunk_idx, (chunk_text, page_numbers) in enumerate(chunks):
        logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunks)} (pages {page_numbers[0]}-{page_numbers[-1]})")
        
        try:
            # Call LLM to extract requirements from this chunk
            # chunk_requirements = llm_wrapper.extract_requirements_via_llm(
            #     chunk_text=chunk_text,
            #     page_numbers=page_numbers,
            #     start_req_id=current_req_id,
            #     model_name=model_name
            # )
            
            # # Accumulate requirements
            # all_requirements.extend(chunk_requirements)
            # current_req_id += len(chunk_requirements)
            
            # logger.info(f"Extracted {len(chunk_requirements)} requirements from chunk {chunk_idx + 1}")
            pass #TODO: Temporarily disable LLM calls
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_idx + 1}: {str(e)}", exc_info=True)
            # Continue processing remaining chunks
            continue
    with open(r"D:\projects\GenAI-Hackathon\test-req_42.json", "r") as f:
        import json #TODO: Remove after testing
        all_requirements = json.load(f)
    
    # Generate hashes and IDs in backend (not from LLM)
    all_requirements = req_utils.enrich_requirements_with_hashes(all_requirements, helper)
    
    logger.info(f"Extracted {len(all_requirements)} requirements from {len(pages_to_process)} changed pages")
    return all_requirements, unchanged_page_numbers
