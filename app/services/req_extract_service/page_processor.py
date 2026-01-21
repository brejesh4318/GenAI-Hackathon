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
    project_id: str,
    version_id: str,
    pages: List[str],
    hashed_pages: List[Dict],
    previous_version_id: str,
    storage_handler: RequirementsStorage,
    helper,
    model_name: str,
    pages_per_chunk: int,
    page_filter_func
) -> List[Dict]:
    """
    Extract requirements from a list of page texts with intelligent diffing.
    
    Args:
        pages (List[str]): List of page texts
        hashed_pages (List[Dict]): Pages with hashes for comparison
        previous_version_id (str): Previous version UUID for diffing (None for V1)
        storage_handler: Storage instance for retrieving previous requirements
        helper: Helper instance for hash generation
        model_name (str): Model name for LLM
        pages_per_chunk (int): Pages to process per LLM call
        page_filter_func: Function to determine if page needs processing
    
    Returns:
        List[Dict]: List of requirement dicts with hashes and IDs
    """
    logger.info(f"Processing {len(pages)} pages for requirement extraction")
    
    # Identify pages that need LLM processing
    pages_to_process = []
    for page_dict in hashed_pages:
        if page_filter_func(page_dict, previous_version_id, project_id, version_id):
            pages_to_process.append(page_dict)
    
    logger.info(f"{len(pages_to_process)}/{len(pages)} pages need LLM processing")
    
    # If no pages changed and we have previous version, copy previous requirements
    if not pages_to_process and previous_version_id:
        logger.info("No pages changed, reusing previous requirements")
        prev_reqs = storage_handler.get_requirements_by_version(project_id, previous_version_id)
        # Regenerate hashes using utils
        prev_reqs = req_utils.enrich_requirements_with_hashes(prev_reqs, helper)
        return prev_reqs
    
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
            chunk_requirements = llm_wrapper.extract_requirements_via_llm(
                chunk_text=chunk_text,
                page_numbers=page_numbers,
                start_req_id=current_req_id,
                model_name=model_name
            )
            
            # Accumulate requirements
            all_requirements.extend(chunk_requirements)
            current_req_id += len(chunk_requirements)
            
            logger.info(f"Extracted {len(chunk_requirements)} requirements from chunk {chunk_idx + 1}")
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_idx + 1}: {str(e)}", exc_info=True)
            # Continue processing remaining chunks
            continue
    
    # Generate hashes and IDs in backend (not from LLM)
    all_requirements = req_utils.enrich_requirements_with_hashes(all_requirements, helper)
    
    logger.info(f"Total requirements extracted: {len(all_requirements)}")
    return all_requirements
