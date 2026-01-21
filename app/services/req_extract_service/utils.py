"""
Utility functions for requirements extraction service.
Handles page chunking, hashing, diffing, and text preparation.
"""
import hashlib
from typing import List, Dict, Tuple
from app.utilities import dc_logger

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})


def create_hashed_pages(pages: List[str], normalize_func) -> List[Dict]:
    """
    Create page dictionaries with normalized text and SHA-256 hashes.
    
    Args:
        pages (List[str]): List of raw page texts
        normalize_func: Function to normalize text (e.g., Helper.normalize_text)
    
    Returns:
        List[Dict]: List of page dicts with page_no, raw_text, normalized_text, page_hash
    """
    hashed_pages = []
    
    for idx, page_text in enumerate(pages):
        # Normalize page text for consistent hashing
        normalized = normalize_func(page_text)
        
        # Calculate SHA-256 hash of normalized text
        page_hash = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        
        hashed_pages.append({
            "page_no": idx + 1,
            "raw_text": page_text,
            "normalized_text": normalized,
            "page_hash": page_hash
        })
    
    logger.debug(f"Created hashes for {len(hashed_pages)} pages")
    return hashed_pages


def chunk_pages(pages: List[str], pages_per_chunk: int = 10) -> List[Tuple[str, List[int]]]:
    """
    Split pages into chunks for LLM processing with page annotations.
    
    Args:
        pages (List[str]): List of page texts
        pages_per_chunk (int): Number of pages per chunk (default: 10)
    
    Returns:
        List[Tuple[str, List[int]]]: List of (chunk_text, page_numbers)
            - chunk_text: Annotated text with page markers
            - page_numbers: List of page numbers in this chunk (1-indexed)
    
    Example:
        >>> pages = ["Page 1 content", "Page 2 content", "Page 3 content"]
        >>> chunks = chunk_pages(pages, pages_per_chunk=2)
        >>> print(len(chunks))  # 2 chunks
        >>> print(chunks[0][1])  # [1, 2]
    """
    chunks = []
    
    for i in range(0, len(pages), pages_per_chunk):
        # Get slice of pages for this chunk
        chunk_pages_list = pages[i:i + pages_per_chunk]
        
        # Calculate page numbers (1-indexed)
        page_numbers = list(range(i + 1, i + len(chunk_pages_list) + 1))
        
        # Prepare annotated chunk text
        chunk_text = prepare_chunk_text(chunk_pages_list, page_numbers)
        
        chunks.append((chunk_text, page_numbers))
    
    return chunks


def chunk_pages_with_numbers(pages_with_numbers: List[Tuple[str, int]], pages_per_chunk: int = 10) -> List[Tuple[str, List[int]]]:
    """
    Split pages into chunks for LLM processing while preserving original page numbers.
    
    Args:
        pages_with_numbers (List[Tuple[str, int]]): List of (page_text, original_page_number)
        pages_per_chunk (int): Number of pages per chunk (default: 10)
    
    Returns:
        List[Tuple[str, List[int]]]: List of (chunk_text, page_numbers)
            - chunk_text: Annotated text with page markers
            - page_numbers: List of ORIGINAL page numbers in this chunk (1-indexed)
    
    Example:
        >>> pages = [("Page 1 content", 1), ("Page 5 content", 5), ("Page 10 content", 10)]
        >>> chunks = chunk_pages_with_numbers(pages, pages_per_chunk=2)
        >>> print(chunks[0][1])  # [1, 5] - original page numbers preserved
    """
    chunks = []
    
    for i in range(0, len(pages_with_numbers), pages_per_chunk):
        # Get slice for this chunk
        chunk_items = pages_with_numbers[i:i + pages_per_chunk]
        
        # Extract page texts and original page numbers
        chunk_pages_list = [item[0] for item in chunk_items]
        page_numbers = [item[1] for item in chunk_items]
        
        # Prepare annotated chunk text with ORIGINAL page numbers
        chunk_text = prepare_chunk_text(chunk_pages_list, page_numbers)
        
        chunks.append((chunk_text, page_numbers))
    
    logger.debug(f"Created chunk with original page numbers: {page_numbers}")
    return chunks


def prepare_chunk_text(pages: List[str], page_numbers: List[int]) -> str:
    """
    Prepare chunk text with page annotations.
    
    Args:
        pages (List[str]): List of page texts for this chunk
        page_numbers (List[int]): Corresponding page numbers (1-indexed)
    
    Returns:
        str: Annotated chunk text with page markers
    
    Format:
        === PAGE 1 ===
        <page 1 content>
        === PAGE 2 ===
        <page 2 content>
    """
    annotated_sections = []
    
    for page_text, page_no in zip(pages, page_numbers):
        # Add page marker
        section = f"=== PAGE {page_no} ===\n{page_text}"
        annotated_sections.append(section)
    
    # Join all sections with double newline separator
    chunk_text = "\n\n".join(annotated_sections)
    return chunk_text


def compare_requirements_by_hash(
    current_requirements: List[Dict],
    previous_requirements: List[Dict]
) -> Dict[str, List[Dict]]:
    """
    Compare current requirements with previous version using hashes.
    
    Args:
        current_requirements (List[Dict]): New requirements with hashes
        previous_requirements (List[Dict]): Previous requirements with hashes
    
    Returns:
        Dict: {
            "unchanged": List[Dict],  # Reuse existing tests
            "new": List[Dict],        # Generate tests
            "modified": List[Dict],   # Regenerate tests
            "removed": List[Dict]     # Mark tests obsolete
        }
    """
    if not previous_requirements:
        logger.warning("No previous requirements found, treating all as new")
        return {
            "unchanged": [],
            "new": current_requirements,
            "modified": [],
            "removed": []
        }
    
    # Build hash lookup
    prev_hash_map = {req["requirement_hash"]: req for req in previous_requirements}
    curr_hash_set = {req["requirement_hash"] for req in current_requirements}
    prev_hash_set = set(prev_hash_map.keys())
    
    # Categorize requirements
    unchanged = []
    new = []
    modified = []
    
    for curr_req in current_requirements:
        curr_hash = curr_req["requirement_hash"]
        
        if curr_hash in prev_hash_map:
            # Hash exists → requirement unchanged
            unchanged.append(curr_req)
        else:
            # Check if req_id exists with different hash (rare: modified)
            prev_req_with_id = next(
                (r for r in previous_requirements if r.get("req_id") == curr_req["req_id"]),
                None
            )
            if prev_req_with_id:
                modified.append(curr_req)
            else:
                new.append(curr_req)
    
    # Find removed requirements
    removed_hashes = prev_hash_set - curr_hash_set
    removed = [prev_hash_map[h] for h in removed_hashes]
    
    logger.info(f"Diff: {len(unchanged)} unchanged, {len(new)} new, "
               f"{len(modified)} modified, {len(removed)} removed")
    
    return {
        "unchanged": unchanged,
        "new": new,
        "modified": modified,
        "removed": removed
    }


def should_process_page(
    page_hash: str,
    page_no: int,
    previous_pages: List[Dict]
) -> bool:
    """
    Determine if a page needs LLM processing based on hash comparison.
    
    Args:
        page_hash (str): SHA-256 hash of current page
        page_no (int): Page number
        previous_pages (List[Dict]): Previous version's pages
    
    Returns:
        bool: True if page should be sent to LLM, False if unchanged
    """
    # For V1 (no previous pages), process all pages
    if not previous_pages:
        logger.debug(f"Page {page_no}: No previous pages, processing")
        return True
    
    # Find matching page in previous version
    prev_page = next((p for p in previous_pages if p["page_no"] == page_no), None)
    
    if not prev_page:
        logger.debug(f"Page {page_no}: New page, processing")
        return True
    
    # Compare hashes
    prev_hash = prev_page.get("page_hash")
    if prev_hash == page_hash:
        logger.debug(f"Page {page_no}: Hash unchanged, skipping LLM")
        return False
    else:
        logger.debug(f"Page {page_no}: Hash changed, processing")
        return True


def enrich_requirements_with_hashes(requirements: List[Dict], helper) -> List[Dict]:
    """
    Generate hashes and IDs for extracted requirements.
    
    Args:
        requirements (List[Dict]): Requirements with text field
        helper: Helper instance with hash generation methods
    
    Returns:
        List[Dict]: Requirements enriched with normalized_text, requirement_hash, req_id, status
    """
    for req in requirements:
        req["normalized_text"] = helper.normalize_text(req["text"])
        req["requirement_hash"] = helper.generate_requirement_hash(req["text"])
        req["req_id"] = helper.generate_requirement_id(req["text"])
        req["status"] = "active"
    
    return requirements
