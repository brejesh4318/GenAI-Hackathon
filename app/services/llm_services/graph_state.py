import operator
from typing import Annotated, TypedDict, List, Dict, Optional
from langchain_core.messages import AnyMessage


class PipelineState(TypedDict):
    file_path: str
    document: str
    test_cases_lv1: str
    test_cases_final: dict
    compliance_plan: str
    messages: Annotated[list[AnyMessage], operator.add]


class ComplianceState(TypedDict, total=False):
    # Input
    requirement: str
    prd: Optional[str]
    mock_text: Optional[str]
    preferred_doc: Optional[str]  # exact standard name(s) optional hint

    # Planning
    search_query: str
    standards_to_check: List[str]

    # Retrieval
    retrieved_docs: List[Dict]
    reranked_docs: List[Dict]

    # Output
    final_answer: str
    citations: List[Dict]