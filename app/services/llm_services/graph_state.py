import operator
from typing import Annotated, TypedDict, List, Dict, Optional
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """
    Represents the state of the agentic workflow.
    Integrates new architecture with existing system compatibility.
    """
    user_request: str
    file_path: list[str]
    document: str
    images: List[str]
    context: str
    context_summary: Optional[str]
    context_built: bool
    test_cases: List[dict]
    test_cases_lv1: Optional[str]
    test_cases_summary: Optional[str]
    test_cases_status: Optional[str]
    validation_status: Optional[str]
    next_node: Optional[str]
    context_agent_messages: Annotated[List[AnyMessage], add_messages]


# Backward compatibility aliases
PipelineState = AgentState


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