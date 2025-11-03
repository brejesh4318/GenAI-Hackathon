from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


class TestCase(BaseModel):
    test_case_id: str = Field(..., description="Unique ID for the test case (e.g., TC_Login_001)")
    # requirement_id: Optional[str] = Field(None, description="Requirement ID from PRD, if available")
    feature: Optional[str] = Field(None, description="Feature or module name")
    title: str = Field(..., description="Short description of the test case")
    type: str = Field(..., description="Type of test case: Positive, Negative, Edge")
    priority: str = Field(..., description="Priority of the test case: High, Medium, Low")
    preconditions: Optional[List[str]] = Field(default_factory=list, description="Conditions required before execution")
    test_data: Optional[Dict[str, str]] = Field(default_factory=dict, description="Input data required for test case")
    steps: List[str] = Field(..., description="Step-by-step instructions to execute the test")
    expected_result: List[str] = Field(..., description="Expected outcome(s) after executing the test")
    postconditions: Optional[List[str]] = Field(default_factory=list, description="System state after execution")
    compliance_reference_standard: Optional[str] = Field(None, description="Applicable compliance clause or control, if relevant (e.g., IEC). Leave empty if not applicable")
    compliance_reference_clause: Optional[str] = Field(None, description="Specific clause or control from the compliance standard that applies to this test case, if relevant. Leave empty if not applicable.")
    compliance_reference_requirement_text: Optional[str] = Field(None, description="The specific requirement text from the compliance standard that applies to this test case, if relevant. Leave empty if not applicable.")


class AgentFormat(BaseModel):
    status: str
    next_action: str
    summary:  str
    scratchpad: dict
    user_interrupt: Optional[str]


class FinalOutput(BaseModel):
    test_cases: List[TestCase]


output_parser = PydanticOutputParser(pydantic_object=FinalOutput)


class ComplianceCitation(BaseModel):
    standard: str = Field(..., description="Standard name, e.g., IEC-62304 or FDA")
    section: Optional[str] = Field(None, description="Section identifier or title")
    page_number: Optional[int] = Field(None, description="Page number if available")


class ComplianceSection(BaseModel):
    standard: str = Field(..., description="Which standard this section maps to")
    section: Optional[str] = Field(None, description="Section identifier or title (e.g., 4.3, 5.1)")
    summary: str = Field(..., description="Concise summary relevant to the requirement")
    citations: List[ComplianceCitation] = Field(default_factory=list)


class ComplianceAnswer(BaseModel):
    requirement: str = Field(..., description="Original requirement or user need")
    standards_covered: List[str] = Field(..., description="List of standards used (e.g., [IEC-62304, FDA])")
    sections: List[ComplianceSection] = Field(..., description="Relevant sections per standard with summaries and citations")
    overall_guidance: str = Field(..., description="Short, actionable guidance synthesizing the standards")


compliance_output_parser = PydanticOutputParser(pydantic_object=ComplianceAnswer)