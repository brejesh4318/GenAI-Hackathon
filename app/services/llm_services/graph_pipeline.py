import operator
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from app.utilities import dc_logger
from app.utilities.constants import Constants
from app.utilities.helper import Helper
from app.services.llm_services.llm_interface import LLMInterface
from app.services.reponse_format import FinalOutput, ComplianceAnswer, compliance_output_parser
from app.utilities.singletons_factory import DcSingleton
from app.services.llm_services.graph_state import PipelineState
from app.services.llm_services.tools.rag_tools import retrieve_by_standards, web_search_tool
from langchain.output_parsers import PydanticOutputParser
import uuid
from langchain_core.messages import ToolMessage
import json
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langchain_core.messages import AnyMessage
from langchain_core.messages import SystemMessage



logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})
class MessagesState(TypedDict):
    compliance_plan: str
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


class GraphPipe(metaclass=DcSingleton):

    def __init__(self, llm: LLMInterface) :
        # Initialize LLM and parser once
        self.llm = llm
        self.output_parser = PydanticOutputParser(pydantic_object=FinalOutput)
        self.graph = self.compile_graph()
        self.tools = []
        self.bind_tools()

    def bind_tools(self):
        tavily = web_search_tool()
        self.tools.append(tavily)
        self.tools.append(retrieve_by_standards)
        llm = self.llm.get_llm()
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.tools_by_name = {tool.name: tool for tool in self.tools}

    def compile_graph(self) -> CompiledStateGraph:
        workflow_graph = StateGraph(PipelineState)

        # ---------- build graph: add nodes ----------
        workflow_graph.add_node("files_parser", self.file_parser)
        workflow_graph.add_node("test_case_generator", self.test_case_generator)
        workflow_graph.add_node("test_case_validator", self.test_case_validator)
        workflow_graph.add_node("test_case_file_generator", self.test_case_file_generator)
        workflow_graph.set_entry_point("files_parser")

        # connect nodes in execution order
        workflow_graph.add_edge("files_parser", "test_case_generator")
        workflow_graph.add_edge("test_case_generator", "test_case_validator")
        workflow_graph.add_edge("test_case_validator", "test_case_file_generator")
        workflow_graph.add_edge("test_case_file_generator", END)

        return workflow_graph.compile()
    
    def invoke_graph(self, document_path) -> PipelineState:
        return self.graph.invoke({"file_path":document_path })

    # ---------------- Graph Nodes ---------------- #

    def file_parser(self, state: PipelineState):
        """Reads input document file"""
        document = Helper.read_file(file_path=state["file_path"])
        return {"document": document}

    def test_case_generator(self, state: PipelineState):
        """Generate initial test cases from document"""
        testcases_details = self.generate_testcase(state["document"])
        return {"test_cases_lv1": testcases_details}

    def test_case_validator(self, state: PipelineState):
        """Validate and refine test cases"""
        testcases_details = self.validator(
            document=state["document"],
            llm_output=state["test_cases_lv1"],
            output_format=self.output_parser.get_format_instructions()
        )
        return {"test_cases_final": testcases_details}

    def test_case_file_generator(self, state: PipelineState):
        """Save final test cases to a JSON file"""
        filename = f"test_case_{uuid.uuid4().hex}.json"
        with open(filename, "w") as f:
            json.dump(state["test_cases_final"], f, indent=4)
        logger.info(f"Test cases saved to {filename}")
        return {"output_file": filename}

    # ---------------- Internal Helpers ---------------- #

    def generate_testcase(self, document: str):
        
        """Calls LLM to generate test cases"""
        prompt = f"""
                You are a **QA test designer**. Analyze the uploaded PRD (Product Requirement Document) and functionality documents. From these, generate **all possible high-level QA test cases**, covering:  
                - Functional scenarios  
                - Negative scenarios  
                - Edge cases  
                - Performance considerations  
                - Security validations  
                
                input document:{document}

            Output the test cases in the following structured template:  
                ---
                **Feature** (derived from requirement)
                **Category:** [Functional | Negative | Edge | Performance | Security]  
                **Test Case ID:** [Unique ID]  
                **Title:** [Concise test case title]  
                **Description:** [High-level explanation of what to validate]  
                **Preconditions:** [If any]  
                **Steps** (clear step-by-step actions)
                **Expected Outcome:** [What should happen]  
                *Test Data** (mock or example data, if needed)
                ---

                **Guidelines:**  
                - Do not go into step-by-step detail; keep them **high-level** for QA engineers to refine.  
                - Ensure **completeness** by covering all categories.  
                - Maintain clarity and consistency in wording.  
                - Where requirements are ambiguous, propose multiple possible test cases.  
                    # Example QC results per chunk
            """
        llm_output = self.llm.generate(prompt)
        logger.info("LLM Test Case Generation Completed")
        return llm_output

    def validator(self, document: str, llm_output: str, output_format: str):
        
        """Validate test cases with LLM and enforce structured format"""
        prompt = f'''You are a **QA test case validator**. Your task is to validate the test cases generated from a Product Requirement Document (PRD) and functionality documents.  
                Input Document: {document}
                Generated Test cases {llm_output}
                
                ### **Validation Scope:**  
                - Ensure test cases cover **all categories**: Functional, Negative, Edge, Performance, Security.  
                - Verify each test case includes the required fields:  
                - test_id  
                - test_case  
                - test_description  
                - expected_outcome  
                - test_severity (Low / High / Critical)  
                - Identify **gaps**: missing categories, vague wording, missing outcomes, or unclear severity.  
                - Detect **duplicates or overlaps**.  
                - Suggest **improvements** for incomplete or ambiguous cases.  

                ### **Output Format:**  
                Return the validation results **strictly as a list of dictionaries**, with each dictionary containing the following fields:  
                {output_format}
                ### **Guidelines:**  

                * Include only validated test cases in the output.  
                *  If issues are found (e.g., missing field, unclear description), **correct them before adding** to the dictionary.  
                *  If a category is missing (Functional, Negative, Edge, Performance, Security), **add new test cases** to cover it.  
                * Maintain **clarity, consistency, and completeness**.  
                * Assign **test_severity** logically:  

                * **Critical** → Security, authentication, and data integrity cases.  
                * **High** → Core functional and performance cases.  
                * **Low** → Minor UI/UX or non-blocking cases.  
            '''

        validated_output = self.llm.generate(prompt)
        logger.info("LLM Test Case Validation Completed")

        parsed_output = self.output_parser.parse(validated_output)
        return parsed_output.model_dump()
    

class ComplianceGraphPipe(metaclass=DcSingleton):
    def __init__(self, llm: LLMInterface) :
        # Initialize LLM and parser once
        self.llm = llm
        # self.output_parser = PydanticOutputParser(pydantic_object=ComplianceAnswer)
        self.graph = self.compile_graph()
        self.tools = []
        self.bind_tools()

    def bind_tools(self):
        tavily = web_search_tool()
        self.tools.append(tavily)
        self.tools.append(retrieve_by_standards)
        llm = self.llm.get_llm()
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
    def compile_graph(self) -> CompiledStateGraph:
        agent_builder = StateGraph(MessagesState)
        agent_builder.add_node("plan_comliance", self.plan_comliance)
        agent_builder.add_node("compliance_answer", self.compliance_answer)
        agent_builder.add_node("tool_node", self.tool_node)

        agent_builder.set_entry_point("plan_comliance")
        agent_builder.add_edge("plan_comliance", "compliance_answer")
        agent_builder.add_conditional_edges("compliance_answer", self.should_continue, {"tool_node": "tool_node", END: END})
        agent_builder.add_edge("tool_node", "compliance_answer")

        return agent_builder.compile()
    
    def plan_comliance(self, process_document):
        """Plan compliance steps based on document"""
        # Example standards to check against
        standards = ["FDA 21 CFR Part 820", "IEC 62304"]

        prompt = Constants.fetch_constant("prompts")["compliance_agent1"].format(process_document=process_document, standards=standards)
        llm_output = self.llm.generate(prompt)
        logger.info("LLM Compliance Planning Completed")
        return {"compliance_plan": llm_output}
    
    def compliance_answer(self, state: dict):
        """
        Enriches compliance requirements with authoritative details using RAG and web search tools.
        """

        logger.info("Sending LLM Compliance Enrichment Request")
        # enriched_output = self.llm_with_tools.invoke({"messages": [prompt] + state["messages"]})
        message = {
        "messages": [
            self.llm_with_tools.invoke(
                [
                    SystemMessage(
                        content= Constants.fetch_constant("prompts")["compliance_agent2"]
                    )
                ] + ["Compliance Plan: " + state["compliance_plan"]]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }
        logger.info("LLM Compliance Enrichment Completed")
        return message
    
    def tool_node(self, state: dict):
        """Performs the tool call"""
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = self.tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}
    
    def should_continue(self, state: MessagesState):
        """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
        messages = state["messages"]
        last_message = messages[-1]
        # If the LLM makes a tool call, then perform an action
        if last_message.tool_calls:
            return "tool_node"
        # Otherwise, we stop (reply to the user)
        return END

