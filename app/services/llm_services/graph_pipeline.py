import operator
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from app.utilities import dc_logger
from app.utilities.constants import Constants
from app.utilities.helper import Helper
from app.services.llm_services.llm_interface import LLMInterface
from app.services.reponse_format import FinalOutput
from app.utilities.singletons_factory import DcSingleton
from app.services.llm_services.graph_state import PipelineState
from app.services.llm_services.tools.rag_tools import retrieve_by_standards, web_search_tool
from langchain.output_parsers import PydanticOutputParser
import uuid
from langchain_core.messages import ToolMessage
import json
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langchain_core.messages import AnyMessage



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
        workflow_graph.set_entry_point("files_parser")
        workflow_graph.add_node("files_parser", self.file_parser)
        workflow_graph.add_node("test_case_generator", self.test_case_generator)
        workflow_graph.add_node("test_case_validator", self.test_case_validator)
        workflow_graph.add_node("test_case_file_generator", self.test_case_file_generator)
        
        workflow_graph.add_node("plan_compliance", self.compliance_planner_agent)
        workflow_graph.add_node("compliance_answer", self.compliance_reacher_agent)
        workflow_graph.add_node("tool_node", self.tool_node)

        workflow_graph.add_edge("files_parser", "plan_compliance")
        workflow_graph.add_edge("plan_compliance", "compliance_answer")
        workflow_graph.add_conditional_edges("compliance_answer", self.should_continue, {"tool_node": "tool_node", "test_case_generator": "test_case_generator"})
        workflow_graph.add_edge("tool_node", "compliance_answer")
        # connect nodes in execution order
        workflow_graph.add_edge("test_case_generator", "test_case_validator")
        workflow_graph.add_edge("test_case_validator", "test_case_file_generator")
        workflow_graph.add_edge("test_case_file_generator", END)

        return workflow_graph.compile()
    
    def invoke_graph(self, document_path) -> PipelineState:
        return self.graph.invoke({"file_path":document_path })

    def file_parser(self, state: PipelineState):
        """Reads input document file"""
        document = Helper.read_file(file_path=state["file_path"])
        return {"document": document}

    def test_case_generator(self, state: PipelineState):
        """Generate initial test cases from document"""
        testcases_details = Helper.generate_testcase(llm = self.llm, document=state["document"])
        return {"test_cases_lv1": testcases_details}

    def test_case_validator(self, state: PipelineState):
        """Validate and refine test cases"""
        testcases_details = Helper.validator(
            llm=self.llm,
            document=state["document"],
            llm_output=state["test_cases_lv1"],
            output_parser=self.output_parser
        )
        return {"test_cases_final": testcases_details}

    def test_case_file_generator(self, state: PipelineState):
        """Save final test cases to a JSON file"""
        filename = f"test_case_{uuid.uuid4().hex}.json"
        with open(filename, "w") as f:
            json.dump(state["test_cases_final"], f, indent=4)
        logger.info(f"Test cases saved to {filename}")
        return {"output_file": filename}
    
    def compliance_planner_agent(self, state: dict):
        return Helper.plan_compliance(self.llm, state=state)
    
    def compliance_reacher_agent(self, state: dict):
        return Helper.compliance_answer(self.llm_with_tools, state=state)

    
    
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
        agent_builder.add_node("plan_compliance", self.plan_compliance)
        agent_builder.add_node("compliance_answer", self.compliance_answer)
        agent_builder.add_node("tool_node", self.tool_node)

        agent_builder.set_entry_point("plan_comliance")
        agent_builder.add_edge("plan_comliance", "compliance_answer")
        agent_builder.add_conditional_edges("compliance_answer", self.should_continue, {"tool_node": "tool_node", END: END})
        agent_builder.add_edge("tool_node", "compliance_answer")

        return agent_builder.compile()
    
    
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

