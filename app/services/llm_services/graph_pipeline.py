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

    def __init__(self, llm: LLMInterface,  llm_tools: LLMInterface) :
        logger.info("Initializing GraphPipe")
        self.llm = llm
        self.llm_tools = llm_tools
        self.output_parser = PydanticOutputParser(pydantic_object=FinalOutput)
        self.graph = self.compile_graph()
        self.tools = []
        self.bind_tools()
        logger.info("GraphPipe initialized successfully")

    def bind_tools(self):
        logger.info("Binding tools to LLM")
        tavily = web_search_tool()
        self.tools.append(tavily)
        self.tools.append(retrieve_by_standards)
        llm = self.llm_tools.get_llm()
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        logger.info(f"Tools bound: {[tool.name for tool in self.tools]}")

    def compile_graph(self) -> CompiledStateGraph:
        logger.info("Compiling workflow graph")
        workflow_graph = StateGraph(PipelineState)

        # ---------- build graph: add nodes ----------
        workflow_graph.set_entry_point("files_parser")
        workflow_graph.add_node("files_parser", self.file_parser)
        workflow_graph.add_node("test_case_generator", self.test_case_generator)
        workflow_graph.add_node("test_case_validator", self.test_case_validator)
        # workflow_graph.add_node("test_case_file_generator", self.test_case_file_generator)
        
        workflow_graph.add_node("plan_compliance", self.compliance_planner_agent)
        workflow_graph.add_node("compliance_answer", self.compliance_reacher_agent)
        workflow_graph.add_node("tool_node", self.tool_node)

        workflow_graph.add_edge("files_parser", "plan_compliance")
        workflow_graph.add_edge("plan_compliance", "compliance_answer")
        workflow_graph.add_conditional_edges("compliance_answer", self.should_continue, {"tool_node": "tool_node", "test_case_generator": "test_case_generator"})
        workflow_graph.add_edge("tool_node", "compliance_answer")
        # connect nodes in execution order
        workflow_graph.add_edge("test_case_generator", "test_case_validator")
        # workflow_graph.add_edge("test_case_validator", "test_case_file_generator")
        workflow_graph.add_edge("test_case_validator", END)

        logger.info("Workflow graph compiled successfully")
        return workflow_graph.compile()
    
    def invoke_graph(self, document_path) -> PipelineState:
        logger.info(f"Invoking graph with document_path: {document_path}")
        result = self.graph.invoke({"file_path":document_path })
        logger.info("Graph invocation completed")
        return result

    def file_parser(self, state: PipelineState):
        logger.info(f"Parsing file: {state['file_path']}")
        document = Helper.read_file(file_path=state["file_path"])
        logger.info(f"Document parsed: {len(document.split())} words")
        return {"document": document}

    def test_case_generator(self, state: PipelineState):
        logger.info("Generating initial test cases from document")
        testcases_details = Helper.generate_testcase(llm = self.llm, document=state["document"], compliance_info=state["messages"][-1].content)
        logger.debug(f"Generated test cases: {testcases_details}")
        return {"test_cases_lv1": testcases_details}

    def test_case_validator(self, state: PipelineState):
        logger.info("Validating and refining test cases")
        testcases_details = Helper.validator(
            llm=self.llm_tools,
            document=state["document"],
            llm_output=state["test_cases_lv1"],
            output_parser=self.output_parser
        )
        logger.debug(f"Validated test cases: {testcases_details}")
        return {"test_cases_final": testcases_details}

    def test_case_file_generator(self, state: PipelineState):
        logger.info("Saving final test cases to a JSON file")
        filename = f"test_case_{uuid.uuid4().hex}.json"
        with open(filename, "w") as f:
            json.dump(state["test_cases_final"], f, indent=4)
        logger.info(f"Test cases saved to {filename}")
        return {"output_file": filename}
    
    def compliance_planner_agent(self, state: dict):
        logger.info("Running compliance planner agent")
        result = Helper.plan_compliance(self.llm, process_document=state["document"])
        logger.debug(f"Compliance plan result: {result}")
        return result
    
    def compliance_reacher_agent(self, state: dict):
        logger.info("Running compliance reacher agent")
        result = Helper.compliance_answer(self.llm_with_tools, state=state)
        logger.debug(f"Compliance answer result: {result}")
        return result

    def tool_node(self, state: dict):
        logger.info("Performing tool calls")
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            logger.info(f"Invoking tool: {tool_call['name']} with args: {tool_call['args']}")
            tool = self.tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            logger.debug(f"Tool {tool_call['name']} observation: {observation}")
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        logger.info("Tool calls completed")
        return {"messages": result}
    
    def should_continue(self, state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        logger.info("Checking if should continue or end")
        if last_message.tool_calls:
            logger.info("Tool call detected, continuing to tool_node")
            return "tool_node"
        logger.info("No tool call detected, ending workflow")
        return "test_case_generator"

