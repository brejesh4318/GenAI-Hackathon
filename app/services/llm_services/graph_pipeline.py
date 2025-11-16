import uuid
import json
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import ToolMessage
from langchain_core.messages import AnyMessage
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver

from app.services.prompt_fetch import PromptFetcher
from app.utilities import dc_logger
from app.utilities.helper import Helper
from app.services.llm_services.llm_interface import LLMInterface 
from app.services.reponse_format import AgentFormat, FinalOutput
from app.utilities.singletons_factory import DcSingleton
from app.services.llm_services.graph_state import PipelineState 
from app.services.llm_services.tools.rag_tools import retrieve_by_standards, web_search_tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import InMemorySaver
import sqlite3

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})
conn = sqlite3.connect('langchain.db', check_same_thread=False)
memory = SqliteSaver(conn)


class GraphPipe(metaclass=DcSingleton):

    def __init__(self, llm: LLMInterface,  llm_tools: LLMInterface) :
        logger.info("Initializing GraphPipe")
        self.llm = llm
        self.llm_tools = llm_tools
        self.output_parser = PydanticOutputParser(pydantic_object=FinalOutput)
        self.brain_output_parser = PydanticOutputParser(pydantic_object=AgentFormat)
        self.tools = []
        self.bind_tools()
        self.graph = self.compile_graph() # Graph compilation is synchronous
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
        workflow_graph.add_node("context_agent",self.context_agent)
        workflow_graph.add_node("brain_agent", self.brain_agent)
        workflow_graph.add_node("interrupt_node", self.interrupt_node)
        workflow_graph.add_node("compliance_research", self.compliance_reacher_agent)
        workflow_graph.add_node("test_case_generator", self.test_case_generator)
        workflow_graph.add_node("test_case_validator", self.test_case_validator)
        # workflow_graph.add_node("test_case_file_generator", self.test_case_file_generator)

        workflow_graph.add_node("tool_node", self.tool_node)
        workflow_graph.add_edge("files_parser", "context_agent")
        workflow_graph.add_edge("context_agent", "brain_agent")
        workflow_graph.add_conditional_edges("brain_agent", self.brain_router, {"compliance_research": "compliance_research", "test_case_generator": "test_case_generator", "interrupt_node": "interrupt_node", "test_case_validator": "test_case_validator"})

        workflow_graph.add_conditional_edges("compliance_research", self.should_continue, {"tool_node": "tool_node", "test_case_generator": "test_case_generator"})
        workflow_graph.add_edge("tool_node", "compliance_research")
        # connect nodes in execution order
        workflow_graph.add_edge("test_case_generator", "brain_agent")
        workflow_graph.add_edge("interrupt_node", "brain_agent")
        # workflow_graph.add_edge("test_case_validator", "test_case_file_generator")
        workflow_graph.add_edge("test_case_validator", END)
        workflow_graph = workflow_graph.compile(checkpointer=memory)
        logger.info("Workflow graph compiled successfully")
        # png = workflow_graph.get_graph().draw_mermaid_png()
        # with open("workflow_graph.png", "wb") as f:
        #     f.write(png)
        return workflow_graph
    
    def invoke_graph(self, document_path, config) -> PipelineState:
        logger.info(f"Invoking graph with document_path: {document_path}")
        result = self.graph.invoke({"file_path":document_path }, config=config)
        logger.info("Graph invocation completed")
        return result
    
    def resume_graph(self, command, config) -> PipelineState:
        logger.info(f"Resuming graph with command: {command}")
        result = self.graph.invoke(Command(resume=command), config=config)
        logger.info("Graph Resumption completed")
        return result

    def file_parser(self, state: PipelineState):
        logger.info(f"Parsing file: {state['file_path']}")
        document = Helper.read_file(file_path=state["file_path"])
        logger.info(f"Document parsed: {len(document.split())} words")
        return {"document": document}



    
    # def compliance_planner_agent(self, state: dict):
    #     logger.info("Running compliance planner agent")
    #     result = Helper.plan_compliance(self.llm, process_document=state["document"])
    #     logger.debug(f"Compliance plan result: {result}")
    #     return result 

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
    
    def should_continue(self, state: PipelineState):
        messages = state["messages"]
        last_message = messages[-1]
        logger.info("Checking if should continue or end")
        if last_message.tool_calls:
            logger.info("Tool call detected, continuing to tool_node")
            return "tool_node"
        logger.info("No tool call detected, ending workflow")
        return "test_case_generator"
    
    def brain_router(self, state: PipelineState):
        next_action = state["next_action"]
        logger.info(f"Checking if checking brain route or end for action {next_action}")
        if next_action == "send_compliance_planner":
            return "compliance_research"
        elif next_action == "test_case_generate":
            return "test_case_generator"
        elif next_action == "user_interrupt":
            return "interrupt_node"
        elif next_action == "process_complete":
            return "test_case_validator"
        else:
            logger.warning(f"Unknown next action: {next_action}, ending workflow")
            return END



    def brain_agent(self, state: PipelineState):
        logger.info("Running brain agent")
        result = Helper.brain_agent(self.llm_tools, state=state, output_parser=self.brain_output_parser)
        state["status"] = result.status
        state["next_action"] = result.next_action
        state["summary"] = result.summary
        state["scratchpad"] = result.scratchpad
        if result.user_interrupt: 
            state["user_interrupt"] = result.user_interrupt 
        logger.debug(f"Brain agent result: {result}")
        return state
    
    def context_agent(self, state: dict):
        logger.info("Running context agent")
        result = Helper.context_builder(self.llm, state=state)
        logger.debug(f"Context agent result: {result}")
        state["document"] = result
        state["brain_agent_message"] = f"Context Generation is completed here is the context information {state["document"]}"
        return state
    
    def interrupt_node(self, state: PipelineState):
        logger.info("Running interrupt node")
        user_messager = interrupt(state["user_interrupt"])
        state["brain_agent_message"] = f"here is the user messages {user_messager} for the question asked {state['user_interrupt']}" 
        state["user_interrupt"] = ""  # reset after processing
        return state
    
    def compliance_reacher_agent(self, state: dict):
        logger.info("Running compliance reacher agent")
        result = Helper.compliance_answer(self.llm_with_tools, state=state)
        logger.debug(f"Compliance answer result: {result}")
        return result
    

    def test_case_file_generator(self, state: PipelineState):
        logger.info("Saving final test cases to a JSON file")
        filename = f"test_case_{uuid.uuid4().hex}.json"
        with open(filename, "w") as f:
            json.dump(state["test_cases_final"], f, indent=4)
        logger.info(f"Test cases saved to {filename}")
        return {"output_file": filename}
    

    def test_case_generator(self, state: PipelineState):
        logger.info("Generating initial test cases from document")
        testcases_details = Helper.generate_testcase(llm = self.llm, document=state["document"], compliance_info=state["messages"][-1].content)
        logger.debug(f"Generated test cases: {testcases_details}")
        state["test_cases_lv1"] = testcases_details
        state["brain_agent_message"] = f"Initial test case generation completed test cases and test cases are {testcases_details}"
        return state #TODO change this return

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