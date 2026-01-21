import uuid
import json
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from langchain_core.messages import AnyMessage
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from app.services.prompt_service import PromptService
from app.utilities import dc_logger
from app.utilities.helper import Helper
from app.utilities.constants import Constants
from app.services.llm_services.llm_interface import LLMInterface 
from app.services.llm_services.reponse_format import AgentFormat, FinalOutput
from app.utilities.singletons_factory import DcSingleton
from app.services.testcase_service.graph_state import AgentState, PipelineState
from app.services.testcase_service.tools.rag_tools import retrieve_by_standards, web_search_tool, interrupt_tool
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from langchain_core.exceptions import OutputParserException
from app.services.req_extract_service.requirements_extractor import RequirementsExtractor
import sqlite3

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})
conn = sqlite3.connect('langchain.db', check_same_thread=False)
memory = SqliteSaver(conn)
# parser = DocumentParser()
prompt_service = PromptService()
# Fetch prompts from Langfuse at startup
validation_agent_prompt = prompt_service.get("validator-agent")
test_case_generator_prompt = prompt_service.get("test-case-generator")
deep_extractor = RequirementsExtractor(pages_per_chunk=10)
class GraphPipe(metaclass=DcSingleton):

    def __init__(self, llm: LLMInterface, llm_tools: LLMInterface):
        logger.info("Initializing GraphPipe with new agentic architecture")
        self.llm = llm
        self.llm_tools = llm_tools
        self.output_parser = PydanticOutputParser(pydantic_object=FinalOutput)
        self.brain_output_parser = PydanticOutputParser(pydantic_object=AgentFormat)
        self.tools = []
        self.bind_tools()
        self.graph = self.compile_graph()  # Graph compilation is synchronous
        logger.info("GraphPipe initialized successfully")

    def bind_tools(self):
        """Bind all tools including interrupt, web search, and RAG tools"""
        logger.info("Binding tools to LLM")
        tavily = web_search_tool()
        self.tools.append(tavily)
        self.tools.append(retrieve_by_standards)
        self.tools.append(interrupt_tool)
        llm = self.llm_tools.get_llm()
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        logger.info(f"Tools bound: {[tool.name for tool in self.tools]}")

    def compile_graph(self) -> CompiledStateGraph:
        """Compile the workflow graph with new architecture"""
        logger.info("Compiling workflow graph with new agentic architecture")
        workflow_graph = StateGraph(AgentState)

        # Add nodes
        
        workflow_graph.add_node("file_parser", self.file_parser)
        workflow_graph.add_node("brain", self.brain_node)
        workflow_graph.add_node("context_builder", self.context_builder_node)
        workflow_graph.add_node("tools", ToolNode([interrupt_tool], messages_key="context_agent_messages"))
        workflow_graph.add_node("test_generator", self.test_generator_node)
        workflow_graph.add_node("validator", self.validator_node)

        # Set entry point and define edges
        workflow_graph.set_entry_point("file_parser")
        workflow_graph.add_edge("file_parser", "brain")

        logger.debug("Adding conditional edges for context_builder")
        workflow_graph.add_conditional_edges(
            "context_builder", 
            self.context_router, 
            {"tools": "tools", "brain": "brain", END: END}
        )

        # logger.info("Adding conditional edges for brain node")
        workflow_graph.add_conditional_edges(
            "brain",
            self.should_continue,
            {
                "context_builder": "context_builder",
                "test_generator": "test_generator",
                "validator": "validator",
                END: END,
            },
        )

        logger.info("Adding edges to route worker nodes back to brain")
        workflow_graph.add_edge("tools", "context_builder")
        workflow_graph.add_edge("test_generator", "brain")
        workflow_graph.add_edge("validator", END)

        # Compile with checkpointer
        workflow_graph = workflow_graph.compile(checkpointer=memory)
        logger.info("Workflow graph compiled successfully")
        return workflow_graph

    # --- Routing Functions ---
    def context_router(self, state: AgentState):
        """Decide if we should continue the loop or stop based on tool calls"""
        try:
            messages_key = "context_agent_messages"
            messages = state.get(messages_key, [])
            logger.info(f"Context router invoked. Messages count: {len(messages) if messages else 0}")

            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
                    logger.info("Tool call detected. Routing to 'tools' node.")
                    return "tools"

                logger.info("No tool call detected. Routing to 'brain' node.")
                return "brain"
            else:
                logger.warning("No valid messages found, routing to brain.")
                return "brain"
        except Exception as e:
            logger.error(f"Error in context router decision: {e}")
            return "brain"

    def should_continue(self, state: AgentState):
        """Conditional routing logic that directs the workflow from the brain"""
        next_node = state.get("next_node")
        logger.info(f"should_continue invoked. Next node decision: {next_node}")
        if next_node == "end" or not next_node:
            logger.info("Workflow ending.")
            return END
        logger.info(f"Routing to next node: {next_node}")
        return next_node

    # --- Node Implementations ---
    def file_parser(self, state: AgentState) -> AgentState:
        """Parse and extract requirements from document with MongoDB storage"""
        logger.info(f"--- Executing File Parser Node (Deep Extractor with MongoDB) ---")
        file_paths = state.get("file_path", [])
        project_id = state.get("project_id")
        version_id = state.get("version_id")
        previous_version_id = state.get("previous_version_id")
        
        if not file_paths:
            logger.warning("No file path provided in state")
            return {"document": "", "images": [], "extraction_result": None, "requirements": []}
        
        file_path = file_paths[0] if isinstance(file_paths, list) else file_paths
        logger.info(f"Parsing file: {file_path}")
        
        # Use extract_and_store for MongoDB persistence if project/version provided
        if project_id and version_id:
            logger.info(f"Storing requirements to MongoDB (project={project_id}, version={version_id})")
            extraction_result = deep_extractor.extract_and_store(
                file_path=file_path,
                project_id=project_id,
                version_id=version_id,
                previous_version_id=previous_version_id
            )
            
            if extraction_result["success"]:
                logger.info(
                    f"Extraction successful: {extraction_result['total_requirements']} requirements "
                    f"from {extraction_result['total_pages']} pages stored in MongoDB"
                )
                
                # Retrieve requirements from MongoDB for test case generation
                requirements = deep_extractor.storage.get_requirements_by_version(project_id, version_id)
                logger.info(f"Retrieved {len(requirements)} requirements for project {project_id}, version {version_id}")
            else:
                logger.error(f"Extraction failed: {extraction_result['message']}")
                requirements = [] #TODO raise exception
        else:
            # Fallback to legacy mode without MongoDB storage
            logger.warning("project_id or version_id missing - using legacy extraction without MongoDB storage")
            raise NotImplementedError("Legacy extraction without MongoDB storage is not implemented in this version.")
        
        return {
            "images": None,
            "file_path": [file_path],
            "extraction_result": extraction_result,
            "requirements": requirements
        }

    def brain_node(self, state: AgentState) -> AgentState:
        """Orchestrator node that decides the next step based on current state"""
        logger.info("--- Executing Brain Node ---")
        
        # Fetch brain orchestrator prompt from Langfuse
        brain_prompt_template = prompt_service.get("brain-orchestrator-agent")
        
        user_request = state.get("user_request", "")
        formatted_prompt = brain_prompt_template.format(
            context_summary=state.get('context_summary', 'Not Completed'),
            test_cases_summary=state.get('test_cases_summary', 'Not Completed'),
            validation_status=state.get('validation_status', 'Not Completed'),
            test_cases_status=state.get("test_cases_status", "Not Completed"),
            context_built=state.get('context_built', False),
            user_request=user_request
        )
        
        llm = self.llm.get_llm()
        next_node_decision = llm.invoke([HumanMessage(content=formatted_prompt)]).content.strip()
        logger.info(f"Brain decided next node: {next_node_decision}")
        
        return {"next_node": next_node_decision}

    def context_builder_node(self, state: AgentState) -> AgentState:
        """Extracts initial context from requirements and document"""
        logger.info("--- Executing Context Builder Node ---")
        user_request = state.get("user_request", "")
        requirements = state.get("requirements", [])
        
        # Build context from requirements if available
        if requirements:
            logger.info(f"Building context from {len(requirements)} requirements")
            
            # Format requirements for context WITHOUT internal hash-based IDs
            # Only send requirement text and document IDs (if present)
            req_context_parts = []
            for req in requirements:
                text = req.get("text", "")
                page = req.get("source_page", "N/A")
                doc_req_id = req.get("document_requirement_id", "")
                
                # # Include document requirement ID if present, otherwise just text
                # if doc_req_id:
                #     req_context_parts.append(f"{doc_req_id} (Page {page}): {text}")
                # else:
                req_context_parts.append(f"(Page {page}): {text}")
            
            requirements_text = "\n".join(req_context_parts)
            context = f"{user_request}\n\nExtracted Requirements:\n{requirements_text}"
            logger.info("Context built from MongoDB requirements (without internal hash IDs).")
        elif state.get("document"):
            logger.info(f"Building context from document (fallback)")
            md = state["document"]
            images = state.get("images", [])
            
            # Fetch context builder prompt from Langfuse
            context_prompt = prompt_service.get("context-builder-agent")
            system_prompt = SystemMessage(content=context_prompt)
            
            llm = self.llm.get_llm()
            
            if images:
                logger.info(f"Found {len(images)} images in document. Building multimodal context.")
                content_parts = [{"type": "text", "text": f'Here is the document: {md}'}]
                for image_uri in images:
                    content_parts.append({"type": "image_url", "image_url": image_uri})
                human_message = HumanMessage(content=content_parts)
                response = llm.invoke([system_prompt, human_message])
            else:
                logger.info("No images found. Building text-only context.")
                response = llm.invoke([system_prompt, HumanMessage(content=f'Here is the document: {md}')])
            
            structured_context = response.content
            context = f"{user_request}\n\nStructured Document Context:\n{structured_context}"
            logger.info("Structured context built using LLM.")
        else:
            logger.info("No requirements or document provided. Using user request as context.")
            context = user_request
            #TODO Raise exception?
        
        context_summary = self.summarize_text(context)
        logger.info("Context summary generated.")
        
        return {
            "context": context,
            "context_summary": context_summary,
            "context_built": True
        }

    def test_generator_node(self, state: AgentState) -> AgentState:
        """Generates new test cases based on requirements with version tracking"""
        logger.info("--- Executing Test Generator Node ---")
        
        if not state.get("context"):
            logger.error("No context found. Cannot proceed with test case generation.")
            raise Exception("No context found. Cannot proceed with test case generation.")
        
        full_context = state["context"]
        requirements = state.get("requirements", [])
        project_id = state.get("project_id")
        version_id = state.get("version_id")
        previous_version_id = state.get("previous_version_id")
        
        # Add metadata about version
        version_context = f"\n\nProject ID: {project_id}\nVersion ID: {version_id}"
        if previous_version_id:
            version_context += f"\nPrevious Version ID: {previous_version_id} (Note: Some requirements may be unchanged from previous version)"
        
        compliance_info = ""  # TODO: Can be enhanced with RAG results
        
        # Enhanced prompt with requirement IDs instructions
        enhanced_prompt = (
            f"{full_context}{version_context}\n\n"
            f"Instructions:\n"
            f"1. Generate test cases for EACH requirement listed above\n"
            f"2. For each test case, include the requirement_id field:\n"
            f"   - If requirement has a document ID (REQ-001, FR-AUTH-01, etc.), use that exact ID\n"
            f"   - If no document ID exists, generate a meaningful system requirement ID based on the requirement text\n"
            f"3. Generate multiple test cases per requirement if needed (positive, negative, edge cases)\n"
            f"4. Ensure test_case_id follows format: TC_<Feature>_<SeqNum> (e.g., TC_Login_001)\n"
            f"5. Include compliance references if applicable\n\n"
            f"Compliance Context: {compliance_info if compliance_info else 'None specified'}"
        )
        
        prompt = test_case_generator_prompt.format(
            document=enhanced_prompt,
            compliance_info=compliance_info
        )
        
        human_message = HumanMessage(content=prompt)
        llm = self.llm.get_llm()
        
        logger.info(f"Generating test cases for {len(requirements)} requirements (version: {version_id})")
        llm_output = llm.invoke([human_message])
        logger.info("LLM Test Case Generation Completed.")
        
        test_cases_summary = self.summarize_text(llm_output.content)
        
        return {
            "test_cases_status": "test case generation completed",
            "test_cases_lv1": llm_output.content,
            "test_cases_summary": test_cases_summary
        }

    @retry(retry=retry_if_exception_type(OutputParserException), stop=stop_after_attempt(2), wait=wait_fixed(2))
    def validator_node(self, state: AgentState) -> AgentState:
        """Validates the structure and content of the generated test cases"""
        logger.info("--- Executing Validator Node ---")
        
        document = state.get("context", "")
        llm_output = state.get("test_cases_lv1", "")
        project_id = state.get("project_id")
        version_id = state.get("version_id")
        requirements = state.get("requirements", [])
        
        output_parser = PydanticOutputParser(pydantic_object=FinalOutput)
        
        prompt = validation_agent_prompt.format(
            document=document,
            llm_output=llm_output,
            output_format=output_parser.get_format_instructions()
        )
        
        llm = self.llm.get_llm()
        validated_output = llm.invoke([HumanMessage(content=prompt)]).content
        logger.info("LLM Test Case Validation Completed.")
        
        try:
            parsed_output = output_parser.parse(validated_output)
        except OutputParserException as e:
            logger.error(f"Parsing failed: {e}. Retrying...")
            raise e
        
        parsed = parsed_output.model_dump()
        
        # Extract test cases from parsed output
        test_cases = parsed.get("test_cases", [])
        
        # Create mapping: document_requirement_id -> internal req_id (hash-based)
        doc_req_to_internal = {}
        for req in requirements:
            doc_req_id = req.get("document_requirement_id", "")
            internal_id = req.get("req_id", "")  # REQ-a1b2c3d4 (hash-based)
            if doc_req_id and internal_id:
                doc_req_to_internal[doc_req_id] = internal_id
        
        # Enrich test cases with metadata and map to internal requirement IDs
        enriched_test_cases = []
        for tc in test_cases:
            # Get system requirement IDs from LLM (document IDs)
            system_req_ids = tc.get("requirement_id", [])
            if not isinstance(system_req_ids, list):
                system_req_ids = [system_req_ids] if system_req_ids else []
            
            # Map to internal requirement IDs for backend tracking
            internal_req_ids = []
            for sys_id in system_req_ids:
                if sys_id in doc_req_to_internal:
                    internal_req_ids.append(doc_req_to_internal[sys_id])
            
            enriched_tc = {
                **tc,
                "requirement_id": system_req_ids,  # System IDs from LLM (REQ-001, FR-AUTH-01)
                "internal_requirement_id": internal_req_ids,  # Backend hash IDs (REQ-a1b2c3d4)
                "project_id": project_id,
                "version_id": version_id,
                "created_at": None,  # Will be set by router/storage
                "updated_at": None
            }
            enriched_test_cases.append(enriched_tc)
        
        test_cases_summary = f"{len(enriched_test_cases)} test cases validated and enriched with metadata."
        validation_status = "Validation successful."
        
        logger.info(f"Validated {len(enriched_test_cases)} test cases for project {project_id}, version {version_id}")
        
        return {
            "test_cases": enriched_test_cases,
            "test_cases_summary": test_cases_summary,
            "validation_status": validation_status,
            "test_cases_lv1": parsed  # Store final validated output
        }

    # --- Helper Methods ---
    def summarize_text(self, text: str) -> str:
        """Generates a summary for a given text using the LLM"""
        if not text:
            return ""
        logger.info("Summarizing text")
        prompt = [HumanMessage(content=f"Please summarize the following text concisely:\n\n{text}")]
        llm = self.llm.get_llm()
        summary = llm.invoke(prompt).content
        return summary

    # --- Graph Invocation Methods ---
    def invoke_graph(self, document_path, config, project_id: str = None, version_id: str = None, previous_version_id: str = None) -> AgentState:
        """Invoke the graph with a document and version tracking"""
        logger.info(f"Invoking graph with document_path: {document_path}, project_id: {project_id}, version_id: {version_id}, previous_version_id: {previous_version_id}")
        result = self.graph.invoke(
            {
                "file_path": [document_path] if not isinstance(document_path, list) else document_path,
                "project_id": project_id,
                "version_id": version_id,
                "previous_version_id": previous_version_id,
                "user_request": "Generate comprehensive test cases based on extracted requirements",
                "context": "",
                "context_summary": None,
                "context_built": False,
                "requirements": None,
                "test_cases": [],
                "test_cases_lv1": None,
                "test_cases_summary": None,
                "test_cases_status": None,
                "validation_status": None,
                "next_node": "context_builder",
                "context_agent_messages": [],
                "document": "",
                "images": [],
                "extraction_result": None
            },
            config=config
        )
        logger.info("Graph invocation completed")
        return result
    
    def resume_graph(self, command, config) -> AgentState:
        """Resume a paused graph execution"""
        logger.info(f"Resuming graph with command: {command}")
        result = self.graph.invoke(Command(resume=command), config=config)
        logger.info("Graph Resumption completed")
        return result
