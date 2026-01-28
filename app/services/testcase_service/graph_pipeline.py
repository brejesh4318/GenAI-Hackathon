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
from app.services.testcase_service.graph_state import AgentState
from app.services.testcase_service.tools.rag_tools import retrieve_by_standards, web_search_tool, interrupt_tool
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from langchain_core.exceptions import OutputParserException
from app.services.req_extract_service.requirements_extractor import RequirementsExtractor
import sqlite3

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})
conn = sqlite3.connect('langchain.db', check_same_thread=False)
memory = SqliteSaver(conn)
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
                
                # Retrieve all requirements from MongoDB
                requirements = deep_extractor.storage.get_requirements_by_version(project_id, version_id)
                logger.info(f"Retrieved {len(requirements)} total requirements for project {project_id}, version {version_id}")
                
                # Determine which requirements need test generation
                if previous_version_id and "generate_tests_for" in extraction_result:
                    # Use diff results: only new/modified requirements
                    requirements_for_testing = extraction_result["generate_tests_for"]# TODO check this follow
                    logger.info(
                        f"Version diff detected: Generating tests for {len(requirements_for_testing)} "
                        f"new/modified requirements (out of {len(requirements)} total)"
                    )
                else:
                    # V1 or no diff: generate tests for all requirements
                    requirements_for_testing = requirements
                    logger.info(f"First version: Generating tests for all {len(requirements_for_testing)} requirements")
            else:
                logger.error(f"Extraction failed: {extraction_result['message']}")
                raise ValueError(f"Extraction failed: {extraction_result['message']}")
        
        # Merge requirements with same requirement_id (concatenate text with \n)
        logger.info(f"Merging requirements with same requirement_id before sending to LLM")
        requirements_for_testing = self._merge_requirements_by_id(requirements_for_testing)
        logger.info(f"After merging: {len(requirements_for_testing)} unique requirements for test generation")
        
        # Sanitize ObjectIds for msgpack serialization (LangGraph checkpointer)
        requirements = self._sanitize_objectids(requirements)
        requirements_for_testing = self._sanitize_objectids(requirements_for_testing)
        extraction_result = self._sanitize_objectids(extraction_result)
        logger.info("Sanitized MongoDB ObjectIds for LangGraph state persistence")
        
        return {
            "images": None,
            "file_path": [file_path],
            "extraction_result": extraction_result,
            "requirements": requirements,
            "requirements_for_testing": requirements_for_testing
        }
    
    def _sanitize_objectids(self, data):
        """Convert MongoDB ObjectIds to strings for msgpack serialization"""
        from bson import ObjectId
        
        if isinstance(data, ObjectId):
            return str(data)
        elif isinstance(data, dict):
            return {key: self._sanitize_objectids(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_objectids(item) for item in data]
        else:
            return data
    
    def _merge_requirements_by_id(self, requirements: list) -> list:
        """Merge requirements with same requirement_id by concatenating text"""
        if not requirements:
            return requirements
        
        merged_dict = {}
        
        for req in requirements:
            req_id = req.get("requirement_id", "")
            
            if not req_id:
                # No requirement_id, keep as-is with unique key
                unique_key = f"_no_id_{id(req)}"
                merged_dict[unique_key] = req
                continue
            
            if req_id in merged_dict:
                # Merge with existing requirement
                existing = merged_dict[req_id]
                
                # Concatenate text with newline
                existing_text = existing.get("text", "")
                new_text = req.get("text", "")
                existing["text"] = f"{existing_text}\n{new_text}" if existing_text else new_text
                
                # Combine source pages
                existing_pages = existing.get("source_page", "")
                new_page = req.get("source_page", "")
                if existing_pages and new_page:
                    pages_list = str(existing_pages).split(", ")
                    if str(new_page) not in pages_list:
                        existing["source_page"] = f"{existing_pages}, {new_page}"
                elif new_page:
                    existing["source_page"] = new_page
                
                logger.debug(f"Merged requirement_id: {req_id}")
            else:
                # First occurrence of this requirement_id
                merged_dict[req_id] = req.copy()
        
        merged_list = list(merged_dict.values())
        logger.info(f"Merged {len(requirements)} requirements into {len(merged_list)} unique requirements")
        
        return merged_list

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
        
        # Remove quotes if LLM returns JSON-formatted string (e.g., '"validator"' -> 'validator')
        next_node_decision = next_node_decision.strip('"').strip("'")
        
        logger.info(f"Brain decided next node: {next_node_decision}")
        
        return {"next_node": next_node_decision}

    def context_builder_node(self, state: AgentState) -> AgentState:
        """Extracts initial context from requirements and document"""
        logger.info("--- Executing Context Builder Node ---")
        user_request = state.get("user_request", "")
        requirements_for_testing = state.get("requirements_for_testing", [])
        
        # Build context from requirements if available
        if requirements_for_testing:
            logger.info(f"Building context from {len(requirements_for_testing)} requirements needing test generation")
            
            # Format requirements for context with document requirement IDs
            req_context_parts = []
            for req in requirements_for_testing:
                text = req.get("text", "")
                page = req.get("source_page", "N/A")
                req_id = req.get("requirement_id", "N/A")  # Document ID like SFSYST1.1
                req_context_parts.append(f"(Page {page}, Requirement ID: {req_id}): {text}")
            
            requirements_text = "\n".join(req_context_parts)
            context = f"{user_request}\n\nExtracted Requirements:\n{requirements_text}"
            logger.info("Context built from MongoDB requirements.")
        else:
            logger.error("No requirements found for test generation")
            raise ValueError("No requirements available for context building")
        
        context_summary = f"Context built successfully from {len(requirements_for_testing)} requirements."
        logger.info(context_summary)
        
        return {
            "context": context,
            "context_summary": context_summary,
            "context_built": True
        }

    def test_generator_node(self, state: AgentState) -> AgentState:
        """Generates new test cases based on requirements with version tracking (batch processing)"""#TODO uncomment
        # logger.info("--- Executing Test Generator Node with Batch Processing ---") 
        
        # user_request = state.get("user_request", "")
        # requirements_for_testing = state.get("requirements_for_testing", [])
        # project_id = state.get("project_id")
        # version_id = state.get("version_id")
        # previous_version_id = state.get("previous_version_id")
        # extraction_result = state.get("extraction_result", {})
        
        # if not requirements_for_testing:
        #     logger.error("No requirements found for test generation")
        #     raise Exception("No requirements available for test case generation.")
        
        # # Add metadata about version and diff status
        # version_context = f"\n\nProject ID: {project_id}\nVersion ID: {version_id}"
        # if previous_version_id:
        #     diff = extraction_result.get("diff", {})
        #     if diff:
        #         version_context += (
        #             f"\nPrevious Version ID: {previous_version_id}"
        #             f"\nGenerating tests for NEW/MODIFIED requirements only:"
        #             f"\n  - New requirements: {len(diff.get('new', []))}"
        #             f"\n  - Modified requirements: {len(diff.get('modified', []))}"
        #             f"\n  - Unchanged requirements: {len(diff.get('unchanged', []))} (skipping test generation)"
        #             f"\n  - Removed requirements: {len(diff.get('removed', []))} (obsolete)"
        #         )
        #     else:
        #         version_context += f"\nPrevious Version ID: {previous_version_id}"
        # else:
        #     version_context += f"\nFirst version - generating tests for all {len(requirements_for_testing)} requirements"
        
        # # Process requirements in batches of 10
        # BATCH_SIZE = 20
        # total_requirements = len(requirements_for_testing)
        # num_batches = (total_requirements + BATCH_SIZE - 1) // BATCH_SIZE
        
        # logger.info(f"Processing {total_requirements} requirements in {num_batches} batches of {BATCH_SIZE}")
        
        # all_llm_outputs = []
        # llm = self.llm.get_llm()
        # compliance_info = ""
        
        # for batch_idx in range(num_batches):
        #     start_idx = batch_idx * BATCH_SIZE
        #     end_idx = min(start_idx + BATCH_SIZE, total_requirements)
        #     batch_requirements = requirements_for_testing[start_idx:end_idx]
            
        #     logger.info(f"Processing batch {batch_idx + 1}/{num_batches}: requirements {start_idx + 1}-{end_idx}")
            
        #     # Build context for this batch
        #     req_context_parts = []
        #     for req in batch_requirements:
        #         text = req.get("text", "")
        #         page = req.get("source_page", "N/A")
        #         req_id = req.get("requirement_id", "N/A")
        #         req_context_parts.append(f"(Page {page}, Requirement ID: {req_id}): {text}")
            
        #     batch_requirements_text = "\n".join(req_context_parts)
        #     batch_context = f"{user_request}\n\nExtracted Requirements (Batch {batch_idx + 1}/{num_batches}):\n{batch_requirements_text}"
            
        #     enhanced_prompt = (
        #         f"{batch_context}{version_context}\n\n"
        #         f"Instructions:\n"
        #         f"1. Generate test cases for EACH of the {len(batch_requirements)} requirements listed above\n"
        #         f"2. For each test case, include the requirement_id field:\n"
        #         f"   - If requirement has a document ID (REQ-001, FR-AUTH-01, etc.), use that exact ID\n"
        #         f"   - If no document ID exists, generate a meaningful system requirement ID based on the requirement text\n"
        #         f"3. Generate multiple test cases per requirement if needed (positive, negative, edge cases)\n"
        #         f"4. Ensure test_case_id follows format: TC_<Feature>_<SeqNum> (e.g., TC_Login_001)\n"
        #         f"5. Include compliance references if applicable\n\n"
        #         f"Compliance Context: {compliance_info if compliance_info else 'None specified'}"
        #     )
            
        #     prompt = test_case_generator_prompt.format(
        #         document=enhanced_prompt,
        #         compliance_info=compliance_info
        #     )
            
        #     human_message = HumanMessage(content=prompt)
            
        #     logger.info(f"Generating test cases for batch {batch_idx + 1} ({len(batch_requirements)} requirements)")
        #     llm_output = llm.invoke([human_message])
        #     all_llm_outputs.append(llm_output.content)
        #     logger.info(f"Batch {batch_idx + 1}/{num_batches} generation completed")
        
        # logger.info(f"All batches completed. Generated {num_batches} batches for {total_requirements} requirements")
        


        test_cases_summary = f"Test case generation completed for 60 requirements in 5 batches." #TODO correct sentence
        with open(r"D:\projects\GenAI-Hackathon\testcase_node_out.json","r") as f:
            all_llm_outputs = json.load(f) #TODO remove after testing
        return {
            "test_cases_status": "test case generation completed",
            "test_cases_lv1": all_llm_outputs,  # Return as list of batches
            "test_cases_summary": test_cases_summary
        }

    @retry(retry=retry_if_exception_type(OutputParserException), stop=stop_after_attempt(2), wait=wait_fixed(2))
    def validator_node(self, state: AgentState) -> AgentState:
        """Validates the structure and content of the generated test cases (batch processing)"""
        logger.info("--- Executing Validator Node with Batch Processing ---")
        
        llm_outputs = state.get("test_cases_lv1", [])
        project_id = state.get("project_id")
        version_id = state.get("version_id")
        requirements = state.get("requirements", [])
        
        # # Handle both list (batch) and string (legacy) formats
        # if isinstance(llm_outputs, str):
        #     llm_outputs = [llm_outputs]
        
        # if not llm_outputs:
        #     logger.error("No LLM outputs found for validation")
        #     raise ValueError("No test cases to validate")
        
        # num_batches = len(llm_outputs)
        # logger.info(f"Processing {num_batches} batches for validation")
        
        # output_parser = PydanticOutputParser(pydantic_object=FinalOutput)
        # llm = self.llm.get_llm()
        
        # all_test_cases = []
        
        # # Process each batch
        # for batch_idx, batch_output in enumerate(llm_outputs):
        #     logger.info(f"Validating batch {batch_idx + 1}/{num_batches}")
            
        #     prompt = validation_agent_prompt.format(
        #         llm_output=batch_output,
        #         output_format=output_parser.get_format_instructions()
        #     )
            
        #     validated_output = llm.invoke([HumanMessage(content=prompt)])
        #     logger.info(f"Batch {batch_idx + 1}/{num_batches} validation completed")
            
        #     try:
        #         parsed_output = output_parser.parse(validated_output.content)
        #     except OutputParserException as e:
        #         logger.error(f"Parsing failed for batch {batch_idx + 1}: {e}. Retrying...")
        #         raise e
        #     parsed = parsed_output.model_dump()
        #     batch_test_cases = parsed.get("test_cases", [])
        #     logger.info(f"Batch {batch_idx + 1} produced {len(batch_test_cases)} test cases")
            
        #     all_test_cases.extend(batch_test_cases)
        
        # logger.info(f"All batches validated. Total test cases: {len(all_test_cases)}") ##TODO uncomment        
        # Create mapping: document requirement_id -> internal req_id (hash-based)

        with open(r"D:\projects\GenAI-Hackathon\validator_out.json","r") as f:
            import json
            all_test_cases = json.load(f) #TODO remove after testing
        doc_req_to_internal = {}
        for req in requirements:
            doc_req_id = req.get("requirement_id", "")  # Document ID like SFSYST1.1
            internal_id = req.get("req_id", "")  # Backend hash ID (REQ-a1b2c3d4)
            if doc_req_id and internal_id:
                doc_req_to_internal[doc_req_id] = internal_id
        
        # Enrich all test cases with metadata and map to internal requirement IDs
        enriched_test_cases = []
        for tc in all_test_cases:
            # Get document requirement IDs from LLM (e.g., SFSYST1.1, REQ-001)
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
        
        test_cases_summary = f"{len(enriched_test_cases)} test cases validated and enriched with metadata from 4 batches."
        validation_status = "Validation successful."
        
        logger.info(f"Validated {len(enriched_test_cases)} total test cases for project {project_id}, version {version_id}")
        
        return {
            "test_cases": enriched_test_cases,
            "test_cases_summary": test_cases_summary,
            "validation_status": validation_status,
            "test_cases_lv1": all_test_cases  # Store all combined test cases
        }

    # --- Graph Invocation Methods ---
    def invoke_graph(self, document_path, config, project_id: int = None, version_id: int = None, previous_version_id: int = None) -> AgentState:
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
                "requirements_for_testing": None,
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
