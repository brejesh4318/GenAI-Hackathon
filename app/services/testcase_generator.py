from app.services.testcase_service.graph_pipeline import GraphPipe
from app.services.llm_services.llm_interface import LLMInterface
from app.services.reponse_format import FinalOutput
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig
from app.utilities import dc_logger
from app.utilities.singletons_factory import DcSingleton
from app.services.testcase_service.graph_state import AgentState


logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})

class TestCaseGenerator(metaclass = DcSingleton):

    def __init__(self, graph_pipe: GraphPipe):
        self.graph_pipe = graph_pipe

    def generate_testcase(self, project_id: str, invoke_type: str = "new", invoke_command: str = "", document_path=None, version_id: str = None) -> dict:
        """Generate test cases using the new agentic graph architecture with MongoDB persistence"""
        config = RunnableConfig(recursion_limit=50, configurable={"thread_id": project_id})
        
        try:
            if invoke_type == "new" and document_path:
                response = self.graph_pipe.invoke_graph(
                    document_path, 
                    config=config, 
                    project_id=project_id,
                    version_id=version_id
                )
            elif invoke_type == "resume" and invoke_command:
                response = self.graph_pipe.resume_graph(command=invoke_command, config=config)
            else:
                raise ValueError("Invalid invoke_type or missing invoke_command for resume")
            
            # Check for interrupts
            if "__interrupt__" in response and response["__interrupt__"]:
                prompt = response["__interrupt__"][-1].value
                logger.info("Workflow interrupted for user input")
                return {"response": f"{response.get('test_cases_lv1')}\n{prompt}", "type": "interrupt"}
            else:
                # Extract test cases from validated output
                if "test_cases" not in response or not response["test_cases"]:
                    logger.error("No test cases found in response")
                    raise ValueError("test_cases not found in response")
                
                test_cases = response["test_cases"]
                logger.info(f"Successfully generated {len(test_cases)} test cases")
                return {"response": test_cases, "type": "final"}
        except Exception as e:
            logger.error(f"Error in generate_testcase: {str(e)}")
            raise e



