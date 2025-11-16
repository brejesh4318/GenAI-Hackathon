from app.services.llm_services.graph_pipeline import GraphPipe
from app.services.llm_services.llm_interface import LLMInterface
from app.services.reponse_format import FinalOutput
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig
from app.utilities import dc_logger
from app.utilities.singletons_factory import DcSingleton
from app.services.llm_services.graph_state import PipelineState



logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})

class TestCaseGenerator(metaclass = DcSingleton):

    def __init__(self, graph_pipe:GraphPipe ):
        self.graph_pipe = graph_pipe

    
    def generate_testcase(self, project_id: str, invoke_type: str = "new", invoke_command: str = "", document_path=None)->dict:
        config = RunnableConfig(recursion_limit=50, configurable={"thread_id": project_id})
        
        try:
            if invoke_type =="new" and document_path:
                response = self.graph_pipe.invoke_graph(document_path, config=config)
            elif invoke_type == "resume" and invoke_command:
                response = self.graph_pipe.resume_graph(command=invoke_command, config=config)
            else:
                raise ValueError("Invalid invoke_type or missing invoke_command for resume")
            if "__interrupt__" in response and response["__interrupt__"]:
                prompt = response["__interrupt__"][-1].value
                # last_msg = response["messages"][-1].content
                # user_input = input(f"{last_msg}\n\n{prompt}\n\nYour Input: ")
                return {"response": f"{response.get("test_cases_lv1")}\n{prompt}", "type": "interrupt"}
            else:
                if "test_cases_final" not in response:
                    raise ValueError("test_cases_final not found in response")
                return {"response": response["test_cases_final"]["test_cases"], "type": "final"}
        except Exception as e:
            logger.error(f"Error in generate_testcase: {str(e)}")
            raise e


