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
    
    def generate_testcase(self, document_path:str):
        config = RunnableConfig(recursion_limit=50)
        test_cases = self.graph_pipe.invoke_graph(document_path, config=config)
        return test_cases["test_cases_final"]["test_cases"]


