from app.services.llm_services.llm_interface import LLMInterface
from app.services.reponse_format import FinalOutput
from app.utilities import dc_logger
from app.utilities.singletons_factory import DcSingleton
from app.services.llm_services.graph_state import PipelineState


logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})

class TestCaseGenerator(metaclass = DcSingleton):

    def __init__(self, graph):
        self.graph = graph
    
    def generate_testcase(self, document_text):
        test_cases = self.graph.invoke({"document":document_text})
        return test_cases


