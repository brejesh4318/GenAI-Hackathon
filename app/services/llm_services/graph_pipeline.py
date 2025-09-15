from langgraph.graph import StateGraph, END, CompiledStateGraph
from app.utilities import dc_logger
from app.utilities.helper import Helper
from app.services.llm_services.llm_interface import LLMInterface
from app.services.reponse_format import FinalOutput
from app.utilities.singletons_factory import DcSingleton
from app.services.llm_services.graph_state import PipelineState

from langchain.output_parsers import PydanticOutputParser
import uuid
import json


logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})


class GraphPipe(metaclass=DcSingleton):

    def __init__(self, llm: LLMInterface) :
        # Initialize LLM and parser once
        self.llm = llm
        self.output_parser = PydanticOutputParser(pydantic_object=FinalOutput)


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
        llm_output = self.llm.generate(prompt).content
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

        validated_output = self.llm.generate(prompt).content
        logger.info("LLM Test Case Validation Completed")

        parsed_output = self.output_parser.parse(validated_output)
        return parsed_output.model_dump()
