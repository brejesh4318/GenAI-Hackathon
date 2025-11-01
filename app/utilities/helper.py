    

from datetime import datetime
from pathlib import Path
from app.services.llm_services.graph_state import PipelineState
from app.services.llm_services.llm_interface import LLMInterface
from app.services.prompt_fetch import PromptFetcher
from app.services.reponse_format import AgentFormat
from app.utilities import dc_logger
from app.utilities.constants import Constants
from langchain.output_parsers import PydanticOutputParser
from app.utilities.singletons_factory import DcSingleton
from langchain_core.messages import SystemMessage, AIMessage
import PyPDF2
import docx  # for .docx
import markdown  # for .md
import os
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from langchain_core.exceptions import OutputParserException

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})
fetch_prompt = PromptFetcher()
brain_agent_prompt = fetch_prompt.fetch("spec2test-brain-agent")
compiance_reacher_prompt = fetch_prompt.fetch("compliance-researcher-agent")
class Helper(metaclass = DcSingleton):
    @staticmethod
    def read_file(file_path: str) -> str:
            """
            Reads and parses text from a file.
            Supports: .docx, .pdf, .txt, .md
            
            Args:
                file_path (str): Path to the file.
                
            Returns:
                str: Extracted text content.
            """
            file_ext = Path(file_path).suffix.lower()

            if file_ext == ".docx":
                return _read_docx(file_path)
            elif file_ext == ".pdf":
                return _read_pdf(file_path)
            elif file_ext == ".txt":
                return _read_txt(file_path)
            elif file_ext == ".md":
                return _read_md(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
    
    @staticmethod
    def save_file(tmp_path: str, content: bytes, filename: str) -> str:
        """
        Save a file from binary content to a specified temporary path.

        Args:
            tmp_path (str): The temporary path to save the file.
            content (bytes): The binary content of the file.
            filename (str): The name of the file.

        Returns:
            str: The path where the file is saved.

        Raises:
            Exception: If there is an error saving the file.
        """
        try:
            if not os.path.exists(tmp_path):
                    os.mkdir(tmp_path)
            path = os.path.join(tmp_path, filename)
            with open(path, 'wb') as f:
                f.write(content)
            return path
        except Exception as exe:
              logger.error("Error occcured in save pdf")
              raise exe 
        
    @staticmethod
    def generate_testcase(llm: LLMInterface, document: str , compliance_info) -> str:
        
        """Calls LLM to generate test cases"""
        prompt = Constants.fetch_constant("prompts")["test_casegenerator"].format(document=document, compliance_info=compliance_info)    
        llm_output = llm.generate(prompt)
        logger.info("LLM Test Case Generation Completed")
        return llm_output
    
    @staticmethod
    @retry(retry=retry_if_exception_type(OutputParserException), stop=stop_after_attempt(2), wait=wait_fixed(2))
    def validator(llm: "LLMInterface", document: str, llm_output: str, output_parser: "PydanticOutputParser"):
        """
        Validate test cases with LLM and enforce structured format.
        Retries parsing if OutputParserException occurs (max 2 attempts, 2 sec delay).
        """
        prompt = Constants.fetch_constant("prompts")["validation_agent"].format(
            document=document,
            llm_output=llm_output,
            output_format=output_parser.get_format_instructions()
        )
        validated_output = llm.generate(prompt)
        logger.info("LLM Test Case Validation Completed")
        try:
            parsed_output = output_parser.parse(validated_output)
        except OutputParserException as e:
            logger.error(f"Parsing failed: {e}. Retrying...")
            raise e
        return parsed_output.model_dump()
    
    @staticmethod
    def plan_compliance(llm: LLMInterface, process_document: str) -> dict:
        """Plan compliance steps based on document"""
        # Example standards to check against
        standards = ["FDA", "IEC-62304", "ISO-13485", "ISO-27001"]

        prompt = Constants.fetch_constant("prompts")["compliance_agent1"].format(process_document=process_document, standards=standards)
        llm_output = llm.generate(prompt)
        logger.info("LLM Compliance Planning Completed")
        return {"compliance_plan": llm_output}
    
    @staticmethod
    def compliance_answer(llm_with_tools , state: dict) -> dict:
        """
        Enriches compliance requirements with authoritative details using RAG and web search tools.
        """

        logger.info("Sending LLM Compliance Enrichment Request")
        # enriched_output = self.llm_with_tools.invoke({"messages": [prompt] + state["messages"]})
        message = {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content= compiance_reacher_prompt
                    )
                ] + ["Document Information: " + state["document"]]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }
        logger.info("LLM Compliance Enrichment Completed")
        return message
    
    @staticmethod
    def time_saved_format(start_time: datetime, end_time: datetime = datetime.now()) -> str:
        """
        Returns a human-readable string representing the time difference between two datetimes.

        Args:
            start_time (datetime): The start time.
            end_time (datetime): The end time.

        Returns:
            str: Time difference in days, hours, minutes, or seconds.
        """
        diff = end_time - start_time
        total_seconds = int(diff.total_seconds())
        days = total_seconds // (3600 * 24)
        hours = (total_seconds % (3600 * 24)) // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if days >= 1:
            return f"{days} day{'s' if days != 1 else ''}"
        elif hours >= 1:
            return f"{hours} hour{'s' if hours != 1 else ''}"
        elif minutes >= 1:
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        else:
            return f"{seconds} second{'s' if seconds != 1 else ''}"
    @staticmethod
    def brain_agent(llm_tools:LLMInterface , state: PipelineState, output_parser: PydanticOutputParser)->AgentFormat:
        try:
            system_prompt = SystemMessage(content=brain_agent_prompt)
            message = [system_prompt] + [state["brain_agent_message"]]
            message += [f"here is your scratchpad {state['brain_agent_scratchpad']}"] if state.get("brain_agent_scratchpad") else []
            # message += [f"here are your notes {state['notes']}"] if state.get("notes") else []
            message += [f"here is your previous status {state['status']}"] if state.get("status") else []
            message += [f"here is your previous next action {state['next_action']}"] if state.get("next_action") else []
            message += [f"here is your previous summary {state['summary']}"] if state.get("summary") else []
            response = llm_tools.generate(message)
            logger.info("LLM Brain Agent Completed")
            response = output_parser.parse(response)
            return response
        except Exception as e:
            logger.error(f"Error in brain agent: {str(e)}")
            raise e
        
    @staticmethod
    def context_builder( llm: LLMInterface, state: PipelineState):
        prompt = Constants.fetch_constant("prompts")["context_agent"] 
        system_prompt= SystemMessage(content=prompt)
        response = llm.get_llm().invoke([system_prompt] + [f'Here is the document: {state["document"]}'])
        logger.info("LLM Context Builder Completed")
        return  response.content
    
def _read_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def _read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def _read_pdf(file_path: str):
    with open(file_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        pages_text = []
        for page in reader.pages:
            pages_text.append(page.extract_text())
    return "\n".join(pages_text)

def _read_md(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_md = f.read()
    # Convert markdown to plain text
    html = markdown.markdown(raw_md)
    # strip HTML tags (optional)
    return "".join(html.split("<")[0::2])  # quick strip, or use BeautifulSoup for better parsing


