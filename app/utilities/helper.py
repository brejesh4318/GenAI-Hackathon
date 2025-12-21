    

from datetime import datetime
from pathlib import Path
from typing import Tuple, List
from app.services.llm_services.graph_state import PipelineState
from app.services.llm_services.llm_interface import LLMInterface
from app.services.prompt_fetch import PromptFetcher
from app.services.reponse_format import AgentFormat
from app.utilities import dc_logger
from app.utilities.constants import Constants
from langchain.output_parsers import PydanticOutputParser
from app.utilities.singletons_factory import DcSingleton
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from app.utilities.document_parser import DocumentParser
import os
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from langchain_core.exceptions import OutputParserException

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})
fetch_prompt = PromptFetcher()
# Fetch all prompts from Langfuse with production labels
brain_agent_prompt = fetch_prompt.fetch("brain-orchestrator-agent")
compliance_researcher_prompt = fetch_prompt.fetch("compliance-researcher-agent")
context_agent_prompt = fetch_prompt.fetch("context-builder-agent")
validation_agent_prompt = fetch_prompt.fetch("validator-agent")

class Helper(metaclass = DcSingleton):
    @staticmethod
    def read_file(file_path: str) -> Tuple[str, List[str]]:
        """
        Reads and parses documents from a file using DocumentParser.
        Supports: .docx, .pdf, .txt, .md
        
        Args:
            file_path (str): Path to the file.
            
        Returns:
            Tuple[str, List[str]]: (markdown_content, list_of_base64_image_uris)
            - markdown_content: Document text as markdown string
            - list_of_base64_image_uris: Images in format "data:image/png;base64,..."
                                        Empty list for text files
        """
        parser = DocumentParser()
        return parser.parse_file(file_path)
    
    @staticmethod
    def extract_doc_pages(file_path: str) -> List[str]:
        """
        Extract raw text per page from PDF/DOCX/TXT/MD files.
        
        Args:
            file_path (str): Path to the document file.
            
        Returns:
            List[str]: List of page texts (one string per page).
                      For text/markdown files, returns single-item list.
        """
        parser = DocumentParser()
        return parser.extract_doc_pages(file_path)
            
    
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
        # Fetch test case generator prompt from Langfuse
        test_case_prompt_template = fetch_prompt.fetch("test-case-generator")
        prompt = test_case_prompt_template.format(document=document, compliance_info=compliance_info)    
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
        prompt = validation_agent_prompt.format(
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
                        content= compliance_researcher_prompt
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
            message += [f"\nhere is your previous status {state['status']}"] if state.get("status") else []
            message += [f"\nhere is your previous next action {state['next_action']}"] if state.get("next_action") else []
            message += [f"\nhere is your previous summary {state['summary']}"] if state.get("summary") else []
            response = llm_tools.generate(message)
            logger.info("LLM Brain Agent Completed")
            response = output_parser.parse(response)
            return response
        except Exception as e:
            logger.error(f"Error in brain agent: {str(e)}")
            raise e
        
    @staticmethod
    def context_builder(llm: LLMInterface, state: PipelineState):
        """
        Build structured context from document using LLM.
        Handles multimodal input if images are present.
        
        Args:
            llm: LLM interface instance
            state: Pipeline state containing document and images
            
        Returns:
            str: Structured context generated by LLM
        """
        system_prompt = SystemMessage(content=context_agent_prompt)
        
        # Check if images are available in state
        images = state.get("images", [])
        
        if images and len(images) > 0:
            # Build multimodal content with text + images
            logger.info(f"Building context with {len(images)} images for multimodal analysis")
            
            # Create content list with text and images
            content_parts = [
                {"type": "text", "text": f'Here is the document: {state["document"]}'}
            ]
            
            # Add all images to the content
            for idx, image_uri in enumerate(images):
                content_parts.append({
                    "type": "image_url",
                    "image_url": image_uri
                })
                logger.debug(f"Added image {idx + 1}/{len(images)} to context builder input")
            
            # Create HumanMessage with multimodal content
            human_message = HumanMessage(content=content_parts)
            response = llm.get_llm().invoke([system_prompt, human_message])
        else:
            # Text-only mode (no images)
            logger.info("Building context with text only (no images)")
            response = llm.get_llm().invoke([system_prompt] + [f'Here is the document: {state["document"]}'])
        
        logger.info("LLM Context Builder Completed")
        return response.content




