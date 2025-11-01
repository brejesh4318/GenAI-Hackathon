from langsmith import Client
from app.utilities import dc_logger
from app.utilities.singletons_factory import DcSingleton
from langchain_core.prompts import SystemMessagePromptTemplate
logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})

class PromptFetcher(metaclass=DcSingleton):
    def __init__(self):
        self.client = Client()
    
    def fetch(self, prompt_name):
        """
        Fetch a prompt from LangSmith using the prompt name

        Args:
            prompt_name (str): Name of the prompt to fetch

        Returns:
            The prompt from LangSmith
        """
        try:
            logger.info(f"Fetching prompt: {prompt_name}")
            prompt = self.client.pull_prompt(prompt_name)
            if not prompt:
                raise ValueError(f"Prompt '{prompt_name}' not found")
            messages = getattr(prompt, "messages", None)
            if not messages:
                raise ValueError(f"Prompt '{prompt_name}' contains no messages")
            if isinstance(messages[-1], SystemMessagePromptTemplate):
                return messages[-1].prompt.template
            else:
                return messages[-1].content
        except Exception as e:
            logger.error(f"Error fetching prompt '{prompt_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to fetch prompt '{prompt_name}': {e}") from e