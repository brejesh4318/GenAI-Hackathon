from typing import Optional, Dict, Any
from app.utilities import dc_logger
from app.utilities.singletons_factory import DcSingleton
from app.services.langfuse_prompt_manager import LangfusePromptManager

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})


class PromptFetcher(metaclass=DcSingleton):
    """
    Unified prompt fetcher using Langfuse for prompt management.
    
    This class provides a simple interface to fetch prompts from Langfuse
    with support for versioning, labels, and variable compilation.
    """
    
    def __init__(self):
        """Initialize with Langfuse prompt manager"""
        self.manager = LangfusePromptManager()
        logger.info("PromptFetcher initialized with Langfuse backend")
    
    def fetch(
        self,
        prompt_name: str,
        version: Optional[int] = None,
        label: str = "production"
    ) -> str:
        """
        Fetch a prompt template from Langfuse.

        Args:
            prompt_name (str): Name of the prompt to fetch (e.g., "test-case-generator")
            version (Optional[int]): Specific version number (optional)
            label (str): Version label to fetch (default: "production")

        Returns:
            str: The prompt template string with {{variable}} placeholders

        Raises:
            RuntimeError: If prompt fetch fails

        Example:
            fetcher = PromptFetcher()
            template = fetcher.fetch("test-case-generator")
            # Returns: "Generate test cases for: {{document}}"
        """
        try:
            logger.info(f"Fetching prompt: {prompt_name} (label: {label})")
            prompt = self.manager.get_prompt(
                name=prompt_name,
                version=version,
                label=label,
                type="text"
            )
            
            if not prompt:
                raise ValueError(f"Prompt '{prompt_name}' not found in Langfuse")
            
            # Return the raw template string
            template = prompt.prompt
            logger.info(f"Successfully fetched prompt '{prompt_name}'")
            return template
            
        except Exception as e:
            logger.error(f"Error fetching prompt '{prompt_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to fetch prompt '{prompt_name}': {e}") from e
    
    def fetch_and_compile(
        self,
        prompt_name: str,
        variables: Dict[str, Any],
        version: Optional[int] = None,
        label: str = "production"
    ) -> str:
        """
        Fetch a prompt and compile it with variables.

        Args:
            prompt_name (str): Name of the prompt to fetch
            variables (Dict[str, Any]): Variables to substitute in the template
            version (Optional[int]): Specific version number (optional)
            label (str): Version label (default: "production")

        Returns:
            str: Compiled prompt with variables replaced

        Example:
            compiled = fetcher.fetch_and_compile(
                "test-case-generator",
                {"document": "PRD content", "compliance_info": "FDA rules"}
            )
        """
        try:
            logger.info(f"Fetching and compiling prompt: {prompt_name}")
            compiled = self.manager.compile_prompt(
                name=prompt_name,
                variables=variables,
                version=version,
                label=label
            )
            logger.info(f"Successfully compiled prompt '{prompt_name}'")
            return compiled
        except Exception as e:
            logger.error(f"Error compiling prompt '{prompt_name}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to compile prompt '{prompt_name}': {e}") from e
    
    def get_config(
        self,
        prompt_name: str,
        version: Optional[int] = None,
        label: str = "production"
    ) -> Dict[str, Any]:
        """
        Get the configuration for a prompt (model, temperature, etc.).

        Args:
            prompt_name (str): Name of the prompt
            version (Optional[int]): Specific version (optional)
            label (str): Version label (default: "production")

        Returns:
            Dict[str, Any]: Configuration dictionary

        Example:
            config = fetcher.get_config("test-case-generator")
            # Returns: {"model": "gemini-2.5-flash", "temperature": 0.7}
        """
        return self.manager.get_prompt_config(
            name=prompt_name,
            version=version,
            label=label
        )