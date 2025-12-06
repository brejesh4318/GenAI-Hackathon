"""
Langfuse Prompt Management Service

This module provides a unified interface for managing prompts with Langfuse,
including creating, fetching, and versioning prompts with production labels.
"""

from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from langfuse import Langfuse
from app.utilities import dc_logger
from app.utilities.singletons_factory import DcSingleton
from app.utilities.env_util import EnvironmentVariableRetriever

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})

load_dotenv()
class LangfusePromptManager(metaclass=DcSingleton):
    """
    Manages prompt operations with Langfuse including:
    - Creating prompts with versioning
    - Fetching prompts by name, version, or label
    - Managing prompt configurations
    - Supporting both text and chat prompt types
    """

    def __init__(self):
        """Initialize Langfuse client with credentials from environment"""
        try:
            # Get Langfuse credentials from environment
            public_key = EnvironmentVariableRetriever.get_env_variable("LANGFUSE_PUBLIC_KEY")
            secret_key = EnvironmentVariableRetriever.get_env_variable("LANGFUSE_SECRET_KEY")
            host = EnvironmentVariableRetriever.get_env_variable("LANGFUSE_HOST")
            
            self.client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            logger.info("Langfuse client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse client: {e}")
            raise RuntimeError(f"Langfuse initialization failed: {e}") from e

    def create_prompt(
        self,
        name: str,
        prompt: str | List[Dict[str, str]],
        type: str = "text",
        labels: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Create or update a prompt in Langfuse.
        
        Args:
            name: Unique prompt identifier (e.g., "test-case-generator")
            prompt: Prompt template string (for text) or list of chat messages (for chat)
            type: Prompt type - "text" or "chat"
            labels: Version labels (e.g., ["production", "staging"])
            config: Model configuration (temperature, model name, etc.)
            tags: Additional tags for organization
            
        Example for text prompt:
            create_prompt(
                name="test-case-generator",
                prompt="Generate test cases for: {{document}}",
                type="text",
                labels=["production"],
                config={"model": "gemini-2.5-flash", "temperature": 0.7}
            )
            
        Example for chat prompt:
            create_prompt(
                name="validator-agent",
                prompt=[
                    {"role": "system", "content": "You are a test validator"},
                    {"role": "user", "content": "Validate: {{test_cases}}"}
                ],
                type="chat",
                labels=["production"]
            )
        """
        try:
            labels = labels or ["production"]  # Default to production label
            config = config or {}
            tags = tags or []
            
            logger.info(f"Creating prompt '{name}' with type '{type}' and labels {labels}")
            
            self.client.create_prompt(
                name=name,
                type=type,
                prompt=prompt,
                labels=labels,
                config=config,
                tags=tags
            )
            
            logger.info(f"Prompt '{name}' created successfully")
        except Exception as e:
            logger.error(f"Failed to create prompt '{name}': {e}")
            raise RuntimeError(f"Failed to create prompt '{name}': {e}") from e

    def get_prompt(
        self,
        name: str,
        version: Optional[int] = None,
        label: str = "production",
        type: str = "text"
    ) -> Any:
        """
        Fetch a prompt from Langfuse.
        
        Args:
            name: Prompt name to retrieve
            version: Specific version number (optional)
            label: Version label to fetch (default: "production")
            type: Expected prompt type for validation
            
        Returns:
            Langfuse prompt object with methods:
                - .compile(**variables) - Compile with variable substitution
                - .prompt - Raw prompt template
                - .config - Configuration dict
                
        Example:
            # Get production version
            prompt = get_prompt("test-case-generator")
            compiled = prompt.compile(document="PRD content here")
            
            # Get specific version
            prompt = get_prompt("test-case-generator", version=3)
            
            # Get staging version
            prompt = get_prompt("test-case-generator", label="staging")
        """
        try:
            logger.info(f"Fetching prompt '{name}' with label '{label}'" + 
                       (f" version {version}" if version else ""))
            
            # Fetch from Langfuse
            if version:
                prompt = self.client.get_prompt(name, version=version, type=type)
            else:
                prompt = self.client.get_prompt(name, label=label, type=type)
            
            if not prompt:
                raise ValueError(f"Prompt '{name}' not found in Langfuse")
            
            logger.info(f"Successfully fetched prompt '{name}'")
            return prompt
            
        except Exception as e:
            logger.error(f"Failed to fetch prompt '{name}': {e}")
            raise RuntimeError(f"Failed to fetch prompt '{name}': {e}") from e

    def get_prompt_template(
        self,
        name: str,
        version: Optional[int] = None,
        label: str = "production"
    ) -> str:
        """
        Fetch only the prompt template string (for text prompts).
        
        Args:
            name: Prompt name
            version: Specific version (optional)
            label: Version label (default: "production")
            
        Returns:
            str: Raw prompt template with {{variable}} placeholders
        """
        prompt = self.get_prompt(name, version=version, label=label, type="text")
        return prompt.prompt

    def compile_prompt(
        self,
        name: str,
        variables: Dict[str, Any],
        version: Optional[int] = None,
        label: str = "production"
    ) -> str:
        """
        Fetch and compile a prompt with variable substitution.
        
        Args:
            name: Prompt name
            variables: Dict of variables to substitute
            version: Specific version (optional)
            label: Version label (default: "production")
            
        Returns:
            str: Compiled prompt with variables replaced
            
        Example:
            compiled = compile_prompt(
                "test-case-generator",
                {"document": "PRD content", "compliance_info": "FDA rules"}
            )
        """
        prompt = self.get_prompt(name, version=version, label=label)
        return prompt.compile(**variables)

    def get_prompt_config(
        self,
        name: str,
        version: Optional[int] = None,
        label: str = "production"
    ) -> Dict[str, Any]:
        """
        Get the configuration object for a prompt.
        
        Args:
            name: Prompt name
            version: Specific version (optional)
            label: Version label (default: "production")
            
        Returns:
            Dict containing model config (temperature, model name, etc.)
        """
        prompt = self.get_prompt(name, version=version, label=label)
        return prompt.config or {}

    def list_prompts(self) -> List[str]:
        """
        List all available prompt names (requires Langfuse API access).
        Note: This is a convenience method - Langfuse may require API calls.
        
        Returns:
            List of prompt names
        """
        try:
            logger.info("Listing all prompts from Langfuse")
            # Note: Langfuse SDK may not have a direct list method
            # This would require API calls or tracking separately
            logger.warning("list_prompts() not fully implemented - track prompts separately")
            return []
        except Exception as e:
            logger.error(f"Failed to list prompts: {e}")
            return []