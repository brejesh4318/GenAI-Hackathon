"""
Unified Prompt Management Service

This service consolidates prompt management with Langfuse integration,
providing efficient caching and startup loading of all prompts.

Features:
- Fetch all prompts at service initialization
- In-memory caching for fast access
- Automatic fallback to Langfuse on cache miss
- Support for template compilation with variables
- Production-ready error handling and logging
"""

from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from langfuse import Langfuse
from app.utilities import dc_logger
from app.utilities.singletons_factory import DcSingleton
from app.utilities.env_util import EnvironmentVariableRetriever
from app.utilities.constants import Constants

logger = dc_logger.LoggerAdap(dc_logger.get_logger(__name__), {"dash-test": "V1"})
load_dotenv()


class PromptService(metaclass=DcSingleton):
    """
    Unified prompt service with caching and Langfuse integration.
    
    All prompts are fetched at initialization for optimal performance.
    Provides simple interface for accessing and compiling prompts.
    """

    def __init__(self):
        """Initialize Langfuse client and load all prompts"""
        self._client = None
        self._prompt_cache: Dict[str, Any] = {}
        self._prompt_names: List[str] = []
        
        self._initialize_client()
        self._load_prompt_names()
        self._prefetch_all_prompts()
        
        logger.info(f"PromptService initialized with {len(self._prompt_cache)} prompts cached")

    def _initialize_client(self) -> None:
        """Initialize Langfuse client with credentials"""
        try:
            public_key = EnvironmentVariableRetriever.get_env_variable("LANGFUSE_PUBLIC_KEY")
            secret_key = EnvironmentVariableRetriever.get_env_variable("LANGFUSE_SECRET_KEY")
            host = EnvironmentVariableRetriever.get_env_variable("LANGFUSE_HOST")
            
            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            logger.info("Langfuse client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse client: {e}")
            raise RuntimeError(f"Langfuse initialization failed: {e}") from e

    def _load_prompt_names(self) -> None:
        """Load prompt names from configuration"""
        try:
            prompt_config = Constants.fetch_constant("prompt_names")
            if isinstance(prompt_config, list):
                self._prompt_names = prompt_config
            elif isinstance(prompt_config, dict):
                self._prompt_names = list(prompt_config.values())
            else:
                logger.warning("No prompt_names found in constants, will fetch on-demand")
                self._prompt_names = []
            
            logger.info(f"Loaded {len(self._prompt_names)} prompt names from configuration")
        except Exception as e:
            logger.warning(f"Failed to load prompt names from config: {e}. Will fetch on-demand.")
            self._prompt_names = []

    def _prefetch_all_prompts(self) -> None:
        """Fetch all configured prompts at startup"""
        if not self._prompt_names:
            logger.info("No prompts to prefetch")
            return
        
        success_count = 0
        failure_count = 0
        
        for prompt_name in self._prompt_names:
            try:
                # Fetch from Langfuse with production label
                prompt = self._fetch_from_langfuse(prompt_name, label="production")
                if prompt:
                    cache_key = self._get_cache_key(prompt_name, label="production")
                    self._prompt_cache[cache_key] = prompt
                    success_count += 1
                    logger.debug(f"Cached prompt: {prompt_name}")
                else:
                    failure_count += 1
                    logger.warning(f"Prompt '{prompt_name}' returned None from Langfuse")
            except Exception as e:
                failure_count += 1
                logger.error(f"Failed to prefetch prompt '{prompt_name}': {e}")
                raise e
        
        logger.info(f"Prefetch complete: {success_count} succeeded, {failure_count} failed")

    def _get_cache_key(self, name: str, version: Optional[int] = None, label: str = "production") -> str:
        """Generate cache key for prompt"""
        if version:
            return f"{name}:v{version}"
        return f"{name}:{label}"

    def _fetch_from_langfuse(
        self,
        name: str,
        version: Optional[int] = None,
        label: str = "production",
        prompt_type: str = "text"
    ) -> Any:
        """
        Fetch prompt from Langfuse API.
        
        Args:
            name: Prompt name
            version: Specific version number (optional)
            label: Version label (default: "production")
            prompt_type: Prompt type for validation ("text" or "chat")
        
        Returns:
            Langfuse prompt object or None
        """
        try:
            if version:
                prompt = self._client.get_prompt(name, version=version, type=prompt_type)
            else:
                prompt = self._client.get_prompt(name, label=label, type=prompt_type)
            
            if not prompt:
                logger.warning(f"Prompt '{name}' not found in Langfuse")
                return None
            
            return prompt
        except Exception as e:
            logger.error(f"Failed to fetch prompt '{name}' from Langfuse: {e}")
            return None

    def get(
        self,
        name: str,
        version: Optional[int] = None,
        label: str = "production"
    ) -> str:
        """
        Get prompt template string.
        
        Args:
            name: Prompt name (e.g., "brain-orchestrator-agent")
            version: Specific version number (optional)
            label: Version label (default: "production")
        
        Returns:
            str: Prompt template with {{variable}} placeholders
        
        Raises:
            RuntimeError: If prompt not found
        
        Example:
            prompt_service = PromptService()
            template = prompt_service.get("test-case-generator")
        """
        cache_key = self._get_cache_key(name, version, label)
        
        # Check cache first
        if cache_key in self._prompt_cache:
            prompt = self._prompt_cache[cache_key]
            return prompt.prompt
        
        # Cache miss - fetch from Langfuse
        logger.info(f"Cache miss for '{name}', fetching from Langfuse")
        prompt = self._fetch_from_langfuse(name, version, label)
        
        if not prompt:
            raise RuntimeError(f"Prompt '{name}' not found in cache or Langfuse")
        
        # Update cache
        self._prompt_cache[cache_key] = prompt
        return prompt.prompt

    
    def get_config(
        self,
        name: str,
        version: Optional[int] = None,
        label: str = "production"
    ) -> Dict[str, Any]:
        """
        Get prompt configuration (model, temperature, etc.).
        
        Args:
            name: Prompt name
            version: Specific version number (optional)
            label: Version label (default: "production")
        
        Returns:
            Dict containing model configuration
        
        Example:
            config = prompt_service.get_config("test-case-generator")
            # Returns: {"model": "gemini-2.5-flash", "temperature": 0.7}
        """
        cache_key = self._get_cache_key(name, version, label)
        
        if cache_key in self._prompt_cache:
            prompt = self._prompt_cache[cache_key]
        else:
            logger.info(f"Cache miss for '{name}', fetching from Langfuse")
            prompt = self._fetch_from_langfuse(name, version, label)
            if not prompt:
                raise RuntimeError(f"Prompt '{name}' not found")
            self._prompt_cache[cache_key] = prompt
        
        return prompt.config or {}

    def create_prompt(
        self,
        name: str,
        prompt: str | List[Dict[str, str]],
        prompt_type: str = "text",
        labels: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Create or update a prompt in Langfuse.
        
        Args:
            name: Unique prompt identifier (e.g., "test-case-generator")
            prompt: Template string (text) or list of messages (chat)
            prompt_type: "text" or "chat"
            labels: Version labels (default: ["production"])
            config: Model configuration (temperature, model name, etc.)
            tags: Additional tags for organization
        
        Example:
            prompt_service.create_prompt(
                name="test-case-generator",
                prompt="Generate test cases for: {{document}}",
                prompt_type="text",
                labels=["production"],
                config={"model": "gemini-2.5-flash", "temperature": 0.7}
            )
        """
        labels = labels or ["production"]
        config = config or {}
        tags = tags or []
        
        try:
            logger.info(f"Creating prompt '{name}' with type '{prompt_type}'")
            
            self._client.create_prompt(
                name=name,
                type=prompt_type,
                prompt=prompt,
                labels=labels,
                config=config,
                tags=tags
            )
            
            # Invalidate cache for this prompt
            for label in labels:
                cache_key = self._get_cache_key(name, label=label)
                if cache_key in self._prompt_cache:
                    del self._prompt_cache[cache_key]
            
            logger.info(f"Prompt '{name}' created successfully")
        except Exception as e:
            logger.error(f"Failed to create prompt '{name}': {e}")
            raise RuntimeError(f"Failed to create prompt '{name}': {e}") from e

    def refresh_cache(self, name: Optional[str] = None) -> None:
        """
        Refresh cached prompts from Langfuse.
        
        Args:
            name: Specific prompt name to refresh (optional)
                  If None, refreshes all cached prompts
        """
        if name:
            # Refresh specific prompt
            logger.info(f"Refreshing cache for prompt '{name}'")
            cache_keys_to_refresh = [k for k in self._prompt_cache.keys() if k.startswith(f"{name}:")]
            
            for cache_key in cache_keys_to_refresh:
                # Extract version/label from cache key
                if ":v" in cache_key:
                    _, version_str = cache_key.split(":v")
                    version = int(version_str)
                    prompt = self._fetch_from_langfuse(name, version=version)
                else:
                    _, label = cache_key.split(":")
                    prompt = self._fetch_from_langfuse(name, label=label)
                
                if prompt:
                    self._prompt_cache[cache_key] = prompt
        else:
            # Refresh all prompts
            logger.info("Refreshing all cached prompts")
            self._prompt_cache.clear()
            self._prefetch_all_prompts()

    def list_cached_prompts(self) -> List[str]:
        """
        List all prompts currently in cache.
        
        Returns:
            List of cached prompt keys (name:label or name:vN)
        """
        return list(self._prompt_cache.keys())

    def clear_cache(self) -> None:
        """Clear all cached prompts"""
        logger.info("Clearing prompt cache")
        self._prompt_cache.clear()

    # Backward compatibility aliases
    def fetch(self, prompt_name: str, version: Optional[int] = None, label: str = "production") -> str:
        """Alias for get() method for backward compatibility"""
        return self.get(prompt_name, version, label)

