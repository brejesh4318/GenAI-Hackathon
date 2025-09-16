from app.services.llm_services.llm_implementation.gemini_models import Gemini25FlashLLM, Gemini25FlashLiteLLM
from app.services.llm_services.llm_interface import LLMInterface
from app.utilities.dc_enums import SupportedLlms


class LlmFactory:
        
    @staticmethod
    def get_llm(type: str) -> LLMInterface:
        """
        Factory method to get an LLM instance based on the given type.

        Args:
            type (str): The type of LLM to instantiate.

        Returns:
            LLMInterface: An instance of the LLM class corresponding to the given type.

        Raises:
            ValueError: If no matching LLM implementation is found for the given type.
        """
        if type == SupportedLlms.LLAMA3_GROQ_70B_VERSATILE.value:
            # import and return the LLAMA3_GROQ_70B_VERSATILE implementation
            # from app.services.llm_services. import Llama3GroqLLM
            # return Llama3GroqLLM()
            pass
        elif type == SupportedLlms.GEMINI_2_5_FASH.value:
            # import and return the GEMINI_2_5_FASH implementation
            return Gemini25FlashLLM()
        elif type == SupportedLlms.GEMINI_2_5_FLASH_LITE.value:
            # return the GEMINI_2_0_FLASH_LITE implementation
            return Gemini25FlashLiteLLM()
        else:
            raise ValueError(f"No matching LLM implementation found for type: {type}")