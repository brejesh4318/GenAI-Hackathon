import os
from langchain_google_genai import ChatGoogleGenerativeAI
from app.services.llm_services.llm_interface import LLMInterface
from app.utilities.env_util import EnvironmentVariableRetriever

class Gemini25FlashLLM(LLMInterface):
    def __init__(self, temperature: float = 1.0):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=temperature,
            api_key=EnvironmentVariableRetriever.get_env_variable("GEMINI_API_KEY")
        )

    def get_llm(self) -> "Gemini25FlashLLM":
        """
        Static method to get an instance of Gemini25FlashLLM.

        Returns:
            Gemini25FlashLLM: An instance of the Gemini25FlashLLM class.
        """
        return self.llm

    def generate(self, prompt: str) -> str:
        """
        Generates output from Gemini 2.5 Flash model for the given prompt.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The generated output from the LLM.
        """
        return self.llm.invoke(prompt).content


class Gemini25FlashLiteLLM(LLMInterface):
    def __init__(self, temperature: float = 1.0):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=temperature,
            api_key=EnvironmentVariableRetriever.get_env_variable("GEMINI_API_KEY")
        )

    def get_llm(self) -> "Gemini25FlashLiteLLM":
        """
        Static method to get an instance of Gemini25FlashLLM.

        Returns:
            Gemini25FlashLLM: An instance of the Gemini25FlashLLM class.
        """
        return self.llm

    def generate(self, prompt: str) -> str:
        """
        Generates output from Gemini 2.5 Flash model for the given prompt.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The generated output from the LLM.
        """
        return self.llm.invoke(prompt).content
