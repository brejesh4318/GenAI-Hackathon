import os
from app.utilities.constants import Constants
from app.utilities.singletons_factory import DcSingleton

class EnvironmentVariableRetriever(metaclass = DcSingleton):
    
    @classmethod
    def get_env_variable(cls,variable_name):
        """
        Retrieves the value of an environment variable.

        Args:
            variable_name (str): The name of the environment variable.

        Returns:
            str: The value of the environment variable, or None if not found.
        """
        # First, check if the variable exists in os.environ
        if variable_name in os.environ:
            return os.environ.get(variable_name)