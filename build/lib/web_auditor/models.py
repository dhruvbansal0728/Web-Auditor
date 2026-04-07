from typing import Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation

class WebAuditorAction(Action):
    """Action for the Web Auditor environment."""
    command: str = Field(..., description="The bash command to execute within the working directory")

class WebAuditorObservation(Observation):
    """Observation from the Web Auditor environment."""
    output: str = Field(default="", description="Terminal output or initial environment state")
    current_directory_structure: str = Field(default="", description="The list of files currently in the working directory")
    file_content: Optional[str] = Field(default=None, description="The raw content of an HTML file if read")
