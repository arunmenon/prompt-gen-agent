# src/prompt_gen/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class PromptGenInput(BaseModel):
    """
    The request body for POST /prompt-gen/create_prompt
    describing user-provided context.
    """
    problem_statement: str
    domain: str
    input_placeholders: List[str]
    output_context: str
    output_schema: Optional[str] = None

class PromptGenConfig(BaseModel):
    """
    The final output from the last task in the agentic flow,
    representing the polished prompt (and optional notes).
    """
    final_prompt: str
    notes: Optional[List[str]] = []
