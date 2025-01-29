# src/prompt_gen/crew.py

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, task, crew, before_kickoff
from .schemas import PromptGenConfig  # or import from your schemas module

@CrewBase
class PromptGenCrew:
    """
    A multi-step meta-crew that:
    1) Gathers user requirements (problem statement, domain, placeholders, etc.).
    2) Interprets those requirements and clarifies how to incorporate them in a prompt.
    3) Crafts a draft prompt.
    4) Refines the prompt into final form.
    """

    def __init__(self, llm_model: str = "openai/gpt-4"):
        self.llm_model = llm_model
        # Example LLM with moderate temperature
        self.llm = LLM(model=self.llm_model, temperature=0.2, verbose=False)
        self.inputs: Dict[str, Any] = {}

    @before_kickoff
    def capture_inputs(self, inputs: Dict[str, Any]):
        """
        This method captures top-level placeholders for CrewAI’s .format(**inputs).
        e.g. {problem_statement}, {domain}, etc.
        """
        self.inputs = inputs
        return inputs

    ##############################
    # Agents
    ##############################

    @agent
    def requirement_analyzer(self) -> Agent:
        return Agent(
            role="Requirements Interpreter",
            goal="Analyze user-supplied problem statement, domain, placeholders, and desired output schema. Distill constraints for building a prompt.",
            backstory="A thorough interpreter with domain knowledge, clarifying the approach to prompt-building.",
            llm=self.llm,
            memory=True,
            verbose=False,
            allow_delegation=False,
            max_iter=5,
            respect_context_window=True,
            use_system_prompt=True,
            cache=False,
            max_retry_limit=2
        )

    @agent
    def prompt_crafter(self) -> Agent:
        return Agent(
            role="Prompt Crafter",
            goal="Construct a well-structured prompt from the interpreted requirements (domain, placeholders, output schema).",
            backstory="A specialized prompter ensuring the user’s problem statement, domain constraints, placeholders, and final output schema are integrated clearly.",
            llm=self.llm,
            memory=True,
            verbose=False,
            allow_delegation=False,
            max_iter=5,
            respect_context_window=True,
            use_system_prompt=True,
            cache=False,
            max_retry_limit=2
        )

    @agent
    def prompt_refiner(self) -> Agent:
        return Agent(
            role="Prompt Refiner",
            goal="Polish the drafted prompt, ensuring placeholders are consistent, instructions unambiguous, and domain references correct.",
            backstory="A meticulous editor who perfects the final prompt for clarity, correctness, and domain alignment.",
            llm=self.llm,
            memory=True,
            verbose=False,
            allow_delegation=False,
            max_iter=5,
            respect_context_window=True,
            use_system_prompt=True,
            cache=False,
            max_retry_limit=2
        )

    ##############################
    # Tasks
    ##############################

    @task
    def gather_user_requirements_task(self) -> Task:
        description = r"""
Below are user inputs:

- Problem Statement: {problem_statement}
- Domain: {domain}
- Input Placeholders: {input_placeholders}
- Output Context: {output_context}
- Output Schema: {output_schema}

**INSTRUCTIONS**:
1. Convert these user inputs into a single JSON object, e.g.:

{
  "problemStatement": "...",
  "domain": "...",
  "inputPlaceholders": [...],
  "outputContext": "...",
  "outputSchema": "..."
}

2. No extra commentary, just that JSON.
"""
        return Task(
            description=description,
            expected_output="JSON with user’s raw requirements.",
            agent=self.requirement_analyzer()
        )

    @task
    def interpret_requirements_task(self) -> Task:
        description = r"""
We have the raw user requirements:
{{output}}

**INSTRUCTIONS**:
1. Clarify how the final prompt should incorporate:
   - The domain knowledge
   - The placeholders
   - The output context
   - The output schema (if any)
2. Return strictly JSON:
   { "clarification": "...some bullet points..." }

No extra commentary.
"""
        return Task(
            description=description,
            expected_output="A JSON with clarifications on prompt-building strategy.",
            agent=self.requirement_analyzer(),
            context=[self.gather_user_requirements_task()]
        )

    @task
    def craft_prompt_task(self) -> Task:
        description = r"""
Here is the clarified approach:
{{output}}

**INSTRUCTIONS**:
1. Draft a single prompt string that includes domain background, placeholders, final instructions about the output, etc.
2. Return strictly JSON with:
   { "draftPrompt": "..." }
No commentary.
"""
        return Task(
            description=description,
            expected_output='{"draftPrompt": "..."}',
            agent=self.prompt_crafter(),
            context=[self.interpret_requirements_task()]
        )

    @task
    def refine_prompt_task(self) -> Task:
        description = r"""
We have a draft prompt:
{{output}}

**INSTRUCTIONS**:
1. Refine & finalize the prompt. Keep placeholders (like <<title>>) if relevant.
2. Return strictly JSON:
   {
     "final_prompt": "...",
     "notes": []
   }

No commentary or extra fields.
"""
        return Task(
            description=description,
            expected_output='{"final_prompt":"...","notes":[]}',
            agent=self.prompt_refiner(),
            context=[self.craft_prompt_task()],
            output_pydantic=PromptGenConfig  # ensures we get final_prompt in structured form
        )

    @crew
    def crew(self) -> Crew:
        """
        The pipeline: gather → interpret → craft → refine.
        """
        return Crew(
            agents=[self.requirement_analyzer(), self.prompt_crafter(), self.prompt_refiner()],
            tasks=[
                self.gather_user_requirements_task(),
                self.interpret_requirements_task(),
                self.craft_prompt_task(),
                self.refine_prompt_task()
            ],
            process=Process.sequential,
            verbose=True
        )
