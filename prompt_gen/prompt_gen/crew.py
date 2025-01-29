# src/prompt_gen/crew.py

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, task, crew, before_kickoff
from .schemas import PromptGenConfig

@CrewBase
class PromptGenCrew:
    """
    A multi-step meta-crew that:
    1) Gathers user requirements (problem statement, domain, placeholders, etc.).
    2) Interprets those requirements and clarifies how to incorporate them in a prompt.
    3) Crafts a draft prompt with a standardized structure (header/system, body, output format).
    4) Refines the prompt into final form.
    """

    def __init__(self, llm_model: str = "openai/gpt-4o"):
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
            backstory="A thorough interpreter with domain knowledge for {domain} , clarifying how to incorporate placeholders, domain constraints, and the final schema.",
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
            goal="Construct a well-structured prompt template from the interpreted requirements, including system instructions, placeholders usage, and output schema details.",
            backstory="An expert prompter who ensures placeholders, domain references, and final output schema are integrated with best practices.",
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
            goal="Polish the drafted prompt template, ensuring placeholders are used consistently and final instructions (schema, domain references) are unambiguous.",
            backstory="A meticulous editor who perfects the final prompt structure for clarity, correctness, and domain alignment.",
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
        # Step 1: Simply echo user inputs as JSON
        description = r"""
Below are user inputs:

- Problem Statement: {problem_statement}
- Domain: {domain}
- Input Placeholders: {input_placeholders}
- Output Context: {output_context}
- Output Schema: {output_schema}

**INSTRUCTIONS**:
1. Convert these user inputs into a single JSON object, e.g.:

{{
  "problem_statement": "...",
  "domain": "...",
  "input_placeholders": [...],
  "output_context": "...",
  "output_schema": "..."
}}

2. No extra commentary, just that JSON.
"""
        return Task(
            description=description,
            expected_output="JSON with user’s raw requirements.",
            agent=self.requirement_analyzer()
        )

    @task
    def interpret_requirements_task(self) -> Task:
        # Step 2: Interpret how to incorporate domain, placeholders, context, schema
        # We also ask it to decode placeholders if they contain colons
        description = r"""
We have the raw user requirements:
{{output}}

**INSTRUCTIONS**:
1. Please analyze the placeholders. If you see placeholders like "<<title: Some description>>", parse it into:
   - name: "<<title>>"
   - description: "Some description"
   If they have no description, keep them as-is.
2. Clarify how the final prompt template should incorporate:
   - The domain knowledge
   - The placeholders (with names & any descriptions)
   - The output context
   - The output schema (if any)
3. Return strictly JSON with a structure like:
   {{
     "clarification": [
       "... bullet points on usage ...",
       "... domain references ...",
       "... placeholders usage ...",
       "... schema instructions ..."
     ],
     "decoded_placeholders": [
       {{
         "name": "<<placeholderName>>",
         "description": "Optional or empty if none"
       }},
       ...
     ]
   }}

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
        # Step 3: Build a prompt template with a standard structure:
        # [SYSTEM], [BODY], [OUTPUT].
        # We'll integrate placeholders & domain constraints here.
        description = r"""
Here is the clarified approach from the previous step:
{{output}}

**INSTRUCTIONS**:
1. Construct a SINGLE prompt template with three sections:

   [SYSTEM SECTION]
   - Summarize the domain or role the assistant should adopt. For example: "You are an expert in {domain}..."

   [BODY SECTION]
   - Incorporate instructions for using the placeholders listed under "decoded_placeholders".
   - Show how to insert them, e.g. "<<title>>" or if there's a description, mention it briefly.
   - Mention any constraints (tone, length, style) from the "clarification".
   - Reference the problem statement or output context from the user if needed.

   [OUTPUT SECTION]
   - Explicitly instruct how the final output from this prompt should be formatted (e.g. in {output_schema}).
   - If the schema is "markdown," mention to use bullet points, headings, etc.
   - If it's "json," mention to return valid JSON, with certain fields.

2. Return strictly JSON with:
   {{
     "draftPrompt": "...(the entire prompt template with the 3 sections)..."
   }}

No commentary outside of that JSON.
"""
        return Task(
            description=description,
            expected_output='{{"draftPrompt": "..."}}',
            agent=self.prompt_crafter(),
            context=[self.interpret_requirements_task()]
        )

    @task
    def refine_prompt_task(self) -> Task:
        # Step 4: Refine the prompt template. Keep the 3 sections, placeholders, etc.
        description = r"""
We have a draft prompt:
{{output}}

**INSTRUCTIONS**:
1. Refine & finalize the prompt template. Keep the same three sections:
   [SYSTEM SECTION], [BODY SECTION], [OUTPUT SECTION].
2. Ensure placeholders like <<title>> are not lost.
3. Make sure the final instructions for output schema are clear and consistent.
4. Return strictly JSON:
   {{
     "final_prompt": "...(the refined final prompt template)...",
     "notes": []
   }}

No commentary or extra fields.
"""
        return Task(
            description=description,
            expected_output='{{"final_prompt":"...","notes":[]}}',
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
