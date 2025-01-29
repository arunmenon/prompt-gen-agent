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

    --- PLAN STAGE ---
    1) Domain Breakdown: Analyze the problem statement precisely in the context of {domain}.
    2) Input Analysis: Understand placeholders and output context in light of the domain breakdown.
    3) Schema Inference: Determine or confirm the final output schema.

    --- EXECUTION STAGE ---
    4) Prompt Construction: Create a prompt template with a header (domain context), body (placeholders), and footer (mandatory schema).
    5) Prompt Refinement: Finalize the prompt template, ensuring placeholders & schema usage are mandatory.
    """

    def __init__(self, llm_model: str = "openai/gpt-4"):
        self.llm_model = llm_model
        self.llm = LLM(model=self.llm_model, temperature=0.2, verbose=False)
        self.inputs: Dict[str, Any] = {}

    # --------------------------------
    # Before Kickoff: capture inputs
    # --------------------------------
    @before_kickoff
    def capture_inputs(self, inputs: Dict[str, Any]):
        """
        This method captures top-level placeholders for CrewAI’s .format(**inputs).
        e.g. {problem_statement}, {domain}, etc.
        """
        self.inputs = inputs
        return inputs

    # --------------------------------
    # Agents
    # --------------------------------

    @agent
    def domain_breakdown_agent(self) -> Agent:
        return Agent(
            role="Domain Breakdown Analyzer for {domain}",
            goal=(
                "Take the user's problem statement and precisely break it down in the context of {domain}. "
                "Identify domain-specific nuances and how they shape the user's main goal."
            ),
            backstory=(
                "An agent specialized in understanding how the {domain} domain influences the problem statement. "
                "Extract key points, constraints, or focus areas unique to {domain}."
            ),
            llm=self.llm,
            memory=True,
            verbose=False,
            allow_delegation=False,
            max_iter=3,
            respect_context_window=True,
            use_system_prompt=True,
            cache=False,
            max_retry_limit=2
        )

    @agent
    def input_analysis_agent(self) -> Agent:
        return Agent(
            role="Input Analysis Agent in {domain}",
            goal=(
                "Examine and interpret the input placeholders and output context, factoring in the domain breakdown "
                "to see how placeholders and output requirements align with {domain} constraints."
            ),
            backstory=(
                "An agent that takes the domain breakdown and applies it to the user's placeholders, understanding how "
                "they should be used given the problem statement and domain specifics."
            ),
            llm=self.llm,
            memory=True,
            verbose=False,
            allow_delegation=False,
            max_iter=3,
            respect_context_window=True,
            use_system_prompt=True,
            cache=False,
            max_retry_limit=2
        )

    @agent
    def schema_inference_agent(self) -> Agent:
        return Agent(
            role="Schema Inference Expert for {domain}",
            goal="Confirm or infer a final output schema (e.g. markdown, JSON) appropriate to {domain} and the user's requirements.",
            backstory=(
                "Ensures the final output schema is not optional, referencing {domain} best practices "
                "and any user-provided output_schema. If not provided, choose an appropriate default."
            ),
            llm=self.llm,
            memory=True,
            verbose=False,
            allow_delegation=False,
            max_iter=3,
            respect_context_window=True,
            use_system_prompt=True,
            cache=False,
            max_retry_limit=2
        )

    @agent
    def prompt_construction_agent(self) -> Agent:
        return Agent(
            role="Prompt Constructor for {domain}",
            goal="Combine domain breakdown, input analysis, and schema inference into a single cohesive prompt template.",
            backstory=(
                "An expert prompter that weaves the domain context (header), placeholders usage (body), and mandatory schema instructions (footer) into one text."
            ),
            llm=self.llm,
            memory=True,
            verbose=False,
            allow_delegation=False,
            max_iter=3,
            respect_context_window=True,
            use_system_prompt=True,
            cache=False,
            max_retry_limit=2
        )

    @agent
    def prompt_refinement_agent(self) -> Agent:
        return Agent(
            role="Prompt Refiner for {domain}",
            goal="Polish the final prompt template, ensuring placeholders are intact, domain references are correct, and the schema is mandatory.",
            backstory=(
                "A meticulous editor who checks that the final prompt is well-structured, references {domain} properly, "
                "and enforces the output schema instructions."
            ),
            llm=self.llm,
            memory=True,
            verbose=False,
            allow_delegation=False,
            max_iter=3,
            respect_context_window=True,
            use_system_prompt=True,
            cache=False,
            max_retry_limit=2
        )

    # --------------------------------
    # TASKS (PLAN STAGE)
    # --------------------------------

    @task
    def domain_breakdown_task(self) -> Task:
        """
        STEP 1 (Plan): Break down the problem statement in context of {domain}.
        """
        description = r"""
Problem Statement: {problem_statement}
Domain: {domain}

**INSTRUCTIONS**:
1. Analyze the user's problem statement, focusing on the {domain}-specific aspects.
2. Identify any nuances or constraints that the {domain} imposes on this problem.
3. Return strictly JSON of the form:
   {{
     "domain_key_points": [
       "... bullet points or short lines about how {domain} shapes the problem statement ..."
     ],
     "relevant_goals": [
       "... derived or restated objectives from problem_statement in domain context ..."
     ]
   }}

No extra commentary.
"""
        return Task(
            description=description,
            expected_output='{"domain_key_points":[],"relevant_goals":[]}',
            agent=self.domain_breakdown_agent()
        )

    @task
    def input_analysis_task(self) -> Task:
        """
        STEP 2 (Plan): Understand placeholders & output context in light of the domain breakdown.
        """
        description = r"""
We have the domain breakdown:
{{output from domain_breakdown_task}}

We also have:
- Placeholders: {input_placeholders}
- Output Context: {output_context}

**INSTRUCTIONS**:
1. For each placeholder, if it has a colon (e.g. "<<title: The product name>>"), split into:
   - name: "<<title>>"
   - description: "The product name"
   Otherwise, leave description blank.
2. Reflect on how each placeholder or the output context aligns with the domain breakdown. 
   For example: "Placeholder <<features>> might describe product attributes, relevant in {domain} because..."
3. Return strictly JSON:
   {{
     "decoded_placeholders": [
       {{
         "name": "<<placeholderName>>",
         "description": "Optional or empty"
       }},
       ...
     ],
     "context_analysis": [
       "... bullet points about how {output_context} fits {domain} or the problem statement ..."
     ]
   }}

No extra commentary outside the JSON.
"""
        return Task(
            description=description,
            expected_output='{"decoded_placeholders":[],"context_analysis":[]}',
            agent=self.input_analysis_agent(),
            context=[self.domain_breakdown_task()]
        )

    @task
    def schema_inference_task(self) -> Task:
        """
        STEP 3 (Plan): Determine or confirm the final output schema.
        """
        description = r"""
We have domain breakdown and input analysis:
{{output from input_analysis_task}}

Also, user-supplied (or empty) 'output_schema': {output_schema}.

**INSTRUCTIONS**:
1. If 'output_schema' is specified and valid, confirm it as final_schema.
2. If it's empty, propose one that fits the {domain} and the above context_analysis.
3. Return JSON:
   {{
     "final_schema": "e.g. markdown or json",
     "schema_details": [
       "... details on how to implement that schema ...",
       "... e.g. bullet points, headings, JSON keys, etc. ..."
     ]
   }}

No extra commentary.
"""
        return Task(
            description=description,
            expected_output='{"final_schema":"","schema_details":[]}',
            agent=self.schema_inference_agent(),
            context=[self.input_analysis_task()]
        )

    # --------------------------------
    # TASKS (EXECUTION STAGE)
    # --------------------------------

    @task
    def prompt_construction_task(self) -> Task:
        """
        STEP 4 (Execution): Create a cohesive prompt template (header/body/footer).
        """
        description = r"""
We have:
- Domain breakdown: {{output from domain_breakdown_task}}
- Placeholders & context analysis: {{output from input_analysis_task}}
- Schema inference: {{output from schema_inference_task}}

**INSTRUCTIONS**:
1. Build a single prompt template text with:
   - A header referencing {domain} context or role (based on "domain_key_points" / "relevant_goals").
   - A body describing how to use each placeholder from "decoded_placeholders" in a meaningful way.
   - A footer with explicit instructions about the "final_schema". This schema is mandatory.

2. Return strictly JSON:
   {{
     "draftPrompt": "...(the entire prompt template text)..."
   }}

No commentary beyond JSON.
"""
        return Task(
            description=description,
            expected_output='{"draftPrompt": "..."}',
            agent=self.prompt_construction_agent(),
            context=[self.schema_inference_task()]
        )

    @task
    def prompt_refinement_task(self) -> Task:
        """
        STEP 5 (Execution): Refine the final prompt template, ensuring placeholders & schema usage are mandatory.
        """
        description = r"""
We have a draft prompt:
{{output}}

**INSTRUCTIONS**:
1. Refine & finalize the prompt text. Keep placeholders intact (e.g. <<title>>).
2. Ensure references to {domain} remain consistent.
3. The schema instructions must be mandatory (remove optional language if present).
4. Return strictly JSON:
   {{
     "final_prompt": "...(the refined final prompt text)...",
     "notes": []
   }}

No commentary or extra fields.
"""
        return Task(
            description=description,
            expected_output='{"final_prompt":"...","notes":[]}',
            agent=self.prompt_refinement_agent(),
            context=[self.prompt_construction_task()],
            output_pydantic=PromptGenConfig
        )

    # --------------------------------
    # The Crew Pipeline
    # --------------------------------

    @crew
    def crew(self) -> Crew:
        """
        The pipeline is organized in 5 tasks across 2 stages:

        --- PLAN STAGE ---
        1) domain_breakdown_task (domain_breakdown_agent)
        2) input_analysis_task (input_analysis_agent)
        3) schema_inference_task (schema_inference_agent)

        --- EXECUTION STAGE ---
        4) prompt_construction_task (prompt_construction_agent)
        5) prompt_refinement_task (prompt_refinement_agent)
        """
        return Crew(
            agents=[
                self.domain_breakdown_agent(),
                self.input_analysis_agent(),
                self.schema_inference_agent(),
                self.prompt_construction_agent(),
                self.prompt_refinement_agent()
            ],
            tasks=[
                # Plan Stage
                self.domain_breakdown_task(),
                self.input_analysis_task(),
                self.schema_inference_task(),

                # Execution Stage
                self.prompt_construction_task(),
                self.prompt_refinement_task()
            ],
            process=Process.sequential,
            verbose=True
        )
