# src/prompt_gen/api/routers/prompt.py
import json
from fastapi import APIRouter, HTTPException
from ...schemas import PromptGenInput
from ...crew import PromptGenCrew

router = APIRouter()

@router.post("/create_prompt")
def create_prompt(input: PromptGenInput):
    """
    Takes user’s prompt-generation requirements, runs the PromptGenCrew,
    and returns the final prompt.
    """
    crew_instance = PromptGenCrew()
    result = crew_instance.crew().kickoff(inputs=input.dict())

    # Try CrewAI’s parsed JSON first:
    final_data = result.json_dict

    if not final_data:
        # If that failed, fallback to manual parsing of the raw text:
        raw_output = result.raw or ""
        try:
            final_data = json.loads(raw_output)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Agent returned non-JSON output. Raw response was:\n{raw_output}"
            )

    if not isinstance(final_data, dict):
        raise HTTPException(status_code=500, detail="No valid JSON structure from final prompt.")

    final_prompt = final_data.get("final_prompt")
    if not final_prompt:
        raise HTTPException(status_code=500, detail="No 'final_prompt' in the final JSON.")

    return {
        "status": "success",
        "final_prompt": final_prompt,
        "notes": final_data.get("notes", [])
    }
