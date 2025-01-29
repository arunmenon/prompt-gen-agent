# src/prompt_gen/api/api.py
from fastapi import FastAPI
from .routers import prompt
# from ..services.prompt_service import init_db  # Optional, if you have a DB

app = FastAPI()

@app.on_event("startup")
def on_startup():
    # init_db()  # e.g. if you have a DB to initialize
    pass

# Include the router for our “prompt-gen” endpoints
app.include_router(prompt.router, prefix="/prompt-gen", tags=["prompt-gen"])
