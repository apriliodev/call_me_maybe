from pydantic import BaseModel, Field
from typing import Literal

class Args(BaseModel):
    type: Literal["string", "number"]

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Args]
    returns: str | None = None

class PromptEntry(BaseModel):
    prompt:str

class FunctionCallResult(BaseModel):
    prompt:str
    name:str
    parameters: dict[str, str | int]

