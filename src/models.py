"""Data models for function calling and schema validation."""

from typing import Any, Literal
from pydantic import BaseModel


class Args(BaseModel):
    """Schema for a single function argument or return type."""

    type: Literal["string", "number"]


class FunctionDefinition(BaseModel):
    """Schema defining an available function and its signature."""

    name: str
    description: str
    parameters: dict[str, Args]
    returns: Args


class PromptEntry(BaseModel):
    """Schema for a test prompt input entry."""

    prompt: str


class FunctionCallResult(BaseModel):
    """Schema for the final function call output result."""

    prompt: str
    name: str
    parameters: dict[str, Any]
