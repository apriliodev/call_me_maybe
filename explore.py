"""Exploration and testing script using the src modules."""

from pathlib import Path
from llm_sdk import Small_LLM_Model
from src.vocabulary import load_vocabulary
from src.pipeline import load_json_list, format_functions_prompt
from src.models import FunctionDefinition
from src.generator import generate_constrained_choice

llm = Small_LLM_Model()
vocab = load_vocabulary(llm)

functions_path = Path("data/input/functions_definition.json")
functions = load_json_list(functions_path, FunctionDefinition)

candidate_names = [fn.name for fn in functions]
prompt_context = format_functions_prompt(functions)

user_query = "Greet John"
full_prompt = f"{prompt_context}\nUser query: {user_query}\nFunction: "

chosen_function = generate_constrained_choice(
    llm, full_prompt, candidate_names, vocab
)
print(f"User query: '{user_query}'")
print(f"Selected function: '{chosen_function}'")