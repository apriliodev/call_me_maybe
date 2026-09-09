from llm_sdk import Small_LLM_Model
import json
import numpy

llm = Small_LLM_Model()
path = llm.get_path_to_tokenizer_file()
print("Fichier tokenizer :", path)

with open(path, encoding="utf-8") as f:
    data = json.load(f)

print("Clés racine :", list(data.keys()))
print("Taille du vocab :", len(data["model"]["vocab"]))
print(llm.encode("Deux fois dix font "))


ids = [1912, 2200, 36191, 294, 941, 3301, 220, 17]
import numpy as np

a = llm.get_logits_from_input_ids(ids)
best_id = int(np.argmax(a))
print(best_id, repr(llm.decode([best_id])))


def closed_word() -> None:
    words = ["chat", "chien"]
    



"""
print(llm.decode(ids))
print(llm.get_logits_from_input_ids(ids))
"""

"""
for text in ["Hello", " Hello", "hello", "42", " 42", '"', "{", '{"name"']:
    ids = llm.encode(text)
    print(repr(text), "->", ids)
"""

"""
# Regarde à quoi ressemblent quelques entrées du vocab
vocab = data["model"]["vocab"]
for i, (token, tid) in enumerate(vocab.items()):
    print(repr(token), "->", tid)
    if i > 15:
        break
"""