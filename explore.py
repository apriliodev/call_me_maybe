from llm_sdk import Small_LLM_Model
import json
import numpy as np

llm = Small_LLM_Model()
path = llm.get_path_to_tokenizer_file()


with open(path, encoding="utf-8") as f:
    data = json.load(f)
"""
print("Fichier tokenizer :", path)
print("Clés racine :", list(data.keys()))
print("Taille du vocab :", len(data["model"]["vocab"]))
print(llm.encode("chien"))


ids = [1912, 2200, 36191, 294, 941, 3301, 220, 17]


a = llm.get_logits_from_input_ids(ids)
best_id = int(np.argmax(a))
print(best_id, repr(llm.decode([best_id])))
"""

"""
 "chi": 14604,
chat = 9686
"""

vocab = data["model"]["vocab"]

texte_genere = ""

def tokens_valides(candidats: list[str], texte_genere: str, vocab: dict[str, int]) -> list[int]:
    tokens_ok = []
    for token_str, token_id in vocab.items():
        nouveau_texte = texte_genere + token_str
        for candidat in candidats:
            if candidat.startswith(nouveau_texte):
                tokens_ok.append(token_id)
                break
    return tokens_ok

tensor_ids = llm.encode("Ecrit le mot chat")



def get_token_list() -> list[int]:
    texte_genere = ""
    candidats_restants = ["chien"]
    tokens_generes: list[int] = []

    prompt_base = "Animal qui aboie: "

    while len(candidats_restants) > 1 or texte_genere not in candidats_restants:
        input_ids = llm.encode(prompt_base + texte_genere)
        ids: list[int] = input_ids[0].tolist()
        logits_brut = np.array(llm.get_logits_from_input_ids(ids))
        tokens_ok = tokens_valides(candidats_restants, texte_genere, vocab)
        
        logits_masque = np.full_like(logits_brut, -np.inf)
        logits_masque[tokens_ok] = logits_brut[tokens_ok]        
        meilleur_id = int(np.argmax(logits_masque))
        id_list = []
        id_list.append(meilleur_id)
        token = llm.decode(id_list)
        texte_genere = texte_genere + token
        tokens_generes.append(int(meilleur_id))
        candidats_restants = [c for c in candidats_restants if c.startswith(texte_genere)]
        if texte_genere in candidats_restants:
            return tokens_generes
    return tokens_generes

print(get_token_list())

"""
ids = tensor_ids[0].tolist()
logits_brut = llm.get_logits_from_input_ids(ids)
logits_brut = np.array(logits_brut)
candidats = ["chat", "chien"]

tokens_ok = tokens_valides(candidats, texte_genere, vocab)

logits_masque = np.full_like(logits_brut, -np.inf)
logits_masque[tokens_ok] = logits_brut[tokens_ok]

meilleur_id = int(np.argmax(logits_masque))


print(meilleur_id, repr(llm.decode([meilleur_id])))


tok: str = llm.decode([meilleur_id])

texte_genere = texte_genere + tok
tokens_ok = tokens_valides(candidats, texte_genere, vocab)
logits_masque = np.full_like(logits_brut, -np.inf)
logits_masque[tokens_ok] = logits_brut[tokens_ok]

meilleur_id = int(np.argmax(logits_masque))


print(meilleur_id, repr(llm.decode([meilleur_id])))

"""




def next_token(candidats: str, bef_token: str, vocab: dict[str, int]) -> list[int]:
    candidats_restants = [c for c in candidats if c.startswith(bef_token)]
    restes = [c.removeprefix(bef_token) for c in candidats_restants]
    next_token = []
    valids: list[int] = tokens_valides(restes, bef_token, vocab)
    return valids



    

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