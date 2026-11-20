"""Mini-avaliação: envia os chamados de teste para a API e compara a rota obtida com a esperada."""

import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

URL = os.getenv("API_URL", "http://localhost:8000/triar")
TOKEN = os.getenv("API_TOKEN")
headers = {"X-API-Key": TOKEN} if TOKEN else {}

with open("chamados_teste.json", encoding="utf-8") as f:
    casos = json.load(f)

acertos = 0
for c in casos:
    inicio = time.time()
    try:
        r = requests.post(
            URL, json={"chamado": c["chamado"], "id_chamado": c["id"]}, headers=headers, timeout=180
        )
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        print(f"{c['id']} | ERRO: {e}")
        continue

    ok = d["rota"] == c["rota_esperada"]
    acertos += ok
    print(
        f"{c['id']} | {'OK  ' if ok else 'FALHA'} | rota={d['rota']} (esperada={c['rota_esperada']}) | "
        f"{d['categoria']} | {d['clausula_citada']} | {time.time() - inicio:.1f}s"
    )
    if d["motivos_rota"]:
        print("        motivos:", "; ".join(d["motivos_rota"]))

print(f"\nTaxa de acerto da rota: {acertos}/{len(casos)} ({100 * acertos / len(casos):.0f}%)")
