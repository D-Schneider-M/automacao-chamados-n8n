"""API de triagem de chamados: o Gemini extrai e classifica, o código Python decide a rota."""

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from motor_rag import BaseConhecimento, TriagemChamado

load_dotenv()

LIMITE_VALOR = float(os.getenv("LIMITE_VALOR_APROVACAO", "100"))  # R$ a partir do qual exige aprovação
API_TOKEN = os.getenv("API_TOKEN")  # se definido, exige o cabeçalho X-API-Key

estado: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    texto = Path(__file__).with_name("politica_parceiros.txt").read_text(encoding="utf-8")
    estado["base"] = BaseConhecimento(texto)  # gera os embeddings uma única vez, na inicialização
    yield
    estado.clear()


app = FastAPI(title="API de Triagem de Chamados de Parceiros", version="1.0.0", lifespan=lifespan)


class EntradaChamado(BaseModel):
    chamado: str = Field(min_length=10, max_length=2000)
    id_chamado: Optional[str] = None


class RespostaTriagem(TriagemChamado):
    id_chamado: Optional[str] = None
    rota: Literal["automatico", "aprovacao_humana"]
    motivos_rota: list[str]
    modelo_usado: str
    trechos_recuperados: list[dict]


def decidir_rota(t: TriagemChamado) -> tuple[str, list[str]]:
    """Regras determinísticas: a decisão de risco não fica a cargo do modelo."""
    motivos = []
    if t.urgencia == "alta":
        motivos.append("urgência alta")
    if t.risco_fraude in ("medio", "alto"):
        motivos.append(f"risco de fraude {t.risco_fraude}")
    if t.valor_disputa is not None and t.valor_disputa >= LIMITE_VALOR:
        motivos.append(f"valor em disputa (R$ {t.valor_disputa:.2f}) acima do limite de R$ {LIMITE_VALOR:.2f}")
    if t.precisa_atendimento_humano:
        motivos.append("caso fora da política ou que exige análise humana")
    return ("aprovacao_humana" if motivos else "automatico"), motivos


@app.get("/saude")
def saude():
    return {"status": "ok", "clausulas_carregadas": len(estado["base"].trechos)}


@app.post("/triar", response_model=RespostaTriagem)
def triar(entrada: EntradaChamado, x_api_key: Optional[str] = Header(default=None)):
    if API_TOKEN and not secrets.compare_digest(x_api_key or "", API_TOKEN):
        raise HTTPException(status_code=401, detail="Token inválido")
    try:
        triagem, trechos, modelo = estado["base"].triar(entrada.chamado)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Falha ao consultar o modelo: {e}")

    rota, motivos = decidir_rota(triagem)
    return RespostaTriagem(
        **triagem.model_dump(),
        id_chamado=entrada.id_chamado,
        rota=rota,
        motivos_rota=motivos,
        modelo_usado=modelo,
        trechos_recuperados=[{"clausula": t.id, "similaridade": round(s, 3)} for t, s in trechos],
    )
