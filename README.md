# Automação de Chamados de Parceiros com IA + n8n

Fluxo automatizado que recebe chamados de restaurantes parceiros por **webhook**, classifica com **Gemini (RAG + JSON estruturado)** e decide a rota: casos simples são resolvidos automaticamente e casos de risco passam por **aprovação humana no Telegram**.

> A política de parceiros ("Sabor Express") e os chamados de exemplo são **fictícios**, criados apenas para demonstração.

## Arquitetura

```
Webhook (n8n) → API FastAPI (RAG + Gemini) → Switch (rota)
                                               ├─ automático → registra em planilha + responde ao parceiro
                                               └─ aprovação humana → alerta no Telegram (aprovar / escalar)
```

- **API (`api.py`)**: reaproveita o motor de RAG do projeto [copiloto-politicas-parceiros](https://github.com/SEU-USUARIO/copiloto-politicas-parceiros) (embeddings do Gemini + similaridade com numpy, sem banco vetorial) e devolve um JSON tipado com Pydantic.
- **Decisão de rota em código**: o modelo extrai categoria, urgência, valor e risco de fraude; quem decide se precisa de humano são regras determinísticas em Python (urgência alta, risco de fraude, valor acima do limite ou caso fora da política).
- **Tolerância a falhas**: repetição automática em erros temporários da API e troca para um modelo reserva.
- **n8n**: orquestra o fluxo. Passo a passo em [`ROTEIRO_N8N.md`](ROTEIRO_N8N.md); o fluxo exportado fica em `fluxo_n8n/`.

## Como rodar a API

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # e preencha GEMINI_API_KEY
python -m uvicorn api:app --reload
```

Documentação interativa em `http://localhost:8000/docs`.

## Avaliação

Com a API rodando, em outro terminal:

```bash
python testar_api.py
```

O script envia os chamados de `chamados_teste.json` e mostra a taxa de acerto da rota escolhida, a cláusula citada e a latência.

## Exemplo de resposta

```json
{
  "categoria": "atraso_entrega",
  "urgencia": "media",
  "valor_disputa": 62.0,
  "risco_fraude": "baixo",
  "elegivel_reembolso": true,
  "clausula_citada": "Cláusula 2.3",
  "rota": "automatico",
  "motivos_rota": []
}
```

## Tecnologias

n8n · FastAPI · Google Gemini (`google-genai`) · Pydantic · NumPy · Telegram Bot

<!-- Adicione aqui um print do fluxo no n8n e um GIF curto do teste funcionando -->
