# Deploy no RunPod (GPU)

Imagem pronta: `ghcr.io/igorosena/council-voice-api:latest` (construída pelo
GitHub Actions a cada push na `main`).

## 1. Criar o pod

RunPod → **Pods** → **Deploy** → escolha a GPU (RTX PRO 4000, 3090, A5000...
qualquer uma com 8 GB+). Em **Edit Template / Customize**:

| Campo | Valor |
|---|---|
| Container Image | `ghcr.io/igorosena/council-voice-api:latest` |
| Container Disk | 20 GB |
| Volume Disk | 20 GB (persistente: modelo + vozes) |
| Volume Mount Path | `/workspace` |
| Expose HTTP Ports | `8000` |
| Expose TCP Ports | (vazio) |

Environment variables:

| Nome | Valor |
|---|---|
| `COUNCIL_API_KEY` | uma string longa e aleatória (`openssl rand -base64 32`) |
| `COUNCIL_DATA_DIR` | `/workspace/data` |
| `COUNCIL_DEVICE` | `cuda` |

Não altere o start command: a imagem já sobe o uvicorn na porta 8000.

## 2. Esperar o modelo baixar

Na primeira subida o pod baixa ~2,4 GB do Hugging Face para o volume. Acompanhe
nos logs do pod até aparecer `model ready`. Nas próximas subidas o modelo já
está no volume e o boot leva ~15 s.

A URL pública fica em **Connect → HTTP Service [Port 8000]**, no formato
`https://<POD_ID>-8000.proxy.runpod.net`.

```bash
export VOICE_URL=https://<POD_ID>-8000.proxy.runpod.net
export COUNCIL_API_KEY=...
curl $VOICE_URL/health
```

## 3. Registrar as 7 vozes

Do Mac, na pasta do projeto (as referências ficam em `voices_src/`):

```bash
python scripts/register_voices.py voices_src --url $VOICE_URL --key $COUNCIL_API_KEY
```

Cada voz leva alguns segundos na GPU (limpeza com DeepFilterNet + codificação).
Elas ficam no volume, então sobrevivem a stop/start do pod.

## 4. Testar

```bash
curl -X POST $VOICE_URL/speak \
  -H "Authorization: Bearer $COUNCIL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"voice":"warren","text":"Good morning, council."}' \
  -D - --output test.mp3 | grep -i x-gen-time
```

`X-Gen-Time` é o tempo de geração em segundos; na GPU deve ficar abaixo de 2 s
a partir da segunda chamada.

## Custo

O pod cobra por segundo enquanto está **rodando**. Parado (Stop), paga só o
volume (centavos por dia). Para pausar fora do horário de uso, pare o pod e
dê start quando precisar; as vozes e o modelo continuam no volume.

## Atualizar a API

Push na `main` → Actions constrói a imagem → no RunPod, **Stop** e **Start**
do pod (ou "Restart") para puxar o `:latest`.
