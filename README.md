# Council Voice API

API headless de clonagem de voz + TTS para o site do Council. Cada agent tem uma
voz clonada registrada uma vez; depois cada request manda `{voice, text}` e
recebe o áudio.

Motor: [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) (`k2-fsa/OmniVoice`,
Apache-2.0, ~2.4 GB, 600+ idiomas incl. pt-BR, saída 24 kHz mono). É o mesmo
modelo padrão do VoiceStudio; aqui só o pacote `omnivoice/` foi mantido, sem UI,
sem banco, sem Whisper, sem dubbing.

## Endpoints

| Método | Rota | O que faz |
|---|---|---|
| `GET` | `/health` | status, device, modelo carregado, nº de vozes (sem auth) |
| `POST` | `/voices` | registra uma voz (multipart: `id`, `ref_audio`, `ref_text`, `name?`, `language?`) |
| `GET` | `/voices` | lista as vozes |
| `DELETE` | `/voices/{id}` | remove uma voz |
| `POST` | `/speak` | JSON `{voice, text, language?, format?, speed?, num_step?, guidance_scale?, seed?}` → áudio |
| `POST` | `/v1/audio/speech` | alias compatível com OpenAI (`input`, `voice`, `response_format`) |

Auth: se `COUNCIL_API_KEY` estiver setada, toda rota (menos `/health`) exige
`Authorization: Bearer <key>` (ou `X-API-Key`).

Formatos: `mp3` (padrão), `wav`, `opus`, `flac`, `pcm` (s16le 24 kHz).
A resposta traz `X-Audio-Duration` e `X-Gen-Time` em segundos.

## Rodar na VPS (Docker)

```bash
cp .env.example .env            # edite COUNCIL_API_KEY
docker compose up -d --build    # GPU NVIDIA (precisa do nvidia-container-toolkit)
docker logs -f council-voice    # 1ª subida baixa ~2.4 GB do modelo
```

Sem GPU: apague o bloco `deploy:` do `docker-compose.yml` e construa a imagem
CPU, bem mais leve:

```bash
docker compose build --build-arg BASE_IMAGE=python:3.12-slim \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
docker compose up -d
```

A porta fica em `127.0.0.1:8000`. Coloque Caddy/Nginx com TLS na frente para
expor na internet; nunca publique a porta direto.

## Rodar local (dev, Mac/Linux)

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
COUNCIL_API_KEY= .venv/bin/uvicorn app.main:app --port 8000
```

## Registrar as 8 vozes

Para cada agent: um clipe limpo de **3 a 15 s** (wav/mp3/m4a/ogg) e a
**transcrição exata** do que é dito nele. A transcrição é obrigatória (não há
Whisper no serviço) e é o que mais afeta a qualidade do clone.

Estrutura:

```
voices_src/
  igor/
    ref.wav
    ref.txt        # "Olá, eu sou o Igor e esta é a minha voz."
    meta.json      # {"name": "Igor", "language": "pt"}  (opcional)
  ana/
    ...
```

```bash
python scripts/register_voices.py voices_src --url http://localhost:8000 --key "$COUNCIL_API_KEY"
```

Ou um por um:

```bash
curl -X POST http://localhost:8000/voices \
  -H "Authorization: Bearer $COUNCIL_API_KEY" \
  -F id=igor -F name=Igor -F language=pt \
  -F ref_audio=@igor.wav \
  -F ref_text="Olá, eu sou o Igor e esta é a minha voz."
```

O clipe é convertido para 24 kHz mono, **limpo com DeepFilterNet** (remove ruído
e reverberação da sala, que o clone copiaria junto com a voz), codificado uma única vez em um
`VoiceClonePrompt` (`data/prompts/<id>.pt`) e reutilizado em todas as
gerações. Registrar de novo com o mesmo `id` substitui a voz.

## Gerar fala

```bash
curl -X POST http://localhost:8000/speak \
  -H "Authorization: Bearer $COUNCIL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"voice":"igor","text":"Bom dia, conselho. Vamos começar a reunião.","format":"mp3"}' \
  --output fala.mp3
```

Do site (Node):

```js
const res = await fetch("https://voice.seudominio.com/speak", {
  method: "POST",
  headers: { Authorization: `Bearer ${KEY}`, "Content-Type": "application/json" },
  body: JSON.stringify({ voice: "igor", text }),
});
const audio = Buffer.from(await res.arrayBuffer()); // mp3
```

Parâmetros úteis em `/speak`:

- `language`: `pt` (padrão por voz), `en`, `es`, ... ou `auto`.
- `num_step`: 16 (rápido, padrão) a 32 (melhor qualidade).
- `speed`: 0.5 a 2.0.
- `seed`: fixa o resultado para o mesmo texto.

Texto até 4096 caracteres; o modelo divide internamente trechos longos.

## Variáveis de ambiente

| Var | Padrão | |
|---|---|---|
| `COUNCIL_API_KEY` | vazio | Bearer exigido em todas as rotas (vazio = sem auth) |
| `COUNCIL_DATA_DIR` | `./data` (`/data` no Docker) | vozes, prompts e cache do modelo |
| `COUNCIL_DEVICE` | `auto` | `cuda` / `mps` / `cpu` |
| `COUNCIL_PRELOAD` | `1` | carrega o modelo no boot |
| `COUNCIL_DEFAULT_LANGUAGE` | `pt` | idioma padrão ao registrar vozes |
| `COUNCIL_DEFAULT_FORMAT` | `mp3` | formato padrão de `/speak` |
| `COUNCIL_NUM_STEP` | `16` | passos de difusão padrão |
| `COUNCIL_CLEAN_REFS` | `1` | limpa referências com DeepFilterNet ao registrar |
| `COUNCIL_DFN_PYTHON` | vazio | python de outro venv com `deepfilternet` (só no Mac dev; ver abaixo) |
| `COUNCIL_IDLE_UNLOAD_SECONDS` | `0` | descarrega o modelo após N s ocioso (0 = nunca) |
| `HF_TOKEN` | vazio | só se a rede exigir |

## Hardware

- GPU NVIDIA com **6 GB+ de VRAM**: ~1 s por frase curta depois do warm-up.
- CPU: funciona, mas 10-50x mais lento (dezenas de segundos por frase). Para
  uso em site com resposta ao vivo, prefira GPU ou gere o áudio
  antecipadamente e cacheie.
- Apple Silicon (dev): use `COUNCIL_DEVICE=cpu`. Com torch 2.14 o load em MPS
  dá *bus error*; se quiser MPS, pine `torch==2.8.0 torchaudio==2.8.0`.
  Medido no M4 em CPU: ~13 s para 5 s de fala (num_step=16).
  O DeepFilterNet exige torch 2.8, que conflita com o torch 2.14 do venv
  principal no Mac; instale-o num venv à parte e aponte `COUNCIL_DFN_PYTHON`:

  ```bash
  uv venv --python 3.11 .venv-tools
  uv pip install --python .venv-tools/bin/python torch==2.8.0 torchaudio==2.8.0 deepfilternet soundfile
  COUNCIL_DFN_PYTHON=$PWD/.venv-tools/bin/python .venv/bin/uvicorn app.main:app --port 8000
  ```
  No Docker nada disso é necessário: a imagem já traz torch 2.8 e o deepfilternet.

Uma geração por vez (lock interno); requests concorrentes enfileiram.

## Licença

O código do modelo em `omnivoice/` vem do VoiceStudio/OmniVoice (Apache-2.0).
O restante deste serviço é seu.
