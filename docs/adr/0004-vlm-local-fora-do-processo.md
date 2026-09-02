---
status: accepted
---

# O VLM local padrão roda fora do processo, servido pelo llama-server

O ticket #21 pedia a "outra metade" da decisão do Scriptwriter local: a pesquisa R1
(`docs/research/vlm-locais.md`, 2026-09-02) mostrou que o pin `transformers<5`
(imposto pelo coqui-tts) decide quais modelos cabem *dentro* do processo Python —
Qwen3-VL e InternVL3.5 sim; **Gemma 4 não** (só existe em transformers 5.5) — e que
no Windows o caminho mais robusto para *qualquer* GGUF é servi-lo fora do processo,
por `llama-server` ou Ollama, com JSON constrangido por gramática. Decisão de
2026-09-02:

1. **O runtime local padrão é o `llama-server` do llama.cpp, em processo separado**,
   atrás da mesma ABC `VisionLanguageEngine` dos outros Scriptwriters (provedor
   `llamacpp`, `mangawhisperer/engines/vlm_llamacpp.py`). O Python fala com ele pelo
   protocolo OpenAI (`/v1/chat/completions`, imagem como data URL), o mesmo que os
   provedores de API já usam — o prompt, o parser tolerante e o passthrough são os
   mesmos.
2. **O motor é dono do ciclo de vida do servidor.** Ele *não* sobe nada ao ser
   construído: inicia o `llama-server` no primeiro painel (`-m` GGUF + `--mmproj`),
   espera `/health` responder 200 com um timeout limitado (erro em PT-BR com o log e
   o comando exato), e o **encerra em `release()`** — que o orquestrador chama antes
   de carregar o TTS. Assim a regra dos 8 GB (`HardwareProfile.heavy_models_coexist`)
   continua valendo com um modelo que vive em outro processo; um painel posterior
   reinicia o servidor. Um servidor que morre no meio da execução é reiniciado; uma
   requisição que falha degrada *aquele* painel para passthrough, nunca aborta o
   volume.
3. **Modo anexado** (`LLAMA_SERVER_URL`): o motor usa um servidor que já está de pé —
   um `llama-server` iniciado à mão, ou o endpoint OpenAI do Ollama (`/v1`) — e
   **nunca** inicia nem encerra nada; `release()` só avisa que a VRAM continua com o
   servidor externo. `--vlm-model` vira o rótulo enviado no campo `model` (para o
   Ollama, a tag: `qwen3-vl:8b`).
4. **JSON por gramática, com queda suave.** A resposta é pedida com
   `response_format` do tipo `json_schema` (forma aninhada da OpenAI — é a que o
   parser do llama-server lê; o exemplo "plano" do README dele não é), com o schema
   do bloco (`text`, `speaker_id`, `is_speech`, `sfx` só como enum fechado das tags
   configuradas). Se o servidor recusar (HTTP 400), o motor repete sem o
   `response_format`; **só quando essa repetição funciona** ele conclui que o schema
   era o problema e segue só com a instrução de JSON do prompt pelo resto da
   execução. Um 400 por outro motivo — prompt maior que o contexto, imagem enviada a
   um servidor sem mmproj, imagem indecodificável — falha também na repetição:
   degrada só aquele painel e mantém o schema para os seguintes.
5. **O motor in-process (`qwen-local`, transformers + bitsandbytes) fica como
   alternativa**, para os modelos que existem em 4.57 e para quem já tem os pesos
   HF; não é mais o caminho padrão para "local". Com `prefer_local`, o `auto`
   resolve para `llamacpp` quando há um GGUF (`LLAMA_MODEL_GGUF` ou
   `--vlm-model x.gguf`) ou um `LLAMA_SERVER_URL`; senão, `qwen-local`.
6. **O Reviewer também roda no mesmo servidor** (texto puro, `create_reviewer
   ("llamacpp")`): mesma URL, mesmo modelo, `chunk_size=8` e `max_tokens=2048` para
   caber no contexto de 4k. Ele é conservador por construção (mantém o original em
   qualquer falha) e continua desligável com `--no-review`.

## Considered Options

- **Só in-process (transformers + NF4)**: um processo, sem binário externo, mas
  prende a escolha do modelo ao pin `<5` (sem Gemma 4), depende do bitsandbytes no
  Windows e não tem decodificação constrangida sem Outlines/Triton (Linux). Fica como
  alternativa, não como padrão.
- **Ollama como runtime padrão**: instalação mais simples e a mesma API OpenAI, mas
  o endpoint `/v1` não tem `keep_alive` (descarregar o modelo exige `/api/generate`
  ou o CLI), os tamanhos do Gemma 4 na biblioteca são anômalos (E4B 9,6 GB > 12B
  7,6 GB) e o ciclo de vida do processo não é nosso. Coberto pelo modo anexado.
- **vLLM / SGLang**: Linux/WSL apenas; descartados para o alvo Windows 11.
- **Porta livre a cada início**: evita colisão com um servidor esquecido, mas o
  Reviewer (construído separadamente pela factory) não saberia onde está o servidor.
  Ficou a porta fixa 8080 (padrão do llama-server), com sondagem *antes* de iniciar:
  se algo já responde nela, o motor recusa iniciar e sugere `LLAMA_SERVER_URL`.
  `port=None` escolhe uma porta livre para uso programático.

## Consequences

- **Distribuição**: o caminho local pede um binário (zip da release do llama.cpp,
  winget ou scoop) mais o download de dois arquivos — o GGUF e o `mmproj-*.gguf` do
  mesmo repositório — em vez de `pip install` de pesos HF, e o pacote `openai`
  (extra `vlm-api`), pelo qual o motor e o Reviewer falam com o servidor; o
  `preflight()` verifica o SDK nos dois modos antes de subir qualquer coisa. O motor
  procura o mmproj ao lado do modelo quando `LLAMA_MMPROJ_GGUF` não está definido e
  avisa quando não acha (um servidor sem mmproj sobe só-texto e todo painel degrada
  para passthrough). Com vários `mmproj*.gguf` na pasta — duas famílias de modelo
  baixadas lado a lado, como no benchmark do ticket #17 — só vale o que traz o nome
  do modelo (o token inicial do stem, `qwen3vl`/`gemma`, desempatando pelo maior
  número de tokens em comum: `8B-Instruct` sobre `4B`); um mmproj sozinho é aceito
  mesmo com nome genérico (`mmproj-F16.gguf`, como o unsloth distribui o do Gemma).
  Pasta ambígua — nenhum candidato com o nome do modelo, ou dois empatados — é erro
  no `preflight()` pedindo `LLAMA_MMPROJ_GGUF`, nunca um palpite: o projetor de
  outro modelo derruba o servidor na inicialização ou, pior, carrega e gera lixo
  que o parser tolerante aceita.
- **O pin `transformers<5` deixa de limitar a escolha do VLM**: qualquer modelo com
  suporte no llama.cpp (Qwen3-VL desde b6887, Gemma 4 desde b8637) entra no
  benchmark do ticket #17 pelo mesmo harness, com VRAM e t/s medidos no mesmo
  caminho.
- **JSON válido vira responsabilidade do runtime**, não do modelo: a taxa de JSON
  válido *sem* schema passa a ser métrica secundária do benchmark (quanto o modelo
  "entende" o contrato), não pré-requisito.
- **Fingerprint** = `vlm-llamacpp:<stem do GGUF>:prompt=<sha1>:json=<schema|prompt>`:
  trocar o GGUF, o elenco, as tags de efeito ou o mecanismo de JSON invalida os
  checkpoints, como nos outros motores. Em modo anexado sem `--vlm-model` o rótulo
  é `llama-server` — o fingerprint não sabe qual modelo o servidor externo carregou.
- **Orçamento em 8 GB é apertado e ainda estimado** (R1 §4): pesos Q4_K_M + mmproj
  do Qwen3-VL-8B ≈ 5,4–6,2 GB; por isso os padrões são `-c 4096`, `-ngl 99`,
  `-np 1`, imagem reduzida a 1024 px de aresta longa (≈ 770 tokens visuais a
  32 px/token) e `max_tokens=2048`. Medir com `nvidia-smi`; `LLAMA_SERVER_ARGS`
  permite ajustar (`--image-max-tokens`, `-c`, `--reasoning off` para Gemma 4).
- **Processo órfão**: `release()`, o `weakref.finalize` e o `atexit` cobrem saídas
  normais e Ctrl+C; um crash duro do Python pode deixar o `llama-server` vivo com a
  VRAM ocupada — o próximo início detecta a porta ocupada e diz o que fazer. Um Job
  Object do Windows fecharia essa brecha; fica registrado como fog.
- A inicialização (carregar 5–6 GB do disco) custa dezenas de segundos por
  execução, pagos de novo após cada `release()`. É o preço da regra dos 8 GB; em GPUs
  que comportam VLM + TTS o orquestrador continua chamando `release()` — relaxar isso
  é um ticket à parte.

## Receita (Windows 11, RTX 5060 Laptop 8 GB)

Fatos verificados em 2026-09-02 contra o README do servidor, `docs/multimodal.md`,
a API de releases do GitHub e os repositórios no Hugging Face; o que não foi
verificado está marcado.

1. **Binário.** Baixar em https://github.com/ggml-org/llama.cpp/releases (tags
   `bNNNN`; a última em 2026-09-02 era b10757):
   `llama-b10757-bin-win-cuda-13.3-x64.zip` (≈150 MB) e, se não houver runtime CUDA
   no PATH, `cudart-llama-bin-win-cuda-13.3-x64.zip` (≈391 MB), extraídos na mesma
   pasta — `llama-server.exe` (e o novo `llama.exe`) ficam na raiz do zip. Preferir o
   build **cuda-13.3**: o build cuda-12.4 não traz kernels nativos para sm_120
   (Blackwell/RTX 50); *não verificado* se ele roda via JIT de PTX. Alternativas:
   `winget install llama.cpp` (id `ggml.llamacpp`, instala o build **Vulkan**, não
   CUDA) ou `scoop install versions/llama.cpp-cu124`.
2. **Modelo + projetor** (mesmo repositório, na mesma pasta):
   - Qwen3-VL-8B-Instruct — `Qwen/Qwen3-VL-8B-Instruct-GGUF`:
     `Qwen3VL-8B-Instruct-Q4_K_M.gguf` (5,03 GB) + `mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf`
     (752 MB; o F16 tem 1,16 GB). Requer llama.cpp ≥ b6887 (o card recomenda b6907+).
     `ggml-org/Qwen3-VL-8B-Instruct-GGUF` **não existe**.
   - Gemma 4 E4B-it — `unsloth/gemma-4-E4B-it-GGUF`: `gemma-4-E4B-it-Q4_K_M.gguf`
     (4,98 GB) + `mmproj-F16.gguf` (990 MB); ou `ggml-org/gemma-4-E4B-it-GGUF`
     (Q4_0 4,59 GB + `mmproj-gemma-4-E4B-it-Q8_0.gguf` 560 MB). Requer ≥ b8637.
     Desligar o raciocínio com `LLAMA_SERVER_ARGS=--reasoning off` (relato de
     usuários; `--reasoning-budget 0` reportadamente **não** funcionou).
   - InternVL3.5-8B: só GGUF comunitário (`bartowski/OpenGVLab_InternVL3_5-8B-GGUF`,
     5,03 GB + mmproj 0,68 GB) — *não verificado* se a visão funciona; a issue
     #15528 (saída distorcida em imagens com texto) é um alerta para mangá. O
     fallback oficial é `ggml-org/InternVL3-8B-Instruct-GGUF`.
3. **Variáveis** (PowerShell):
   ```powershell
   $env:LLAMA_SERVER_BIN  = "C:\llama\llama-server.exe"
   $env:LLAMA_MODEL_GGUF  = "C:\models\Qwen3VL-8B-Instruct-Q4_K_M.gguf"
   $env:LLAMA_MMPROJ_GGUF = "C:\models\mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"  # opcional se estiver ao lado
   python main_demo.py --vlm llamacpp --pages 2
   ```
   O motor executa, e registra no log em INFO, o equivalente a
   `llama-server -m <gguf> --mmproj <mmproj> --host 127.0.0.1 --port 8080 -ngl 99 -c 4096 -np 1 --jinja`
   (`--jinja` já é o padrão nas builds novas; `-hf <repo>` baixaria modelo e mmproj
   sozinho, mas o motor usa caminhos explícitos para nunca baixar pesos por conta
   própria). Saída do servidor: `%TEMP%\mangawhisperer\llama-server.log`.
4. **Servidor já em execução / Ollama**: `LLAMA_SERVER_URL=http://127.0.0.1:8080`
   (ou `http://localhost:11434/v1` com `--vlm-model qwen3-vl:8b`). O Ollama aceita
   `image_url` em base64 e `response_format` com schema no `/v1` (schema com raiz
   `array`: *não verificado* no Ollama), não tem `keep_alive` no `/v1` — descarregar
   o modelo antes do TTS é `ollama stop <modelo>` — e *não verificado*: sua
   documentação não lista `/health`, por isso o motor trata um 404 nessa rota como
   "servidor de pé".
