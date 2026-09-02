# Referência de narração humana (uso local)

Ferramenta de desenvolvimento — **não faz parte do pipeline**. Serve para estudar
como um narrador humano conduz um mangá em PT-BR (ritmo, pausas, vocabulário de
descrição) e transformar isso em exemplos para o roteirista.

`tools/transcribe_reference.py` recebe um **arquivo de áudio local**, transcreve
com faster-whisper (timestamps por segmento e por palavra) e gera, em
`assets/reference/<nome do áudio>/` (pasta ignorada pelo git):

| Arquivo | Conteúdo |
|---|---|
| `segments.json` | segmentos brutos (`start`, `end`, `text`, `words`) e metadados do modelo |
| `transcript.md` | uma linha `[HH:MM:SS] texto` por segmento, com um título quando o arco muda |
| `pacing.json` | palavras/min, pausas (média, mediana, p90, pausas longas >= 1,5 s), duração dos segmentos |
| `style_excerpts.md` | N janelas de ~90 s espalhadas pelo arquivo, ordenadas por "densidade de narração" |

## Aviso legal — leia antes

* A narração é **obra protegida do criador do vídeo**, e a transcrição e os
  excertos derivam dela. Tudo fica **local, privado e sem fins comerciais**: não
  redistribua o áudio, a transcrição ou os excertos, não os publique (nem em
  issues/PRs) e não treine com eles nada que vá ser distribuído.
* Baixar o áudio de um vídeo **pode violar os Termos de Serviço do YouTube**
  ("Permissions and Restrictions": é proibido baixar qualquer parte do serviço
  ou do conteúdo, salvo com autorização expressa do YouTube ou do detentor dos
  direitos). A decisão e a responsabilidade são suas — leia as cláusulas em
  <https://www.youtube.com/t/terms> antes.
* A ferramenta **não baixa nem envia nada**: só lê o arquivo local indicado.
  A pasta de saída está no `.gitignore` (`assets/reference/`).
* Se os excertos forem para o prompt de um roteirista **de API** (Qwen, Claude,
  OpenAI...), o texto do narrador sai da máquina para o provedor. Quando isso
  importar, prefira o VLM local (`--vlm llamacpp`) ou diretrizes parafraseadas
  (seção 5).

## 1. Obter o áudio (fora da ferramenta)

Com `yt-dlp`, que exige `ffmpeg`/`ffprobe` no PATH e, para o YouTube, um runtime
JS (Deno é o recomendado):

```powershell
.venv\Scripts\python -m pip install -U "yt-dlp[default]"
winget install DenoLand.Deno

# Só a descrição do vídeo (lista de capítulos/arcos), sem baixar mídia
yt-dlp --skip-download --write-description -o "assets/reference/%(title)s [%(id)s].%(ext)s" URL

# Só o áudio, e só um trecho de 10-15 min (aqui: 00:10:15 a 00:25:00)
yt-dlp -f bestaudio -x --audio-format m4a --download-sections "*00:10:15-00:25:00" -o "assets/reference/%(title)s [%(id)s].%(ext)s" URL
```

* `-x --audio-format m4a` mantém o AAC original sem recodificar; `--audio-format
  best` também serve (o faster-whisper decodifica m4a, opus, mp3 e wav via PyAV).
* `--download-sections "*INÍCIO-FIM"` corta pelo tempo (`*` = intervalo de
  tempo; sem `*` seria um regex sobre títulos de capítulo). Um recorte começa em
  00:00 no arquivo, mas os tempos da descrição continuam sendo os do vídeo
  inteiro — por isso existe `--arcs-offset` (seção 3).
* O arquivo `.description` pode ser passado **inteiro** em `--arcs`: só as linhas
  `HH:MM:SS - Título` (também `H:MM:SS`, `MM:SS`, travessão ou sem separador)
  contam; o resto é ignorado.

## 2. Instalar o faster-whisper

```powershell
.venv\Scripts\python -m pip install faster-whisper
# ou, pelo extra do projeto:
.venv\Scripts\python -m pip install -e ".[reference]"
```

Na primeira execução o faster-whisper baixa o modelo do Hugging Face para o
cache do hub (`large-v3-turbo` ≈ 1,6 GB; `distil-large-v3` ≈ 1,5 GB;
`medium` ≈ 1,5 GB; `small` ≈ 0,5 GB). Nenhum peso entra no repositório.

### GPU no Windows (RTX 5060 Laptop, 8 GB)

* Os wheels do `ctranslate2` carregam `cublas64_12.dll` pelo nome, via PATH (a
  partir do 4.6.3 o cuDNN deixou de ser obrigatório). A ferramenta importa o
  `torch` antes do `faster_whisper` quando o dispositivo é `cuda`: o torch do
  venv (2.11+cu128) traz cuBLAS 12 e cuDNN 9 em
  `.venv\Lib\site-packages\torch\lib`, carrega essas DLLs no processo e a pasta
  entra no PATH como reserva. Normalmente basta.
* Se ainda aparecer `Library cublas64_12.dll is not found` ou `Could not locate
  cudnn_ops64_9.dll`: coloque `...\torch\lib` no PATH do shell antes de rodar,
  ou instale `nvidia-cublas-cu12` e `nvidia-cudnn-cu12` (wheels win_amd64) e
  ponha as pastas `site-packages\nvidia\cublas\bin` e `...\cudnn\bin` no PATH,
  ou use o pacote de DLLs "libs" do whisper-standalone-win (Purfview).
* Blackwell (sm_120): o `ctranslate2` 4.6.2 desligou int8 nessas GPUs e o
  4.7.0 religou. Mantenha `ctranslate2>=4.7` (`pip install -U "ctranslate2>=4.7"`).
  Por segurança o padrão da ferramenta na GPU é `float16` (~2,5 GB de VRAM para
  o turbo com beam 5); `--compute-type int8_float16` reduz quase à metade, só
  com >= 4.7.
* Não rode enquanto o pipeline estiver com TTS ou VLM na GPU: 8 GB não
  comportam os dois.
* Sem GPU: `--device cpu` usa `int8`; `small`/`medium` são as escolhas práticas
  (o turbo na CPU fica várias vezes mais lento — não medido).

## 3. Rodar

```powershell
# teste rápido: só os primeiros 60 s
.venv\Scripts\python -m tools.transcribe_reference --audio "assets/reference/narracao.m4a" --limit-seconds 60

# arquivo inteiro, com a lista de arcos vinda da descrição
.venv\Scripts\python -m tools.transcribe_reference --audio "assets/reference/narracao.m4a" --arcs "assets/reference/narracao.description"

# só um arco (sai em assets/reference/narracao/a_era_de_ouro/)
.venv\Scripts\python -m tools.transcribe_reference --audio "assets/reference/narracao.m4a" --arcs "assets/reference/narracao.description" --arc "Era de Ouro"

# recorte baixado com --download-sections a partir de 00:10:15 do vídeo
.venv\Scripts\python -m tools.transcribe_reference --audio "assets/reference/recorte.m4a" --arcs "assets/reference/narracao.description" --arcs-offset 00:10:15
```

| Opção | Padrão | Efeito |
|---|---|---|
| `--audio` | — | arquivo de áudio local (obrigatório) |
| `--out-dir` | `assets/reference/<nome>/` | pasta de saída; com `--arc`, uma subpasta com o nome do arco |
| `--model` | `large-v3-turbo` | alias do faster-whisper ou id de repo do HF (ex.: `deepdml/faster-whisper-large-v3-turbo-ct2`) |
| `--device` | `auto` | `cuda` se o torch enxergar a GPU, senão `cpu` |
| `--compute-type` | `auto` | `float16` na GPU, `int8` na CPU |
| `--language` | `pt` | idioma falado (ISO 639-1) |
| `--arcs` / `--arcs-offset` | — | lista de arcos e, para recortes, onde o áudio começa no vídeo |
| `--arc` | — | restringe tudo a um arco (trecho do título, sem diferenciar maiúsculas/acentos) |
| `--excerpts` / `--window` | 8 / 90 | quantos excertos de estilo e a duração aproximada de cada um |
| `--limit-seconds` | — | transcreve só os N primeiros segundos (do arco, se houver) |

A transcrição roda com VAD (`vad_filter`, silêncio mínimo de 500 ms) e beam 5;
`--arc`/`--limit-seconds` recortam a forma de onda decodificada, por isso o VAD
continua ativo no trecho. A saída é reescrita a cada execução.

Com `--arcs-offset`, o `--arc` pedido precisa estar (ao menos em parte) dentro
do recorte: um arco que termina antes do ponto em que o áudio começa (ex.:
`--arcs-offset 00:10:15 --arc Abertura` quando "Abertura" vai de 00:00 a 05:30)
é recusado com erro, em vez de transcrever o arquivo inteiro com o nome errado.
Um arco que começa antes do recorte e termina dentro dele é transcrito só na
parte presente (o trecho aparece como `00:00:00 a HH:MM:SS`). `--limit-seconds`
precisa ser maior que zero.

## 4. Ler os resultados

`pacing.json` é o que mais interessa para calibrar a entrega do TTS:

* `words_per_minute_speech` (só sobre o tempo falado) e `words_per_minute_total`
  (inclui as pausas) → comparar com o ritmo atual do XTTS e ajustar
  `tts_speed` do `NarrationStyle` (`mangawhisperer/engines/styles.py`).
* `pauses.median_s` / `p90_s` → referência para `gap_ms` entre blocos.
* `long_pauses` (>= 1,5 s, as 50 maiores com o instante) → onde o narrador
  segura o silêncio; ouça esses pontos para entender o que os antecede
  (normalmente uma descrição curta antes de um impacto).
* `segment_duration_s` e `words_per_segment_mean` → tamanho típico das frases.

`style_excerpts.md` traz janelas escolhidas por uma heurística simples e
documentada em `narration_score`: contagem de palavras/expressões de descrição
("enquanto", "de repente", "silêncio", "sombra"...), verbos com cara de narração
em 3ª pessoa (terminações -ou/-eu/-iu, -ava, -ando/-endo/-indo) e onomatopeias
com letra repetida ("booom"), dividida pelo número de palavras. É um filtro
grosseiro: revise à mão e fique com 3 a 5 trechos que representem o tom.

## 5. Próximo passo: alimentar o roteirista

O prompt do roteirista já tem o gancho: `build_scriptwriter_prompt(...,
style_addendum=...)` (`mangawhisperer/engines/vlm.py`) recebe o
`prompt_addendum` do `NarrationStyle` escolhido. O plano de continuação é:

1. Uma opção (ex.: `--style-examples assets/reference/narracao/style_excerpts.md`)
   ou um preset que leia o arquivo local e acrescente ao addendum um bloco
   "Exemplos de tom — imitar ritmo e foco descritivo, **não copiar frases**"
   com os excertos revisados.
2. Como o addendum entra no hash do prompt (fingerprint do motor), roteiros
   antigos são invalidados e refeitos — comportamento esperado.
3. O arquivo de excertos continua fora do git; só o código que o lê é versionado.
   Quem clonar o projeto sem o arquivo roda com o addendum vazio.

Alternativa mais conservadora: em vez dos trechos literais, derivar diretrizes
parafraseadas a partir de `pacing.json` e da leitura dos excertos (frases de N
palavras, pausa antes do impacto, vocabulário de atmosfera) e escrevê-las como um
novo `NarrationStyle` — nada do texto original sai da máquina.
