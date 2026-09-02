---
ticket: R3
status: concluído
data: 2026-09-02
ambiente-alvo: Windows 11 · Python 3.11 · torch 2.11.0+cu128 · transformers 4.57.6 · coqui-tts 0.27.5 · RTX 5060 Laptop 8 GB (Blackwell sm_120)
---

# TTS locais expressivos para PT-BR em GPU de 8 GB (R3)

## English abstract

We compared five local TTS candidates for Brazilian Portuguese against MangaWhisperer's
pinned Windows environment (torch 2.11+cu128, transformers 4.57.6, coqui-tts 0.27.5,
numpy 2.4.6, 8 GB VRAM): XTTS v2 (baseline), Chatterbox Multilingual V3 (+ official pt-BR
language pack), Kokoro-82M, F5-TTS with community pt-BR finetunes, and OpenAudio S1-mini.
Only Kokoro and F5-TTS install cleanly on the current pins; Chatterbox hard-pins
`torch==2.6.0` (no Blackwell support) and `transformers==5.2.0` (0.1.7) / `4.46.3` (0.1.6)
plus `numpy<2`, so it needs `--no-deps` or its own venv; S1-mini has no pinned code release,
pins `numpy<=1.26.4`, and its docs now target S2 (24 GB, Linux/WSL). Measured VRAM: XTTS
~2.1 GB, Chatterbox ~5 GB (community), Kokoro <1 GB weights; F5-TTS and S1-mini have no
official number. Recommendation for the A/B listening test: **XTTS v2 (baseline) vs
Chatterbox V3 pt-BR pack vs Kokoro-82M pt-BR**, with F5-TTS-pt-br as an optional fourth
arm; exclude S1-mini for now. Only Kokoro (and marginally XTTS/F5) can share the 8 GB GPU
with a 3B VLM in 4-bit; nothing coexists with a 7B VLM, so the existing
load→use→`release()` sequencing stays.

---

## Pergunta

Qual TTS local expressivo para português brasileiro cabe numa GPU Windows de 8 GB e
convive com o ambiente pinado do projeto (torch 2.11+cu128, transformers 4.57.x,
coqui-tts 0.27.5)? Para Chatterbox Multilingual, Kokoro-82M, F5-TTS (+ finetunes pt-BR),
OpenAudio S1-mini e XTTS v2: qualidade documentada em PT-BR, VRAM, fator de tempo real
(RTF) em GPU de consumo, controles de expressividade, clonagem zero-shot, viabilidade de
`pip install` no Windows com ESTES pins, e se divide a GPU com um VLM de 3B–7B. Quais
entram num teste cego A/B?

## Resumo executivo

| Motor | Licença (pesos) | pt-BR documentado | VRAM (fonte) | RTF | Expressividade | Clonagem | Instala nos pins atuais? |
|---|---|---|---|---|---|---|---|
| **XTTS v2** (coqui-tts 0.27.5) | CPML (não-comercial; aceito pelo ADR-0002) | `pt` genérico; 2.386,8 h de português no treino (paper), variante não declarada | **2,1 GB** (issue coqui-ai/TTS#3976) | streaming <200 ms (docs); ~0,3 (secundário) | temperature, repetition_penalty, speed, top_k/p; sem tags de emoção | sim, 6 s | **Sim** — já instalado e integrado |
| **Chatterbox Multilingual V3** + pack pt-BR | MIT | **Sim**: pack dedicado `ResembleAI/Chatterbox-Multilingual-pt-br` (locale pt-BR); CER pt na faixa 1–5 % (blog V3) | ~**5 GB** observado em GTX 1060 (issue #508); sem número oficial | H100: ~5× tempo real (RTF ≈ 0,2); GTX 1060: ~15 s/frase | `exaggeration`, `cfg_weight`, `temperature`, `repetition_penalty`, `min_p`, `top_p` | sim, ~10 s | **Não** — pina `torch==2.6.0`, `transformers==5.2.0` (0.1.7) ou `==4.46.3` (0.1.6), `numpy<2`; só com `--no-deps` ou venv própria |
| **Kokoro-82M** | Apache-2.0 | `p` = pt-BR com 3 vozes (pf_dora, pm_alex, pm_santa) **sem nota de qualidade**; G2P via espeak-ng | <1 GB de pesos (82M); sem medição oficial | 36× (T4) a 96× (A10G) tempo real (gist) | só `speed` e mistura de vozes; sem emoção, sem tags | **não** | **Sim** — deps sem pin (torch, transformers) |
| **F5-TTS** + `firstpixel/F5-TTS-pt-br` | código MIT; pesos CC-BY-NC-4.0 | finetune comunitário (~447 h, 3.500+ falantes, Common Voice); relatos de "gibberish" na config (disc. #774) | sem número oficial; checkpoint 1,35 GB (335,8M params) | 0,15 @ 32 NFE em RTX 3090 (paper) | `speed`, `nfe_step`, `cfg_strength`, `sway_sampling_coef`; expressividade só via referência | sim, 5–12 s | **Sim** — `torch>=2.0`, transformers sem pin; precisa FFmpeg/torchcodec no Windows |
| **OpenAudio S1-mini** (fish-speech) | CC-BY-NC-SA-4.0 | lista "Portuguese" entre 13 idiomas; sem amostras/eval pt | S1-era docs: 12 GB; S2 atual: 24 GB; sem número para S1-mini isolado | ~1:7 em RTX 4090 com `torch.compile` (README S1) | 30+ marcadores `(angry)`, `(whispering)`, `(laughing)`… | sim | **Não** — sem release pinada para S1-mini; `numpy<=1.26.4`, `pydantic==2.9.2`; docs atuais só Linux/WSL, `compile` não roda no Windows |

**Recomendação:** teste cego A/B com **XTTS v2 (baseline) × Chatterbox V3 pt-BR × Kokoro-82M pt-BR**, cada um em venv própria atrás da factory de TTS; F5-TTS-pt-br como quarto braço opcional; S1-mini fora por ora. Detalhes na seção "Recomendação".

---

## Achados

### 0. Ambiente verificado (o que os pins realmente são)

Lido em `.venv/Lib/site-packages` em 2026-09-02:

- `torch 2.11.0+cu128`, `torchaudio 2.11.0+cu128`, `torchvision 0.26.0+cu128`, `torchcodec 0.15.0`
- `transformers 4.57.6`, `coqui_tts 0.27.5`, `coqui_tts_trainer 0.3.3`
- `numpy 2.4.6`, `gradio 6.17.3`, `accelerate 1.14.0`, `bitsandbytes 0.50.0`, `soundfile 0.14.0`, `easyocr 1.7.2`, `edge_tts 7.2.8`

Fatos de plataforma que afetam todos os candidatos:

- **Blackwell exige torch ≥ 2.7 com cu128.** O PyTorch 2.7 introduziu o suporte à arquitetura NVIDIA Blackwell e wheels CUDA 12.8 ([blog PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/)). Qualquer pacote que pine `torch==2.6.0` não consegue usar a RTX 5060.
- **torchcodec 0.14–0.16 ↔ torch ≥ 2.11** (tabela do README do torchcodec). Suporta FFmpeg 4–9, mas no Windows precisa do build *shared* do FFmpeg (DLLs separadas) ([torchcodec README](https://github.com/meta-pytorch/torchcodec)). Há um issue aberto exatamente com XTTS + RTX 5060 + Windows 11 falhando em "Could not load libtorchcodec" (torch 2.9.1+cu128, torchcodec 0.9.1, FFmpeg 7.1.1) ([torchcodec#1147](https://github.com/meta-pytorch/torchcodec/issues/1147)). O projeto já contorna decodificando via `soundfile` (memória do projeto, `SFXLibrary`).
- **coqui-tts 0.27.5 declara `transformers>=4.57` sem teto**, mas quebra com transformers ≥ 5.1 por importar `isin_mps_friendly` (removido) — issue aberto, PR #560 não mesclado, workaround = 4.57.6 ([idiap/coqui-ai-TTS#558](https://github.com/idiap/coqui-ai-TTS/issues/558)). É o motivo do pin `<5` no `pyproject.toml`.

### 1. XTTS v2 — baseline (coqui-tts 0.27.5)

**Pacote.** `coqui-tts 0.27.5` (26/01/2026), Python `>=3.10,<3.15`; desde 0.27.4 o torch não vem por padrão (`[cpu]`/`[cuda]` → `torch>=2.2`, `torchaudio>=2.2`; `[codec]` → `torchcodec>=0.8`); `transformers>=4.57`, `numpy>=1.26` sem teto. Código MPL-2.0 ([PyPI](https://pypi.org/project/coqui-tts/), [pyproject](https://raw.githubusercontent.com/idiap/coqui-ai-TTS/main/pyproject.toml)).

**Modelo.** 17 idiomas incluindo `pt`; clonagem com clipe de 6 s; "emotion and style transfer by cloning"; clonagem cross-lingual; múltiplas referências e interpolação de falantes; 24 kHz. Licença **CPML** (Coqui Public Model License, não-comercial) ([model card](https://huggingface.co/coqui/XTTS-v2)). O ADR-0002 do repositório aceita explicitamente essa licença.

**Tamanho e dados (paper XTTS, Interspeech 2024).** Encoder GPT-2 443M + VQ-VAE 13M + decoder 26M ≈ **482M parâmetros**; 27.281,6 h de áudio no total, das quais **2.386,8 h de português**; para idiomas fora do núcleo "a maior parte dos dados vem do Common Voice". O paper **não** diz se o português é pt-BR ou pt-PT ([arXiv 2406.04904](https://arxiv.org/html/2406.04904)). Na prática, o Common Voice pt é majoritariamente brasileiro, mas isso é inferência, não fato documentado.

**Controles (docs oficiais).** `temperature` (0,65), `length_penalty`, `repetition_penalty` (2,0), `top_k` (50), `top_p` (0,8), `speed` (1,0; extremos geram artefatos), `enable_text_splitting`; vozes de estúdio embutidas (`speaker="Ana Florence"`); streaming com latência <200 ms; `use_deepspeed=True` opcional ([docs XTTS](https://coqui-tts.readthedocs.io/en/latest/models/xtts.html)). Não há tags de emoção; a receita expressiva do projeto (`temperature 0.75`, `repetition_penalty 4.0`, `speed`) já está integrada no `XTTSEngine`.

**VRAM/RAM.** Relato com números no issue oficial: **2,1 GB de VRAM** e ~4,6 GB de RAM de sistema durante inferência (pico ~5 GB de RAM ao inicializar o bloco GPT); fechado como *wontfix* ([coqui-ai/TTS#3976](https://github.com/coqui-ai/TTS/issues/3976)). RTF ≈ 0,3 em GPU de consumo aparece só em fontes secundárias.

**Compatibilidade.** É o único motor já instalado e validado nos pins; os riscos conhecidos (transformers 5, torchcodec no Windows) estão documentados e contornados.

### 2. Chatterbox Multilingual V3 (Resemble AI) + pack pt-BR

**Modelo.** Backbone Llama de **0,5B**; 23 idiomas com `pt` no dicionário `SUPPORTED_LANGUAGES` de `mtl_tts.py`; treino em 0,5M h de dados limpos; watermark Perth em todo áudio gerado; MIT ([README](https://github.com/resemble-ai/chatterbox), [HF card](https://huggingface.co/ResembleAI/chatterbox)). V3 (blog de 10/06/2026): dados de treino de 25,6k → **36,7k h**, foco em fala expressiva/conversacional; 25 idiomas com 4 dialetos e 6 *Language Packs*, incluindo **pt-BR e pt-PT**; CER por faixa: variantes de português na faixa "production-ready 1–5 %"; TTFB <300 ms e ~5× tempo real em **H100** ([blog V3](https://www.resemble.ai/resources/chatterbox-multilingual-v3-tts-with-embedded-watermarking-for-25-languages)).

**Pack pt-BR.** `ResembleAI/Chatterbox-Multilingual-pt-br`: finetune de `ResembleAI/chatterbox`, MIT, language ID `pt`, locale `pt-BR`; arquivos `t3_pt_br.safetensors` (2,14 GB), `s3gen_v3.pt/.safetensors` (decoder V3, 1,06 GB), `grapheme_mtl_merged_expanded_v1.json`; "use quando quiser controle de qualidade pt-BR mais apertado que o checkpoint multilíngue amplo" ([model card](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-pt-br)). A página comercial fala em "clone any voice from 10 seconds of reference audio" e "stronger dialect behavior and prosody control for Brazilian Portuguese" ([Resemble](https://www.resemble.ai/learn/models/chatterbox-multilingual)). Há um issue aberto (12/09/2025) relatando que o demo HF saía pt-BR e o pacote pip saía pt-PT — ou seja, no checkpoint genérico o sotaque não é garantido; o pack existe justamente por isso ([#281](https://github.com/resemble-ai/chatterbox/issues/281)).

**Controles (`mtl_tts.py`, master).** `generate(text, language_id, audio_prompt_path=None, exaggeration=0.5, cfg_weight=0.5, temperature=0.8, repetition_penalty=1.2, min_p=0.05, top_p=1.0)`; sem `audio_prompt_path` usa `conds.pt` (voz embutida) ([fonte](https://raw.githubusercontent.com/resemble-ai/chatterbox/master/src/chatterbox/mtl_tts.py)). É o único candidato com controle explícito de intensidade emocional (`exaggeration`) além de referência.

**Tamanho em disco.** `t3_mtl23ls_v2/v3.safetensors` 2,14 GB cada, `s3gen*.pt` 1,06 GB, `ve.pt` 5,7 MB; repositório total ~13,9 GB (baixar seletivamente) ([tree](https://huggingface.co/ResembleAI/chatterbox/tree/main)).

**VRAM/velocidade em GPU de consumo.** Não há número oficial. Relato com números: **GTX 1060 6 GB, Windows 10: ~15 s por frase, ~5 GB de VRAM, até 7 GB de RAM**, com travamentos esporádicos de 200+ s ([#508](https://github.com/resemble-ai/chatterbox/issues/508)). Com 8 GB, cabe sozinho; não cabe ao lado de VLM.

**Pins — o problema central.**

| versão | data | torch | transformers | numpy | outros pins exatos |
|---|---|---|---|---|---|
| 0.1.4 | 04/09/2025 | ==2.6.0 | ==4.46.3 | >=1.24,<1.26 | `pkuseg==0.0.25` (sem wheel; quebra builds) |
| 0.1.6 | 15/12/2025 | ==2.6.0 | ==4.46.3 | >=1.24,<1.26 | `gradio==5.44.1`, `safetensors==0.5.3`, `diffusers==0.29.0`, `spacy-pkuseg` |
| 0.1.7 | 26/03/2026 | ==2.6.0 (py<3.14) | **==5.2.0** | <2 (py<3.13) | `gradio==6.8.0`, `safetensors==0.5.3` |
| master | jul/2026 | idem 0.1.7 | ==5.2.0 | idem | `resemble-perth @ git+…` (exige git) |

Fontes: PyPI JSON [0.1.4](https://pypi.org/pypi/chatterbox-tts/0.1.4/json), [0.1.6](https://pypi.org/pypi/chatterbox-tts/0.1.6/json), [0.1.7](https://pypi.org/pypi/chatterbox-tts/0.1.7/json); [pyproject master](https://raw.githubusercontent.com/resemble-ai/chatterbox/master/pyproject.toml).

Consequências para o ambiente do projeto:

- `torch==2.6.0` não tem Blackwell → o pin tem de ser ignorado de qualquer forma.
- `transformers==5.2.0` (0.1.7/master) colide com o `4.57.6` exigido pelo coqui-tts; `4.46.3` (0.1.6) também colide. O commit que subiu para 5.2.0 (`eaf9354`, 18/03/2026) mudou só três coisas no código: força `_attn_implementation` de `sdpa` para `eager` no `alignment_stream_analyzer.py`, troca `HF_TOKEN or True` por `or None` no Turbo, e sobe gradio ([commit](https://github.com/resemble-ai/chatterbox/commit/eaf9354)). Isso sugere que o código não depende de APIs exclusivas do transformers 5 — mas **rodar master com 4.57.6 não está verificado**. O issue [#339](https://github.com/resemble-ai/chatterbox/issues/339) (SDPA não aceita `output_attentions=True` com referência de áudio em transformers ≥ 4.36; workaround `eager`) é o mesmo mecanismo que o commit acima endereça.
- `numpy<2` colide com `numpy 2.4.6` (usado por EasyOCR/OpenCV 5/pymupdf). `gradio==6.8.0` colide com `6.17.3`.
- **Suporte a V3 e ao pack pt-BR só existe no git master**: commits "Add opt-in v3 multilingual checkpoint" (01/05/2026) e "v3 multilingual and single language pack release" (10/06/2026) são posteriores ao 0.1.7 do PyPI ([commits](https://github.com/resemble-ai/chatterbox/commits/master)).
- Receita que funcionou no Windows com CUDA: instalar torch cu-específico primeiro, depois `pip install --no-deps .` e completar dependências à mão ([#159](https://github.com/resemble-ai/chatterbox/issues/159)). Histórico de falhas de build com `pkuseg==0.0.25` nas versões antigas ([#367](https://github.com/resemble-ai/chatterbox/issues/367)); resolvido com `spacy-pkuseg` a partir da 0.1.6.

**Veredito de instalação:** viável só em **venv separada** (Python 3.11, torch 2.11+cu128, transformers 4.57.6 ou 5.2.0 conforme teste, `pip install --no-deps git+https://github.com/resemble-ai/chatterbox` + deps manuais) exposta ao pipeline por subprocesso/HTTP atrás da ABC de TTS. Instalar na venv principal quebraria coqui-tts ou numpy.

### 3. Kokoro-82M (hexgrad)

**Modelo.** 82M parâmetros, Apache-2.0, v1.0 (27/01/2025): 8 idiomas e 54 vozes; treinado "exclusivamente em áudio permissivo/não-copyright", "algumas centenas de horas", ~1.000 GPU-h A100; exclui clones de voz sintéticos ([model card](https://huggingface.co/hexgrad/Kokoro-82M)). **Não faz clonagem** — vozes fixas em `voices/*.pt`, com mistura por soma ponderada de tensores.

**pt-BR.** Código `p` → `pt-br`; vozes **pf_dora (F), pm_alex (M), pm_santa (M)**. Na tabela `VOICES.md`, as vozes em inglês têm *Target Quality*, *Training Duration* (HH = 10–100 h, H = 1–10 h, MM = 10–100 min, M = 1–10 min) e *Overall Grade*; as vozes pt-BR aparecem **só com nome, traço e SHA — sem nota**. O arquivo avisa que as línguas não-inglesas usam fallback espeak-ng e podem ter representação "ausente ou rala" por G2P fraco e dados limitados ([VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)). Não localizei avaliação formal de pt-BR; relatos de uso são anedóticos.

**G2P/Windows.** `pipeline.py`: para `p`, `self.g2p = espeak.EspeakG2P(language='pt-br')`; escolhe `cuda` automaticamente se disponível ([pipeline.py](https://raw.githubusercontent.com/hexgrad/kokoro/main/kokoro/pipeline.py)). O extra `misaki[en]` (dependência obrigatória do `kokoro`) já traz `phonemizer-fork` + `espeakng-loader`, que empacotam os binários do espeak-ng ([misaki pyproject](https://raw.githubusercontent.com/hexgrad/misaki/main/pyproject.toml)); o README do kokoro ainda documenta o instalador `.msi` do espeak-ng no Windows como alternativa ([kokoro README](https://github.com/hexgrad/kokoro)).

**Pins.** `kokoro 0.9.4`: `python>=3.10,<3.14`; deps `torch`, `transformers`, `numpy`, `huggingface_hub`, `loguru` **sem pin**, `misaki[en]>=0.9.4` ([pyproject](https://raw.githubusercontent.com/hexgrad/kokoro/main/pyproject.toml)). Compatível com torch 2.11 / transformers 4.57.6 / numpy 2.4.6 sem conflito. Alternativa `kokoro-onnx` (MIT): modelo ~300 MB, variantes fp16 e int8 (~80 MB), `onnxruntime-gpu` opcional, Python 3.10–3.13 ([kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)).

**Velocidade.** Benchmark de terceiros com Kokoro-82M-v1.0: PyTorch CUDA **36× tempo real em T4, 96× em A10G, 81× em L4**; CPU 32 vCPU 5× ([gist](https://gist.github.com/efemaer/23d9a3b949b751dde315192b4dcf0653)). VRAM não medida nessa fonte; com 82M parâmetros os pesos ficam bem abaixo de 1 GB — é o único candidato que cabe folgado ao lado de um VLM.

**Controles.** `KPipeline(lang_code='p')(text, voice='pf_dora', speed=1, split_pattern=r'\n+')`; só velocidade e mistura de vozes; sem emoção, sem referência.

### 4. F5-TTS + finetunes pt-BR

**Modelo/código.** Código MIT; pesos oficiais **CC-BY-NC-4.0** (dataset Emilia). F5-TTS Base **335,8M** parâmetros; RTF **0,15 com 32 NFE em RTX 3090** (paper); CFG 2, sway sampling −1; ~95k h zh/en ([arXiv 2410.06885](https://arxiv.org/html/2410.06885)). Checkpoints oficiais só zh/en; `F5TTS_v1_Base/model_1250000.safetensors` = 1,35 GB ([HF](https://huggingface.co/SWivid/F5-TTS/tree/main/F5TTS_v1_Base)). Com Triton/TensorRT-LLM em L20: RTF 0,0394; PyTorch puro 0,1467 ([README](https://github.com/SWivid/F5-TTS)).

**Pins.** `f5-tts 1.1.22`: `torch>=2.0`, `torchaudio>=2.0`, `transformers` sem pin, `numpy<=1.26.4` só para py≤3.10, `accelerate>=0.33`, `gradio>=6.15`, `bitsandbytes>0.37` ([pyproject](https://raw.githubusercontent.com/SWivid/F5-TTS/main/pyproject.toml)). README recomenda `torch==2.8.0+cu128` mas não pina → **instala nos pins atuais**. Requer FFmpeg (`conda install ffmpeg`). Ponto de atenção: com torchaudio ≥ 2.9 a carga de áudio passa por torchcodec, e há issue aberto ("help wanted") de Windows 11 falhando em `Could not load libtorchcodec` (torch 2.9.1, FFmpeg 8.0); a solução sugerida pela comunidade é patch para `soundfile` ([F5-TTS#1234](https://github.com/SWivid/F5-TTS/issues/1234)). Mesmo contorno que o projeto já usa.

**Controles.** `--speed`, `--nfe_step` (32), `--cfg_strength` (2), `--sway_sampling_coef` (−1), `--cross_fade_duration`, `--remove_silence`; referência <~12 s a 24 kHz com 1 s de silêncio no fim; `--ref_text` vazio carrega um ASR e consome VRAM extra; limite de 30 s por chunk (ref + gerado); MAIÚSCULAS são soletradas ([infer/README](https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/infer/README.md)). Expressividade só por referência — sem tags nem `exaggeration`.

**Finetunes pt-BR.**

- `firstpixel/F5-TTS-pt-br` (última atualização fev/2025): ~**447 h**, 90.947 amostras, **3.500+ falantes** (Mozilla Common Voice pt + datasets Facebook), vocab de 2.545 caracteres (= o vocab `Emilia_ZH_EN_pinyin` do F5TTS_Base, ou seja, arquitetura v0, não v1), `pt-br/model_last.safetensors` **1,35 GB** (também `.pt` de 5,39 GB), **CC-BY-NC-4.0**; entrada em minúsculas, números via `num2words`, referência de 5–9 s; marcadores `[speaker:name, emotion:type]` são orquestração do script de exemplo (troca de referência), não capacidade do modelo ([card](https://huggingface.co/firstpixel/F5-TTS-pt-br), [arquivos](https://huggingface.co/firstpixel/F5-TTS-pt-br/tree/main/pt-br)). Na discussão oficial de configuração, usuários relataram saída ininteligível e dúvida sobre vocab/limites de duração, sem resolução confirmada ([disc. #774](https://github.com/SWivid/F5-TTS/discussions/774)).
- `ModelsLab/F5-tts-brazilian`: `model_2600000.pt`, sem dados de treino, licença ou data; instruções legadas (CUDA 11.8, `inference-cli.py`) ([card](https://huggingface.co/ModelsLab/F5-tts-brazilian)).
- A tabela oficial de modelos comunitários (`SHARED.md`) **não lista nenhum checkpoint português** ([SHARED.md](https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/infer/SHARED.md)) — nenhum dos dois passou por curadoria dos mantenedores.

**VRAM.** Sem número oficial; pesos de 1,35 GB + Vocos sugerem ordem de 2 GB, próximo do XTTS. Tratar como estimativa até medir.

### 5. OpenAudio S1-mini / fish-speech

**Modelo.** S1-mini = **0,5B** parâmetros (destilado do S1 de 4B); 13 idiomas incluindo **Portuguese**; >2M h de áudio; WER 0,011 / CER 0,005 em inglês; pesos **CC-BY-NC-SA-4.0**; 30+ marcadores de emoção/tom/efeito — `(angry)`, `(sad)`, `(excited)`, `(whispering)`, `(shouting)`, `(laughing)`, `(sobbing)`… ([HF card](https://huggingface.co/fishaudio/openaudio-s1-mini)). README da era S1: RTF ~**1:7 em RTX 4090 com `torch.compile`**; "suporte nativo a Linux e Windows" ([espelho S1](https://github.com/ai-audio/fish-speech-s1)). O número "1:5 em RTX 4060 laptop, 1:15 em RTX 4090" que circula é do **Fish Speech 1.5**, não do S1-mini ([README v1.5.1](https://raw.githubusercontent.com/fishaudio/fish-speech/v1.5.1/README.md)). Sem amostras nem avaliação em português publicadas (issue pedindo amostras multilíngues fechou como *stale* — [#1088](https://github.com/fishaudio/fish-speech/issues/1088)). A dúvida de licença sobre uso não-comercial ficou sem resposta dos mantenedores ([disc. #1001](https://github.com/fishaudio/fish-speech/discussions/1001)).

**Código — o problema.** O repositório `fishaudio/fish-speech` **não tem tag/release para S1**: as releases pulam de v1.5.1 para v2.0.0-beta (S2 Pro) ([releases](https://github.com/fishaudio/fish-speech/releases)). Rodar S1-mini com as tags v1.5.1/v1.4.x falha (`DualARModelArgs.__init__() got an unexpected keyword argument 'attention_o_bias'`), issue aberto sem resposta ([#1026](https://github.com/fishaudio/fish-speech/issues/1026)) — é preciso cravar um commit de `main` de meados de 2025. Os pins dessa era: `torch>=2.5.1`, `transformers>=4.45.2`, **`numpy<=1.26.4`**, `pydantic==2.9.2`, `datasets==2.18.0`, `modelscope==1.17.1`, `pyaudio`, `descript-audio-codec` ([pyproject S1-era](https://raw.githubusercontent.com/ai-audio/fish-speech-s1/main/pyproject.toml)); docs da mesma era: "GPU Memory: 12 GB (Inference); System: Linux, WSL" ([install.md S1-era](https://raw.githubusercontent.com/ai-audio/fish-speech-s1/main/docs/en/install.md)). O `main` atual (2.0.0, S2 Pro): `torch==2.8.0`, `transformers<=4.57.3`, **24 GB de VRAM**, Linux/WSL, Python 3.12, licença própria "FISH AUDIO RESEARCH LICENSE", "`compile` não suportado no Windows" ([pyproject main](https://raw.githubusercontent.com/fishaudio/fish-speech/main/pyproject.toml), [install](https://speech.fish.audio/install/), [inference](https://speech.fish.audio/inference/)).

**Veredito:** `numpy<=1.26.4` e `pydantic==2.9.2` colidem com a venv principal; sem release pinada, sem número de VRAM para o S1-mini isolado, sem `compile` no Windows (logo sem o RTF anunciado). Só faria sentido em venv própria com commit cravado — custo alto para um candidato sem evidência de qualidade em pt-BR.

### 6. Coexistência com VLM de 3B–7B em 8 GB

Fatos: a memória do projeto registra empiricamente que **Qwen2.5-VL 7B em 4-bit + XTTS não cabem juntos** em 8 GB, e o orquestrador já chama `vlm_engine.release()` entre estágios. Qwen2.5-VL vem em 3B e 7B ([3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)); em NF4 os pesos ficam na ordem de 0,5–0,6 byte/parâmetro, mais o encoder de visão e ativações de imagem (estimativa, sem número oficial).

Orçamento aproximado (8 GB nominais, ~7 GB úteis no Windows/WDDM):

| TTS residente | VRAM TTS | + VLM 3B 4-bit (~2–3 GB) | + VLM 7B 4-bit (~5–5,5 GB) |
|---|---|---|---|
| Kokoro-82M | <1 GB | **cabe** | provavelmente cabe (justo) |
| XTTS v2 | 2,1 GB | cabe (justo) | **não cabe** (confirmado) |
| F5-TTS | ~2 GB (est.) | cabe (justo) | não cabe |
| Chatterbox | ~5 GB (obs.) | não cabe | não cabe |
| S1-mini | desconhecido (docs: 12 GB) | não avaliar | não cabe |

Conclusão: manter o sequenciamento carrega→usa→`release()`. Só Kokoro (e, com folga mínima, XTTS/F5) justifica pensar em residência simultânea com VLM de 3B; com 7B nada convive. Como o VLM padrão do projeto é API (Qwen/Claude), a coexistência local é caso secundário.

---

## Recomendação

**Teste cego A/B com três braços, e um quarto opcional:**

1. **XTTS v2** (baseline obrigatório): já integrado, 2,1 GB, controles conhecidos, pt no treino. Serve de âncora para o painel.
2. **Chatterbox Multilingual V3 — pack pt-BR** (`ResembleAI/Chatterbox-Multilingual-pt-br`): único candidato MIT com finetune oficial pt-BR e controle explícito de emoção (`exaggeration`/`cfg_weight`); custo: venv própria, `--no-deps`, instalar de git master (V3 só lá), ~5 GB de VRAM, RTF de consumo desconhecido (medir). É o principal desafiante.
3. **Kokoro-82M pt-BR** (`pf_dora`, `pm_alex`, `pm_santa`): Apache, instala limpo nos pins, dezenas de vezes mais rápido que tempo real, cabe ao lado de qualquer coisa. Sem clonagem e sem emoção — entra como candidato a **voz de narrador/descrições** (grande volume de texto) e como piso de custo, não como elenco.
4. *(opcional)* **F5-TTS + `firstpixel/F5-TTS-pt-br`**: CC-BY-NC (aceitável pelo ADR-0002), instala nos pins, melhor RTF entre os clonadores (0,15 em 3090); mas finetune comunitário sem curadoria, relatos de configuração instável, e exige contorno torchcodec/FFmpeg. Incluir só se o setup levar <1 dia; caso contrário, registrar como fog.

**Excluir por ora:** OpenAudio S1-mini — sem release pinada compatível, pins colidentes, docs migradas para S2/24 GB, sem evidência pt-BR, `compile` indisponível no Windows.

**Desenho mínimo do teste.** Mesmo roteiro de 6–8 falas de Berserk em pt-BR (narração, grito, sussurro, pergunta, fala longa com vírgulas), mesmo clipe de referência de ~8 s por personagem para os motores que clonam, presets de expressividade fixados por motor; 3+ ouvintes (incluindo pelo menos um leitor cego) em ordem aleatória, escala 1–5 para naturalidade, inteligibilidade e emoção; medir por motor `torch.cuda.max_memory_allocated()` e RTF na RTX 5060. Cada motor numa venv própria, plugado pela factory (`tts_factory.py`) via subprocesso — a ABC de `interfaces.py` já isola isso.

## Riscos e incertezas

- **Chatterbox master com transformers 4.57.6 não foi executado por ninguém nas fontes** — o commit para 5.2.0 mudou pouco código, mas o teste real é obrigatório; fallback é venv com transformers 5.2.0, aceitável porque a venv é isolada.
- **VRAM de Chatterbox, F5-TTS e S1-mini não têm número oficial**; os valores acima são relatos de usuários ou estimativa por tamanho de pesos.
- **Qualidade pt-BR do Kokoro é não-graduada** e depende de espeak-ng; pode soar robótico em nomes próprios do mangá (Guts, Griffith) — testar explicitamente.
- **Sotaque do Chatterbox genérico oscila pt-BR/pt-PT** (#281); usar sempre o pack pt-BR.
- **XTTS: variante do português não documentada** no paper; o teste cego resolve na prática.
- **torchcodec no Windows** permanece frágil para qualquer motor que passe por `torchaudio.load` (F5-TTS, XTTS); manter o contorno via `soundfile`.
- **Licenças**: só Chatterbox (MIT) e Kokoro (Apache) são *license-clean*; XTTS (CPML), F5-pt-br (CC-BY-NC) e S1-mini (CC-BY-NC-SA) dependem do ADR-0002 continuar válido.
- Números de RTF em H100/L20/3090 não se transferem para uma RTX 5060 Laptop (TDP limitado); medir localmente antes de decidir.

## Fontes

Ambiente / plataforma
- https://pytorch.org/blog/pytorch-2-7/ (Blackwell + cu128 a partir do 2.7)
- https://pytorch.org/get-started/previous-versions/ (torch 2.11.0 cu128)
- https://github.com/meta-pytorch/torchcodec (tabela torchcodec↔torch, FFmpeg 4–9, builds shared no Windows)
- https://github.com/meta-pytorch/torchcodec/issues/1147 (XTTS + RTX 5060 + Windows)
- https://github.com/idiap/coqui-ai-TTS/issues/558 (transformers ≥ 5.1 quebra coqui-tts)

XTTS v2
- https://pypi.org/project/coqui-tts/
- https://raw.githubusercontent.com/idiap/coqui-ai-TTS/main/pyproject.toml
- https://huggingface.co/coqui/XTTS-v2
- https://coqui-tts.readthedocs.io/en/latest/models/xtts.html
- https://arxiv.org/html/2406.04904 (paper XTTS)
- https://github.com/coqui-ai/TTS/issues/3976 (2,1 GB VRAM)

Chatterbox
- https://github.com/resemble-ai/chatterbox
- https://raw.githubusercontent.com/resemble-ai/chatterbox/master/pyproject.toml
- https://raw.githubusercontent.com/resemble-ai/chatterbox/master/src/chatterbox/mtl_tts.py
- https://pypi.org/pypi/chatterbox-tts/0.1.4/json · /0.1.6/json · /0.1.7/json · https://pypi.org/project/chatterbox-tts/#history
- https://github.com/resemble-ai/chatterbox/commit/eaf9354 · https://github.com/resemble-ai/chatterbox/commits/master
- https://github.com/resemble-ai/chatterbox/issues/339 · /159 · /281 · /367 · /508
- https://huggingface.co/ResembleAI/chatterbox · https://huggingface.co/ResembleAI/chatterbox/tree/main
- https://huggingface.co/ResembleAI/Chatterbox-Multilingual-pt-br
- https://www.resemble.ai/learn/models/chatterbox-multilingual
- https://www.resemble.ai/resources/chatterbox-multilingual-v3-tts-with-embedded-watermarking-for-25-languages

Kokoro
- https://huggingface.co/hexgrad/Kokoro-82M · https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
- https://github.com/hexgrad/kokoro · https://raw.githubusercontent.com/hexgrad/kokoro/main/pyproject.toml · https://raw.githubusercontent.com/hexgrad/kokoro/main/kokoro/pipeline.py
- https://github.com/hexgrad/misaki · https://raw.githubusercontent.com/hexgrad/misaki/main/pyproject.toml
- https://github.com/thewh1teagle/kokoro-onnx
- https://gist.github.com/efemaer/23d9a3b949b751dde315192b4dcf0653 (benchmark T4/A10G/L4)

F5-TTS
- https://github.com/SWivid/F5-TTS · https://raw.githubusercontent.com/SWivid/F5-TTS/main/pyproject.toml
- https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/infer/README.md · https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/infer/SHARED.md
- https://arxiv.org/html/2410.06885 (paper F5-TTS)
- https://huggingface.co/SWivid/F5-TTS/tree/main/F5TTS_v1_Base
- https://github.com/SWivid/F5-TTS/issues/1234 · https://github.com/SWivid/F5-TTS/discussions/774
- https://huggingface.co/firstpixel/F5-TTS-pt-br · https://huggingface.co/firstpixel/F5-TTS-pt-br/tree/main/pt-br
- https://huggingface.co/ModelsLab/F5-tts-brazilian

OpenAudio S1-mini / fish-speech
- https://huggingface.co/fishaudio/openaudio-s1-mini
- https://github.com/fishaudio/fish-speech · https://raw.githubusercontent.com/fishaudio/fish-speech/main/pyproject.toml · https://github.com/fishaudio/fish-speech/releases
- https://speech.fish.audio/install/ · https://speech.fish.audio/inference/
- https://github.com/fishaudio/fish-speech/issues/1026 · /1088 · https://github.com/fishaudio/fish-speech/discussions/1001
- https://github.com/ai-audio/fish-speech-s1 (espelho da era S1) · https://raw.githubusercontent.com/ai-audio/fish-speech-s1/main/pyproject.toml · https://raw.githubusercontent.com/ai-audio/fish-speech-s1/main/docs/en/install.md
- https://raw.githubusercontent.com/fishaudio/fish-speech/v1.5.1/README.md (números "1:5 em RTX 4060" são do 1.5)

VLM
- https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct · https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
