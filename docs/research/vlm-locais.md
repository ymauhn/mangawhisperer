# R1 — VLMs locais para roteirização de mangá em 8 GB de VRAM

| Campo | Valor |
|---|---|
| Ticket | R1 (mapa Wayfinder, Ago/2026) |
| Data da pesquisa | 2026-09-02 |
| Ambiente-alvo | Windows 11 · Python 3.11 · torch 2.11.0+cu128 · `transformers>=4.57,<5` · RTX 5060 Laptop 8 GB (piso de VRAM) |
| Status | concluído — aguardando benchmark (shortlist na seção *Recomendação*) |

## Abstract (EN)

We surveyed open-weight vision-language models that fit an 8 GB consumer GPU at 4-bit and can act as the "scriptwriter" stage of MangaWhisperer (speaker attribution + scene description, Brazilian-Portuguese output, structured JSON). Facts were taken from model cards, official GGUF repositories (file sizes via the Hugging Face API), technical reports and runtime documentation (transformers, llama.cpp, Ollama, vLLM, bitsandbytes). Key findings: (1) **Qwen3-VL-8B/4B/2B** (Apache 2.0, `transformers>=4.57.0`, official GGUFs of 4.68/2.33/1.03 GB at Q4_K_M) are the best-supported family inside the project's transformers pin; (2) **Gemma 4 E4B/12B** (Apache 2.0, released 2026-04-02) has the strongest documented multilingual scores but needs `transformers>=5.5.0`, so on this project it is only reachable out-of-process (llama.cpp / Ollama); (3) **InternVL3.5-8B** is the only candidate with a published Portuguese-specific number (MMMB pt = 81.4) and is natively supported by transformers 4.57 via its `-HF` checkpoints; (4) **Gemma 3 12B** does not fit 8 GB with its vision projector at 4-bit; **Moondream 3** is excluded (BSL license, `.compile()`/FlexAttention requires Triton, which is Linux-only, no official int4, no llama.cpp support, no multilingual claim); (5) JSON reliability should come from **grammar-constrained decoding** (llama-server `json_schema`, Ollama `format`, Outlines for transformers), not from prompting. Recommended benchmark shortlist: **Qwen3-VL-8B-Instruct, Gemma 4 E4B-it, InternVL3.5-8B**, plus Qwen3-VL-4B as a cheap headroom variant and Qwen3-VL-2B for the no-GPU tier. A manga-specific finding (MangaVQA paper): off-the-shelf VLMs cannot read manga text (Qwen2.5-VL-7B scored 0.9 % on MangaOCR), which validates keeping EasyOCR as the text source and using the VLM only for attribution and description.

## Pergunta

Quais VLMs open-source cabem em 8 GB de VRAM (4-bit) e produzem texto confiável em português brasileiro e JSON estruturado para roteirizar painéis de mangá (atribuição de falante + descrição de cena)? Para cada candidato: VRAM real em 4-bit (bitsandbytes/AWQ/GGUF), evidência de qualidade em português, confiabilidade de saída estruturada, licença, compatibilidade com `transformers>=4.57,<5` (ou alternativas llama.cpp/Ollama/vLLM no Windows) e tokens/s em GPU consumer. Entregar uma shortlist de 3 para benchmark.

## Resumo executivo

1. **O pin `transformers<5` é a restrição decisiva.** A última versão 4.x é a 4.57.6 (2026-01-16); a 5.0.0 saiu em 2026-01-26 [S31][S32]. Dentro do pin cabem, com suporte nativo (páginas de documentação existentes em `v4.57.0`): Qwen3-VL, Qwen2.5-VL, Gemma 3 e InternVL (checkpoints `-HF`) [S33][S34][S35]. **Ficam de fora in-process**: Gemma 4 (adicionado em transformers 5.5.0 [S36]; documentação inexistente em 4.57.0 [S37]) e MiniCPM-V 4.6 (`transformers[torch]>=5.7.0` [S18]).
2. **O caminho mais robusto no Windows é fora do processo Python**: `llama-server` (binários oficiais `win-cuda-12.4` e `win-cuda-13.3`, release b10752 de 2026-09-02 [S40]) ou Ollama (Windows 10 22H2+, driver 551.61+, compute capability 12.0/RTX 50 suportada [S43][S44]). Ambos oferecem **JSON constrangido por schema** com imagens [S41][S45]. vLLM **não suporta Windows nativamente** (só Linux/WSL) [S46]. Isso desacopla a escolha do VLM do pin do transformers.
3. **VRAM real em 4-bit (peso do texto + projetor de visão, GGUF oficial)**: Qwen3-VL-8B 4.68 + 1.08 GB; Qwen3-VL-4B 2.33 + 0.78 GB; Qwen3-VL-2B 1.03 + 0.76 GB; InternVL3.5-8B 4.68 + 0.63 GB; MiniCPM-V 4.5 4.68 + 1.02 GB; Gemma 4 E4B 4.63 + 0.92 GB; Gemma 4 12B 6.64 + 0.16 GB; Gemma 3 12B 6.80 + 0.80 GB; Gemma 3 4B 2.49 GB (+ mmproj) [S8][S9][S10][S27][S21][S38][S39][S24][S25]. Somando KV cache e buffers (~1 GB), **Gemma 3 12B não cabe em 8 GB** e **Gemma 4 12B só cabe em IQ4_XS/Q4_K_S com contexto curto**; os demais cabem com folga de 1,5–4 GB.
4. **Evidência de português**: só InternVL publica número por idioma (MMMB pt: InternVL3.5-8B 81.4, -4B 81.0, -2B 75.9; InternVL3-8B 82.5) [S28][S29]. Qwen3-VL declara OCR em 32 idiomas incluindo "línguas de escrita latina" e herda o LLM Qwen3, que lista **Portuguese** explicitamente entre 119 idiomas [S4][S5][S30]. Gemma 3/4 declaram 140+ idiomas de pré-treino (Gemma 4: 35+ "out-of-the-box") com MMMLU 76.6 (E4B) / 83.4 (12B) [S22][S23]. MiniCPM-V 4.5 diz "30+ idiomas" sem nomear português [S17]. Moondream não faz nenhuma afirmação multilíngue [S19][S20].
5. **Shortlist para benchmark**: (1) **Qwen3-VL-8B-Instruct** (padrão), (2) **Gemma 4 E4B-it** (família diferente, melhor pedigree multilíngue documentado, só via llama.cpp/Ollama), (3) **InternVL3.5-8B** (único com número de PT publicado; funciona in-process em 4.57). Extras de custo quase zero no mesmo harness: **Qwen3-VL-4B** (decisão de folga/velocidade) e **Qwen3-VL-2B** (camada sem GPU, ≤3B conforme ADR-0001).
6. **Achado específico de mangá**: no benchmark MangaOCR (páginas de mangá em japonês), Qwen2.5-VL-7B fez 0,9 % de Hmean e GPT-4o/Gemini 2.5 fizeram 0,0 %; só o modelo fine-tunado (MangaLMM) chegou a 71,5 % [S47]. Conclusão: **o VLM não deve ser a fonte do texto** — manter EasyOCR como OCR e usar o VLM para atribuição de falante + descrição, com o texto OCR no prompt (como o pipeline já prevê).

## Achados

### 1. Runtime no Windows 11 (fatos que condicionam tudo)

| Componente | Fato verificado | Fonte |
|---|---|---|
| transformers 4.x | Última 4.x = **4.57.6** (2026-01-16); 5.0.0 lançada 2026-01-26 (tokenizers unificados, `WeightConverter`, remoções de API) | [S31][S32] |
| Modelos com doc em `v4.57.0` | `qwen3_vl` (Qwen3VLForConditionalGeneration), `gemma3` (Gemma3ForConditionalGeneration), `internvl` (InternVLForConditionalGeneration) existem; `gemma4` **não existe** ("doesn't exist in v4.57.0, but exists on the main version") | [S33][S34][S35][S37] |
| bitsandbytes (NF4) | Wheels oficiais **Windows x86-64** para CUDA 12.8 com alvos `sm100, sm120` (Blackwell/RTX 50); NF4/FP4 exigem CC 6.0+ | [S48] |
| llama.cpp | Multimodal via `libmtmd` + `--mmproj`, suportado em `llama-cli`, `llama-server` (API OpenAI `/v1/chat/completions` com `image_url`) e `llama-mtmd-cli`; release **b10752 (2026-09-02)** traz `llama-b10752-bin-win-cuda-12.4-x64.zip` e `llama-b10752-bin-win-cuda-13.3-x64.zip` | [S40][S41][S42] |
| llama.cpp — saída estruturada | `json_schema` (grammar-based sampling), `grammar` GBNF, e `response_format: {"type":"json_object","schema":{...}}` no endpoint de chat | [S41] |
| Ollama | Windows 10 22H2+ / driver NVIDIA 551.61+; CC **12.0 (RTX 5060…5090)** suportada; `format` aceita JSON Schema e "Vision models accept the same `format` parameter" | [S43][S44][S45] |
| vLLM | "OS: Linux" — "vLLM does not support Windows natively" (WSL como alternativa); CC 7.5+ | [S46] |
| Outlines (in-process) | Geração estruturada com modelos multimodais do transformers (exemplo oficial com `Qwen/Qwen2.5-VL-3B-Instruct`) | [S49] |
| Triton (torch.compile) | "Supported Platforms: Linux" — Windows não listado | [S50] |
| GPU alvo | RTX 5060 Laptop: **8 GB GDDR7, 384 GB/s, 3328 CUDA cores** | [S51] |

Implicação: qualquer modelo pode ser servido no Windows por `llama-server`/Ollama independentemente do pin do transformers; in-process (transformers + bitsandbytes NF4) só os que existem em 4.57.x.

### 2. Tabela comparativa

Tamanhos de arquivo obtidos pela API do Hugging Face (`/api/models/<repo>/tree/main`); "Q4 total" = texto Q4_K_M + mmproj F16 (Q8_0 do mmproj entre parênteses quando existe).

| Modelo | Params | Licença | transformers | Q4 total (GB) | Ollama (GB) | Evidência PT | Notas |
|---|---|---|---|---|---|---|---|
| **Qwen3-VL-8B-Instruct** | 8,77 B [S6] | Apache-2.0 | `>=4.57.0` [S3]; doc em 4.57.0 [S33] | 4,68 + 1,08 (0,70) = **5,76** [S8] | 6,1 [S11] | OCR 32 idiomas incl. escrita latina [S4]; LLM Qwen3 lista Portuguese [S30] | llama.cpp PR #16780 [S52]; GGUF oficial |
| **Qwen3-VL-4B-Instruct** | 4,44 B [S7] | Apache-2.0 | idem | 2,33 + 0,78 (0,42) = **3,11** [S9] | 3,3 [S11] | idem | GGUF oficial |
| **Qwen3-VL-2B-Instruct** | 2,13 B [S53] | Apache-2.0 | idem | 1,03 + 0,76 (0,41) = **1,79** [S10] | 1,9 [S11] | idem | candidato à camada sem GPU (≤3B) |
| Qwen2.5-VL-7B-Instruct | 7 B | Apache-2.0 [S12] | desde 4.49.0 (2025-02-17) [S54] | 4,36 + 1,26 (0,79) = **5,62** [S13] | — | OCRBench 864, DocVQA 95,7 [S12]; sem número PT | superado pelo Qwen3-VL; 28 px/token (mais tokens por página) [S15] |
| Qwen2.5-VL-3B-Instruct | 3 B | **qwen-research** (não Apache) [S14] | idem | 1,93 + mmproj [S55] | — | — | licença restritiva → descartar |
| **InternVL3.5-8B** | 8,5 B (0,3 ViT + 8,2 Qwen3-8B) [S26] | Apache-2.0 | `>=4.52.1` (código custom) ou `InternVL3_5-8B-HF` nativo [S26][S35] | 4,68 + 0,63 = **5,31** (GGUF comunitário bartowski) [S27] | não verificado | **MMMB pt 81,4** [S29] | GGUF oficial ggml-org só para InternVL3-8B (4,36 + 0,62) [S56] |
| MiniCPM-V 4.5 | 8,70 B [S57] | Apache-2.0 (pesos e código) [S17] | `trust_remote_code`; versão não fixada no card (o irmão MiniCPM-o 4.5 exige `==4.51.0`) [S17][S18] | 4,68 + 1,02 = **5,70** (GGUF oficial) [S21] | 6,1 (ctx 40K) [S58] | "30+ idiomas", PT não nomeado [S17] | llama.cpp PR #15575 [S59]; int4 oficial sem VRAM declarada |
| MiniCPM-V 4.6 | 1 B (Qwen3.5-0.8B) [S18] | Apache-2.0 | **`>=5.7.0`** → fora do pin [S18] | — | 1,6 [S60] | — | lançado 2026-05-11; pequeno demais para roteirizar |
| Gemma 3 4B-it | 4 B | Gemma Terms of Use [S22][S61] | desde 4.50.0 (2025-03-21) [S62]; doc em 4.57.0 [S34] | 2,49 + mmproj [S25] | 3,3 [S63] | 140+ idiomas; Global-MMLU-Lite 57,0 (PT-base) [S22] | 256 tokens/imagem fixos (896×896) [S22] |
| Gemma 3 12B-it | 12,19 B [S64] | Gemma Terms of Use | idem | 6,80 + 0,80 = **7,60** [S24]; QAT q4_0 oficial 7,52 + 0,80 [S65] | 8,1 [S63] | Global-MMLU-Lite 69,4; DocVQA 87,1 [S22] | **não cabe em 8 GB** com KV (ver §4) |
| **Gemma 4 E4B-it** | 4,5 B efetivos (8 B c/ embeddings) [S66] | **Apache-2.0** [S23][S67] | **`>=5.5.0`** → só llama.cpp/Ollama [S36][S37] | 4,63 + 0,92 = **5,55** (GGUF unsloth) [S38] | 9,6 (!) [S68] | MMMLU **76,6**; OmniDocBench 0,181 [S23] | budget visual 70–1120 tokens [S23] |
| Gemma 4 12B-it | 11,95 B | Apache-2.0 | idem | 6,64 + 0,16 = **6,80**; IQ4_XS 5,94 [S39] | 7,6 [S68] | MMMLU **83,4**; OmniDocBench 0,164 [S23] | limítrofe em 8 GB |
| Moondream 3 (preview) | 9 B MoE / 2 B ativos [S19] | BSL 1.1 + Additional Use Grant [S19] | `trust_remote_code` + `.compile()` "critical" (FlexAttention) [S69] | sem int4 oficial; bf16 ≈ 18 GB | — | nenhuma afirmação de idioma [S19][S20] | docs citam A100/H100/RTX 4090 [S20]; llama.cpp só moondream2 [S42] |

### 3. Fichas por modelo

#### Qwen3-VL (2B / 4B / 8B Instruct) — Alibaba, Apache-2.0
- Lançamento: 4B/8B em 2025-10-15, 2B em 2025-10-21; relatório técnico 2025-11-27 [S3]. Requisito oficial: "The Qwen3-VL model requires transformers >= 4.57.0" [S3] — exatamente o piso do pin do projeto.
- Visão: `patch_size 16`, `spatial_merge_size 2` → **32 px por token visual** (Qwen2.5-VL: 14×2 = 28 px) [S15][S16]. Texto (8B): 36 camadas, 32 heads, **8 KV heads**, contexto 262 144 [S16].
- OCR: card e docs oficiais falam em **32 idiomas** ("up from 10/19"), com categorias "Latin-script languages, Chinese, Japanese, Korean, Arabic, Hebrew, Cyrillic, Indic" [S2][S4]; o relatório técnico fala em 39 [S5]. Português não é nomeado explicitamente em nenhuma dessas listas — a evidência direta de PT vem do LLM base Qwen3 (119 idiomas, "English, French, **Portuguese**, German…") [S30].
- Distribuição: GGUF **oficial** da Qwen para 2B/4B/8B com mmproj F16/Q8_0 [S8][S9][S10]; Ollama `qwen3-vl:{2b,4b,8b}` = 1,9/3,3/6,1 GB, contexto 256K [S11]; suporte llama.cpp mesclado no PR #16780 [S52].
- Saída estruturada: nada específico no card; o README fala de grounding com coordenadas relativas [S3]. Confiar em decodificação constrangida (§6).

#### Qwen2.5-VL-7B / 3B — Alibaba
- 7B: Apache-2.0, OCRBench 864, DocVQA 95,7, TextVQA 84,9 [S12]; no transformers desde 4.49.0 [S54]; GGUF oficial ggml-org (Q4_K_M 4,36 GB + mmproj 1,26/0,79) [S13]; listado em `multimodal.md` do llama.cpp [S42].
- **3B: licença `qwen-research`**, não Apache [S14] — mesmo sendo aceitável pelo ADR-0002, há alternativa Apache melhor (Qwen3-VL-4B).
- Veredito: superado pelo Qwen3-VL na mesma licença (mais idiomas de OCR, menos tokens por página, mesmo pin). Útil só como baseline de regressão.

#### InternVL3.5-8B — OpenGVLab, Apache-2.0
- Lançado 2025-08-25; ViT InternViT-300M + LLM Qwen3-8B (8,5 B) [S26]. "transformers>=4.52.1" para o código custom; checkpoints **`InternVL3_5-8B-HF`, -4B-HF, -2B-HF, -1B-HF** para a classe nativa `InternVLForConditionalGeneration`, presente em 4.57.0 [S26][S35].
- **Única família com número por idioma**: Tabela 8 do paper (MMMB, pt): 1B 67,2 · 2B 75,9 · 4B 81,0 · **8B 81,4** · 14B 82,7; Qwen2-VL-7B 81,2 [S29]. InternVL3 (Tabela 7): 8B MMMB pt 82,5 / Multilingual MMBench pt 83,2 [S28].
- Distribuição: GGUF oficial ggml-org só para InternVL3 (8B: 4,36 + 0,62 GB) [S56]; para 3.5 há conversões comunitárias (bartowski: Q4_K_M 4,68 GB, mmproj f16 0,63 GB) [S27]. Visão em tiles de 448×448, patch 14 [S35].
- Quantização in-process: card demonstra `load_in_8bit=True` (bitsandbytes) [S26]; NF4 pela mesma via.

#### MiniCPM-V 4.5 (e 4.6) — OpenBMB, Apache-2.0
- 4.5: Qwen3-8B + SigLIP2-400M, 8,7 B [S17][S57]; "more than 30 languages" sem nomear PT [S17]; card afirma liderar OCRBench "surpassing GPT-4o-latest and Gemini 2.5" (auto-reportado) [S17]. Requer `trust_remote_code`; o card não fixa versão do transformers, mas o irmão MiniCPM-o 4.5 exige `==4.51.0` "as other versions may have compatibility issues" [S18] — sinal de fragilidade do código custom frente a 4.57.
- Formatos: int4 oficial (sem VRAM declarada) [S70], AWQ, **GGUF oficial** (Q4_K_M 4,68 GB + mmproj f16 1,02 GB) [S21]; llama.cpp PR #15575 [S59]; Ollama `minicpm-v4.5` 6,1 GB, contexto 40K [S58]. O guia BnB do cookbook reporta "GPU memory usage after quantization: 18.97GB" — número que não corresponde a um modelo de 8B em 4-bit e não deve ser usado como referência [S71].
- 4.6 (2026-05-11): 1 B (Qwen3.5-0.8B), `transformers[torch]>=5.7.0` [S18] — fora do pin; Ollama 1,6 GB [S60]. Alternativa a estudar para a camada sem GPU, mas via Ollama.

#### Gemma 3 (4B / 12B) — Google, Gemma Terms of Use
- Multimodal em 4B/12B/27B; "over 140 languages"; imagens normalizadas para 896×896 e codificadas em **256 tokens** [S22]. Transformers desde 4.50.0 [S62]. Licença "Gemma Terms of Use" (modificada 2026-04-01; Gemma 4 tem licença própria) [S61].
- Benchmarks (IT): DocVQA 75,8 (4B) / 87,1 (12B); TextVQA 57,8 / 67,7; MMMU 48,8 / 59,6. Multilíngue (PT-base): Global-MMLU-Lite 57,0 / 69,4; MGSM 34,7 / 64,3; WMT24++ 48,4 / 53,9 [S22].
- Memória: Google reporta 12B int4 (QAT) em 6,6 GB e cita a "RTX 4060 Laptop GPU (8GB VRAM)" [S72], mas esse número é só o LLM: o GGUF QAT oficial tem **7,52 GB + mmproj 0,80 GB** [S65] e o Q4_K_M ggml-org 6,80 + 0,80 GB [S24]; Ollama `gemma3:12b` = 8,1 GB [S63]. Com KV cache e buffers CUDA, **não cabe** em 8 GB sem offload para CPU.
- Veredito: 4B cabe mas é o mais fraco do grupo em documento/OCR; 12B não cabe. Ambos superados por Gemma 4 na mesma casa.

#### Gemma 4 (E2B / E4B / 12B / 26B-A4B / 31B) — Google, **Apache-2.0**
- Lançado 2026-04-02 ("The release of Gemma 4 under the Apache 2.0 license") [S67]; card: E2B 2,3 B ef., E4B 4,5 B ef., 12B 11,95 B; "out-of-the-box support for 35+ languages, pre-trained on 140+ languages"; budget visual configurável **70/140/280/560/1120 tokens** [S23].
- Benchmarks: MMLU Pro 69,4 (E4B) / 77,2 (12B); **MMMLU 76,6 / 83,4**; OmniDocBench 1.5 (edit distance, menor é melhor) 0,181 / 0,164 [S23].
- Suporte: transformers **5.5.0** [S36]; ausente em 4.57.0 [S37] → no projeto só via llama.cpp/Ollama. GGUF (unsloth): E4B Q4_K_M 4,63 GB + mmproj 0,92; 12B Q4_K_M 6,64 GB + mmproj 0,16, IQ4_XS 5,94 [S38][S39]. Ollama: `gemma4:12b` 7,6 GB, `gemma4:e4b` 9,6 GB, `gemma4:e2b` 7,2 GB [S68] — os tamanhos E2B/E4B do Ollama são anômalos (maiores que o 12B); usar os GGUF Q4 diretamente no llama.cpp evita a dúvida. A própria doc de structured outputs do Ollama usa `gemma4` com imagem como exemplo [S45].

#### Moondream 3 (preview / 3.1) — M87 Labs
- 9 B MoE, 2 B ativos; preview em 2025-09-18 [S19][S73]. Licença: "Business Source License 1.1 with an Additional Use Grant (No Third-Party Service)" (preview) [S19]; 3.1 usa "Moondream Model License" v1.0 [S74].
- Execução: `trust_remote_code` + `moondream.compile()` — "calling `.compile()` is critical for fast decoding", FlexAttention [S69]. FlexAttention/torch.compile dependem de Triton, cujo README lista **só Linux** [S50] → risco alto no Windows nativo. Docs de execução local citam A100/H100/RTX 4090 e dizem que um Mac de 16 GB não comporta os pesos [S20]. Não há int4 oficial (só conversão comunitária), o llama.cpp suporta apenas moondream2 [S42], e nenhuma página oficial menciona idiomas.
- Veredito: **excluir**.

### 4. Orçamento de VRAM em 8 GB (estimativa a partir de fatos medidos)

Tokens visuais por página de mangá (≈1000×1500 px, sem reamostragem além do nativo):
- Qwen3-VL: 32 px/token → (1000/32)×(1500/32) ≈ 31×47 ≈ **1 450 tokens** [S15].
- Qwen2.5-VL: 28 px/token → ≈ 36×54 ≈ **1 940 tokens** [S16].
- Gemma 3: **256 tokens fixos** (896×896) — perde detalhe de balões pequenos, mas prefill barato [S22].
- Gemma 4: budget de **70 a 1 120 tokens** escolhido pelo chamador [S23].
- InternVL: tiles de 448 px, patch 14 [S35] (contagem depende do tiling dinâmico; medir no benchmark).

KV cache (F16) por token, a partir dos configs:
- Qwen3-VL-8B: 36 camadas × 8 KV heads × 128 × 2 (K,V) × 2 bytes ≈ **147 KB/token** → 4 k tokens ≈ 0,6 GB [S16]. InternVL3.5-8B e MiniCPM-V 4.5 usam o mesmo Qwen3-8B → mesma ordem.
- Qwen2.5-VL-7B: 28 × 4 × 128 × 2 × 2 ≈ 57 KB/token [S16].

Orçamento ilustrativo (pesos Q4 + mmproj + KV 4 k + ~0,5 GB de buffers/contexto CUDA; o Windows ainda reserva VRAM para o desktop na dGPU):

| Modelo (Q4_K_M) | Pesos+mmproj | +KV 4k +buffers | Folga em 8 GB |
|---|---|---|---|
| Qwen3-VL-8B (mmproj Q8_0) | 5,38 | ≈ 6,5 | ≈ 1,5 GB |
| Qwen3-VL-4B | 3,11 | ≈ 4,0 | ≈ 4 GB |
| InternVL3.5-8B | 5,31 | ≈ 6,4 | ≈ 1,6 GB |
| MiniCPM-V 4.5 | 5,70 | ≈ 6,8 | ≈ 1,2 GB |
| Gemma 4 E4B | 5,55 | ≈ 6,5 | ≈ 1,5 GB |
| Gemma 4 12B (IQ4_XS) | 6,10 | ≈ 7,2 | < 1 GB (limítrofe) |
| Gemma 3 12B | 7,60 | > 8 | **não cabe** |

In-process com bitsandbytes NF4 os pesos ficam ~10–15 % maiores que Q4_K_M e o encoder de visão roda em bf16; para o 8B esperar ≈ 6–6,5 GB antes do KV — viável, mas com menos folga que o GGUF.

### 5. Velocidade (tokens/s) em GPU consumer

Medições da tabela oficial de desempenho CUDA do llama.cpp (`llama-bench -m llama-2-7b.Q4_0.gguf -ngl 99`, modelo de 7 B em Q4_0 ≈ 3,8 GB) [S75]:

| GPU | VRAM | pp512 (t/s) | tg128 (t/s) |
|---|---|---|---|
| RTX 5060 (desktop) | 8 GB | 3 269 | 96,7 |
| RTX 5060 Ti | 16 GB | 3 737 | 90,9 |
| RTX 4060 Ti | 8 GB | 3 395 | 63,9 |
| RTX 4050 Laptop | 6 GB | 1 726 | 43,7 |
| RTX PRO 500 Blackwell Laptop | 6 GB | 1 440 | 48,9 |

Estimativa para a RTX 5060 Laptop (384 GB/s [S51]; decodificação limitada por largura de banda, eficiência típica ~0,7–0,85 na tabela acima):
- Qwen3-VL-8B Q4_K_M (4,68 GB): teto 384/4,68 ≈ 82 t/s → **esperar 45–65 t/s** de geração; prefill de uma página (~1 500 tokens visuais + prompt) na casa de 1 s mais o encoder de visão.
- Qwen3-VL-4B Q4_K_M (2,33 GB): **~90–130 t/s**.
- Gemma 4 E4B / InternVL3.5-8B / MiniCPM-V 4.5 (4,6–4,7 GB): faixa do 8B.
- Gemma 3 12B com offload parcial para CPU: uma ordem de grandeza mais lento — motivo adicional para descartar.

Nenhum card oficial publica t/s para GPUs consumer; esses números devem ser medidos no benchmark (`llama-bench` e tempo por página end-to-end).

### 6. Saída estruturada (JSON) — mecanismo, não promessa do modelo

- Nenhum dos cards publica taxa de validade de JSON. A confiabilidade vem do runtime: `llama-server` (`json_schema`/`response_format` com schema; GBNF) [S41], Ollama (`format` com JSON Schema, válido para modelos de visão; recomenda incluir o schema no prompt e temperatura 0) [S45], Outlines para transformers multimodal (exemplo com Qwen2.5-VL-3B) [S49]. vLLM também tem decodificação guiada, mas é Linux-only [S46].
- Recomendação de medição: rodar sempre **com** schema (produção) e registrar, como métrica secundária, a taxa de JSON válido **sem** schema — indica o quanto o modelo "entende" o contrato e reduz dependência de gramática (que pode degradar a fluência quando força tokens improváveis).
- O schema Pydantic do pipeline (falante, tipo de linha, descrição de ação em PT-BR) mapeia direto para JSON Schema; manter enums fechados para `speaker_id` (elenco conhecido do capítulo) reduz alucinação de nomes.

### 7. Evidência específica de mangá

- **MangaVQA/MangaLMM (arXiv 2505.20298)**: em MangaOCR, GPT-4o 0,0 %, Gemini 2.5 Flash 0,0 %, Claude Sonnet 4.5 0,0 %, **Qwen2.5-VL-7B 0,9 %**, MangaLMM (Qwen2.5-VL fine-tunado) 71,5 %; em MangaVQA (0–10): Gemini 2.5 Flash 7,26, MangaLMM 6,68, GPT-4o 6,00, Qwen2.5-VL-7B 5,65. Citação: "All LMMs except MangaLMM show near-zero scores on the MangaOCR benchmark" [S47]. O texto é japonês, então não transfere diretamente para PT-BR, mas mostra que VLMs genéricos não devem ler balões — confirma a arquitetura EasyOCR → VLM.
- **Magi v2 (`ragavsachdeva/magiv2`)**: modelo especializado que faz detecção de personagens, OCR e **associação texto↔personagem** capítulo inteiro (`text_character_associations`); licença livre para uso pessoal/pesquisa/não comercial (aceitável pelo ADR-0002); `trust_remote_code` [S76]. Não é VLM generativo, mas é um baseline natural para a métrica de atribuição de falante e um possível componente auxiliar — vale um ticket próprio.

## Recomendação

**Harness**: servir todos os candidatos por `llama-server` (build `win-cuda-13.3` para Blackwell; alternativamente Ollama) com `response_format`+schema, atrás da ABC de motor do VLM. Isso (a) torna o pin do transformers irrelevante para a escolha, (b) unifica o mecanismo de JSON e (c) mede VRAM/t/s no mesmo caminho. Manter um adaptador transformers+NF4 só para os modelos que existem em 4.57 (Qwen3-VL, InternVL3.5-HF) se a integração in-process for desejada.

**Shortlist (3) para o benchmark contra o Benchmark Set, com Opus como referência de ouro (ADR-0001):**

1. **Qwen3-VL-8B-Instruct** — `Qwen/Qwen3-VL-8B-Instruct-GGUF` Q4_K_M + `mmproj-…-Q8_0` (5,38 GB). Melhor combinação de licença, suporte oficial (GGUF da própria Qwen, transformers 4.57), OCR multilíngue e folga de VRAM. Candidato a padrão.
2. **Gemma 4 E4B-it** — `unsloth/gemma-4-E4B-it-GGUF` Q4_K_M + mmproj (5,55 GB), via llama.cpp/Ollama. Família diferente, Apache-2.0, melhor pedigree multilíngue documentado na faixa (MMMLU 76,6) e budget visual ajustável; testa se a qualidade de prosa em PT-BR compensa rodar fora do processo.
3. **InternVL3.5-8B** — `OpenGVLab/InternVL3_5-8B-HF` (in-process NF4) ou GGUF comunitário (5,31 GB). Único com número de português publicado (MMMB pt 81,4); serve de controle para a hipótese "o LLM Qwen3-8B é o que importa" (MiniCPM-V 4.5 e Qwen3-VL-8B usam o mesmo LLM).

**Extras no mesmo harness** (custo marginal): **Qwen3-VL-4B** (decide folga para múltiplos painéis/contexto de capítulo e ~2× a velocidade) e **Qwen3-VL-2B** (camada sem GPU, ≤3B). **Stretch** só se sobrar tempo: Gemma 4 12B em IQ4_XS com contexto ≤4 k.

**Descartados e por quê**: Gemma 3 12B (não cabe), Gemma 3 4B (superado pelo Gemma 4 E4B), Qwen2.5-VL (superado pelo Qwen3-VL; 3B com licença research), MiniCPM-V 4.5 (mesmo LLM que dois da shortlist, código custom sem pin, sem evidência PT) e 4.6 (transformers 5.7, 1 B), Moondream 3 (BSL, Triton/Linux, sem int4 oficial, sem claim multilíngue).

**Métricas sugeridas**: acurácia de atribuição de falante (vs. gabarito humano e vs. Magi v2), qualidade da descrição em PT-BR (rubrica + julgamento do Opus), taxa de JSON válido com/sem schema, VRAM de pico (`nvidia-smi`), t/s de prefill e geração, tempo por página.

## Riscos e incertezas

- **Português nunca é nomeado nos cards da Qwen e da OpenBMB**; a evidência é indireta (LLM base, "escrita latina", "30+ idiomas"). O benchmark precisa de rubrica de PT-BR (concordância, gírias, acentuação) — não assumir.
- **Qualidade em 4-bit**: nenhum card publica degradação Q4 vs bf16 para tarefas multilíngues; medir com pelo menos uma amostra em Q8_0 do Qwen3-VL-8B (8,11 GB — só cabe com offload) ou do 4B (3,99 GB) para calibrar.
- **Números de VRAM da §4 são estimativas** a partir de tamanhos de arquivo e configs; os 1–1,5 GB de folga podem evaporar com imagens múltiplas, contexto de capítulo ou o desktop do Windows na dGPU. Medir com `nvidia-smi` no laptop real.
- **Binários CUDA do llama.cpp**: o build `win-cuda-12.4` pode não trazer SASS para sm_120 (Blackwell); preferir `win-cuda-13.3` e confirmar que carrega na RTX 5060 Laptop. Independente do CUDA 12.8 do torch (processos separados).
- **Tamanhos anômalos do Gemma 4 no Ollama** (E4B 9,6 GB > 12B 7,6 GB): não explicado nas páginas oficiais; usar GGUF explícito.
- **transformers v5.5.0**: a data de lançamento não pôde ser lida com segurança da página (o extrator retornou "April 2, 2025", inconsistente com o lançamento do Gemma 4 em 2026-04-02); a afirmação relevante — Gemma 4 entra na 5.5.0 e não existe na 4.57.0 — foi confirmada pela documentação versionada [S36][S37].
- **MiniCPM-V 4.5 in-process**: o código custom pode quebrar em 4.57 (o irmão fixa 4.51.0); só testar via GGUF/Ollama.
- **Licença do Gemma 4**: o card e o blog dizem Apache 2.0 [S23][S67]; o texto da licença em `ai.google.dev/gemma/docs/gemma_4_license` não foi lido integralmente — confirmar antes de fixar como padrão (irrelevante para o ADR-0002, relevante se o horizonte mudar).
- **Novidades pós-pesquisa**: o ecossistema move rápido (Gemma 4 e MiniCPM-V 4.6 surgiram após o mapa); rever esta nota antes de fechar o benchmark.

## Fontes

- [S1] Qwen/Qwen3-VL-8B-Instruct — https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- [S2] Qwen/Qwen3-VL-4B-Instruct e 2B — https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct · https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- [S3] QwenLM/Qwen3-VL README (lineup, datas, "requires transformers >= 4.57.0") — https://github.com/QwenLM/Qwen3-VL
- [S4] Qwen3-VL docs — OCR & Key Information Extraction ("supports OCR in 32 languages") — https://qwenlm-qwen3-vl.mintlify.app/capabilities/ocr
- [S5] Qwen3-VL Technical Report (arXiv 2511.21631) — https://arxiv.org/html/2511.21631
- [S6] HF API Qwen3-VL-8B-Instruct (8 767 123 696 params) — https://huggingface.co/api/models/Qwen/Qwen3-VL-8B-Instruct
- [S7] HF API Qwen3-VL-4B-Instruct (4 437 815 808 params) — https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct
- [S8] Qwen3-VL-8B-Instruct-GGUF (tree) — https://huggingface.co/api/models/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/main
- [S9] Qwen3-VL-4B-Instruct-GGUF (tree) — https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct-GGUF/tree/main
- [S10] Qwen3-VL-2B-Instruct-GGUF (tree) — https://huggingface.co/api/models/Qwen/Qwen3-VL-2B-Instruct-GGUF/tree/main
- [S11] Ollama library qwen3-vl — https://ollama.com/library/qwen3-vl
- [S12] Qwen/Qwen2.5-VL-7B-Instruct — https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- [S13] ggml-org/Qwen2.5-VL-7B-Instruct-GGUF (tree) — https://huggingface.co/api/models/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF/tree/main
- [S14] Qwen/Qwen2.5-VL-3B-Instruct README (license: qwen-research) — https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/README.md
- [S15] Qwen3-VL-8B-Instruct config.json — https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/main/config.json
- [S16] Qwen2.5-VL-7B-Instruct config.json — https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/raw/main/config.json
- [S17] openbmb/MiniCPM-V-4_5 — https://huggingface.co/openbmb/MiniCPM-V-4_5
- [S18] OpenBMB/MiniCPM-V README (4.6, transformers>=5.7.0; MiniCPM-o 4.5 ==4.51.0) — https://github.com/OpenBMB/MiniCPM-V
- [S19] moondream/moondream3-preview — https://huggingface.co/moondream/moondream3-preview
- [S20] Moondream docs — Run locally — https://docs.moondream.ai/running-locally/
- [S21] openbmb/MiniCPM-V-4_5-gguf (tree) — https://huggingface.co/api/models/openbmb/MiniCPM-V-4_5-gguf/tree/main
- [S22] Gemma 3 model card — https://ai.google.dev/gemma/docs/core/model_card_3
- [S23] Gemma 4 model card — https://ai.google.dev/gemma/docs/core/model_card_4
- [S24] ggml-org/gemma-3-12b-it-GGUF (tree) — https://huggingface.co/api/models/ggml-org/gemma-3-12b-it-GGUF/tree/main
- [S25] ggml-org/gemma-3-4b-it-GGUF — https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF
- [S26] OpenGVLab/InternVL3_5-8B README — https://huggingface.co/OpenGVLab/InternVL3_5-8B/raw/main/README.md
- [S27] bartowski/OpenGVLab_InternVL3_5-8B-GGUF (tree) — https://huggingface.co/api/models/bartowski/OpenGVLab_InternVL3_5-8B-GGUF/tree/main
- [S28] InternVL3 paper, Tabela 7 (arXiv 2504.10479) — https://arxiv.org/html/2504.10479
- [S29] InternVL3.5 paper, Tabela 8 (arXiv 2508.18265) — https://arxiv.org/html/2508.18265
- [S30] Qwen3 blog (119 idiomas, lista com Portuguese) — https://qwenlm.github.io/blog/qwen3/
- [S31] PyPI transformers 4.57.6 (2026-01-16) — https://pypi.org/project/transformers/4.57.6/
- [S32] transformers v5.0.0 release (26 Jan 2026) — https://github.com/huggingface/transformers/releases/tag/v5.0.0
- [S33] transformers v4.57.0 docs — qwen3_vl — https://huggingface.co/docs/transformers/v4.57.0/en/model_doc/qwen3_vl
- [S34] transformers v4.57.0 docs — gemma3 — https://huggingface.co/docs/transformers/v4.57.0/en/model_doc/gemma3
- [S35] transformers v4.57.0 docs — internvl — https://huggingface.co/docs/transformers/v4.57.0/en/model_doc/internvl
- [S36] transformers v5.5.0 release (Gemma 4) — https://github.com/huggingface/transformers/releases/tag/v5.5.0
- [S37] transformers v4.57.0 docs — gemma4 (inexistente) — https://huggingface.co/docs/transformers/v4.57.0/en/model_doc/gemma4
- [S38] unsloth/gemma-4-E4B-it-GGUF (tree) — https://huggingface.co/api/models/unsloth/gemma-4-E4B-it-GGUF/tree/main
- [S39] unsloth/gemma-4-12b-it-GGUF (tree) — https://huggingface.co/api/models/unsloth/gemma-4-12b-it-GGUF/tree/main
- [S40] llama.cpp releases (b10752, assets Windows CUDA) — https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=3
- [S41] llama.cpp tools/server README (json_schema, grammar, response_format, image_url) — https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md
- [S42] llama.cpp docs/multimodal.md — https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/multimodal.md
- [S43] Ollama docs — Windows — https://docs.ollama.com/windows
- [S44] Ollama docs — GPU (compute capability 12.0) — https://docs.ollama.com/gpu
- [S45] Ollama docs — Structured outputs — https://docs.ollama.com/capabilities/structured-outputs
- [S46] vLLM — GPU installation requirements — https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html
- [S47] MangaVQA and MangaLMM (arXiv 2505.20298) — https://arxiv.org/html/2505.20298
- [S48] bitsandbytes installation (Windows CUDA 12.8, sm120) — https://huggingface.co/docs/bitsandbytes/main/en/installation
- [S49] Outlines — Transformers multimodal — https://dottxt-ai.github.io/outlines/latest/features/models/transformers_multimodal/
- [S50] Triton README (Supported Platforms: Linux) — https://github.com/triton-lang/triton
- [S51] NVIDIA GeForce RTX 50 Laptop specs (RTX 5060 Laptop: 8 GB GDDR7, 384 GB/s) — https://www.nvidia.com/en-us/geforce/laptops/50-series/
- [S52] llama.cpp PR #16780 (qwen3vl) — https://github.com/ggml-org/llama.cpp/pull/16780
- [S53] HF API Qwen3-VL-2B-Instruct (2 127 532 032 params) — https://huggingface.co/api/models/Qwen/Qwen3-VL-2B-Instruct
- [S54] transformers v4.49.0 release (Qwen2.5-VL) — https://github.com/huggingface/transformers/releases/tag/v4.49.0
- [S55] ggml-org/Qwen2.5-VL-3B-Instruct-GGUF — https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF
- [S56] ggml-org/InternVL3-8B-Instruct-GGUF (tree) — https://huggingface.co/api/models/ggml-org/InternVL3-8B-Instruct-GGUF/tree/main
- [S57] HF API MiniCPM-V-4_5 (8 695 895 280 params) — https://huggingface.co/api/models/openbmb/MiniCPM-V-4_5
- [S58] Ollama library minicpm-v4.5 — https://ollama.com/library/minicpm-v4.5
- [S59] llama.cpp PR #15575 (MiniCPM-V 4.5) — https://github.com/ggml-org/llama.cpp/pull/15575
- [S60] Ollama library minicpm-v4.6 — https://ollama.com/library/minicpm-v4.6
- [S61] Gemma Terms of Use — https://ai.google.dev/gemma/terms
- [S62] transformers v4.50.0 release (Gemma 3) — https://github.com/huggingface/transformers/releases/tag/v4.50.0
- [S63] Ollama library gemma3 — https://ollama.com/library/gemma3
- [S64] HF API google/gemma-3-12b-it — https://huggingface.co/api/models/google/gemma-3-12b-it
- [S65] google/gemma-3-12b-it-qat-q4_0-gguf (tree) — https://huggingface.co/api/models/google/gemma-3-12b-it-qat-q4_0-gguf/tree/main
- [S66] google/gemma-4-E4B-it — https://huggingface.co/google/gemma-4-E4B-it
- [S67] Google Open Source Blog — Gemma 4 (2026-04-02, Apache 2.0) — https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html
- [S68] Ollama library gemma4 — https://ollama.com/library/gemma4
- [S69] moondream3-preview README (compile, FlexAttention, BSL) — https://huggingface.co/moondream/moondream3-preview/raw/main/README.md
- [S70] openbmb/MiniCPM-V-4_5-int4 — https://huggingface.co/openbmb/MiniCPM-V-4_5-int4
- [S71] MiniCPM-V CookBook — BnB quantization guide — https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/quantization/bnb/minicpm-v4_5_bnb_quantize.md
- [S72] Google Developers Blog — Gemma 3 QAT (2025-04-18) — https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/
- [S73] Moondream blog — Moondream 3 Preview (2025-09-18) — https://moondream.ai/blog/moondream-3-preview
- [S74] moondream/moondream3.1-9B-A2B — https://huggingface.co/moondream/moondream3.1-9B-A2B
- [S75] llama.cpp — Performance of llama.cpp on NVIDIA CUDA (discussion #15013) — https://github.com/ggml-org/llama.cpp/discussions/15013
- [S76] ragavsachdeva/magiv2 — https://huggingface.co/ragavsachdeva/magiv2
