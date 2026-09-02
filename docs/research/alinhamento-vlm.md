# R4 — Alinhamento de um VLM pequeno para o roteirista (destilação do Opus)

Ticket de pesquisa do mapa Wayfinder · Data: 2026-09-02 · Escopo: viabilidade de destilar/alinhar um VLM local (Qwen2.5-VL 3B/7B, Qwen3-VL 2B/4B/8B ou similar) para a tarefa de roteirista do MangaWhisperer usando saídas do Claude Opus como ouro, sob o teto de VRAM de 8 GB (RTX 5060 Laptop, Blackwell sm_120, Windows 11) ou GPUs gratuitas (Colab/Kaggle T4 16 GB).

## English abstract

Feasibility study of distilling Claude Opus (the project's default scriptwriter) into a small open-weight VLM, against primary sources (model cards, TRL/Unsloth/LLaMA-Factory/ms-swift/EasyR1 docs, Anthropic pricing/vision docs). Findings: (a) QLoRA SFT of a 3B–4B VLM fits an 8 GB GPU (Unsloth lists 3.5 GB minimum for a 3B QLoRA; image tokens must be capped via `max_pixels`); TRL, Unsloth, LLaMA-Factory and ms-swift all support VLM SFT with a conversational dataset that carries an `images` column. (b) DPO with images is supported by TRL (`DPOTrainer`, `image`/`images` column) and LLaMA-Factory (`ranking: true` + `images`); ORPO in TRL is `experimental` with no VLM support. (c) GRPO with rule-based rewards is supported by TRL (`GRPOTrainer`, plain-Python reward functions that receive every dataset column, so an answer key column can be passed), Unsloth (Qwen3-VL-8B GSPO/GRPO on a free Colab T4, notebook uses 4-bit + vLLM, `num_generations=4`) and ms-swift; research frameworks (EasyR1, VLM-R1) assume 40 GB+ multi-GPU. GRPO does not fit 8 GB in practice (Unsloth reports 13–14.5 GiB peak for a 4B text model on a T4) and vLLM does not run natively on Windows; run it on Colab/Kaggle T4. (d) Judge distillation: the Anthropic API exposes no logprobs, so only black-box (sequence-level) distillation is possible; text-only judges (Prometheus-2 7B, M-Prometheus 3B–14B, JudgeLM) and a Qwen2.5-VL-7B-based multimodal reward model (Skywork-VL-Reward-7B, MIT) exist, but the cheapest, most reliable path for this task is a programmatic checker (PAJAMA-style) with a small text judge only for the subjective residue. Cost of a 500-panel Opus gold set: roughly US$ 20–40 synchronous or US$ 10–20 via the Batch API (image ≈ 1,800 tokens/panel at 28×28 px per visual token; Opus 4.8 = $5/$25 per MTok, 50% off in batch). Recommended model: Qwen3-VL-4B-Instruct (Apache-2.0, loads under the repo's `transformers>=4.57,<5` pin; Qwen3.5 needs transformers ≥ 5.2 and is therefore out). Staged plan: rule checker → SFT QLoRA (local) → on-policy DPO (local or T4) → GRPO/GSPO with verifiable rewards (T4) → optional small judge.

---

## Pergunta

Vale a pena — e cabe no hardware disponível — destilar/alinhar um VLM pequeno para a tarefa de roteirista (diarização de falantes + descrição de ação + normalização de onomatopeias + tags SFX, tudo em JSON `PanelScript`) usando as saídas do Claude Opus como ouro? Em particular:

- (a) SFT com QLoRA: VRAM mínima, ferramentas com suporte a VLM (TRL, Unsloth, LLaMA-Factory, ms-swift), formato do dataset.
- (b) DPO/ORPO com pares de preferência (Opus vs. modelo local).
- (c) GRPO com recompensas verificáveis por regra (falante vs. gabarito, JSON válido, sem bloco vazio, onomatopeia normalizada, tag SFX válida): memória, cabe em 8 GB? Cabe no T4 16 GB do Colab/Kaggle?
- (d) Destilar um juiz LLM em um modelo pequeno de recompensa/avaliação.
- Custo de montar um dataset de ~500 painéis com o Opus. Plano concreto em etapas.

## Resumo executivo

1. **SFT QLoRA cabe nos 8 GB** para um VLM de 3B–4B. A tabela oficial da Unsloth lista **3,5 GB (3B) / 5 GB (7B) / 6 GB (8B)** como mínimos para QLoRA 4-bit (texto; VLMs gastam mais por causa dos tokens de imagem). Para caber, é obrigatório limitar `max_pixels` (Qwen-VL usa 1 token visual por bloco de 28×28 px; um painel de 1000×1400 px custa ~1.800 tokens sem limite). TRL `SFTTrainer`, Unsloth, LLaMA-Factory e ms-swift têm suporte oficial a VLM; o formato é um dataset conversacional com coluna `images` (PIL) e `content` em lista `[{"type":"image"},{"type":"text",...}]` (TRL) ou ShareGPT com placeholder `<image>` (LLaMA-Factory).
2. **DPO com imagem é suportado** por TRL (`DPOTrainer` com coluna `image`/`images`, colator `DataCollatorForVisionPreference`) e por LLaMA-Factory (`"ranking": true` + coluna `images`; entradas `rlhf_v`, `vlfeedback`, `rlaif_v` no `dataset_info.json`). **ORPO** no TRL vive em `trl.experimental.orpo` e não tem suporte a VLM — descartar. DPO com PEFT não duplica o modelo de referência (usa o policy inicial) e `precompute_ref_log_probs=True` tira o ref da memória; ainda assim, o passo processa `chosen` e `rejected` juntos, então conte ~1,5–2× a memória do SFT. Viável em 8 GB para 3B–4B com `batch=1` e `max_pixels` baixo; 7B não.
3. **GRPO com recompensa verificável é suportado** (TRL `GRPOTrainer`: funções Python puras recebem `prompts`, `completions` e **todas as colunas do dataset** como kwargs — logo uma coluna `gabarito` chega à recompensa; Unsloth: notebooks Qwen3-VL-8B e Qwen2.5-VL-7B GSPO/GRPO "em T4 gratuito do Colab", também Kaggle T4; ms-swift: modo `colocate` com `offload_model/offload_optimizer/sleep_level`). **Não cabe nos 8 GB**: a Unsloth mede 13–14,5 GiB de pico para GRPO de um Qwen3-4B (texto) no T4, e a regra oficial "VRAM em GB ≈ parâmetros em B" vale para QLoRA sem contar a geração de N amostras com imagem. Além disso vLLM (que torna o GRPO viável) **não roda nativamente no Windows**. Frameworks de pesquisa (EasyR1: 7B LoRA = 2×32 GB; VLM-R1: exemplos com 8 GPUs; ms-swift best practice: 8 GPUs) assumem 40 GB+. Conclusão: GRPO só no Colab/Kaggle T4 16 GB via Unsloth.
4. **Destilação de juiz**: a API da Anthropic **não expõe logprobs**, portanto só há destilação caixa-preta (SFT sobre saídas, preferências, RL com regra) — sem GKD/destilação de logits. Para esta tarefa, a maior parte dos critérios é verificável por programa (JSON, contagem de bolhas, falante ∈ elenco, tag SFX ∈ biblioteca, onomatopeia normalizada); o paper PAJAMA mostra juízes-programa mais consistentes que LLM-juiz e destiláveis em modelos pequenos. Juízes textuais abertos: Prometheus-2 7B (Mistral base, Apache-2.0, inglês), M-Prometheus 3B–14B (>20 idiomas, CC-BY-4.0), JudgeLM (7B–33B, >90% de concordância com o GPT-4 professor). Modelo de recompensa multimodal aberto: Skywork-VL-Reward-7B (Qwen2.5-VL-7B + value head, MIT) — 7B bf16 não cabe em 8 GB. `RewardTrainer` do TRL é só texto. Recomendação: checker programático primeiro; juiz textual pequeno (o revisor do projeto já é text-only) só para o resíduo subjetivo (vivacidade das descrições de ação).
5. **Custo do ouro (500 painéis, Opus 4.8 = US$ 5/25 por MTok)**: ~3,3k tokens de entrada por painel (prompt de sistema ~1,2–1,5k + imagem ~1,8k + OCR) e 0,8–2,5k de saída (JSON + thinking) → **≈ US$ 20–40 síncrono; ≈ US$ 10–20 com Batch API (−50%)**. Os checkpoints `workspace/<slug>/script/panels_raw.json` já gerados são ouro de custo marginal zero. Julgar 500 saídas locais com o Opus (text-only) custa ≈ US$ 6–11 a mais.
6. **Modelo recomendado: Qwen3-VL-4B-Instruct** (Apache-2.0, 256K de contexto, suportado desde transformers 4.57.0 — compatível com o pin `>=4.57,<5` do repositório). Qwen3-VL-2B (Apache-2.0, ~2,1B) é o plano B para os 8 GB; Qwen3-VL-8B (Apache-2.0, "9B") para o T4. **Qwen3.5** (nativamente multimodal, Apache-2.0) exige transformers ≥ 5.2 e conflita com o pin do coqui-tts — fora por ora. Qwen2.5-VL-3B usa **Qwen Research License (não comercial)**; 7B é Apache-2.0.
7. **Plano em etapas**: (0) checker de regras como módulo puro + coletar ouro dos checkpoints existentes; (1) SFT QLoRA local do Qwen3-VL-4B; (2) DPO on-policy (rejeitados = saídas do modelo SFT que falham no checker); (3) GRPO/GSPO no T4 com recompensa = formato + F1 de falante vs. gabarito + validade SFX + onomatopeia; (4) opcional: juiz pequeno text-only. Treinar num venv/Colab separado; no pipeline só carregar o LoRA mesclado sob transformers 4.57.

---

## Achados

### 1. Modelos candidatos (números e licenças)

| Modelo | Params | Licença | transformers mínimo | Observações |
|---|---|---|---|---|
| Qwen2.5-VL-3B-Instruct | 3B | **Qwen Research License** (`license_name: qwen-research`): "FOR NON-COMMERCIAL PURPOSES ONLY"; uso comercial exige licença; derivados devem exibir "Built with Qwen"/"Improved using Qwen" | ≥ 4.49 (card manda instalar de source para evitar `KeyError: 'qwen2_5_vl'`) | `min_pixels=256*28*28`, `max_pixels=1280*28*28` → 256–1.280 tokens por imagem |
| Qwen2.5-VL-7B-Instruct | 7B | Apache-2.0 | idem | Base do Skywork-VL-Reward-7B e do notebook GRPO da Unsloth |
| Qwen3-VL-2B-Instruct | ~2,1B (safetensors 2,13 GB bf16) | Apache-2.0 | 4.57.0 (release 2025-10-03, PR #40795) | Criado 2025-10-19 |
| Qwen3-VL-4B-Instruct | 4B (texto: hidden 2560, 36 camadas; ViT: 24 blocos, hidden 1024) | Apache-2.0 | 4.57.0 | Interleaved-MRoPE, DeepStack; 256K nativo, 1M expansível |
| Qwen3-VL-8B-Instruct | "9B" no card | Apache-2.0 | 4.57.0 | Notebooks Unsloth SFT e GSPO/GRPO "free Colab T4" |
| Qwen3.5-4B / 9B | 4B / 9B | Apache-2.0 | **≥ 5.2** (contribuído em 2026-02-09; config.json gravado com build interna "4.57.0.dev0", mas o `model_type: qwen3_5` só existe no transformers 5.x público) | Nativamente multimodal (Gated DeltaNet + MoE esparso); lançado 2026-02-16. **Incompatível com o pin `transformers<5`** exigido pelo coqui-tts 0.27.5 |

Implicação para o repositório: o pin `transformers>=4.57,<5` (memória de arquitetura, issue idiap/coqui-ai-TTS#558) cobre Qwen2.5-VL e Qwen3-VL, não Qwen3.5. Como treino e inferência podem viver em ambientes distintos, Qwen3.5 só entraria se a inferência local do VLM rodasse num processo/venv próprio — complexidade que não compensa agora.

Orçamento de tokens de imagem no Qwen-VL: 1 token por bloco 28×28 px (patch 14 + merge 2×2). Painel de 1000×1400 px em resolução nativa ≈ 36×50 = 1.800 tokens; página inteira 1600×2400 ≈ 4.900 tokens. Para caber em 8 GB, fixar `max_pixels` em algo como `640*28*28` (~500k px → ≤ 640 tokens/painel). A ms-swift precisou reduzir `MAX_PIXELS` para 262.144 (≈ 334 tokens) para evitar OOM no GRPO multimodal — o mesmo botão.

### 2. (a) SFT com QLoRA — ferramentas, VRAM, formato

**VRAM (Unsloth, tabela oficial de requisitos, QLoRA 4-bit, "mínimos, variam por modelo")**: 3B → 3,5 GB; 7B → 5 GB; 8B → 6 GB. LoRA 16-bit: 3B → 8 GB; 7B → 19 GB. LLaMA-Factory (README): 7B/8B QLoRA 4-bit ≈ 12 GB, LoRA 16-bit ≈ 16 GB (estimativa conservadora com batch maior). Para VLM some: pesos do ViT (Qwen3-VL-4B: 24 blocos × 1024) + ativações dos tokens de imagem. Leitura prática para os 8 GB: **3B–4B em 4-bit com `max_pixels` limitado, batch 1, gradient checkpointing (default no TRL) e `chunked_nll` (default no TRL) cabem; 7B/8B ficam no limite e não valem o risco no laptop** (vão para o T4).

**TRL `SFTTrainer`** ("fully supports training VLMs"): dataset com coluna `image` (uma PIL) ou `images` (lista); `processing_class` = `AutoProcessor` (carregado automaticamente do nome do modelo); `DataCollatorForVisionLanguageModeling` selecionado automaticamente; **`max_length=None` obrigatório** (truncar pode remover tokens de imagem e quebrar o treino); QLoRA = `quantization_config=BitsAndBytesConfig(...)` + `peft_config=LoraConfig(...)`; `assistant_only_loss=True` para perder só na resposta (TRL patcha o template do Qwen3 automaticamente); LR típico de adapter ≈ 1e-4. Exemplo oficial usa `Qwen/Qwen2.5-VL-3B-Instruct`. Misturar exemplos texto-only e com imagem exige transformers ≥ 4.57.0.

Formato TRL (conversacional, prompt-completion, visão):

```python
{
  "prompt": [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
             {"role": "user",   "content": [{"type": "image"},
                                            {"type": "text", "text": "Bolhas (OCR, ordem de leitura):\n1. ...\n2. ..."}]}],
  "completion": [{"role": "assistant", "content": [{"type": "text", "text": '{"blocks":[{"text":"...","speaker_id":"Guts","is_speech":true,"sfx":null}, ...]}'}]}],
  "images": [PIL.Image.open("workspace/berserk-01/panels/p008_02.png")]
}
```

**Unsloth**: `FastVisionModel` com `finetune_vision_layers` / `finetune_language_layers`; SFT de Qwen3-VL "1,7× mais rápido, 60% menos VRAM"; notebooks Qwen3-VL-8B e Qwen2.5-VL-7B "free Colab T4"; Python 3.11–3.13; GPUs CC ≥ 7.0, Blackwell RTX 50 suportado. **Windows**: Unsloth Desktop/Studio rodam nativos; o núcleo (`unsloth` pip) via Conda, Docker ou WSL; **vLLM só via WSL/Linux**; exemplos usam `dataset_num_proc=1`.

**LLaMA-Factory**: Qwen2.5-VL (template `qwen2_vl`) e Qwen3-VL (`qwen3_vl`) na tabela; formato ShareGPT/OpenAI com coluna `images` (caminhos) e um `<image>` por imagem na primeira mensagem do usuário ("número de imagens deve ser igual ao de tokens `<image>`"); guia de instalação Windows no README. v0.9.5 adiciona transformers v5 e Qwen3.5.

**ms-swift**: SFT/DPO/GRPO/RM/PPO/KTO para modelos de texto e multimodais; orientado a Linux + vLLM.

### 3. (b) DPO / ORPO com pares de preferência

- **TRL `DPOTrainer`** "fully supports training VLMs": mesmo esquema de coluna `image`/`images`; dataset de preferência `{"prompt", "chosen", "rejected", "images"}`; `max_length=None`; colator `DataCollatorForVisionPreference`; exemplo oficial com `Qwen/Qwen2.5-VL-3B-Instruct` e `HuggingFaceH4/rlaif-v_formatted`. Com `ref_model=None` o trainer usa o policy inicial como referência (com PEFT = adapters desligados, sem segunda cópia); `precompute_ref_log_probs=True` calcula os logprobs de referência antes e libera memória (incompatível com `sync_ref_model` e Liger). Suporta perdas combinadas — o exemplo documentado é o **MPO** (`loss_type=["sigmoid","bco_pair","sft"]`, `loss_weights=[0.8,0.2,1.0]`), proposto justamente para VLMs. LR típico com adapters ≈ 1e-5; `beta` default 0,1.
- **LLaMA-Factory**: preferência multimodal = `"ranking": true` + `chosen`/`rejected` + `"images"`; existem entradas prontas com essa forma (`rlhf_v`, `vlfeedback`, `rlaif_v`) no `dataset_info.json`; DPO/KTO/ORPO/SimPO listados. Issue #8692 (formato DPO Qwen2.5-VL) está marcado "solved".
- **ORPO no TRL**: `trl.experimental.orpo.ORPOTrainer` — experimental, sem menção a VLM/imagens; o exemplo oficial usa 8 GPUs. Descartar em favor do DPO (ou do MPO via `DPOTrainer`).
- Memória: cada passo processa `chosen` + `rejected` (sequências com imagem repetida) → ~1,5–2× o SFT. Em 8 GB: 3B–4B em 4-bit, batch 1, `max_pixels` baixo, `precompute_ref_log_probs`; 7B → T4.
- Como gerar os pares sem custo extra de Opus: `chosen` = saída do Opus (já no checkpoint); `rejected` = saída do **modelo local pós-SFT** que **falha no checker** (bloco vazio, falante fora do elenco, SFX inválido, onomatopeia crua, contagem de bolhas errada). Pares on-policy tendem a ensinar mais que pares contra o modelo base.

### 4. (c) GRPO com recompensas verificáveis — memória e onde rodar

**Suporte**:
- **TRL `GRPOTrainer`**: testado com Gemma3, LLaVA-NeXT, Qwen2-VL, Qwen2.5-VL-3B, SmolVLM2 ("compatibilidade com todos os VLMs não é garantida"). Dataset prompt-only + coluna `image`/`images`. Funções de recompensa = callables Python (sync ou async) que recebem `prompts`, `completions`, `completion_ids`, `trainer_state` **e todas as colunas do dataset como kwargs** → uma coluna `gabarito` (falantes por bolha, nº de bolhas, tags permitidas) chega direto à recompensa; podem devolver `None` quando não se aplicam. Defaults: `num_generations=8`, `max_completion_length=512`. vLLM opcional (`use_vllm`, `vllm_mode="colocate"` com `vllm_gpu_memory_utilization=0.3`, `vllm_enable_sleep_mode`); dicas oficiais: LoRA + 4-bit, batch pequeno, ZeRO-3 para VLMs grandes. O exemplo oficial (`grpo_visual_math`) lança com DeepSpeed ZeRO-3, PEFT em `q_proj,v_proj`, `max_completion_length 1024`, `--use_vllm --vllm_mode colocate`; o cookbook correspondente usa **A100**.
- **Unsloth**: "Qwen3-VL-8B com GSPO/GRPO num T4 gratuito do Colab"; notebooks Qwen3-VL-8B-GRPO (T4), Qwen2.5-VL-7B-GRPO (T4 e Kaggle T4), Gemma3-4B-GRPO (T4; Gemma prefere L4 por bf16 no vLLM). Parâmetros reais do notebook Qwen2.5-VL-7B: `load_in_4bit=True`, `fast_inference=True` (vLLM), `gpu_memory_utilization=0.8`, `max_seq_length=16384`, LoRA r=16/alpha=16, **`finetune_vision_layers=False`** (vLLM não suporta LoRA nas camadas de visão), `finetune_language_layers=True`, `num_generations=4`, `max_completion_length=1024`, batch 1 × acum. 2, LR 5e-6, `importance_sampling_level="sequence"` (GSPO), 0,5 época; recompensas = `formatting_reward_func` + `correctness_reward_func`. Requisito: vLLM ⇒ Linux/WSL.
- **ms-swift**: GRPO com modo `colocate` (treino e inferência na mesma GPU), `--offload_model/--offload_optimizer`, `--sleep_level 1`, `--vllm_gpu_memory_utilization`, LoRA (`--vllm_enable_lora`), "Multimodal ViT LoRA Sync" para Qwen2.5-VL/Qwen3-VL, plugin de recompensas externas (`external_r1v_acc`, `format`). O best-practice multimodal usou Qwen2.5-VL-3B **full-parameter em 8 GPUs** (6 treino + 2 vLLM), `num_generations` 8–24, LR 1e-6, `beta` 0,001, `MAX_PIXELS=262144` por OOM.
- **EasyR1** (GRPO/DAPO/GSPO; Qwen2/2.5/3-VL): 7B full BF16 = 4×40 GB; **7B LoRA = 2×32 GB**; issue #450: Qwen2.5-VL-3B deu OOM num L20 de 48 GB apesar de "~40 GB" documentados. **VLM-R1**: exemplos `torchrun --nproc_per_node=8`, LoRA disponível. Não são opção para 8–16 GB.

**Memória medida (Unsloth, doc "Memory Efficient RL")**: Qwen3-4B (texto) GRPO no T4: **14,5 GiB de pico** com Standby e `gpu_memory_utilization=0.95`; ~13 GiB a 0,7 com Standby; 15,1 GiB sem Standby. Guia de RL: "5 GB bastam para modelos ≤ 1,5B"; "regra geral para QLoRA 4-bit: parâmetros (B) ≈ VRAM (GB)". Um VLM de 4B com 4–8 gerações de ~300–500 tokens sobre ~500 tokens de imagem fica **acima de 8 GB**; sem vLLM (Windows nativo), a geração por `transformers.generate` é lenta demais para GRPO prático.

**Veredito**: GRPO/GSPO → **Colab ou Kaggle T4 16 GB via Unsloth** (Qwen3-VL-8B ou 4B em 4-bit). Não no laptop de 8 GB. Se quiser insistir localmente: WSL2 + vLLM + Qwen3-VL-2B, `num_generations=4`, `max_completion_length≈400`, `max_pixels≈256*28*28` — sem dado primário que garanta caber.

**Desenho das recompensas verificáveis (todas puras, sem modelo)**: `json_valido` (parse `PanelScript` via Pydantic) → 0/1 gate; `sem_bloco_vazio` (já é `min_length=1` no schema; penalizar `text` só com espaços/pontuação); `contagem_bolhas` (nº de blocos `is_speech=true` == nº de bolhas OCR; evita o hack de "sumir" com bolhas difíceis); `falante_f1` (F1 posicional entre `speaker_id` das falas e o gabarito do Opus; rótulos fora do elenco ∪ rótulos descritivos permitidos = 0); `onomatopeia` (regex: sem 3+ letras repetidas, sem CAIXA-ALTA integral, sem tokens de SFX escrito — "CRASH", "VROOM" — em falas); `sfx_valido` (`sfx ∈ tags` ou `None`; ≤ N por painel conforme `sfx_intensity`); `pt_br` (detector de idioma sobre os textos); `acao_curta` (blocos `Narrator` ≤ 2 por painel, ≤ ~40 palavras). Formato + acurácia é exatamente o par usado por VLM-R1/ms-swift.

### 5. (d) Juiz LLM destilado em modelo pequeno

- **Limite estrutural**: a API Messages da Anthropic não retorna logprobs (só blocos `text`/`thinking`/`tool_use`); a destilação é caixa-preta — SFT sobre saídas, preferências, RL com verificador. Sem GKD/logit matching contra o Opus.
- **Programa antes de modelo**: PAJAMA (arXiv 2506.10403) — LLM sintetiza programas juízes executáveis; +15,8% de consistência vs. Qwen2.5-14B como juiz, −23,7% de respostas enviesadas, custo ~3 ordens de grandeza menor; **destilado num modelo pequeno**, bate LLM-juiz no CHAT-HARD do RewardBench (+2,19% vs. Prometheus, +8,67% vs. JudgeLM). Para o roteirista, 80–90% dos critérios já são programáveis (seção 4).
- **Juízes textuais abertos**: Prometheus-2 7B (`prometheus-eval/prometheus-7b-v2.0`, base Mistral-7B-Instruct-v0.2, Apache-2.0, 100k Feedback Collection + 200k Preference Collection, avaliação direta 1–5 e pareada, **inglês**); **M-Prometheus 3B–14B** (>20 idiomas, dados sintéticos multilíngues, CC-BY-4.0, modelos + dados + código liberados) — candidato natural para PT-BR; JudgeLM 7B/13B/33B (treinado em vereditos do GPT-4; concordância com o professor > 90%, acima da concordância humano-humano; mitiga viés de posição via swap augmentation, de conhecimento via referência, de formato via reference drop).
- **Modelo de recompensa multimodal**: Skywork-VL-Reward-7B (Qwen2.5-VL-7B-Instruct + value head, MIT; VL-RewardBench 73,1; RewardBench 90,1). 7B em bf16 ≈ 16 GB → só em 4-bit e só no T4. `trl.RewardTrainer` (Bradley-Terry, `AutoModelForSequenceClassification`, `modules_to_save=["score"]`, LR ≈ 1e-3 com LoRA) é **text-only** (`processing_class` = tokenizer).
- **Encaixe no projeto**: o Reviewer Agent já é text-only (chunks de 40 painéis). Um juiz text-only pequeno (Qwen3-1.7B/4B como classificador, ou M-Prometheus-3B) treinado em vereditos do Opus sobre `(OCR, script)` cabe em 8 GB e cobre o resíduo subjetivo (vivacidade/adequação da descrição de ação). Volume: JudgeLM usou ~100k amostras; para um domínio estreito, 1–3k vereditos (≈ US$ 20–60 no Opus) é o ponto de partida; validar com 100 painéis anotados à mão.

### 6. Custo do dataset de ~500 painéis com o Opus

Preços (docs oficiais, set/2026): Claude Opus 4.8 (modelo default em `engines/vlm.py`) e Opus 5 = **US$ 5 / MTok entrada, US$ 25 / MTok saída**; Batch API **−50%** (US$ 2,50 / 12,50); cache: escrita 5 min 1,25×, leitura 0,1× (US$ 0,50); descontos de batch e cache se somam. Tokenizador do 4.7+ gera ~30% mais tokens que o do 4.6.

Tokens de imagem: `⌈w/28⌉ × ⌈h/28⌉`; modelos 4.7+ (tier high-res) aceitam até 2.576 px de lado e 4.784 tokens antes de reduzir; 1000×1000 = 1.296 tokens (US$ 6,48 por mil imagens no Opus 5). Um recorte de painel típico (~1000×1400) ≈ 1.800 tokens; página inteira ≈ 4.784 (teto).

Estimativa por painel (engine atual: 1 imagem + prompt de sistema `build_scriptwriter_prompt` ≈ 700 palavras ≈ 1,2–1,5k tokens + esquema `PanelScript` + OCR ~150 tokens; `max_tokens=8192` com thinking):

| Item | Tokens/painel | 500 painéis | Síncrono | Batch |
|---|---|---|---|---|
| Entrada (sistema + imagem + OCR) | ~3.300 | 1,65M | US$ 8,3 | US$ 4,1 |
| Saída (JSON ~300 + thinking 0,5–2,2k) | 800–2.500 | 0,4–1,25M | US$ 10–31 | US$ 5–16 |
| **Total** | | | **≈ US$ 18–40** | **≈ US$ 9–20** |

Notas: (i) o prompt de sistema só é cacheável se ultrapassar o mínimo do modelo (512–4.096 tokens conforme o modelo) — conferir `usage.cache_read_input_tokens`; o ganho é pequeno porque a imagem domina a entrada; (ii) reduzir `effort`/thinking na coleta de ouro corta a maior parcela (saída); (iii) 2 amostras do Opus por painel (para filtrar por auto-consistência de falante) dobram o custo; (iv) julgar 500 saídas locais com o Opus em modo texto (~2,5k entrada / ~400 saída) ≈ US$ 6–11 síncrono; (v) **custo marginal zero**: cada volume já narrado deixa `script/panels_raw.json` (Opus bruto) e `panels.json` (pós-revisor) — 500 painéis ≈ 100 páginas de Berserk ≈ metade de um volume já processado.

### 7. Hardware e ambiente

- **RTX 5060 Laptop 8 GB (Blackwell sm_120) no Windows**: as wheels oficiais do bitsandbytes para Windows x86-64 com CUDA 12.8 têm alvos `sm70…sm120` (idem CUDA 13.0–13.2), e NF4/FP4 exigem CC ≥ 6.0 — ou seja, QLoRA deve funcionar com o `torch 2.11.0+cu128` do projeto. Contra-evidência: issue #1937 (maio/2026) relata "no kernel image is available" numa 5070 Ti/Windows com driver CUDA 13.2 — aberta, sem resposta do mantenedor. Ação: smoke test de `BitsAndBytesConfig(load_in_4bit=True)` antes de planejar.
- **Windows sem vLLM**: Unsloth core via Conda/WSL; vLLM só WSL/Linux → SFT/DPO local ok; GRPO local não.
- **Colab gratuito**: T4 16 GB (≈15 GiB utilizáveis — o pico de 14,5 GiB da Unsloth foi medido nele); Google não publica limites ("flutuam"); sessões até 12 h; GPU "fortemente restrita" para não pagantes. **T4 não tem bf16** (CC 7.5; bf16 exige CC ≥ 8.0) → treinar em fp16 (os notebooks Unsloth já lidam com isso).
- **Kaggle** (fontes secundárias — a página de docs é renderizada em JS e não pôde ser lida): ~30 h/semana de GPU, T4×2 ou P100 16 GB, sessões de ~9 h; a Unsloth publica um notebook GRPO Qwen2.5-VL-7B específico para Kaggle T4.

---

## Recomendação

**Sim, é viável e barato — mas por etapas, com o GRPO fora do laptop.** Modelo alvo: **Qwen3-VL-4B-Instruct** (Apache-2.0; carrega sob o pin atual do transformers; 2B como reserva para os 8 GB; 8B no T4).

| Etapa | O quê | Onde | Ferramenta | Critério de saída |
|---|---|---|---|---|
| 0 | `checker` puro (seção 4) como módulo do pacote; script que exporta `panels_raw.json` + imagens de painel + OCR para JSONL no formato TRL; conferência humana de ~50 painéis do "ouro" (o Opus também erra falante) | laptop, US$ 0 | Pydantic, regex, `datasets` | Métrica de referência: acurácia de falante do Opus vs. humano; do Qwen3-VL-4B zero-shot vs. Opus |
| 1 | **SFT QLoRA** do Qwen3-VL-4B: 4-bit NF4, LoRA r=16 (só LLM; ViT congelado), `max_length=None`, `max_pixels≈640*28*28`, batch 1 × acum. 8, 1–2 épocas, LR 1e-4, `assistant_only_loss` | laptop 8 GB (WSL2 ou Conda) ou Colab T4 | TRL `SFTTrainer` ou Unsloth `FastVisionModel` | F1 de falante ≥ 90% do Opus no held-out (100 painéis); 100% JSON válido |
| 2 | **DPO (ou MPO)** on-policy: `chosen` = Opus, `rejected` = saída do modelo SFT reprovada no checker; `precompute_ref_log_probs=True`, `beta` 0,1, LR 1e-5 | laptop (4B) ou T4 | TRL `DPOTrainer` (ou LLaMA-Factory `ranking:true`+`images`) | Taxa de reprovação no checker < 2%; F1 sobe |
| 3 | **GRPO/GSPO** com recompensas verificáveis: `num_generations=4`, `max_completion_length≈512`, `importance_sampling_level="sequence"`, 4-bit + vLLM standby, `finetune_vision_layers=False` | Colab/Kaggle T4 16 GB | Unsloth (notebook Qwen3-VL-8B-GRPO adaptado) | Recompensa média ↑ sem cair contagem de bolhas; F1 vs. gabarito ≥ SFT+DPO |
| 4 | (opcional) juiz text-only pequeno para o resíduo subjetivo, treinado em 1–3k vereditos do Opus; validar em 100 painéis humanos | laptop | TRL `RewardTrainer` (Bradley-Terry) ou SFT de M-Prometheus-3B | Concordância com Opus ≥ 85% no held-out |

Integração no pipeline: exportar o LoRA mesclado e carregá-lo pelo `QwenVisionLanguageEngine` existente (`engines/vlm_local.py`) sob transformers 4.57; a assinatura de checkpoint (`run_config.json` + sha1 do prompt) já invalida scripts antigos. Treino sempre num venv/Colab separado, porque o TRL `main` já assume transformers 5.x.

Ordem de decisão: se a etapa 1 já entregar F1 ≥ 95% do Opus, pare — o custo/benefício do RL é baixo para uma saída curta e estruturada. GRPO só se o DPO estacionar em erros que o checker detecta (contagem de bolhas, SFX inválido, onomatopeia).

## Riscos e incertezas

1. **Ouro ≠ verdade**: o gabarito de falante é do Opus (o próprio usuário relatou que o Haiku "não funcionou"; o Opus ignorou a regra de SFX até o prompt ser reescrito). Sem amostra anotada por humano, F1 vs. Opus mede imitação, não acerto. Mitigar com 50–100 painéis anotados.
2. **Dependências**: TRL `main` documenta comportamento "since Transformers v5"; sob o pin 4.57 é preciso fixar uma release do TRL de fim de 2025 — ou treinar em ambiente separado (recomendado). LLaMA-Factory v0.9.5 também migrou para transformers v5.
3. **bitsandbytes em Blackwell/Windows**: wheels oficiais com sm_120 existem, mas há relato aberto de falha (#1937). Smoke test obrigatório; plano B = WSL2.
4. **Sem vLLM no Windows nativo** → GRPO local inviável; Colab gratuito não garante GPU e não publica limites; Kaggle: números de quota vêm de fontes secundárias.
5. **Memória com imagem**: todos os mínimos de VRAM citados são de modelos de texto; a única medida VLM-GRPO primária é "roda no T4" (Unsloth). Nada garante 4B-VLM GRPO em 8 GB.
6. **Reward hacking no GRPO**: modelo pode omitir bolhas difíceis ou inventar rótulos genéricos ("Desconhecido") para maximizar recompensa — por isso `contagem_bolhas` e penalidade a "Desconhecido" acima de uma taxa.
7. **Dataset pequeno (500)**: overfitting a Berserk/vol. 1; validar em páginas de outro arco/volume; LoRA de rank baixo, ≤ 2 épocas.
8. **Licenças**: Qwen2.5-VL-3B é não comercial (aceitável pelo ADR-0002, mas obriga "Built with Qwen"); Qwen3-VL/Qwen3.5 são Apache-2.0. Skywork-VL-Reward é MIT; Prometheus-2 Apache-2.0; M-Prometheus CC-BY-4.0.
9. **Qwen3.5** fica de fora enquanto o coqui-tts prender `transformers<5` (idiap/coqui-ai-TTS#560 não mesclado).
10. **Custos**: estimativas assumem ~1.800 tokens/imagem e thinking moderado; `effort` alto ou páginas inteiras (4.784 tokens) podem dobrar a conta. Medir com `messages.count_tokens` antes de rodar em lote.

## Fontes

Modelos e licenças
- https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct e https://huggingface.co/api/models/Qwen/Qwen2.5-VL-3B-Instruct (license_name `qwen-research`)
- https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/LICENSE (Qwen Research License)
- https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct (apache-2.0; `min_pixels`/`max_pixels`)
- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct · https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct · https://huggingface.co/api/models/Qwen/Qwen3-VL-2B-Instruct
- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/raw/main/config.json (arquitetura, transformers_version)
- https://github.com/QwenLM/Qwen3-VL (datas de lançamento)
- https://github.com/huggingface/transformers/releases/tag/v4.57.0 (suporte Qwen3-VL, 2025-10-03)
- https://huggingface.co/Qwen/Qwen3.5-4B · https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json · https://github.com/QwenLM/Qwen3.5 (lançamento 2026-02-16)
- https://huggingface.co/docs/transformers/model_doc/qwen3_5 · https://github.com/NVIDIA/TensorRT-LLM/issues/12321 (Qwen3.5 ⇒ transformers ≥ 5.2)

Ferramentas de treino
- https://huggingface.co/docs/trl/main/en/sft_trainer (VLM, `max_length=None`, QLoRA, `assistant_only_loss`)
- https://huggingface.co/docs/trl/main/en/dpo_trainer (VLM, `precompute_ref_log_probs`, MPO)
- https://huggingface.co/docs/trl/main/en/grpo_trainer (VLM, assinatura das recompensas, vLLM colocate)
- https://huggingface.co/docs/trl/main/en/orpo_trainer (experimental, sem VLM)
- https://huggingface.co/docs/trl/main/en/reward_trainer (text-only, Bradley-Terry)
- https://huggingface.co/docs/trl/main/en/dataset_formats (seção "Vision datasets")
- https://huggingface.co/learn/cookbook/fine_tuning_vlm_grpo_trl (GRPO VLM em A100)
- https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements (tabela de VRAM, GPUs, Python)
- https://unsloth.ai/docs/get-started/install/windows-installation (Conda/Docker/WSL; vLLM só WSL)
- https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide (5 GB para ≤1,5B; regra params≈VRAM)
- https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl (picos no T4)
- https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/vision-reinforcement-learning-vlm-rl (Qwen3-VL-8B GRPO em T4; `finetune_vision_layers=False`)
- https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune
- https://unsloth.ai/docs/get-started/unsloth-notebooks (lista de notebooks VLM, T4/Kaggle)
- https://raw.githubusercontent.com/unslothai/notebooks/main/nb/Qwen2_5_7B_VL_GRPO.ipynb (hiperparâmetros reais)
- https://github.com/hiyouga/LLaMA-Factory (métodos, tabela de modelos, VRAM, Windows)
- https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md e https://raw.githubusercontent.com/hiyouga/LLaMA-Factory/main/data/dataset_info.json (`rlhf_v`, `vlfeedback`, `rlaif_v`)
- https://github.com/hiyouga/LlamaFactory/issues/8692 · https://github.com/hiyouga/LlamaFactory/releases/tag/v0.9.5
- https://swift.readthedocs.io/en/latest/Instruction/GRPO/GetStarted/GRPO.html · https://swift.readthedocs.io/en/latest/BestPractices/GRPO-Multi-Modal-Training.html
- https://github.com/hiyouga/EasyR1 (tabela de hardware) · https://github.com/hiyouga/EasyR1/issues/450
- https://github.com/om-ai-lab/VLM-R1

Juízes e modelos de recompensa
- https://arxiv.org/abs/2506.10403 (PAJAMA)
- https://arxiv.org/abs/2405.01535 · https://huggingface.co/prometheus-eval/prometheus-7b-v2.0
- https://arxiv.org/abs/2504.04953 (M-Prometheus)
- https://arxiv.org/abs/2310.17631 (JudgeLM)
- https://huggingface.co/Skywork/Skywork-VL-Reward-7B · https://arxiv.org/abs/2505.07263
- https://github.com/anerli/anthropic-logprobs (ausência de logprobs na API)

Custo, hardware e plataformas
- https://platform.claude.com/docs/en/about-claude/pricing (Opus 4.8/5, Batch, cache, tokenizador)
- https://platform.claude.com/docs/en/build-with-claude/vision (28×28 px por token, tiers de resolução, limites)
- https://huggingface.co/docs/bitsandbytes/main/en/installation (wheels Windows CUDA 12.8 com sm120; CC mínima)
- https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1937 (falha relatada em Blackwell/Windows)
- https://research.google.com/colaboratory/faq.html (limites não publicados, 12 h)
- https://github.com/vllm-project/vllm/issues/1157 (T4 CC 7.5 sem bf16)
- https://www.kaggle.com/general/108481 · https://www.kaggle.com/datasets/headsortails/kaggle-weekly-gpu-quotas (quota Kaggle — secundárias)

Código do repositório consultado
- `mangawhisperer/engines/vlm.py` (prompt de sistema, `PanelScript`, modelo default `claude-opus-4-8`, `max_tokens=8192`)
- `mangawhisperer/models.py` (`ContextualizedBlock`: `text`, `speaker_id`, `is_speech`, `sfx`)
