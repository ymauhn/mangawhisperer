# R5 — Identificação de personagens no painel sem LLM ("speaker hints" a custo zero de tokens)

**Data:** 2026-09-02 · **Ticket:** R5 do mapa Wayfinder · **Status:** concluído (pesquisa; nada implementado)

## English abstract

Can we tell *which* cast members appear in a manga panel using only classical/small models, and hand that to the VLM as a cheap speaker hint? Yes, with caveats. The manga-specific literature (Manga109 re-ID, Magi v1/v2, Manga109Dialog, CoMix) converges on a **detector → crop embedding → gallery match** pattern. Manga-trained parts exist off the shelf: `deepghs/manga109_yolo` (body/face/frame/text, F1 0.88–0.92, 2.6–68 M params) and `ragavsachdeva/magiv2-crop-embedder` (85.8 M params, supervised-contrastive on manga). With roughly **one exemplar per character**, Magiv2 names principal characters with **0.73–0.75 accuracy** chapter-wide (0.80–0.85 with perfect page constraints, ~0.87 with well-chosen exemplars); generic encoders (DINOv2/CLIP) are markedly weaker on comics (CoMix AMI 0.29). Fully zero-shot naming (no exemplars, names mined from dialogue) tops out at ~35–44 %. For injecting hints, AutoAD-Zero's ablation shows that a **plain text list of who is present** captures most of the gain; overlaying coloured circles/marks on the image (Set-of-Mark style) adds a further boost in harder scenes. Recommended minimal pipeline: manga109_yolo-s body boxes → Magiv2 crop embedder → cosine match against 3–10 manga-sourced exemplars per cast member with an "unknown" threshold → advisory hint block in the VLM user message (optionally with numbered marks on the panel). Expected reliability: ~70–80 % correct names on the principal cast, lower for look-alike extras; all components fit in well under 1 GB of VRAM and must be released before XTTS. All Magi weights are non-commercial (acceptable under ADR-0002).

---

## Pergunta

Dá para identificar **quais** personagens aparecem num painel de mangá **sem** usar um LLM, e entregar isso ao VLM roteirista como "dica de falante" que não custa tokens de raciocínio? O que existe de re-identificação de personagens em mangá (benchmarks Manga109, clustering do Magi, reconhecimento de rosto de anime), quais modelos de embedding servem para recortes de mangá (CLIP, SigLIP, DINOv2, específicos de anime), quais detectores de rosto/corpo funcionam em mangá, que acurácia esperar com poucas imagens de referência por personagem, e como alimentar as dicas no prompt do VLM? Recomendar um pipeline mínimo (detector + embedding + galeria) e sua confiabilidade esperada.

## Resumo executivo

1. **Sim, é viável e a literatura converge no mesmo padrão**: detector de personagem → embedding do recorte → comparação com uma galeria de exemplares nomeados. É exatamente o que o Magiv2 faz para nomear personagens ao longo de um capítulo (Sachdeva, Shin & Zisserman, 2024), e o que o CoMix e o "accessible comics" (2024) usam como baseline.
2. **Peças prontas, treinadas em mangá, existem**: `deepghs/manga109_yolo` (classes body/face/frame/text, treinado em Manga109-s; variante *s* com 9,4 M parâmetros, F1 0,90, mAP50 0,938) e `ragavsachdeva/magiv2-crop-embedder` (ViT de 85,8 M parâmetros, treinado com Supervised Contrastive Loss em recortes de mangá). Ambos cabem folgadamente na RTX 5060 de 8 GB (menos de 1 GB somados).
3. **Acurácia esperada com poucas referências**: o banco de personagens do Magiv2 tem ≈1 exemplar por personagem (11,5 K imagens para >11 K personagens) e obtém **0,727–0,753 de acurácia de nomeação** com restrições preditas, 0,799–0,853 com restrições perfeitas, e ~0,87 quando o exemplar é bem escolhido (vs ~0,73 com exemplar aleatório). Sem nenhum exemplar (zero-shot puro, nomes minerados do diálogo por GPT-4), a identificação cai para 35–44 %. Embeddings genéricos (DINOv2, CLIP) são bem piores fora do domínio: AMI 0,29 no CoMix, e features ResNet50-ImageNet dão só 2,8 % de rank-1 em corpos do Manga109.
4. **Rosto vs corpo**: em mangá, rosto é mais discriminativo (rank-1 78,4 % vs 40,3 % para corpo no FSAC), mas corpo cobre costas, capacetes e criaturas — o Magi e o zero-shot da ACM MM 2024 usam **corpo**. Recomendação: detectar corpo, e usar rosto como sinal auxiliar quando houver.
5. **Como injetar a dica**: a ablação do AutoAD-Zero mostra que **uma lista textual "quem está presente"** captura quase todo o ganho (CRITIC 0,1 → 27,4 no TV-AD); marcar a imagem com círculos coloridos + legenda "Nome (cor)" acrescenta mais em cenas difíceis (CMD-AD 34,7 → 43,7). O padrão Set-of-Mark (marcas numeradas sobre a imagem) é agnóstico ao provedor — funciona com Claude, Qwen API e GPT sem depender de tokens especiais de grounding.
6. **Confiabilidade realista** do pipeline mínimo (sem otimização global por capítulo): ~70–80 % de nomes corretos para o elenco principal com 3–10 exemplares extraídos do próprio mangá; menos para figurantes parecidos (soldados, criaturas). Por isso a dica deve ser **consultiva** ("provável Guts, confiança 0,82"), nunca imposta; o VLM continua dono da atribuição.
7. **Licenças**: todos os pesos Magi são "uso pessoal, pesquisa, não-comercial" — aceitável pelo ADR-0002. DINOv2 é Apache-2.0, SigLIP2 Apache-2.0, CLIP MIT, CCIP OpenRAIL. O card do `manga109_yolo` **não declara licença** (risco a registrar). O dataset Manga109 em si é acadêmico (não precisamos dele; só dos modelos derivados).

## Achados

### A. O que a literatura de mangá já resolveu

#### A.1 Manga109 e os benchmarks derivados

- **Manga109**: 109 volumes, 21 142 páginas, anotações de frames, textos, rostos e corpos (Aizawa et al.). As anotações somam **118 715 rostos, 157 152 corpos, 103 900 frames e 147 918 textos** (Ogawa et al., 2018, Tabela 1).
- **Termos de uso**: instituições acadêmicas podem usar o Manga109 completo; organizações não-acadêmicas só o **Manga109-s** (87 títulos). Publicar modelos pré-treinados é permitido desde que o uso do dataset seja indicado; redistribuir imagens é proibido; o pedido de acesso exige e-mail acadêmico (card HF `hal-utokyo/Manga109-s`). Para o MangaWhisperer isso é irrelevante na prática: usaremos apenas **modelos** treinados nele, não as imagens.
- **Detecção histórica (Ogawa 2018, SSD300)**: AP corpo 79,1 %, rosto 67,1 %, frame 97,1 %, texto 82,0 % — rosto é a classe mais difícil de detectar em mangá (linhas simplificadas, expressões exageradas).
- **Manga109Dialog** (Li, Hinami, Aizawa, Matsui — ICME 2024): 132 692 pares falante→texto; método por *scene graph generation* atinge >75 % de acurácia de atribuição de falante. Confirma que atribuir a fala **geometricamente** (balão → personagem mais próximo/cauda) é um problema tratável sem LLM, mas com teto ~75–85 %.

#### A.2 Re-identificação não supervisionada (FSAC, Zhang & Wang, 2022)

Protocolo de re-ID no Manga109: 186 817 imagens, **1 506 identidades**; recortes de rosto 112×112 e corpo 128×256; backbone ResNet50.

| Configuração | mAP | Rank-1 | Rank-5 |
|---|---|---|---|
| FSAC rosto | 32,2 % | **78,4 %** | 88,1 % |
| FSAC corpo | 10,2 % | 40,3 % | 66,8 % |
| FSAC rosto+corpo | 9,8 % | 52,8 % | 68,6 % |
| ResNet50-ImageNet rosto (sem adaptação) | 7,1 % | 22,5 % | — |
| ResNet50-ImageNet corpo (sem adaptação) | 0,6 % | **2,8 %** | — |

Duas lições: (1) **features genéricas de ImageNet são quase inúteis** em corpos de mangá; (2) rosto é o sinal mais forte quando visível. O trabalho seguinte OAM-ReID (MMAsia 2023) adiciona oclusão sintética por balões e corpos incompletos e supera o estado da arte de person-ReID, mas as métricas não estavam acessíveis (ACM 403).

#### A.3 Magi v1 — "The Manga Whisperer" (Sachdeva & Zisserman, 2024)

- Arquitetura: ResNet-50 + DETR condicional (6+6 camadas, 300 queries), entrada redimensionada para lado curto 800 px / longo ≤1333 px. Módulo de embedding de recortes: ViT de 12 camadas × 768, inicializado com MAE, treinado com Supervised Contrastive Loss por página; limiar de cluster τ = 0,65.
- Detecção (AP@0,5): personagens 0,849/0,862 (PopManga Test-S/U); no Manga109: corpo **0,9015**, painel 0,9357.
- Clustering de personagens por página (AMI): PopManga 0,657/0,653; **Manga109 rosto 0,625, corpo 0,635**.
- Associação texto→falante (Recall@#text): 0,845/0,831.
- Pesos: 0,5 B parâmetros, F32. Treino em 2×A40. Licença não-comercial.

#### A.4 Magi v2 — "Tails Tell Tales" (Sachdeva, Shin & Zisserman, 2024)

O trabalho mais próximo do que o ticket pede: nomeação de personagens ao longo de um capítulo a partir de um **banco de personagens** (`{"images": [...], "names": [...]}`).

- Banco **PopCharacters**: >11 K personagens de 76 séries com **11,5 K imagens** — ou seja, ≈**1 exemplar por personagem**, com exemplares extras minerados do próprio mangá para os personagens frequentes (miniaturas da web vinham do anime, com *distribution shift*).
- Nomeação = comparação de embeddings com o banco + otimização por **MILP** com restrições *must-link/cannot-link* vindas do clustering por página; personagem (k+1) "outro" para quem fica abaixo do limiar de outlier **η = 0,75**.
- **Acurácia de nomeação (Tabela 4)**: 0,7273 (Test-S) / 0,7530 (Test-U) com restrições preditas; 0,7987 / 0,8526 com restrições reais; baselines K-means 0,38–0,51. **Escolha do exemplar importa**: ~0,87 com exemplar "ótimo" vs ~0,73 com exemplar fixo aleatório (Tabela 6).
- Clustering por página (AMI): 0,6745 / 0,6650; Manga109 0,6456. Diarização (AP): 0,7499 / 0,7512 (Magi v1: 0,52–0,56). Detecção de personagens mAP 0,854 / 0,872.
- Limitação explícita: figurantes viram um único "other"; não distingue "desconhecido 1" de "desconhecido 2".
- Artefatos: `ragavsachdeva/magiv2` (pytorch_model.bin **2,06 GB** F32; API `do_chapter_wide_prediction(pages, character_bank, use_tqdm, do_ocr)`, saída com `character_names` e `text_character_associations`; lê páginas como `convert("L").convert("RGB")`) e o embedder isolado `ragavsachdeva/magiv2-crop-embedder` (**85,8 M parâmetros**, `AutoModel.from_pretrained(..., trust_remote_code=True)`). Licença: "unrestricted use in personal, research, non-commercial, and not-for-profit endeavors".
- **Magi v3** (2025, `ragavsachdeva/magiv3`): 0,8 B parâmetros, F16, base Florence-2, gera texto autoregressivo para detecção/OCR/grounding de personagens em legendas; o card não documenta nomeação por banco. Menos útil para este ticket que o v2.

#### A.5 Zero-shot puro — "Iterative Multimodal Fusion" (Li et al., ACM MM 2024)

Sem nenhum exemplar: classificador ResNet50 sobre **corpos 270×270** (escolha deliberada: cobre não-humanos e costas), nomes extraídos do diálogo por GPT-4, pseudo-rótulos propagados e refinados em até 3 iterações. Resultados no Manga109 (23 volumes de teste): identificação de personagens **35–44 %**, predição de falante 38–52 %; baselines K-means+distância 33–37 %; teto com relações reais 54 % / 60 %. Código MIT (`liyingxuan1012/zeroshot-speaker-prediction`), exige API GPT-4. Conclusão: **sem exemplares a confiabilidade é baixa demais para servir de dica**; o custo de montar 3–10 recortes por membro do elenco é pequeno e vale a pena.

#### A.6 Encoders genéricos fora do domínio — CoMix (NeurIPS 2024 D&B)

Clustering de personagens por página com caixas reais, em mangá + quadrinhos ocidentais: **DINOv2 AMI 0,29 / NMI 0,51** foi o melhor baseline e superou o Magi *fine-tuned* fora do domínio de mangá ("Magi is not able to retain recognition performances out of its manga domain"). Comparado ao AMI ≈0,63–0,67 do Magi **dentro** do domínio (A.3/A.4), o recado é: encoder treinado em mangá para mangá; DINOv2 como reserva *license-clean*.

#### A.7 Precedente de acessibilidade — "Toward accessible comics for BLV readers" (2024)

Pipeline: CLIP ViT-L/14 → UMAP(5) → HDBSCAN(min 15) para IDs temporários (c0, c1…), nomes inferidos por LLM a partir do roteiro, e **nomes gravados na imagem** (retângulo branco, fonte comum) antes de passar o painel ao VLM, com o prompt "describe this panel using character's names written in white rectangles". Associação balão→personagem 77,7–88,9 %. Não reporta acurácia de nomeação.

### B. Detectores de personagem/rosto utilizáveis

| Modelo | Domínio de treino | Classes | Tamanho | Métricas | Licença |
|---|---|---|---|---|---|
| `deepghs/manga109_yolo` v2023.12.07 (YOLOv8 e YOLO11) | **Manga109-s** (P&B) | body, face, frame, text | n 2,59 M / 6,44 GFLOPs; **s 9,43 M / 21,6 G**; m 20,1 M; l 25,3 M; x 68,2 M | n F1 0,88 mAP50 0,916 mAP50-95 0,691; **s F1 0,90 mAP50 0,938 mAP50-95 0,729**; m 0,92 / 0,948 / 0,757; l 0,92 / 0,952 / 0,763; x 0,92 / 0,950 / 0,766 | **não declarada** no card |
| Magi v1/v2 (DETR) | mangá (Mangadex-1.5M + PopManga) | painel, personagem, texto, cauda | 0,5 B (v1) / 2,06 GB (v2) | corpo AP 0,90 (Manga109) | não-comercial |
| `hysts/anime-face-detector` (YOLOv3 / Faster R-CNN) | anime colorido, "near-frontal" | face | — | não reportadas | MIT |
| `Fuyucch1/yolov8_animeface` (YOLOv8x6, 1280 px) | 10 K imagens safebooru (coloridas) | face | 68 M+ | P 0,957 R 0,924 mAP50 0,955 mAP50-95 0,534; 81,9 ms/img | AGPL-3.0 |
| `deepghs/anime_face_detection` via `imgutils.detect.detect_faces` (YOLOv8 n/s, v1.4) | "Anime Face CreateML" (colorido) | face | — | só gráfico de benchmark | — |

Observações: os detectores de anime foram treinados em ilustração **colorida**; nenhum documenta desempenho em mangá P&B. O `manga109_yolo` é o único treinado em páginas de mangá e ainda entrega **body** (o que queremos) e **frame** (poderia substituir o `ClassicalLayoutParser` que sub-segmenta o Berserk — fora do escopo deste ticket, mas vale anotar). Carrega-se com `dghs-imgutils`: `yolo_predict(image, repo_id="deepghs/manga109_yolo", model_name="v2023.12.07_s_yv11")`, backend ONNX Runtime (CPU ou CUDA).

### C. Modelos de embedding para recortes

| Modelo | Treino | Params | Dim | Licença | Evidência em mangá/anime |
|---|---|---|---|---|---|
| **`ragavsachdeva/magiv2-crop-embedder`** | SupCon em recortes de mangá (PopManga) | 85,8 M | 768 (ViT-B) | não-comercial | AMI 0,65–0,67 por página; nomeação 0,73–0,85 com ≈1 exemplar |
| DINOv2 ViT-S/B/L/g-14 | auto-supervisionado, fotos | 21 M / 86 M / 300 M / 1,1 B | 384 / 768 / 1024 / 1536 | **Apache-2.0** (código e pesos) | CoMix AMI 0,29 (melhor genérico) |
| DINOv3 ViT-S…7B/16 | auto-supervisionado, 1,7 B imagens | 21 M – 6,7 B | — | licença própria Meta, *gated* | sem evidência em mangá; usado com ArcFace em retrieval de identidades estilizadas (Fursee, 2026) |
| SigLIP 2 base-patch16-224 | imagem-texto | 0,4 B (visão+texto) | 768 | Apache-2.0 | sem evidência em mangá |
| CLIP ViT-B/32 · ViT-L/14 | imagem-texto | ~0,15 B / ~0,4 B | 512 (proj.) / 768 | MIT | 224 px; usado no pipeline BLV (A.7) via UMAP+HDBSCAN; abaixo do DINOv2 no CoMix |
| **CCIP** (`deepghs/ccip`, CaFormer) | contrastivo em **240 K imagens / 3 982 personagens de anime colorido** | ckpt 154 MB (77 MB fp16) | 768 | OpenRAIL | F1 0,917 (limiar 0,178) no modelo padrão `ccip-caformer-24-randaug-pruned`; 0,941 no `caformer_b36-24`; clustering OPTICS/DBSCAN embutido (`ccip_clustering`); limitações documentadas: viés para penteado, pouca sensibilidade a cor, **não sabe dizer "nenhum dos conhecidos"** |
| AniWho (EfficientNet-B7, classificação fechada) | rostos de anime | 66 M | — | — | 85,08 % top-1 em conjunto fechado |

Leitura: para mangá P&B o **embedder do Magiv2 é a única opção com evidência no domínio**. CCIP é excelente em anime colorido mas seu viés para penteado/cor é justamente o que o P&B de Berserk (capacetes, armaduras, sombras) tira. DINOv2-B é a reserva Apache com ~86 M parâmetros — mesma ordem de VRAM, pior qualidade esperada.

### D. Quantas referências por personagem e que acurácia esperar

- **1 exemplar** (Magiv2, com otimização global por capítulo): 0,73–0,75; **exemplar vindo do próprio mangá** e bem escolhido: ~0,87. Exemplares de arte do anime/colorida sofrem *distribution shift* — o paper abandonou miniaturas da web para personagens frequentes.
- **Sem otimização global** (comparar cada recorte independentemente contra a galeria, que é o pipeline mínimo): esperar abaixo do Magiv2 — a diferença entre restrições preditas e reais (0,73 → 0,80–0,85) mostra que as restrições de página valem 7–10 pontos. Estimativa conservadora: **60–75 % por recorte** para o elenco principal com 1 exemplar; **70–80 %** com 3–10 exemplares por personagem cobrindo variações (com/sem capacete, close/corpo inteiro, expressão), pois a média/k-NN de vários exemplares reduz a variância do "exemplar aleatório" (Tabela 6 do Magiv2).
- **Rosto** quando detectado: rank-1 78 % (FSAC) sugere que combinar score de rosto e corpo (média ponderada) recupera casos de corpo ocluído por balão.
- **Figurantes**: precisam de um limiar de "desconhecido" (η = 0,75 no Magiv2 para o embedder dele; calibrar no Berserk com um punhado de páginas rotuladas à mão). O CCIP documenta explicitamente que não resolve "não é nenhum dos conhecidos" — qualquer galeria pequena precisa desse limiar.
- **Zero exemplares**: 35–44 % (A.5). Não serve como dica.

### E. Como alimentar as dicas no VLM

Evidência direta (AutoAD-Zero, ECCV 2024, VideoLLaMA2-7B; banco de rostos InsightFace + IMDb top-10 por episódio):

| Exp. | Informação de personagem | TV-AD CRITIC | CMD-AD CRITIC |
|---|---|---|---|
| A | nenhuma | 0,1 | 0,8 |
| B | **só nomes em texto** | 27,4 | 34,7 |
| C | nomes em texto + círculos sem cor | 24,2 | 35,0 |
| D | caixas em volta dos rostos | 21,1 | 32,5 |
| E | círculos coloridos sem nomes | 21,1 | 36,6 |
| G (padrão) | **círculos coloridos + "Nome (cor)" no texto** | 27,6 | 43,7 |

Ou seja: a lista textual de presentes entrega quase todo o ganho no dataset fácil; a marcação visual com legenda adiciona ~9 pontos no dataset difícil (cenas com vários personagens). O Set-of-Mark (Yang et al., 2023) generaliza a ideia — marcas alfanuméricas sobre regiões fazem o GPT-4V zero-shot superar modelos *fine-tuned* em RefCOCOg — e é **agnóstico ao provedor**: não precisa de tokens de grounding do Qwen (o card do Qwen2.5-VL-7B documenta caixas só na **saída**, em JSON `bbox_2d`, não como entrada). O pipeline BLV (A.7) usa a variante "nome escrito na imagem".

No código atual, `VisionLanguageEngine.contextualize(panel_image, bubbles)` não tem *slot* para dicas, e o prompt de sistema (`build_scriptwriter_prompt`) já lista o elenco (`{cast}`) e pede rótulos curtos para desconhecidos. A dica entra naturalmente como um bloco **na mensagem de usuário** (por painel), não no sistema — para não invalidar o *fingerprint* do prompt de sistema a cada painel:

```
Personagens detectados neste painel (dica automática, pode estar errada;
confie na imagem se discordar): [1] Guts (0.83), [2] Casca (0.61), [3] desconhecido.
Os números correspondem às marcas circulares sobre a imagem.
```

Três regras que a evidência sustenta: (1) enviar a dica como **consultiva com confiança**, porque 20–30 % estarão erradas; (2) marcar a imagem só quando houver ≥2 personagens detectados (é onde o overlay ajuda — CMD-AD); (3) registrar a configuração de dicas (modelo, limiares, galeria) no `run_config.json`, já que o roteiro muda com ela (mesma lição do sha1 do prompt).

### F. Custo de VRAM / tempo no alvo (RTX 5060 Laptop 8 GB)

| Componente | Pesos | VRAM estimada em inferência | Nota |
|---|---|---|---|
| manga109_yolo `s_yv11` (ONNX) | ~40 MB (9,43 M × 4 B) | < 300 MB | roda em CPU também (21,6 GFLOPs/imagem) |
| magiv2-crop-embedder | ~343 MB F32 (85,8 M) | < 1 GB em lotes de recortes | pode rodar em fp16 |
| DINOv2 ViT-B/14 (reserva) | ~346 MB | < 1 GB | idem |
| Magiv2 completo (opção "máxima") | 2,06 GB F32 | **estimativa** 3–4 GB a 800×1333 px (o card não informa) | dá diarização também; testar compatibilidade com `transformers 4.57` + `trust_remote_code` |

Todos cabem ao lado do EasyOCR; nenhum cabe **junto** com XTTS + Qwen-VL local — seguir o padrão existente de `release()` entre estágios. Como o VLM padrão é API (Qwen/Claude), o único vizinho de VRAM é o XTTS.

## Recomendação

**Pipeline mínimo (ordem de implementação):**

1. **Detector**: `deepghs/manga109_yolo`, variante `v2023.12.07_s_yv11` via `dghs-imgutils.yolo_predict` (ONNX). Rodar na **página inteira** (não no recorte do painel — o detector foi treinado em páginas) e atribuir cada caixa `body`/`face` ao painel pelo IoU com `PanelData.bbox`. Guardar como `BoundingBox` normalizado ao painel, coerente com `SpeechBubble.bbox`.
2. **Embedding**: `ragavsachdeva/magiv2-crop-embedder` sobre o recorte de **corpo** (P&B → RGB como o Magi faz). Reserva *license-clean*: DINOv2 ViT-B/14 (Apache-2.0), sabendo que perde qualidade.
3. **Galeria**: pasta por volume `workspace/<slug>/cast/<Nome>/*.png` com **3–10 recortes extraídos do próprio mangá** por membro do elenco (o usuário recorta uma vez, com o próprio detector ajudando). Similaridade cosseno contra a média dos exemplares e contra o vizinho mais próximo; aceitar quando `max_sim ≥ τ_aceite` e `top1 − top2 ≥ margem`; senão "desconhecido". Calibrar τ com ~20 páginas rotuladas à mão (ponto de partida: η = 0,75 do Magiv2 para este embedder).
4. **Rosto como auxiliar**: se houver caixa `face` dentro da caixa `body`, embutir também o rosto e combinar scores (média ponderada); resolve oclusão por balão.
5. **Dica no VLM**: bloco textual consultivo na mensagem do usuário + marcas numeradas na imagem quando ≥2 detecções (formato da seção E). Fingerprint do checkpoint inclui `{repo, model_name, embedder, τ, margem, sha1 da galeria}`.
6. **Bônus sem LLM** (fase 2): balão → personagem mais próximo/cauda dá um palpite de falante geométrico (Manga109Dialog: baseline por regras; SGG >75 %). Pode virar campo `speaker_hint` por balão.

**Confiabilidade esperada**: 70–80 % de nomes corretos no elenco principal do Berserk com galeria bem construída; 60–70 % em painéis escuros/close de armadura; figurantes majoritariamente "desconhecido" (correto por construção). Isso é suficiente como *dica* — o VLM, que já vê a imagem, usa a dica para desempatar, não para decidir.

**Opção "máxima"** se a mínima não bastar: rodar `magiv2.do_chapter_wide_prediction(..., do_ocr=False)` por volume e consumir `character_names` + `text_character_associations` como dicas de presença **e** de falante (AP 0,75), a custo de 2 GB de pesos e uma dependência `trust_remote_code` sensível à versão do `transformers`.

## Riscos / incertezas

- **Licença do `manga109_yolo` não declarada** no card (dataset de origem Manga109-s permite publicar modelos com atribuição, mas o autor não escolheu licença). Mitigação: abrir *discussion* no HF; reserva = detector de personagens do Magi v1 (não-comercial, aceitável).
- **Pesos Magi são não-comerciais** — coberto pelo ADR-0002; se o horizonte mudar, trocar embedder por DINOv2 (Apache) e aceitar perda.
- **Compatibilidade** dos modelos `trust_remote_code` do Magi com o pin `transformers>=4.57,<5` não foi verificada — validar em ambiente isolado antes de integrar.
- **VRAM do Magiv2 completo é estimativa**; o card não informa.
- **Domain shift do Berserk**: armaduras, capacetes, Guts com/sem tapa-olho e braço mecânico, criaturas. Galeria precisa cobrir "eras" do personagem; o Magiv2 reconhece esse tipo de variação como caso de falha aberto.
- **Números de nomeação vêm do PopManga** (títulos populares, capítulo inteiro, com MILP); o pipeline mínimo por painel deve ficar abaixo — os 70–80 % são projeção, não medição. Medir em 20 páginas antes de decidir.
- **CCIP** foi treinado em anime colorido; seu viés para penteado/cor o torna suspeito em P&B. Não usar sem teste.
- **Detectores de rosto de anime** (hysts, yolov8_animeface) não documentam desempenho em mangá P&B — não confiar neles para rosto; usar a classe `face` do `manga109_yolo`.
- **"M109NC"** (tarefa de nomeação no Manga109) apareceu em um snippet de busca atribuído ao Manga109-v2026, mas **não consta do abstract oficial** — não usado aqui.
- Overlays na imagem alteram a imagem enviada à API (custo de tokens de imagem inalterado, mas muda o cache/fingerprint) — manter a versão sem marcas no workspace.

## Fontes

- Magi v1 — The Manga Whisperer (arXiv 2401.10224): https://arxiv.org/html/2401.10224
- Magi v2 — Tails Tell Tales (arXiv 2408.00298): https://arxiv.org/html/2408.00298
- Card `ragavsachdeva/magiv2` (uso, banco de personagens, licença): https://huggingface.co/ragavsachdeva/magiv2 · árvore de arquivos: https://huggingface.co/ragavsachdeva/magiv2/tree/main
- Card `ragavsachdeva/magiv2-crop-embedder`: https://huggingface.co/ragavsachdeva/magiv2-crop-embedder
- Card `ragavsachdeva/magi` (v1): https://huggingface.co/ragavsachdeva/magi
- Card `ragavsachdeva/magiv3`: https://huggingface.co/ragavsachdeva/magiv3
- Repositório Magi (versões, licença): https://github.com/ragavsachdeva/magi/blob/main/readme.md
- Zero-Shot Character Identification and Speaker Prediction in Comics (ACM MM 2024, arXiv 2404.13993): https://arxiv.org/html/2404.13993 · código MIT: https://github.com/liyingxuan1012/zeroshot-speaker-prediction
- FSAC — Unsupervised Manga Character Re-identification (arXiv 2204.04621): https://ar5iv.labs.arxiv.org/html/2204.04621
- OAM-ReID — Occlusion-Aware Manga Character Re-identification (MMAsia 2023): https://dl.acm.org/doi/10.1145/3595916.3626401
- Manga109Dialog (arXiv 2306.17469): https://arxiv.org/abs/2306.17469 · repositório: https://github.com/liyingxuan1012/SGG_based_speaker_prediction
- Object Detection for Comics using Manga109 Annotations (Ogawa et al., arXiv 1803.08670): https://arxiv.org/html/1803.08670v2
- Termos de uso Manga109 / Manga109-s: https://huggingface.co/datasets/hal-utokyo/Manga109-s
- Manga109-v2026 (arXiv 2605.21182, abstract): https://arxiv.org/abs/2605.21182
- CoMix (NeurIPS 2024 D&B, arXiv 2407.03550): https://arxiv.org/html/2407.03550
- Toward accessible comics for blind and low vision readers (arXiv 2407.08248): https://arxiv.org/html/2407.08248
- AutoAD-Zero (arXiv 2407.15850, ablação de dicas de personagem): https://arxiv.org/html/2407.15850
- Set-of-Mark Prompting (arXiv 2310.11441): https://arxiv.org/abs/2310.11441
- `deepghs/manga109_yolo` (card e tabela de métricas): https://huggingface.co/deepghs/manga109_yolo
- `dghs-imgutils` — `yolo_predict` genérico: https://dghs-imgutils.deepghs.org/main/api_doc/generic/yolo.html · `detect_faces`: https://dghs-imgutils.deepghs.org/main/api_doc/detect/face.html · CCIP API: https://dghs-imgutils.deepghs.org/main/api_doc/metrics/ccip.html
- CCIP — card `deepghs/ccip`: https://huggingface.co/deepghs/ccip · guia conceitual: https://deepghs.org/waifuc/main/advanced_guides/ccip/index.html
- `hysts/anime-face-detector`: https://github.com/hysts/anime-face-detector
- `Fuyucch1/yolov8_animeface`: https://github.com/Fuyucch1/yolov8_animeface
- DINOv2 (README, licença Apache-2.0, tabela de modelos): https://github.com/facebookresearch/dinov2 · card `facebook/dinov2-base`: https://huggingface.co/facebook/dinov2-base
- DINOv3 card `facebook/dinov3-vitb16-pretrain-lvd1689m` (licença própria): https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
- SigLIP 2 card `google/siglip2-base-patch16-224`: https://huggingface.co/google/siglip2-base-patch16-224
- CLIP (repositório MIT): https://github.com/openai/CLIP · config `openai/clip-vit-base-patch32`: https://huggingface.co/openai/clip-vit-base-patch32/raw/main/config.json · card `openai/clip-vit-large-patch14`: https://huggingface.co/openai/clip-vit-large-patch14
- AniWho (arXiv 2208.11012): https://arxiv.org/abs/2208.11012
- Fursee — YOLO + DINOv3 para retrieval de identidades estilizadas (arXiv 2606.22872): https://arxiv.org/abs/2606.22872
- Qwen2.5-VL-7B-Instruct (grounding só na saída): https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- Contexto interno: `mangawhisperer/interfaces.py` (`VisionLanguageEngine.contextualize`), `mangawhisperer/engines/vlm.py` (`build_scriptwriter_prompt`, `{cast}`), `docs/adr/0002-licencas-nao-comerciais-aceitaveis.md`.
