---
ticket: R2
status: concluído
data: 2026-09-02
autor: agente de pesquisa (Wayfinder)
---

# Segmentação de painéis e ordem de leitura RTL com zero tokens

> **English abstract.** This note surveys the practical state of the art for
> *zero-token* manga panel segmentation and right-to-left reading order using
> classical OpenCV techniques (background connected components, projection
> profiles, recursive X-Y cut, double-page splitting, Kovanen-style cut trees
> and Magi-style DAG ordering), then the small trained detectors available as
> a fallback (comic-text-detector, Magi v1/v2/v3, YOLO26 models trained on
> Manga109-s). Concrete numbers: Pang et al. 2014 reach 91.3% panel / 87.9%
> page success on 104 pages; Kovanen & Aizawa 2015 report >95% transition
> accuracy on 1,769 pages; an SSD300 already scores frame AP 97.1 on Manga109;
> Magi v2 reaches panel AP 0.9405 but weighs 2.06 GB (fp32, Conditional-DETR
> ResNet-50) and its ordering is a *geometric heuristic*, not a learned head;
> a YOLO26-nano fine-tuned on Manga109-s (2.57 M params, 15.5 MB `.pt`,
> 10.1 MB ONNX at 1024 px) reports panel mAP50-95 0.953. We recommend a
> six-stage heuristic battery (Pang-style panel block + projection-profile
> recursive split + Kovanen/Magi cut-tree ordering + KCC-style spread split),
> gated by a confidence score, with the YOLO26n ONNX model as the small
> fallback detector run through `onnxruntime` (avoiding the AGPL-3.0
> `ultralytics` dependency and its `opencv-python` conflict), and Magi v2 kept
> offline as an evaluation oracle only.

## Pergunta

Qual é o estado da arte *prático* para segmentar painéis de mangá e inferir a
ordem de leitura direita-para-esquerda **sem gastar tokens** (só OpenCV):
perfis de projeção para sarjetas, componentes conexos, X-Y cut recursivo,
divisão de páginas duplas, ordenação RTL com painéis diagonais/sobrepostos, e
quais implementações open-source existem? Em seguida, como *fallback*, quais
detectores pequenos treinados (comic-text-detector, Magi v1/v2, YOLO em
Manga109) valem a pena — tamanho, licença, acurácia, compatibilidade com
torch 2.11 / transformers 4.57? Que bateria heurística prototipar e qual o
melhor detector pequeno?

Contexto do repo: `mangawhisperer/engines/layout.py` já tem um
`ClassicalLayoutParser` (dilatação de tinta → contornos → caixas) e um
`sort_reading_order` por linhas com RTL dentro da linha. O que falta está
mapeado na seção de recomendação.

## Resumo executivo

1. **A literatura clássica converge num pipeline de 3 passos** (Pang et al.,
   ACM MM 2014): (a) *bloco de painéis* = página menos a máscara de fundo
   obtida por componentes conexos nos pixels **brancos** (não na tinta),
   depois de "fechar" painéis abertos na borda da página; (b) *divisão binária
   recursiva* do bloco com uma linha de corte escolhida por um custo sobre o
   perfil de projeção (`C_h(y) = 2g(y) − g(y+1) − g(y−1)`), com suporte a
   sarjetas **diagonais**; (c) recuperação da forma (4 cantos via convex hull).
   Resultado: 91,3% dos painéis e 87,9% das páginas corretas (IoU > 0,9) em
   104 páginas de Naruto/Slam Dunk, contra 74,1%/73,9% de erosão-dilatação
   sobre CCL. Isso é exatamente "X-Y cut recursivo + componentes conexos",
   mas aplicado ao *bloco* e não à página crua.
2. **Ordem de leitura RTL é um problema resolvido por heurística**, não por
   modelo: Kovanen & Aizawa (ICIP 2015) reportam >95% de acurácia de transição
   com uma árvore de cortes (horizontal antes de vertical; topo→base,
   direita→esquerda) em 1.769 páginas / 14.726 textos; o Magi (CVPR 2024,
   ACCV 2024) usa um **DAG geométrico + ordenação topológica** com erosão
   iterativa das caixas para sobreposições, também sem nada aprendido. O
   `manga109/panel-order-estimator` (MIT) implementa a árvore de Kovanen,
   incluindo `--initial-cut two-page` para páginas duplas.
3. **Página dupla**: o Kindle Comic Converter decide *spread* por
   `largura/altura > 1.16`, bissecciona no meio (`width/2`) e, em RTL, emite a
   metade **direita** primeiro. Nenhuma ferramenta pesquisada procura a sarjeta
   central; é uma melhoria barata via perfil de projeção vertical.
4. **Implementações open-source utilizáveis**: `adenzu/Manga-Panel-Extractor`
   (MIT; máscara de fundo por CC em pixels ≥ 240, Laplaciano + dilatação,
   contornos, filtro `área < página/32`, divisão de painéis unidos por
   *thinning* + casamento de templates — precisa de `opencv-contrib`; **não
   ordena**), `njean42/kumiko` (**AGPL-3.0+**; Sobel → limiar 100 → contornos →
   LSD → split/merge/deoverlap/expand → numeração `rtl`), `hummat/panelizer`
   (MIT, arquivado; desenho de "clássico + score de confiança + fallback ML").
   `kumpel/panel-extractor` **não foi localizado** em nenhuma busca.
5. **Detectores pequenos**: painel é a classe "fácil" — um SSD300 de 2018 já
   faz frame AP 97,1 em Manga109. O melhor custo-benefício hoje é o
   **YOLO26-nano fine-tuned em Manga109-s** (`leoxs22`, 2,57 M params,
   15,5 MB `.pt`; panel mAP50 0,985 / mAP50-95 0,953; classes *panel* e
   *text*), com export **ONNX a 1024 px** (`mednasserallah`, 10,1 MB) que roda
   em `onnxruntime` sem `ultralytics`. `dmMaze/comic-text-detector` (GPL-3.0,
   76,3 MB) detecta **só texto**, não painéis. Magi v2 (2,06 GB fp32,
   Conditional-DETR ResNet-50, entrada 1333×800, licença não-comercial —
   aceitável pelo ADR 0002) é o oráculo de referência (panel AP 0,9405), não
   um detector "pequeno".

## Achados

### A. Métodos clássicos (OpenCV)

#### A1. X-Y cut recursivo e perfis de projeção (origem)

- A árvore X-Y (Nagy & Seth, ICPR 1984, pp. 347–349) alterna cortes
  horizontais/verticais onde o perfil de projeção mostra faixas vazias.
  Ha, Haralick & Phillips (ICDAR 1995) mostram que aplicar o X-Y cut sobre
  **caixas de componentes conexos** em vez de pixels é bem mais barato e
  igualmente eficaz. Limitação estrutural (repetida em toda a literatura de
  quadrinhos): exige sarjeta limpa que atravesse o bloco inteiro; painéis
  irregulares, diagonais ou unidos por balões quebram o corte.
- Pang et al. 2014 resolvem isso operando o corte **no bloco de painéis** e
  não na página, com um custo sobre o perfil acumulado `g(y) = Σ I(p)` ao
  longo da linha: candidatos onde `C_h(y) = 2g(y) − g(y+1) − g(y−1) > 0.1·W`
  **ou** `g(y) > 0.1·W` (este segundo termo captura sarjetas diagonais);
  linhas a < 30 px (horizontais) / < 50 px (diagonais) são agrupadas em faixas
  e substituídas pela linha central; faixas horizontais que cortam preto por
  mais de `0.7·W` são descartadas; para diagonais, CCL na região local, rejeita
  se > 5 componentes, ajusta elipse (razão eixos < 0,2 ⇒ faixa) e uma reta aos
  centróides das colunas. A linha ótima em cada nível é a de maior
  *confiança* = fração de pixels brancos ao longo dela (1,0 = atravessa sem
  cortar tinta). A forma final vem do convex hull: centróide, um vértice por
  quadrante, refinamento local (raio 10 px) maximizando a área do
  quadrilátero.
- Números (Tabela 1 do paper): páginas 87,9%, painéis 91,3%, IoU médio 0,97
  vs. 73,9% / 74,1% / 0,94 para erosão-dilatação sobre CCL (Ho, Burie &
  Ogier, GREC 2011). Falha conhecida: onomatopeia gigante que corta o painel
  gera forma incompleta.
- Reparo de painéis abertos na borda (Pang §2.1): binariza com limiar 235 e,
  para cada borda da página, se `max(φ(x)·x) − min(φ(x)·x) > 0.7·L` e
  `Σ|φ'(x)| > 10`, substitui a borda por uma linha preta antes do CCL.
  Componentes cujo convex hull se sobrepõe > 50% ao de outro são fundidos
  (efeito "quebra da quarta parede").

#### A2. Componentes conexos no fundo vs. na tinta

- Rigaud, Tsopze, Burie & Ogier (GREC 2011, LNCS 7423) exploram que desenhos
  são cercados por linhas pretas e rotulam componentes conexos para extrair
  quadros e texto. Stommel, Merhej & Müller (ICCVG 2012, LNCS 7594,
  pp. 633–640) propõem detecção "segmentation-free". Pang 2014 inverte o
  sentido — CCL nos pixels **brancos** — porque o fundo da página é quase
  sempre um único componente, enquanto um painel pode ser vários. Essa
  inversão é a diferença prática mais importante em relação ao
  `ClassicalLayoutParser` atual do repo, que dilata a **tinta**.
- `adenzu/Manga-Panel-Extractor` (MIT) faz o mesmo: `generate_background_mask`
  escolhe a borda menos variada como fundo, `cv2.threshold` em ≥ 240,
  `connectedComponentsWithStats`, mantém componentes grandes que atravessam a
  página, dilata 3×3 ×2; `preprocess_image_with_dilation` = GaussianBlur 3×3
  → Laplaciano → dilate 5×5 → inversão; `findContours(RETR_EXTERNAL,
  CHAIN_APPROX_SIMPLE)`; descarta contornos com área < `página/32`. O
  *split joint panels* usa `cv2.ximgproc.thinning` + `matchTemplate` (limiar
  0,9) com 8 padrões direcionais — exige **`opencv-contrib-python`**
  (`requirements.txt` do projeto lista `opencv-contrib-python`,
  `opencv_python_headless`, `torch`, `yolov5`). O *fallback* (`get_fallback_panels`,
  quando < 2 painéis) é limiar adaptativo. Painéis saem **na ordem do
  `findContours`** — sem ordem de leitura. O README avisa que não funciona bem
  em manhwa/webtoon.

#### A3. Kumiko (AGPL-3.0+)

- Pipeline (`Page` / `Panel`): cinza → Sobel X+Y (`addWeighted` 0,5/0,5) →
  limiar 100 → `findContours(RETR_EXTERNAL)` → `approxPolyDP(ε = 0.001·arco)`;
  descarta painéis com largura/altura < 1% da imagem. `get_segments()` usa o
  **Line Segment Detector** (`cv2.createLineSegmentDetector`; removido do
  OpenCV 4.1.0–4.5.3 por licença e restaurado em 4.5.4+ — o projeto exige
  ≥ 4.9, então está disponível) e filtra segmentos menores que
  `min(img)·small_panel_ratio`. `Panel.split()` tenta dividir polígonos por
  vértices próximos com cobertura de segmentos > 50%; `group_small_panels()`
  agrupa centros a < 75% das dimensões combinadas; `merge()` quando um painel
  contém > 50% de outro; `deoverlap_panels()`; `expand_panels()` até a sarjeta
  real (`actual_gutters()`). Ordem: `fix_panels_numbering(numbering="rtl")`
  move painéis que aparecem antes do vizinho superior/direito até convergir.
- Licença **AGPL-3.0 ou posterior** — copiar código exige AGPL no repo;
  reimplementar a ideia é livre.

#### A4. Ordem de leitura RTL com diagonais e sobreposições

- Kovanen & Aizawa, "A layered method for determining manga text bubble
  reading order", ICIP 2015, pp. 4283–4287: três camadas (página → painel →
  texto) ordenadas hierarquicamente; >95% de acurácia de transição em 1.769
  páginas / 14.726 textos. O `manga109/panel-order-estimator` (MIT,
  depende de `manga109api`) implementa a versão para painéis: recursivamente
  procura um separador **horizontal** primeiro, depois **vertical**; ao
  interpretar a árvore, visita topo→base e **direita→esquerda**; conjuntos
  inseparáveis viram uma folha com um único número (desenhados pontilhados);
  `--initial-cut two-page` (padrão) separa a página dupla antes de tudo e
  `two-page-four-panel` trata yonkoma. Entrada: caixas já detectadas — ele
  **não** detecta painéis.
- Magi (`utils.py` de `ragavsachdeva/magiv2`): `sort_panels()` monta um DAG
  com `is_there_a_directed_edge()` (A estritamente acima / à direita de B ⇒
  A→B), erosão iterativa dos retângulos para desfazer sobreposições,
  `use_cuts_to_determine_edge_from_a_to_b()` como fallback por cortes
  horizontais/verticais (`merge_overlapping_ranges`), remoção de ciclos com
  `nx.simple_cycles` (mantendo as arestas mais curtas) e
  `nx.topological_sort`. `sort_text_boxes_in_reading_order()` mapeia texto →
  painel por interseção/distância (Shapely) e ordena dentro do painel pela
  proximidade ao **canto superior direito**. Os papers (CVPR 2024 §4; ACCV
  2024) confirmam: "heurística, não aprendida", sem métrica de ordem
  reportada. Dependências extras: `networkx`, `shapely`, `pulp`, `scipy`,
  `einops`.
- Documentos genéricos: XY-Cut++ (arXiv 2504.10258, 2025) melhora o X-Y cut
  com pré-máscara, multi-granularidade e casamento cross-modal, BLEU 98,8 em
  DocBench-100 — mas é LTR e não menciona quadrinhos; serve só como
  referência de que "X-Y cut + pré-processamento" continua competitivo.

#### A5. Página dupla (spread)

- Kindle Comic Converter (`kindlecomicconverter/image.py`,
  `ComicPageParser.splitCheck`): spread se `width/height > 1.16`; se a razão
  for < `BISECT_THRESHOLD = 1.8` divide, senão rotaciona; corte geométrico em
  `int(width/2)`; com `righttoleft`, `pageone = crop(rightbox)`. Não procura
  sarjeta.
- Manga109 é anotado em **páginas duplas** (10.130 páginas duplas em 109
  volumes, Ogawa et al. 2018), o que explica o `--initial-cut two-page` do
  panel-order-estimator e é um alerta: modelos treinados em Manga109 viram
  spreads inteiros no treino.

#### A6. O que o repo já tem vs. o que falta

`layout.py` hoje: `ink < 220` → dilatação `min(h,w)//100` ×2 → contornos →
área ≥ 2% da página → `sort_reading_order` (linhas por sobreposição vertical
> 0,5, RTL dentro da linha). Faltam: máscara de fundo por CC em branco (A2),
reparo de bordas abertas e divisão recursiva de blocos unidos com perfil de
projeção (A1), diagonais, árvore de cortes/DAG para ordem (A4 — o agrupamento
por linhas falha em layouts "escada" e em painéis altos que cruzam linhas),
divisão de spreads (A5) e um score de confiança para acionar fallback.

### B. Detectores treinados (fallback)

| Modelo | Tarefa / classes | Arquitetura · tamanho | Acurácia reportada | Licença | Stack |
|---|---|---|---|---|---|
| `leoxs22/manga-panel-detector-yolo26n` | panel, text | YOLO26-n (COCO→Manga109-s, 87 títulos, ~18k págs, ~32k anotações); 2,57 M params; `.pt` fp32 15,5 MB; TFLite int8 2,84 MB; 640 px | INT8: panel mAP50 0,985 / mAP50-95 0,953; text 0,928 / 0,740; ~100–180 ms CPU | Card Apache-2.0 (com ressalva Manga109-s) | `ultralytics` (AGPL-3.0) |
| `mednasserallah/manga-panel-detector-yolo26n-onnx` | idem (export do anterior) | ONNX fp32 **10,1 MB**, letterbox **1024** px (640 perdia painéis finos laterais) | herda | Apache-2.0 | `onnxruntime` — **sem ultralytics** |
| `ShadowB/Manga109-panel-balloon-text-yolov26-segmentation` | frame, text, balloon (seg) | YOLO26-s-seg; 11,44 M params; 23,4 MB; treino 1280 px; ultralytics 8.4.43 | box P 0,965 R 0,952 mAP50 0,975 mAP50-95 0,900; mask mAP50 0,970 / 0,846 (split por livro) | MIT (card); dados MS92/MangaSegmentation exigem crédito "Copyrighted by Minshan Xie" | `ultralytics` |
| `mosesb/best-comic-panel-detection` | comic panel (1 classe) | YOLOv12-**x** (Roboflow, quadrinhos ocidentais) | mAP50 0,991 / 0,985 | Apache-2.0 | `ultralytics`; pesado; sofre com painéis irregulares |
| `dmMaze/comic-text-detector` | **texto** (blocos, linhas, máscara) — **não** painéis | YOLOv5 + DBNet + UNet; `comictextdetector.pt` 76,3 MB / `.onnx` 90,3 MB (release beta-0.2.1 do manga-image-translator); ~13k imgs (⅓ Manga109-s, ⅓ DCM, ⅓ sintético) | não reportada | **GPL-3.0** | torch |
| `ragavsachdeva/magi` (v1, CVPR 2024) | painel, texto, personagem + associação + ordem | Conditional-DETR ResNet-50 (enc/dec 6+6, d=256) + ViT crop; **2,06 GB** (`model.safetensors`) | Manga109 panel AP **0,9357**, char 0,9015; PopManga text 0,92 | não-comercial (uso pessoal/pesquisa livre) | transformers `trust_remote_code` |
| `ragavsachdeva/magiv2` (ACCV 2024) | + caudas de balão, texto essencial, OCR (TrOCR) | mesmo detector; `pytorch_model.bin` **2,06 GB** fp32; entrada 1333×800; config salva com transformers 4.34.0.dev0 | Manga109 panel AP **0,9405**, char 0,9046; PopManga-X text 0,937, tails 0,877 | não-comercial | transformers + `pulp`, `networkx`, `shapely`, `einops`, `scipy` |
| `ragavsachdeva/magiv3` (2025) | VLM autoregressivo (localização, OCR, grounding) | 0,8 B params, F16 safetensors | n/d no card | não-comercial | `AutoModelForCausalLM` |
| `pedrovgs/DeepPanel` | segmentação painel/borda/fundo | U-Net TensorFlow/TFLite; dataset privado 550/83 págs | só gráficos | Apache-2.0 | TF — fora do stack |

Observações de compatibilidade:

- **Ultralytics** (8.4.138, 1/set/2026): `torch>=1.8.0` (Windows: `!=2.4.0`)
  → torch 2.11 ok; exige **`opencv-python`** (não headless) `>=4.7.0,
  !=4.13.0.90` — instalar junto com o `opencv-python-headless` do projeto
  duplica o `cv2` e é fonte clássica de import quebrado. Licença
  **AGPL-3.0**; a página de licença afirma que "todos os modelos treinados
  com Ultralytics caem sob AGPL-3.0 por padrão" (código de treino *e* pesos).
  Para um projeto pessoal/filantrópico (ADR 0002) isso é aceitável, mas
  contamina a licença do repo se ele um dia for aberto sob MIT/Apache.
  Rodar o **ONNX em `onnxruntime`** evita a dependência de código.
- **Magi × transformers 4.57**: `modelling_magiv2.py` importa de
  `transformers` apenas `ConditionalDetrModel`, `ConditionalDetrMLPPredictionHead`,
  `ConditionalDetrModelOutput`, `inverse_sigmoid`, `ViTMAEModel`,
  `VisionEncoderDecoderModel`, `center_to_corners_format` — todos presentes em
  `v4.57.0/modeling_conditional_detr.py`. O `ConditionalDetrHungarianMatcher`
  (removido do transformers a partir de 4.35.0, issue magi#11) foi copiado
  para o código remoto em abr/2025 (PR de emanuelevivoli, mesclado pelo
  autor). Logo, carrega no pin `>=4.57,<5` do repo; não foi testado por mim.
  VRAM de inferência não é documentada em lugar nenhum — 2,06 GB de pesos
  fp32 mais ativações a 1333×800 devem caber nos 8 GB, mas é estimativa.
- **YOLO26** (jan/2026): inferência *end-to-end sem NMS* por padrão e sem DFL
  — o export ONNX tende a já sair com caixas finais, mas o card do
  `mednasserallah` não especifica shapes de saída; verificar ao integrar.
- Baseline histórico útil: Ogawa et al. 2018 (Manga109-annotations:
  103.900 frames, 147.918 textos em 109 volumes; treino 99 vols/9.250 págs,
  teste 10 vols/880 págs) — SSD300 frame AP **97,1** vs. texto 82,0, face
  67,1. Painel é a classe mais fácil; texto/balão é onde um detector ajuda de
  verdade.

### C. Dados e licenças de treino

- **Manga109**: 109 volumes, 21.142 páginas; "solely for academic purposes",
  só organizações não-comerciais, sem redistribuição, crédito "© Autor" e
  citação obrigatória. **Manga109-s**: 87 volumes (~3,3 GB), acesso *gated*
  no HF (formulário, 2–7 dias); permite usar **resultados de ML**
  comercialmente, mas proíbe redistribuir o dataset, vender imagens e
  publicar > 20% das páginas de um volume; exige indicar o uso do Manga109-s.
  Os modelos YOLO acima herdam essas condições ("ressalva" no card do
  leoxs22).
- **MS92/MangaSegmentation** (base do ShadowB): < 1k imagens, 614 MB, JSON;
  uso acadêmico e comercial permitidos com o crédito obrigatório.
- `manga109api` (MIT): só parseia o XML (frame/text/face/body); a ordem dos
  tags "não carrega informação" — não serve para ordem de leitura.

## Recomendação

### Bateria heurística a prototipar (zero tokens, só `cv2` + `numpy`)

Ordem sugerida de implementação, cada estágio com teste unitário sintético
(páginas geradas por código, como `tests/test_engines.py` já faz para o
parser atual):

- **H0 — Normalização + spread.** Cinza, estimativa de fundo pela borda menos
  variada (adenzu). Se `w/h > 1.16` (KCC) e < 1.8: calcular o perfil de
  projeção vertical de pixels brancos e procurar a coluna de máximo dentro de
  ±5% do centro; se existir sarjeta (branco ≥ 0,9 da altura) cortar nela,
  senão em `w/2`. Emitir metade **direita** antes da esquerda. Isso melhora o
  KCC sem custo.
- **H1 — Bloco de painéis (Pang §2.1).** Binarizar em 235; reparar bordas
  abertas com o teste `0.7·L` / `Σ|φ'| > 10`; CCL nos pixels **brancos**;
  fundo = componentes que tocam a borda e atravessam a página; bloco =
  complemento; fundir componentes cujos convex hulls se sobrepõem > 50%.
- **H2 — Candidatos por CC + filtro.** `findContours(RETR_EXTERNAL)` no bloco;
  descartar área < `página/32` (adenzu) ou < 2% (parâmetro atual); se restar
  < 2 painéis, fallback de limiar adaptativo e, se ainda assim nada, página
  inteira (comportamento atual).
- **H3 — Divisão recursiva de blocos unidos (Pang §2.2).** Para cada
  componente com razão de preenchimento baixa ou muito grande: candidatos
  horizontais e verticais por `C(y) = 2g(y) − g(y±1) > 0.1·W` ∪ `g(y) > 0.1·W`;
  agrupar em faixas (30 px / 50 px, escalar pela resolução); descartar faixas
  que cortam tinta por > 0,7·W; diagonais via elipse (razão < 0,2) + reta
  pelos centróides; escolher a de maior fração de branco; recursar. Este é o
  "X-Y cut recursivo" pedido no ticket, na forma que sobrevive a mangá.
- **H4 — Forma.** Convex hull → 4 cantos por quadrante → refino local (raio
  10 px). Guardar o polígono; a `BoundingBox` normalizada continua sendo o
  contrato com o resto do pipeline.
- **H5 — Ordem de leitura.** Substituir o agrupamento por linhas por uma
  **árvore de cortes** (Kovanen / panel-order-estimator): separador
  horizontal primeiro, depois vertical; visitar topo→base, direita→esquerda;
  nós inseparáveis (diagonais, sobreposições) resolvidos pelo **DAG do Magi**
  (arestas "estritamente acima / à direita", erosão iterativa, quebra de
  ciclos pela aresta mais curta, ordenação topológica) e, em último caso,
  pela distância ao canto superior direito. Texto dentro do painel: mesmo
  critério do canto superior direito.
- **H6 — Confiança + gate.** Score = cobertura da página pelos painéis,
  sobreposição residual, nº de nós inseparáveis, razão de painéis com
  IoU-vs-retângulo baixa (Panelizer). Abaixo do limiar ⇒ fallback B.

Evidência de que isso basta: 91,3% de painéis (Pang) e >95% de transições
(Kovanen) em mangá comercial, sem GPU.

### Melhor detector pequeno (fallback)

**`mednasserallah/manga-panel-detector-yolo26n-onnx`** (10,1 MB, 1024 px, classes
*panel* + *text*), executado com `onnxruntime` (CPU ou `onnxruntime-gpu`):
panel mAP50-95 0,953 no treino original, 2,57 M params, zero dependência de
`ultralytics` (evita AGPL no código e o conflito `opencv-python` vs.
`-headless`). Encaixa como um segundo `MangaLayoutParser` na factory de
motores; H5 continua sendo a camada de ordem. Alternativa quando máscaras de
balão forem necessárias: exportar o **YOLO26-s-seg do ShadowB** (23,4 MB,
frame/text/balloon) para ONNX num ambiente separado e consumir só o `.onnx`.

**Magi v2** fica como **oráculo offline** para avaliar H0–H6 nas páginas do
piloto (sua ordem é a mesma heurística, então serve de referência de
detecção, não de ordem) — 2,06 GB, `trust_remote_code`, licença não-comercial
compatível com o ADR 0002. **Não** usar `comic-text-detector` para painéis
(é só texto, GPL-3.0, 76 MB); se a classe *text* do YOLO26n for insuficiente
para balões, ele é o candidato para essa outra tarefa.

### Protocolo de avaliação sugerido

Solicitar o **Manga109-s** (gated, sem redistribuição): frames como GT →
taxa de painéis com IoU ≥ 0,9 (métrica de Pang) e IoU ≥ 0,5; ordem → acurácia
de transição (Kovanen). Registrar o uso do Manga109-s no README conforme os
termos.

## Riscos / incertezas

- **`kumpel/panel-extractor` não existe** nos resultados de busca (só
  Kumiko, C.A.P.E, comics-splitter, DeepPanel, ehsanx/Comic-Panel-Extractor);
  o ticket pode ter o nome errado.
- **Licenças**: Kumiko é AGPL-3.0+ (não copiar código); Ultralytics AGPL-3.0
  reivindica os pesos treinados; comic-text-detector GPL-3.0; Magi
  não-comercial; Manga109-s exige menção e proíbe redistribuir. Tudo
  compatível com o ADR 0002, mas incompatível com um futuro repo MIT — o gate
  ONNX minimiza a superfície.
- **Dependências**: `cv2.ximgproc.thinning` (adenzu) exige `opencv-contrib`;
  a bateria acima evita isso usando o corte por projeção de Pang. LSD
  (Kumiko) requer OpenCV ≥ 4.5.4 — ok com `>=4.9`. Magi v2 puxa `pulp`,
  `networkx`, `shapely`, `einops`.
- **Compatibilidade Magi × transformers 4.57**: verificada por inspeção de
  símbolos, não por execução. `trust_remote_code` continua frágil (issues
  transformers#44561/#45020 em 2026 sobre remoções internas); o pin `<5` do
  repo protege por ora.
- **VRAM do Magi v2** não documentada; estimativa de ~3–4 GB para inferência
  fp32 a 1333×800 precisa ser medida antes de contar com ele no laptop.
- **Domínio**: todos os métodos clássicos assumem fundo branco e sarjetas
  claras; manhwa/webtoon vertical, páginas com fundo preto (cenas noturnas) e
  onomatopeias gigantes cruzando painéis (falha declarada por Pang) são os
  casos onde o gate H6 precisa mandar para o YOLO. Manga109 é dos anos
  1970–2010 e anotado em página dupla — spreads modernos e escaneamentos com
  borda preta são pouco representados.
- **Métricas do YOLO26n** são sobre o TFLite INT8 e o split do próprio autor
  (não há split canônico); os números do ShadowB têm métricas por classe
  marcadas como TODO.
- **Ordem de leitura**: nenhum dos modelos reporta métrica de ordem em
  Manga109; a única cifra quantitativa é a de Kovanen 2015 (texto, não
  painel). Um teste próprio é indispensável.

## Fontes

Papers
- Pang, Cao, Lau, Chan. *A Robust Panel Extraction Method for Manga*. ACM MM 2014. PDF do autor: https://www.cs.cityu.edu.hk/~rynson/papers/mm14a.pdf · DOI https://doi.org/10.1145/2647868.2654990
- Kovanen, Aizawa. *A layered method for determining manga text bubble reading order*. ICIP 2015. https://www.semanticscholar.org/paper/9839f650dc890fd1cfb6c0eb900b77bf8ecad626
- Rigaud, Tsopze, Burie, Ogier. *Robust Frame and Text Extraction from Comic Books*. GREC 2011 / LNCS 7423. https://link.springer.com/chapter/10.1007/978-3-642-36824-0_13
- Stommel, Merhej, Müller. *Segmentation-Free Detection of Comic Panels*. ICCVG 2012 / LNCS 7594. https://link.springer.com/chapter/10.1007/978-3-642-33564-8_76
- Ha, Haralick, Phillips. *Recursive X-Y cut using bounding boxes of connected components*. ICDAR 1995. https://haralick.org/conferences/71280952.pdf · https://ieeexplore.ieee.org/document/602059/
- Nagy, Seth. *Hierarchical representation of optically scanned documents*. ICPR 1984. https://sites.ecse.rpi.edu/~nagy/PDF_chrono/1984_Seth_icpr84-xy-tree.pdf
- Ogawa et al. *Object Detection for Comics using Manga109 Annotations*. arXiv 1803.08670. https://ar5iv.labs.arxiv.org/html/1803.08670
- Sachdeva, Zisserman. *The Manga Whisperer* (Magi v1). CVPR 2024. https://arxiv.org/html/2401.10224
- Sachdeva, Shin, Zisserman. *Tails Tell Tales* (Magi v2). ACCV 2024. https://arxiv.org/html/2408.00298
- Sachdeva, Zisserman. *From Panels to Prose* (Magi v3). 2025. https://arxiv.org/abs/2503.23344
- Liu, Li, Wei. *XY-Cut++*. arXiv 2504.10258. https://arxiv.org/abs/2504.10258

Código clássico
- adenzu/Manga-Panel-Extractor (MIT): https://github.com/adenzu/Manga-Panel-Extractor · `src/image_processing/panel.py` · `requirements.txt`
- njean42/kumiko (AGPL-3.0+): https://github.com/njean42/kumiko · LICENSE · pipeline: https://deepwiki.com/njean42/kumiko/4.2-processing-pipeline
- manga109/panel-order-estimator (MIT): https://github.com/manga109/panel-order-estimator
- manga109/manga109api (MIT): https://github.com/manga109/manga109api
- hummat/panelizer (MIT, arquivado): https://github.com/hummat/panelizer
- Kindle Comic Converter `image.py` (`splitCheck`): https://raw.githubusercontent.com/ciromattia/kcc/master/kindlecomicconverter/image.py
- OpenCV LSD (restaurado em 4.5.4): https://github.com/opencv/opencv_contrib/issues/2524 · https://docs.opencv.org/4.13.0/db/d73/classcv_1_1LineSegmentDetector.html
- pedrovgs/DeepPanel (Apache-2.0): https://github.com/pedrovgs/DeepPanel

Modelos
- https://huggingface.co/leoxs22/manga-panel-detector-yolo26n (+ `/tree/main`)
- https://huggingface.co/mednasserallah/manga-panel-detector-yolo26n-onnx
- https://huggingface.co/ShadowB/Manga109-panel-balloon-text-yolov26-segmentation
- https://huggingface.co/mosesb/best-comic-panel-detection
- https://github.com/dmMaze/comic-text-detector · pesos: https://api.github.com/repos/zyddnys/manga-image-translator/releases/tags/beta-0.2.1
- https://huggingface.co/ragavsachdeva/magi (+ `/tree/main`) · https://huggingface.co/ragavsachdeva/magiv2 (+ `/tree/main`, `config.json`, `utils.py`, `modelling_magiv2.py`) · https://huggingface.co/ragavsachdeva/magiv3
- https://github.com/ragavsachdeva/magi · issue #11 (matcher removido no transformers 4.35): https://github.com/ragavsachdeva/magi/issues/11
- transformers v4.57.0 `modeling_conditional_detr.py`: https://raw.githubusercontent.com/huggingface/transformers/v4.57.0/src/transformers/models/conditional_detr/modeling_conditional_detr.py

Ultralytics / datasets
- https://docs.ultralytics.com/models/yolo26 · https://www.ultralytics.com/license · https://raw.githubusercontent.com/ultralytics/ultralytics/main/pyproject.toml · https://pypi.org/project/ultralytics/
- Manga109: https://manga109.github.io/manga109-project-website/en/index.html · Manga109-s: https://huggingface.co/datasets/hal-utokyo/Manga109-s
- MS92/MangaSegmentation: https://huggingface.co/datasets/MS92/MangaSegmentation
