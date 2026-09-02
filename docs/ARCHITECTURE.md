# Arquitetura do MangaWhisperer

Decisões estruturais e as razões por trás delas — o "porquê" que o código sozinho
não conta.

## 1. Contratos primeiro, modelos depois

Todo o pipeline foi construído em TDD **antes** de qualquer modelo de deep
learning existir no projeto: cinco ABCs (`interfaces.py`) com contratos de dados
Pydantic v2 (`models.py`), um orquestrador que só conhece as interfaces, e mocks
determinísticos provando o fluxo de ponta a ponta. Modelos reais (EasyOCR, XTTS,
VLMs) chegaram depois, um por vez, cada um atrás de uma interface já testada.

Consequência prática: trocar qualquer estágio é uma linha. O mesmo orquestrador
roda com o roteirista na Anthropic, na Alibaba (Qwen), na OpenAI, na Moonshot
(Kimi) ou num Qwen2.5-VL local na GPU — e com TTS XTTS, Edge ou silencioso.

## 2. Workspace persistente com retomada por fingerprint

Cada volume tem um workspace em disco com artefatos e checkpoints JSON por
estágio (`script/panels.json`, `audio/segments.json`). Execuções longas caem —
Colab desconecta, APIs falham — e o resume evita repagar chamadas de VLM.

A lição mais cara do projeto: **um checkpoint sem identidade é um bug à espera**.
Na primeira versão, um checkpoint gerado por motores *placeholder* (TTS de
silêncio) foi retomado por uma execução real e produziu um áudio mudo. Hoje o
`run_config.json` grava a *fingerprint* de cada motor — provedor, modelo e um
hash do prompt de sistema — e qualquer divergência invalida o estágio. Roteiro
refeito invalida o áudio em cascata.

## 3. Saída de LLM: campos estruturados + parser tolerante

Nada de markup inline (`[sfx: explosao]`) no texto: efeitos, falante e tipo de
bloco são **campos estruturados** validados por Pydantic (na Anthropic, via
structured outputs nativos; nos demais, instrução JSON + parser tolerante que
salva blocos válidos de arrays truncados e descarta os inválidos, bloco a bloco,
em vez de rejeitar o painel inteiro). Toda falha degrada para *passthrough* — o
diálogo bruto nunca se perde.

## 4. Duas passadas de LLM: roteirista (visão) + revisor (texto)

O roteirista vê um painel por vez — ótimo para descrição, ruim para coerência
global. O revisor recebe o roteiro inteiro (em blocos de 40 painéis, com memória
dos rótulos entre blocos) e corrige consistência de nomes, pronúncia para leitura
em voz alta e sanidade dos efeitos. Por ser texto puro, custa ~10% do estágio de
visão. Ambos os estágios têm checkpoint próprio (`panels_raw.json` → o bruto é
sempre preservado para auditoria).

## 5. Áudio: convenção única, mixagem em dois barramentos

Tudo converge para 24 kHz mono 16-bit (o formato do XTTS): efeitos são
normalizados na entrada da biblioteca, o Edge-TTS converte o mp3 na saída, e o
mixer soma dois barramentos — timeline de narração+SFX (com ganhos por canal) e
trilha de fundo em loop com fade — em numpy puro, sem ffmpeg.

## 6. Sonoplastia em camadas com garantia determinística

1. O **roteirista** escolhe efeitos com base na cena (prompt com mapa cena→tag e
   três níveis de intensidade);
2. um **tagger por palavras-chave** garante os momentos óbvios que o modelo pular;
3. **uploads do usuário** entram com classificação zero-shot (CLAP) e tabela
   editável antes de confirmar.

Variantes de um mesmo efeito são escolhidas por hash do texto do bloco — a mesma
cena sempre soa igual, inclusive após resume.

## 7. GPU de 8 GB como restrição de projeto

Modelos pesados nunca coexistem: o VLM local expõe `release()` e o orquestrador o
descarrega antes do TTS subir; o CLAP carrega no upload e é liberado em seguida.
O VLM local padrão roda **fora do processo** (`llama-server`, ADR-0004): o motor
`llamacpp` sobe o servidor no primeiro painel e o encerra no `release()`, então a
mesma regra vale para um modelo que vive em outro processo.
Pins de versão que importam: `transformers>=4.57,<5` (o coqui-tts quebra com a
5.x — issue idiap/coqui-ai-TTS#558) e torch do canal cu128 (Blackwell/sm_120).

## 8. Uma única montagem do pipeline (`PipelineConfig`)

CLI e interface web montavam os motores cada um do seu jeito — e um bug de
ordem de inicialização (`bgm_name` usado antes de existir) só aparecia num
deles. Hoje toda entrada constrói um `PipelineConfig` (Pydantic, imutável,
validado) e chama `build_pipeline`, que monta os estágios numa ordem fixa:
estilo ➜ efeitos ➜ trilha ➜ roteirista (com *preflight* de credenciais) ➜
revisor ➜ voz ➜ orquestrador. Valores `None` herdam do preset de estilo;
valores explícitos sempre vencem. O mesmo módulo é dono dos invariantes que os
motores compartilhavam em silêncio (`constants.py`: 24 kHz mono 16-bit) e do
perfil de hardware (`HardwareProfile`: 8 GB ⇒ modelos pesados não coexistem).
`PipelineResources` guarda OCR e XTTS carregados para a UI reaproveitar entre
cliques; o CLI monta tudo do zero. A regra dos 8 GB é aplicada na montagem:
com roteirista local numa GPU pequena, o XTTS em cache é liberado
(`evict()`) e reconstruído por execução — como ele carrega sob demanda, só
toca a GPU depois que o orquestrador descarregou o VLM.

## 9. Heurísticas sem token antes de qualquer modelo

Três filtros determinísticos rodam antes do VLM ver um pixel ou o LLM ler um
caractere:

1. **Layout** (`engines/layout.py`) — escada de heurísticas que só divide, nunca
   funde: divisão de *spreads* (imagem paisagem = duas páginas), **rede de
   sarjetas** (o papel conectado à margem é inundado; os buracos são painéis —
   sobrevive a arte vazando na sarjeta e a balões sobre a borda), **corte X-Y
   por perfis de projeção** dentro de cada região (sarjetas de 3 px sobrevivem;
   um corte atravessado por uma linha reta longa é interior de painel, nunca
   sarjeta), contornos como fallback e a página inteira por último. A ordem
   de leitura respeita a flag `reading_order` (rtl/ltr).
2. **Lixo de OCR** (`engines/text_cleaning.py`) — regex/heurísticas para
   descartar o que o EasyOCR alucina em mangá (`{`, `@`, "Jbl@jw", números de
   página, sopa de glifos sem vogais) sem perder reações como "?!" ou "Hmph".
3. **Plano de narração** (`engines/narration.py`) — anúncios de personagem,
   posição dos efeitos e o tagger por palavras-chave são funções puras sobre o
   roteiro; o orquestrador só renderiza o plano. Cada regra tem teste de
   milissegundos, sem TTS.

Limite honesto: arte que atravessa a sarjeta em toda a largura e balões cujo
contorno cruza a sarjeta (Berserk faz as duas coisas com frequência) ainda
fundem painéis. Uma segunda passada que apagava traços finos recuperava
algumas dessas sarjetas, mas também fatiava colunas de balões dentro de
painéis — perder diálogo é pior do que fundir painéis, então foi descartada;
essas páginas esperam o detector treinado previsto no mapa de decisões.

## 10. Gabarito humano para o Juiz Automático

`python -m tools.label_benchmark` mostra recortes do workspace e grava rótulos
por tecla de atalho em `tests/benchmark/gabarito/<volume>.json` (retomável,
salva a cada tecla). O gabarito referencia os recortes pelo nome — as páginas
em si nunca entram no repositório.

## 11. Vozes por perfil, fixadas por volume

O roteirista, que vê o painel, declara em cada fala o **perfil de voz** do
falante como desenhado (`voice`: homem, mulher, idoso, idosa, menino, menina,
criatura). Antes do TTS, o orquestrador consolida um perfil por personagem
(maioria dos votos, contando também o roteiro bruto, caso o revisor descarte o
campo) e o motor de voz **escala** cada falante uma única vez: elenco curado
primeiro, depois uma voz ainda não usada do banco daquele perfil (o XTTS traz
58 vozes de estúdio; o Edge, três vozes com deslocamentos de tom). A escolha
fica em `cast_voices.json` no workspace do volume e entra na fingerprint do
áudio — o "Sacerdote" da página 31 tem a mesma voz na página 200 e na
execução do mês que vem, e nunca recebe uma voz feminina. Sem perfil (modelo
que omitiu o campo), vale o pool antigo. Timbres de idade/criatura foram
agrupados de ouvido e podem ser ajustados em `engines/tts.py`.

## 12. Audiodescrição sóbria

O prompt do roteirista passou a exigir descrições só do que está desenhado
(quem, onde, o quê), sem atmosfera, sons ou sentimentos inventados, curtas e
sem repetir o que o diálogo já diz — a reclamação real do autor ouvindo o
primeiro áudio foi "inventa e enfeita demais". O preset `sobrio` aperta mais
(≤ 12 palavras, um bloco por painel, pular quando o diálogo basta). Excertos
de um narrador humano (`tools/transcribe_reference.py`) entram como
referência de tom por `--style-examples`, com a instrução de imitar ritmo e
foco, nunca copiar frases; como o addendum faz parte do hash do prompt,
roteiros antigos são invalidados de propósito.
