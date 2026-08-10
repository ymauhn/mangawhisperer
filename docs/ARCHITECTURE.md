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
Pins de versão que importam: `transformers>=4.57,<5` (o coqui-tts quebra com a
5.x — issue idiap/coqui-ai-TTS#558) e torch do canal cu128 (Blackwell/sm_120).
