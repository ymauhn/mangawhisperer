# MangaWhisperer

Pipeline que transforma um mangá em PDF em um audiodrama multi-voz em PT-BR para
leitores cegos. Este glossário fixa os termos do domínio; decisões vivem em `docs/adr/`.

## Language

### Fonte (o mangá)

**Volume**:
Um PDF de mangá processado como unidade de trabalho, com workspace próprio.
_Avoid_: livro, arquivo, capítulo

**Page**:
Uma página do volume rasterizada em imagem.

**Panel**:
Uma região da página que contém um momento da cena; a unidade que o roteirista lê.
_Avoid_: quadro, frame, cena

**Bubble**:
Uma região de texto dentro de um painel (balão de fala ou caixa), de onde o OCR extrai texto.
_Avoid_: balão, caption, caixa de texto

**Reading Order**:
A ordem de leitura de mangá: painéis e balões da direita para a esquerda, de cima para baixo.
_Avoid_: sequência, ordenação

### Roteiro

**Script**:
O roteiro completo de um volume: painéis em ordem de leitura, cada um com seus blocos.
_Avoid_: transcrição, texto

**Block**:
A menor unidade narrável do roteiro: uma fala ou uma descrição de ação, com um rótulo de falante.
_Avoid_: linha, fala (é só um tipo de bloco), segmento (é áudio)

**Action Description**:
Bloco do Narrador que descreve o que só existe visualmente no painel — o núcleo da acessibilidade.
Regra de sobriedade: descreve apenas o que está desenhado (quem, onde, o quê), sem atmosfera, sons ou sentimentos inventados, e sem repetir o diálogo.

**Tone Reference**:
Excertos de um narrador humano (transcritos de uma narração de referência) anexados ao prompt do roteirista para imitar ritmo e foco — nunca para copiar frases; material local, fora do repositório.
_Avoid_: few-shot (o termo técnico; aqui é referência de tom), exemplo
_Avoid_: legenda, descrição de cena, caption

**Speaker Label**:
O nome atribuído a um bloco: um membro do elenco, um rótulo descritivo ou o Narrador.
_Avoid_: voz, personagem, speaker id

**Cast**:
O conjunto curado de personagens conhecidos, cada um com voz fixa.
_Avoid_: personagens principais, lista de vozes

**Descriptive Label**:
Rótulo curto para um falante fora do elenco ("Criatura", "Soldado"), reutilizado de forma consistente.
_Avoid_: Desconhecido (é o último recurso, não a regra)

**Narrator**:
A voz que lê descrições de ação e anúncios de falante.
_Avoid_: locutor, voz off

**Announcement**:
Bloco do Narrador que fala o rótulo do falante antes de uma fala, quando o falante muda.
_Avoid_: intro, tag de personagem

**Passthrough**:
Roteiro de último recurso: cada balão vira uma fala sem atribuição, sem descrições.
_Avoid_: fallback (genérico demais), modo cru

### Motores

**Scriptwriter**:
O estágio de visão-linguagem que produz os blocos de um painel a partir da imagem e do OCR.
_Avoid_: writer agent, VLM engine, roteirista-visão

**Reviewer**:
O estágio de LLM que revisa o roteiro inteiro depois do roteirista.
_Avoid_: critic, revisor de qualidade, segunda passada

**Provider**:
Um backend concreto (API ou local) por trás de um estágio de LLM.
_Avoid_: vendor, API

**Style**:
Uma direção de narração pré-definida que ajusta roteiro, ritmo e pausas de uma vez.
_Avoid_: preset (é o mecanismo), tom, modo

### Áudio

**Segment**:
Um arquivo de áudio na timeline: uma fala, um anúncio ou um efeito.
_Avoid_: clip, trecho, bloco

**Timeline**:
A sequência de segmentos com silêncios entre eles, antes da mixagem.
_Avoid_: faixa, track

**Effect**:
Um som curto disparado por um bloco, identificado por uma tag.
_Avoid_: SFX (sigla aceitável no código), sample

**Tag**:
A chave da biblioteca que nomeia um efeito ("espada", "trovão"); nome de arquivo ou entrada do dicionário.

**Voice Profile**:
Como o falante soa segundo o que está desenhado (homem, mulher, idoso, idosa, menino, menina, criatura); declarado pelo roteirista em cada fala e consolidado por personagem antes do TTS.
_Avoid_: gênero (é só uma das dimensões), timbre (é a voz concreta, não o perfil)

**Cast Voices Registry**:
A tabela persistente por volume (`cast_voices.json`) que fixa a voz concreta de cada personagem a partir do elenco curado e do banco do seu perfil; garante que o mesmo personagem tenha a mesma voz em todas as execuções.
_Avoid_: mapa de vozes, cache de vozes
_Avoid_: keyword, label

**Variant**:
Um dos vários arquivos sob a mesma tag, escolhido de forma determinística pela cena.
_Avoid_: alternativa, versão

**Intensity**:
Nível de 0 a 3 que regula quantos efeitos o roteiro recebe.
_Avoid_: frequência, densidade

**BGM Bed**:
Música ou ambiência em loop sob a timeline inteira.
_Avoid_: trilha sonora, soundtrack, música de fundo

### Qualidade

**Benchmark Set**:
Conjunto fixo de páginas de referência, cada uma com gabarito humano, contra o qual todo candidato a motor é medido.
_Avoid_: teste, amostra, eval

**Gabarito**:
A resposta humana de referência para uma página do Benchmark Set: falantes corretos, contagem de painéis, descrições consideradas usáveis.
_Avoid_: ground truth (no código pode aparecer, mas em prosa é Gabarito), label

**Rubric**:
Os critérios e a barra de aprovação aplicados ao Benchmark Set (atribuição de falante, descrições usáveis, onomatopeias tratadas).
_Avoid_: métrica, score

**Checker**:
Conjunto de verificações determinísticas sobre um roteiro (falante vs. gabarito, JSON válido, blocos vazios, onomatopeias, tags de efeito) que produz recompensas verificáveis.
_Avoid_: validador, linter, reward function (no código pode aparecer)

**Judge**:
O avaliador automático da parte subjetiva da rúbrica (qualidade de descrição, naturalidade), primeiro um LLM, depois um modelo pequeno destilado.
_Avoid_: crítico, avaliador, reviewer (é outro estágio)

### Identidade

**Gallery**:
Recortes de referência de cada membro do elenco, extraídos do próprio mangá, guardados por volume.
_Avoid_: banco de rostos, dataset

**Speaker Hint**:
Sugestão consultiva de quem está no painel (e, por geometria, quem fala cada balão), produzida sem LLM a partir da galeria; o scriptwriter usa para desempatar, não para decidir.
_Avoid_: diarização (é o resultado final), detecção de personagem (é só um passo)

### Execução

**Reading Order Flag**:
Declaração por volume de que a leitura é RTL (padrão) ou LTR; nunca inferida automaticamente.
_Avoid_: detecção de orientação

**Workspace**:
A pasta em disco de um volume com artefatos e checkpoints por estágio.
_Avoid_: output, pasta de saída

**Checkpoint**:
A saída persistida de um estágio, reutilizável em uma retomada.
_Avoid_: cache, snapshot

**Fingerprint**:
A identidade de configuração de um motor (provedor, modelo, prompt) que decide se um checkpoint é válido.
_Avoid_: hash, versão
