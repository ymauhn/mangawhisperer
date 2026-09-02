---
status: accepted
---

# As três trilhas de ML fazem parte do protótipo v2, nesta ordem

Um leitor pode estranhar que um protótipo "econômico e determinístico" carregue
pipelines de treino. É deliberado: o autor (cientista da computação) quer que o
projeto demonstre uso genuíno de sistemas inteligentes, e cada trilha **serve ao
destino** em vez de enfeitá-lo. Decisão de 2026-08-10, após as pesquisas #4–#8:

1. **RAG de identidade visual** — galeria do elenco + embeddings de recortes →
   *dica de falante* sem gastar tokens de LLM. Entra primeiro por ser barata,
   determinística na inferência e útil para qualquer scriptwriter.
2. **Juiz automático** — LLM-as-judge (Opus) sobre a rúbrica, destilado num juiz
   pequeno quando houver vereditos suficientes. É o que permite medir tudo o mais
   sem o autor ler cada roteiro — e serve de recompensa à trilha 3.
3. **Destilação do Opus num VLM pequeno** — por último, porque envolve espera de
   treino e GPU do Colab/Kaggle (GRPO não cabe em 8 GB). Conduzida como sessão
   guiada passo a passo com o autor, seguindo o plano por etapas registrado em
   `docs/research/alinhamento-vlm.md`, com regra de parada (SFT com F1 ≥ 95% do
   Opus dispensa o RL).

## Considered Options

- Deixar o ML fora do protótipo e tratá-lo como mapa futuro: mais simples, mas
  o destino "local com qualidade" provavelmente **depende** da destilação — os
  VLMs pequenos zero-shot não tinham entregado diarização aceitável até aqui.
- Bandits sobre feedback de ouvintes e fine-tuning de TTS: descartados (sem
  usuários ainda; ganho difuso).

## Consequences

- O Benchmark Set (#3) precisa de rótulos humanos em escala maior (50–100
  painéis) do que a rúbrica de 3 páginas pedia — o Opus também erra falante.
- Ambientes de treino ficam **separados** do runtime (TRL/Unsloth assumem
  transformers 5.x; o runtime pina <5 por causa do coqui-tts).
