---
status: accepted
---

# Caminho padrão 100% local e offline; APIs só como provedores opcionais

O protótipo v2 (mapa Wayfinder de 2026-08-10) é **local-first**: um comando instala e
roda na máquina do usuário com GPU consumer (piso: 8 GB de VRAM), sem rede e sem
custo por uso. Provedores de API — pagos (Anthropic, OpenAI, Kimi, Qwen) ou em
free-tier — continuam disponíveis como opção plugável, mas **nunca como padrão**:
free-tiers expiram e mudam de regra, e uma dependência de endpoint alheio não é
"grátis", é emprestada. Pelo mesmo motivo o Edge-TTS (endpoint não-oficial da
Microsoft) fica como fallback explícito para máquinas sem GPU, não como base.

## Considered Options

- Free-tier de API como padrão (DashScope 1M tokens/90 dias; Gemini): melhor
  qualidade imediata, mas fundação instável e exige rede.
- Hospedagem (HF Space/servidor) para uso sem instalação: custo contínuo de GPU,
  o oposto de "cost-efficient"; fica para um mapa próprio quando os modelos leves
  estiverem escolhidos.

## Consequences

- Toda escolha de modelo (VLM, TTS, classificador) é avaliada primeiro em
  qualidade *local*, contra o Benchmark Set, com o Opus como referência de ouro.
- Máquinas sem GPU recebem um caminho degradado (VLM ≤3B, Edge/Kokoro), não uma
  rebaixa do caminho principal.
