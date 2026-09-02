# Benchmark Set (Ticket #3 — Gabarito)

Conjunto de avaliação para o **Juiz Automático** e para as heurísticas de
layout/OCR. Aqui ficam apenas os **gabaritos** (rótulos humanos em JSON);
as páginas do mangá são material protegido e **nunca** entram no repositório
— o gabarito referencia os recortes pelo nome (`page003_panel01`), que o
pipeline regenera deterministicamente a partir do PDF local.

## Estrutura

```
tests/benchmark/
├── README.md
└── gabarito/
    └── <volume>.json      # ex.: berserk_vol_01.json (gerado pela ferramenta)
```

Formato de `gabarito/<volume>.json`:

```json
{
  "volume": "berserk_vol_01",
  "speakers": {
    "page_031_panel01/b2": {"text": "Vamos.", "label": "Guts", "predicted": "Desconhecido"}
  },
  "panels": {
    "page_031": {"predicted": 4, "verdict": "under", "true_count": 6}
  }
}
```

As chaves usam a página **real do PDF** (`page_031`, lida do `run_config.json`
do workspace), então gabaritos de execuções com `--start` diferentes se
alinham; um balão cujo texto OCR mudou entre execuções volta para a fila.

* `speakers` — quem fala cada balão (rótulo humano vs. previsão do pipeline).
  Rótulos especiais: `Narrador` e `__lixo__` (OCR que não é fala).
* `panels` — veredito sobre a segmentação da página: `ok`, `under`
  (sub-segmentada), `over` (super-segmentada) ou `wrong`, com a contagem real.

## Como rotular

1. Rode o pipeline no trecho desejado (o gabarito usa o workspace resultante):

   ```bash
   python main_demo.py --vlm passthrough --tts silent --no-review --start 8 --pages 20
   ```

2. Abra a ferramenta (janela matplotlib; as teclas aparecem no título):

   ```bash
   python -m tools.label_benchmark --volume berserk_vol_01 --mode speakers
   ```

   ```bash
   python -m tools.label_benchmark --volume berserk_vol_01 --mode panels
   ```

   Teclas em `speakers`: `1..9` personagem do elenco, `0` Narrador, `o` outro
   (digita o nome no terminal), `x` lixo de OCR, `s` pular, `q` sair.
   Teclas em `panels`: `y` ok, `u` sub-segmentada, `v` super-segmentada,
   `n` errada, `c` digitar a contagem real, `s` pular, `q` sair.

A ferramenta salva a cada tecla e retoma de onde parou (itens já rotulados
não reaparecem). Meta inicial: 50–100 balões e 20–50 páginas por volume.

Os arquivos `gabarito/*.json` guardam o texto OCR dos balões (falas da obra),
por isso ficam **fora do git** (`.gitignore`); compartilhe-os por outro canal
se precisar reproduzir o benchmark em outra máquina.
