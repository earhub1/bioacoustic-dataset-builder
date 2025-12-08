# Fluxo de extração de fragmentos

O script [`src/extract_fragments.py`](../src/extract_fragments.py) implementa uma CLI para gerar fragmentos de áudio anotados e
seus coeficientes de Mel (MFCC) a partir de um manifesto CSV.

## Entrada
- CSV de detecções (padrão: `data/events/labels_0_30kHz_reapath.csv`).
- Colunas esperadas: `filepath` (ou `file`), `onset_s`, `offset_s`, `label`.

## Parâmetros principais
- `--target-sr` (padrão `64000`): taxa de amostragem usada ao carregar o trecho; realiza downsample se o arquivo tiver SR maior.
- `--n-mels` (padrão `9`): número de coeficientes MFCC por frame.
- `--window` (padrão `hann`): janela aplicada ao cálculo dos MFCCs.
- `--frame-length` (padrão `6400`): tamanho da FFT em amostras na SR de destino.
- `--hop-length` (padrão `6400`): salto entre frames; com SR=64 kHz, cada frame representa 0,1 s.
- `--limit` e `--seed` (padrões `None` e `42`): amostra aleatoriamente até `limit` linhas para extração reprodutível.
- `--min-duration` / `--max-duration` (padrões `None`): filtram linhas com `offset_s - onset_s` fora do intervalo desejado antes
  de extrair; o extrator registra quantas foram mantidas ou descartadas.
- `--max-per-label` / `--max-nothing` (padrões `None`): aplicam tetos por rótulo antes da amostragem global; se um grupo excede
  o limite, o extrator sorteia um subconjunto reprodutível por `label` (com `--seed`) e registra quantas linhas foram mantidas
  ou descartadas. `--max-nothing` vale apenas para `Nothing`; os demais rótulos usam `--max-per-label`.
- `--output-dir` (padrão `data/results/fragments`): destino dos `.npy` e do `manifest.csv`.

Para eventos "Nothing", gere primeiro um manifesto dedicado ou combinado com o `generate_nothing_manifest.py` (detalhado abaixo) e em seguida execute o `extract_fragments.py` apontando para esse CSV.

## Processo
1. **Carregamento do CSV**: lê o manifesto de detecções, aplica filtros opcionais de duração (`--min-duration`/`--max-duration`)
   e tetos por rótulo (`--max-per-label`/`--max-nothing`) registrando quantas linhas foram descartadas. Se `limit` for definido,
   uma amostra aleatória estável (`seed`) é aplicada depois desses filtros.
2. **Resolução dos caminhos**: caminhos relativos são resolvidos em relação ao diretório do CSV para localizar o arquivo de áudio correspondente.
3. **Recorte do áudio**: usa `librosa.load` para buscar apenas o trecho entre `onset_s` e `offset_s` já na SR alvo (`target_sr`). Se o áudio tiver múltiplos canais, apenas um canal é utilizado.
4. **Extração de MFCC**: calcula `n_mels` coeficientes com `frame_length`, `hop_length` e `window` definidos, produzindo uma matriz `[n_mels, n_frames]`.
5. **Persistência dos fragmentos**: salva os MFCCs em `.npy` dentro de uma subpasta com o nome do `label` (`output_dir/<label>/`), usando o índice da linha para compor o nome do arquivo.
6. **Manifesto de saída**: gera `output_dir/manifest.csv` contendo: `index`, `snippet_path`, `label`, `source_filepath`, `onset_s`, `offset_s`, `duration_s`, `n_frames`. Fragmentos seguem os campos já presentes no CSV de entrada (se você apontar para um manifesto que contenha `Nothing`, eles serão extraídos e registrados da mesma forma, preservando `index=-1`).

## Fluxo em duas etapas com manifestos de Nothing
Para reduzir o acoplamento e inspecionar os trechos de fundo antes de extrair MFCCs, você pode gerar um manifesto dedicado de `Nothing` e, opcionalmente, um manifesto combinado:

1. **Gerar manifesto Nothing**: use [`src/generate_nothing_manifest.py`](../src/generate_nothing_manifest.py) para amostrar intervalos livres por arquivo de áudio (respeitando `--non-event-count`, `--non-event-duration` ou `--non-event-duration-range`, e `--seed`). O script grava um `manifest_nothing.csv` com colunas `index=-1`, `label="Nothing"`, `filepath`, `onset_s`, `offset_s`, `duration_s` e `n_frames` estimados.
2. **Manifesto combinado (opcional)**: forneça `--combined-manifest-path` para gravar um CSV que concatena as anotações originais com as linhas `Nothing` (mantendo `index` original ou `-1` para Nothing).
3. **Extrair MFCCs**: a qualquer momento, aponte o `extract_fragments.py` para o manifesto desejado (original, Nothing ou combinado) para materializar os fragmentos como `.npy`.

### Exemplo: gerar e extrair Nothing separadamente
```bash
# Passo 1: gerar manifesto dedicado de Nothing e um combinado
python src/generate_nothing_manifest.py \
  --csv-path data/events/labels_0_30kHz_reapath.csv \
  --non-event-count 3 \
  --non-event-duration-range 0.1 0.3 \
  --nothing-manifest-path data/events/manifest_nothing.csv \
  --combined-manifest-path data/events/manifest_combined.csv

# Passo 2: extrair MFCCs apenas para Nothing
python src/extract_fragments.py \
  --csv-path data/events/manifest_nothing.csv \
  --output-dir data/results/fragments_nothing

# Passo 3 (opcional): extrair MFCCs de todos os eventos usando o manifesto combinado
python src/extract_fragments.py \
  --csv-path data/events/manifest_combined.csv \
  --output-dir data/results/fragments_combined
```

### Exemplo: limitar por rótulo antes de extrair
```bash
python src/extract_fragments.py \
  --csv-path data/events/manifest_combined.csv \
  --output-dir data/results/fragments_limited \
  --max-per-label 1000 \
  --max-nothing 2000 \
  --max-duration 67 \
  --seed 42
```
O comando acima mantém no máximo 1000 linhas para cada rótulo diferente de `Nothing` e até 2000 para `Nothing` antes da
amostragem global (`--limit`), além de descartar eventos acima de 67 s. O log informa quantas linhas por label foram mantidas
ou descartadas; o `manifest.csv` final reflete apenas esse subconjunto.

## Análise exploratória das durações (suporte para calibrar Nothing)
Antes de definir os ranges de duração para Nothing, você pode inspecionar a distribuição das anotações originais com `src/analyze_event_durations.py` e calcular o intervalo central que cobre a maior parte dos eventos:

```bash
python src/analyze_event_durations.py \
  --csv-path data/events/labels_0_30kHz_reapath.csv \
  --output-dir docs/figures \
  --coverage 0.9 \
  --include-labels DELFIN BALEIA \   # opcional
  --min-duration 0.02 --max-duration 5  # opcional
```

O script calcula `duration_s = offset_s - onset_s`, salva `duration_stats.csv` (média, mediana, p05/p95, min/max por label e geral) e gera `duration_histograms.png` e `duration_boxplots.png` no diretório escolhido (padrão `docs/figures/`). A tabela inclui `central_lower_s`/`central_upper_s` correspondentes ao `--coverage` escolhido (padrão 90%), também destacados no histograma com linhas verticais. Use esse intervalo central para definir `--non-event-duration-range` ao rodar o `generate_nothing_manifest.py`, descartando outliers ao selecionar trechos de fundo.
