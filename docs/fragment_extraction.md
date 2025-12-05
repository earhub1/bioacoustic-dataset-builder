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
- `--output-dir` (padrão `data/results/fragments`): destino dos `.npy` e do `manifest.csv`.
- `--inject-non-event`: ativa a geração de fragmentos adicionais rotulados como `Nothing` a partir de regiões não anotadas.
- `--non-event-count`: quantidade de fragmentos `Nothing` a gerar **por arquivo de áudio** quando a flag anterior está ativa.
- `--non-event-duration` ou `--non-event-duration-range MIN MAX`: controla a duração (fixa ou intervalo aleatório) dos fragmentos `Nothing`;
  caso omita ambos, a duração padrão é `frame_length / target_sr` (0,1 s com os padrões atuais).

## Processo
1. **Carregamento do CSV**: lê o manifesto de detecções e, se `limit` for definido, seleciona uma amostra aleatória estável (`seed`).
2. **Resolução dos caminhos**: caminhos relativos são resolvidos em relação ao diretório do CSV para localizar o arquivo de áudio correspondente.
3. **Recorte do áudio**: usa `librosa.load` para buscar apenas o trecho entre `onset_s` e `offset_s` já na SR alvo (`target_sr`). Se o áudio tiver múltiplos canais, apenas um canal é utilizado.
4. **Extração de MFCC**: calcula `n_mels` coeficientes com `frame_length`, `hop_length` e `window` definidos, produzindo uma matriz `[n_mels, n_frames]`.
5. **Persistência dos fragmentos**: salva os MFCCs em `.npy` dentro de uma subpasta com o nome do `label` (`output_dir/<label>/`), usando o índice da linha para compor o nome do arquivo.
6. **Manifesto de saída**: gera `output_dir/manifest.csv` contendo: `index`, `snippet_path`, `label`, `source_filepath`, `onset_s`, `offset_s`, `duration_s`, `n_frames`.
   - Fragmentos `Nothing` usam `label="Nothing"` e `index=-1` para indicar que não derivam de uma linha do CSV original.

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
