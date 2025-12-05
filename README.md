# bioacoustic-dataset-builder

Ferramentas para preparar datasets bioacústicos (especialmente gravações marinhas) com recortes anotados, fragmentos de fundo e extração de features para modelos de Deep Learning / Machine Learning.

## Pré-requisitos
- Python 3.10+ (testado com CPython)
- Dependências listadas em `requirements.txt` (instale com `pip install -r requirements.txt`)

## Estrutura esperada
- Manifesto inicial de detecções (ex.: `data/events/labels_0_30kHz_reapath.csv`) contendo ao menos as colunas `filepath` (ou `file`), `onset_s`, `offset_s`, `label`. Caminhos relativos são resolvidos em relação ao diretório do CSV.
- Diretórios de saída (padrões):
  - `data/events/manifest_nothing.csv` e/ou `data/events/manifest_combined.csv` para manifestos de fundo.
  - `data/results/fragments/` para os `.npy` de MFCC e `manifest.csv` gerado pelo extrator.

## Fluxo recomendado (duas etapas)
1. **Gerar manifestos Nothing** com `generate_nothing_manifest.py` (sem extrair features):
   ```bash
   python src/generate_nothing_manifest.py \
     --csv-path data/events/labels_0_30kHz_reapath.csv \
     --non-event-count 3 \
     --non-event-duration-range 0.1 0.3 \
     --nothing-manifest-path data/events/manifest_nothing.csv \
     --combined-manifest-path data/events/manifest_combined.csv
   ```
   - `manifest_nothing.csv` contém apenas `Nothing` com `index=-1`, `label="Nothing"`, `filepath`, `onset_s`, `offset_s`, `duration_s`, `n_frames` e `source_filepath` para rastrear o trecho original.
   - `manifest_combined.csv` (opcional) concatena o CSV original com as linhas Nothing, preservando os índices originais.

2. **Extrair fragmentos e MFCCs** com `extract_fragments.py`, apontando para o manifesto desejado (original, Nothing ou combinado):
   ```bash
   # Extrair apenas Nothing
   python src/extract_fragments.py \
     --csv-path data/events/manifest_nothing.csv \
     --output-dir data/results/fragments_nothing

   # Extrair todos os eventos (originais + Nothing) usando o manifesto combinado
   python src/extract_fragments.py \
     --csv-path data/events/manifest_combined.csv \
     --output-dir data/results/fragments_combined
   ```
   O extrator lê cada linha do manifesto, recorta o trecho solicitado (downsample opcional via `--target-sr`), calcula MFCCs (`--n-mels`, `--frame-length`, `--hop-length`, `--window`) e grava os arquivos em subpastas por `label`, além de `manifest.csv` com `snippet_path`, `label`, `source_filepath`, `onset_s`, `offset_s`, `duration_s`, `n_frames` e `index` (herdado do CSV de entrada).

## Próximos passos
- A partir dos fragmentos extraídos, você pode aplicar rotinas de balanceamento, split de treino/validação/teste e data augmentation conforme as necessidades do modelo alvo.
