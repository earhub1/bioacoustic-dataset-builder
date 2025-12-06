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

## Passo a passo para calibrar Nothing sem viciar o modelo
Para evitar que o dataset fique dominado por silêncios muito homogêneos, ajuste a quantidade e a duração dos trechos Nothing antes de gerar o manifesto:

1. **Inspecione as durações positivas**: observe a distribuição de `offset_s - onset_s` no CSV original (por exemplo, com um histograma rápido no notebook ou em Python) para saber os limites típicos das classes anotadas.
2. **Defina um intervalo de duração variável**: escolha `--non-event-duration-range <min> <max>` inspirado nesses limites (ex.: 0.1–0.4 s), em vez de uma duração fixa. Isso cria fragmentos de fundo com tamanhos variados, mais próximos dos eventos reais.
3. **Balanceie a quantidade por arquivo**: use `--non-event-count` para manter uma razão controlada com os eventos anotados (ex.: 1:1 ou 1:2 por arquivo). Se um áudio tem poucas anotações, considere reduzir o count para não gerar um volume excessivo de negativos.
4. **Gere o manifesto Nothing**: execute o gerador com as escolhas acima. Exemplo:
   ```bash
   python src/generate_nothing_manifest.py \
     --csv-path data/events/labels_0_30kHz_reapath.csv \
     --non-event-count 2 \
     --non-event-duration-range 0.12 0.35 \
     --seed 42 \
     --nothing-manifest-path data/events/manifest_nothing.csv
   ```
5. **(Opcional) Crie o manifesto combinado**: inclua `--combined-manifest-path` para já obter um CSV com eventos + Nothing e reutilizá-lo diretamente no extrator.
6. **Extraia as features**: aponte `extract_fragments.py` para o manifesto escolhido (Nothing ou combinado) e gere os MFCCs.

## Montar sequências sintéticas com `build_dataset.py` (opcional)
Após extrair os fragmentos em `.npy`, você pode criar “fitas” sintéticas concatenando Nothing e eventos, ignorando rótulos indesejados (ex.: `NI`). Isso ajuda a treinar modelos com sequências mais realistas e balanceadas.

1. **Escolha os fragmentos de entrada**: defina os diretórios com `manifest.csv` resultantes da extração (padrão: `data/results/fragments`). Use `--fragments-dir` múltiplas vezes se quiser combinar fontes.
2. **Filtre labels**: por padrão, `NI` é excluído. Para incluir/excluir explicitamente, use `--include-labels` e/ou `--exclude-labels`.
3. **Defina duração e balanceamento**: use `--sequence-duration` para a duração alvo (s) e `--nothing-ratio` para controlar a razão Nothing:eventos (ex.: 1.0 ≈ 1:1 quando ambos existem). Ajuste `--num-sequences` para quantas fitas deseja.
4. **Gere as sequências**:
   ```bash
   python src/build_dataset.py \
     --fragments-dir data/results/fragments_combined \
     --exclude-labels NI \
     --sequence-duration 6.0 \
     --nothing-ratio 0.8 \
     --num-sequences 20 \
     --output-dir data/results/sequences \
     --seed 7
   ```
5. **Saídas**:
   - Sequências salvas como `.npy` em `data/results/sequences/sequence_<n>.npy`.
   - `manifest_sequences.csv` no mesmo diretório, contendo `sequence_path`, `total_duration_s`, `total_frames`, `n_segments`, `seed` e uma coluna `segments` (JSON) com a ordem, rótulo e posição (frames/segundos) de cada fragmento dentro da sequência.

## Próximos passos
- A partir dos fragmentos extraídos, você pode aplicar rotinas de balanceamento, split de treino/validação/teste e data augmentation conforme as necessidades do modelo alvo.
