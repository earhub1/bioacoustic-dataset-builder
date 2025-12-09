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
     
    # Exemplo com filtros de duração e tetos por rótulo
    python src/extract_fragments.py \
      --csv-path data/events/manifest_combined.csv \
      --output-dir data/results/fragments_limited \
      --max-per-label 1000 \          # limite por rótulo (exceto Nothing)
      --max-nothing 2000 \            # limite específico para Nothing
      --max-duration 67 \             # descarta outliers longos
      --seed 42
   ```
   O extrator lê cada linha do manifesto, pode aplicar filtros de duração (`--min-duration`/`--max-duration` em segundos) antes de
   processar, aplicar tetos por rótulo (`--max-per-label` e `--max-nothing`) antes de qualquer amostragem global (`--limit`),
   recortar o trecho solicitado (downsample opcional via `--target-sr`), calcular MFCCs (`--n-mels`, `--frame-length`,
   `--hop-length`, `--window`) e gravar os arquivos em subpastas por `label`, além de `manifest.csv` com `snippet_path`, `label`,
   `source_filepath`, `onset_s`, `offset_s`, `duration_s`, `n_frames` e `index` (herdado do CSV de entrada).

## Passo a passo para calibrar Nothing sem viciar o modelo
Para evitar que o dataset fique dominado por silêncios muito homogêneos, ajuste a quantidade e a duração dos trechos Nothing antes de gerar o manifesto:

1. **Inspecione as durações positivas**: use o CLI `src/analyze_event_durations.py` para gerar estatísticas e gráficos (histogramas e boxplots) das durações `offset_s - onset_s` por label e no geral, incluindo o intervalo central que cobre a fração desejada dos eventos (`--coverage`, padrão 0.9). Exemplo:
   ```bash
   python src/analyze_event_durations.py \
     --csv-path data/events/labels_0_30kHz_reapath.csv \
     --output-dir docs/figures \
     --coverage 0.9 \
     --include-labels DELFIN BALEIA  # opcional
   ```
   O comando grava `duration_stats.csv`, `duration_histograms.png` e `duration_boxplots.png` (por padrão em `docs/figures/`). A tabela inclui as colunas `central_lower_s`/`central_upper_s` para o intervalo central (ex.: 90%) e os histogramas destacam esse range com linhas vermelhas, ajudando a escolher um `--non-event-duration-range` que desconsidere outliers.
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

> Veja `docs/build_dataset.md` para uma descrição detalhada das entradas, parâmetros e do manifesto de saída.

1. **Escolha os fragmentos de entrada**: defina os diretórios com `manifest.csv` resultantes da extração (padrão: `data/results/fragments`). Use `--fragments-dir` múltiplas vezes se quiser combinar fontes.
2. **Filtre labels**: por padrão, `NI` é excluído. Para incluir/excluir explicitamente, use `--include-labels` e/ou `--exclude-labels`.
3. **Defina duração, balanceamento e splits**: use `--sequence-duration` para a duração alvo (s) e `--nothing-ratio` para controlar a razão Nothing:eventos (ex.: 1.0 ≈ 1:1 quando ambos existem). Ajuste `--num-sequences` para quantas fitas deseja. Se precisar limitar quantos trechos entram em cada sequência, use `--max-fragments-per-sequence`; para aceitar fragmentos maiores que o orçamento restante, ative `--allow-partial-fragments` (por padrão eles são descartados e um novo trecho é sorteado). Controle o split de saída com `--train-ratio`, `--val-ratio` e `--test-ratio` (padrão 0.7/0.15/0.15); as sequências serão gravadas em subpastas `train/`, `val/` e `test` sob `--output-dir`.
4. **Gere as sequências (modo padrão)**:
   ```bash
  python src/build_dataset.py \
    --fragments-dir data/results/fragments_combined \
    --exclude-labels NI \
    --sequence-duration 6.0 \
  --nothing-ratio 0.8 \
  --num-sequences 20 \
    --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 \
  --max-fragments-per-sequence 8 \
  --output-dir data/results/sequences \
  --seed 7
  ```
5. **(Opcional) Modo exaustivo**: se quiser usar todos os fragmentos uma única vez, sem reposição, e alocar frames por orçamento de split, execute com `--pack-all-fragments`. Inclua `--max-sequence-duration` para abrir novas fitas quando a atual atingir esse limite; se omitir, será gerada uma única sequência por split com todos os frames atribuídos. Exemplo:
   ```bash
   python src/build_dataset.py \
     --fragments-dir data/results/fragments_combined \
     --exclude-labels NI \
     --pack-all-fragments \
     --max-sequence-duration 30 \
     --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 \
     --output-dir data/results/sequences_pack_all \
     --seed 7
   ```
6. **Saídas**:
   - Sequências salvas como `.npy` em `data/results/sequences/{train,val,test}/sequence_<n>.npy`.
   - `manifest_sequences_summary.csv` na raiz de `data/results/sequences` e por split, com uma linha por fita: `sequence_path`, `sequence_idx`, `split`, `total_duration_s`, `total_frames`, `n_segments`, `seed`, `skipped_too_long`, `fragment_limit_reached`, `truncated_segments`, `pack_all_mode`.
   - `manifest_sequences.csv` (manifesto por segmento) na raiz de `data/results/sequences` e por split, com uma linha por trecho usado: `sequence_path`, `sequence_idx`, `split`, `segment_idx`, `label`, `snippet_path`, `start_frame`, `end_frame`, `duration_frames`, `start_s`, `end_s`, `duration_s`, `truncated`.
7. **(Opcional) Visualizar sequências**: execute `python src/visualize_sequences.py --sequence-manifest data/results/sequences/manifest_sequences.csv --fragments-dir data/results/fragments_combined --output-dir data/results/sequence_viz --splits train val` para gerar PNGs com waveform reconstruído, espectrograma, MFCC armazenado e uma máscara binária (0 = Nothing, 1 = demais classes). Use `--max-plot-duration` para recortar fitas longas e `--viz-sr` para downsample só do plot, evitando estouro de memória em espectrogramas. Consulte `docs/visualize_sequences.md` (e o resumo em `docs/build_dataset.md`) para mais detalhes — e use `src/visualize_sequences_minimal.py` se quiser renderizar apenas algumas sequências específicas ou um intervalo de `segment_idx` dentro delas.

## Próximos passos
- A partir dos fragmentos extraídos, você pode aplicar rotinas de balanceamento, split de treino/validação/teste e data augmentation conforme as necessidades do modelo alvo.
