# Guia do `build_dataset.py`

Este documento detalha o funcionamento do CLI responsável por montar sequências sintéticas a partir dos fragmentos já extraídos pelo `extract_fragments.py`. Ele explica as entradas esperadas, as principais flags, a lógica de montagem das sequências e o formato do manifesto gerado.

## Visão geral
O `build_dataset.py` lê um ou mais `manifest.csv` produzidos pelo extrator, carrega os arquivos `.npy` de features correspondentes e os concatena para formar sequências mais longas. O processo é reprodutível (via `--seed`), permite filtrar labels (por exemplo, excluir `NI`), e balancear a proporção de trechos `Nothing` em relação às demais classes com `--nothing-ratio`.

## Entradas
- **Fragmentos**: diretórios contendo subpastas por label e um `manifest.csv` com as colunas `snippet_path`, `label`, `source_filepath`, `onset_s`, `offset_s`, `duration_s`, `n_frames` e `index` (Nothing com `index=-1`). Por padrão, o script procura em `data/results/fragments`, mas você pode passar um ou mais caminhos com `--fragments-dir`.
- **Filtros de label**: use `--include-labels` para listar explicitamente quais rótulos usar ou `--exclude-labels` (padrão: `NI`) para ignorar classes. Os filtros são aplicados após concatenar todos os manifests encontrados.

## Parâmetros principais
- `--sequence-duration`: duração alvo (em segundos) de cada sequência gerada. O script converte essa duração em número de frames usando `frame_length` e `hop_length` (padrões: 6400 cada, com `target_sr=64000`, equivalendo a ~0,1 s por frame).
- `--max-fragments-per-sequence`: limite opcional de quantos fragmentos podem ser concatenados. Se atingido, a sequência é finalizada mesmo que a duração alvo não tenha sido alcançada.
- `--allow-partial-fragments`: por padrão, fragmentos maiores que o orçamento restante são ignorados e um novo trecho é sorteado. Ative esta flag para permitir incluir fragmentos longos mesmo que excedam o alvo; eles serão cortados na etapa final de truncamento.
- `--num-sequences`: quantas sequências gerar.
- `--train-ratio`, `--val-ratio`, `--test-ratio`: proporções (padrão 0.7/0.15/0.15) usadas para direcionar cada sequência gerada para as pastas `train/`, `val/` ou `test` sob `--output-dir`. Os valores devem somar 1.0.
- `--nothing-ratio`: controla a probabilidade relativa de amostrar fragmentos `Nothing` versus demais labels quando ambos estão disponíveis. Por exemplo, 1.0 tende a um equilíbrio 1:1 entre `Nothing` e eventos; valores menores reduzem a presença de `Nothing`.
- `--seed`: fixa o gerador pseudoaleatório para que a escolha de trechos e a ordem se repitam entre execuções.

## Lógica de montagem
1. **Carregamento e filtros**: todos os manifests encontrados em `--fragments-dir` são lidos e concatenados; cada linha recebe a coluna auxiliar `_manifest_dir` para resolver `snippet_path` relativo. Aplica-se então `--include-labels`/`--exclude-labels`.
2. **Pool por label**: o script agrupa os índices das linhas por `label`, mantendo listas de candidatos para amostragem.
3. **Seleção de label**: a cada iteração, escolhe-se um label via `nothing_ratio` (função `pick_label`):
   - se só houver eventos (nenhum `Nothing`), amostra-se entre os eventos;
   - se só houver `Nothing`, amostra-se dele;
   - se ambos existirem, sorteia-se `Nothing` com peso `nothing_ratio` e os demais labels com peso 1.
4. **Amostragem de fragmento**: seleciona-se aleatoriamente uma linha do pool do label escolhido e carrega-se o `.npy` correspondente. O script ignora fragmentos ausentes ou com `n_frames <= 0`.
5. **Concatenação temporal**: os fragmentos são empilhados na dimensão temporal (`axis=1`). O processo continua até atingir ou ultrapassar o número de frames alvo derivado de `--sequence-duration`, respeitando `--max-fragments-per-sequence` (quando definido) e um limite de tentativas para evitar laços infinitos.
6. **Tratamento de fragmentos longos**: por padrão, se um fragmento exceder o orçamento restante de frames, ele é ignorado e outro trecho é sorteado. Com `--allow-partial-fragments`, o fragmento pode ser usado mesmo que ultrapasse o limite; a sequência será truncada no ajuste final, marcando o segmento como truncado.
7. **Ajuste final**: se a sequência exceder os frames alvo, é truncada. Cada segmento recebe `start_frame`, `end_frame`, `start_s`, `end_s` e `truncated` (quando houve corte) calculados a partir de `frame_length`/`hop_length`/`target_sr`.

## Saídas
- **Sequências**: salvas como `.npy` em subpastas de split sob `--output-dir` (padrão `data/results/sequences/{train,val,test}`) com o padrão `sequence_<n>.npy`. Cada arquivo contém um tensor de features concatenadas (mesma dimensão de frequência dos fragmentos de entrada).
- **Manifesto de sequências** (`manifest_sequences.csv`): salvo no diretório raiz de `--output-dir`, com uma coluna `split` indicando o destino (`train`, `val` ou `test`). Um manifesto separado é salvo em cada subpasta de split, contendo apenas as sequências daquele conjunto. Cada linha descreve uma sequência gerada com as colunas:
  - `sequence_path`: caminho para o `.npy` salvo.
  - `total_frames` / `total_duration_s`: frames e duração total da sequência.
  - `n_segments`: quantidade de fragmentos concatenados.
  - `segments`: JSON com a lista ordenada dos trechos incluídos, contendo `label`, `snippet_path`, `start_frame`, `end_frame`, `start_s`, `end_s` e `truncated` para cada segmento.
  - `seed`: valor usado para a amostragem reprodutível.
  - `skipped_too_long`: quantos fragmentos foram descartados por excederem o orçamento restante sem `--allow-partial-fragments`.
  - `fragment_limit_reached`: indica se a sequência encerrou por atingir `--max-fragments-per-sequence`.
  - `truncated_segments`: quantos segmentos foram cortados na etapa final de truncamento.
  - `split`: conjunto destino da sequência.

## Exemplo de uso
```bash
python src/build_dataset.py \
  --fragments-dir data/results/fragments_combined \
  --exclude-labels NI \
  --sequence-duration 6.0 \
  --nothing-ratio 0.8 \
  --num-sequences 20 \
  --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 \
  --output-dir data/results/sequences \
  --seed 7
```

Este comando gera 20 sequências de aproximadamente 6 s cada, balanceando a seleção de `Nothing` e eventos com `nothing-ratio=0.8`, ignorando a label `NI`, e grava as sequências nas subpastas `train/`, `val/` e `test` de `data/results/sequences`, além do `manifest_sequences.csv` agregado (com coluna `split`).

## Visualizar sequências geradas (opcional)
Use `src/visualize_sequences.py` para inspecionar cada fita sintética. O script reconstrói a linha do tempo aproximada em áudio usando as anotações do manifesto de fragmentos e gera um subplot com quatro faixas: waveform reconstituído, espectrograma, MFCC armazenado e máscara binária (0 = Nothing, 1 = demais classes).

Entradas principais:
- `--sequence-manifest`: caminho para o `manifest_sequences.csv` produzido pelo builder (padrão: `data/results/sequences/manifest_sequences.csv`).
- `--fragments-dir`: um ou mais diretórios contendo os fragmentos originais e respectivos `manifest.csv` (padrão: `data/results/fragments`).
- Filtros: `--splits` para focar em `train/val/test` específicos e `--max-sequences` para limitar quantos arquivos são renderizados.
- Parâmetros de tempo: `--frame-length`, `--hop-length`, `--target-sr` para alinhar os eixos de tempo; `--n-fft` e `--spectrogram-hop-length` para o espectrograma.

Exemplo:
```bash
python src/visualize_sequences.py \
  --sequence-manifest data/results/sequences/manifest_sequences.csv \
  --fragments-dir data/results/fragments_combined \
  --splits train val \
  --max-sequences 3 \
  --output-dir data/results/sequence_viz
```

Saídas:
- PNGs em `--output-dir`, um por sequência (`sequence_<n>.png`).
- Cada figura inclui: (1) waveform reconstruído somando os trechos originais na posição indicada; (2) espectrograma (dB); (3) MFCC da sequência (carregado do `.npy` da sequência); (4) máscara binária que marca `Nothing` como 0 e demais classes como 1.
