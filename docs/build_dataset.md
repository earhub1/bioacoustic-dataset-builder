# Guia do `build_dataset.py`

Este documento detalha o funcionamento do CLI responsável por montar sequências sintéticas a partir dos fragmentos já extraídos pelo `extract_fragments.py`. Ele explica as entradas esperadas, as principais flags, a lógica de montagem das sequências e o formato do manifesto gerado.

## Visão geral
O `build_dataset.py` lê um ou mais `manifest.csv` produzidos pelo extrator, carrega os arquivos `.npy` de features correspondentes e os concatena para formar sequências mais longas. O processo é reprodutível (via `--seed`), permite filtrar labels (por exemplo, excluir `NI`), e balancear a proporção de trechos `Nothing` em relação às demais classes com `--nothing-ratio`.

## Entradas
- **Fragmentos**: diretórios contendo subpastas por label e um `manifest.csv` com as colunas `snippet_path`, `label`, `source_filepath`, `onset_s`, `offset_s`, `duration_s`, `n_frames` e `index` (Nothing com `index=-1`). Por padrão, o script procura em `data/results/fragments`, mas você pode passar um ou mais caminhos com `--fragments-dir`.
- **Filtros de label**: use `--include-labels` para listar explicitamente quais rótulos usar ou `--exclude-labels` (padrão: `NI`) para ignorar classes. Os filtros são aplicados após concatenar todos os manifests encontrados.

## Parâmetros principais
- `--sequence-duration`: duração alvo (em segundos) de cada sequência gerada no modo padrão de amostragem. O script converte essa duração em número de frames usando `frame_length` e `hop_length` (padrões: 6400 cada, com `target_sr=64000`, equivalendo a ~0,1 s por frame).
- `--pack-all-fragments`: ativa o modo exaustivo, que consome cada fragmento exatamente uma vez, sem reposição, e distribui os frames entre os splits (`train`/`val`/`test`) conforme o orçamento definido pelas razões de split. Nesse modo, `--sequence-duration` não é usado para limitar as fitas; em vez disso você pode opcionalmente definir `--max-sequence-duration`.
- `--max-sequence-duration`: (apenas com `--pack-all-fragments`) duração máxima de cada sequência gerada. Se omitido, o script cria **uma sequência por split** contendo todos os frames atribuídos àquele conjunto. Se definido, o builder abre novas sequências sempre que a atual atingiria o limite, mantendo todos os fragmentos (sem truncar) e marcando o manifesto como `pack_all_mode=True`.
- `--max-fragments-per-sequence`: limite opcional de quantos fragmentos podem ser concatenados. Se atingido, a sequência é finalizada mesmo que a duração alvo não tenha sido alcançada.
- `--allow-partial-fragments`: por padrão, fragmentos maiores que o orçamento restante são ignorados e um novo trecho é sorteado. Ative esta flag para permitir incluir fragmentos longos mesmo que excedam o alvo; eles serão cortados na etapa final de truncamento.
- `--num-sequences`: quantas sequências gerar.
- `--train-ratio`, `--val-ratio`, `--test-ratio`: proporções (padrão 0.7/0.15/0.15) usadas para direcionar cada sequência gerada para as pastas `train/`, `val/` ou `test` sob `--output-dir`. Os valores devem somar 1.0.
- `--nothing-ratio`: controla a probabilidade relativa de amostrar fragmentos `Nothing` versus demais labels quando ambos estão disponíveis. Por exemplo, 1.0 tende a um equilíbrio 1:1 entre `Nothing` e eventos; valores menores reduzem a presença de `Nothing`.
- `--seed`: fixa o gerador pseudoaleatório para que a escolha de trechos e a ordem se repitam entre execuções.

## Lógica de montagem
1. **Carregamento e filtros**: todos os manifests encontrados em `--fragments-dir` são lidos e concatenados; cada linha recebe a coluna auxiliar `_manifest_dir` para resolver `snippet_path` relativo. Aplica-se então `--include-labels`/`--exclude-labels`.
   - `snippet_path` pode ser absoluto ou relativo. Se for relativo e o caminho já existir tal como está, ele é usado diretamente; caso contrário, é resolvido em relação à pasta do manifest para evitar duplicar prefixos como `data/results/fragments/...`.
2. **Pool por label**: o script agrupa os índices das linhas por `label`, mantendo listas de candidatos para amostragem.
3. **Modo padrão (amostragem com reposição)**:
   - **Seleção de label**: a cada iteração, escolhe-se um label via `nothing_ratio` (função `pick_label`):
     - se só houver eventos (nenhum `Nothing`), amostra-se entre os eventos;
     - se só houver `Nothing`, amostra-se dele;
     - se ambos existirem, sorteia-se `Nothing` com peso `nothing_ratio` e os demais labels com peso 1.
   - **Amostragem de fragmento**: seleciona-se aleatoriamente uma linha do pool do label escolhido e carrega-se o `.npy` correspondente. O script ignora fragmentos ausentes ou com `n_frames <= 0`.
   - **Concatenação temporal**: os fragmentos são empilhados na dimensão temporal (`axis=1`). O processo continua até atingir ou ultrapassar o número de frames alvo derivado de `--sequence-duration`, respeitando `--max-fragments-per-sequence` (quando definido) e um limite de tentativas para evitar laços infinitos.
   - **Tratamento de fragmentos longos**: por padrão, se um fragmento exceder o orçamento restante de frames, ele é ignorado e outro trecho é sorteado. Com `--allow-partial-fragments`, o fragmento pode ser usado mesmo que ultrapasse o limite; a sequência será truncada no ajuste final, marcando o segmento como truncado.
   - **Ajuste final**: se a sequência exceder os frames alvo, é truncada. Cada segmento recebe `start_frame`, `end_frame`, `start_s`, `end_s` e `truncated` (quando houve corte) calculados a partir de `frame_length`/`hop_length`/`target_sr`.
4. **Modo exaustivo (`--pack-all-fragments`)**:
   - Todos os fragments são embaralhados e alocados aos splits por **orçamento de frames** derivado de `train/val/test` (o último split recebe qualquer resíduo para cobrir 100% dos frames).
   - Dentro de cada split, os fragments são concatenados **sem reposição**, preservando cada entrada exatamente uma vez. Se `--max-sequence-duration` for definido, o builder abre novas fitas sempre que a sequência atual atingiria o limite; se não for definido, gera uma única fita por split com todos os frames atribuídos.
   - Os segmentos mantêm os metadados (`label`, `snippet_path`, `start_frame`, `end_frame`, `start_s`, `end_s`) e o manifesto marca `pack_all_mode=True` para essas fitas.

## Saídas
- **Sequências**: salvas como `.npy` em subpastas de split sob `--output-dir` (padrão `data/results/sequences/{train,val,test}`) com o padrão `sequence_<n>.npy`. Cada arquivo contém um tensor de features concatenadas (mesma dimensão de frequência dos fragmentos de entrada).
- **Manifestos**:
  - `manifest_sequences_summary.csv`: resumo por fita, salvo na raiz de `--output-dir` (e em cada subpasta de split). Colunas: `sequence_path`, `sequence_idx`, `split`, `total_frames`, `total_duration_s`, `n_segments`, `pack_all_mode`, `seed`, `skipped_too_long`, `fragment_limit_reached`, `truncated_segments`.
  - `manifest_sequences.csv`: manifesto detalhado por **segmento**, salvo na raiz (e por split). Cada linha indica um trecho dentro de uma sequência com: `sequence_path`, `sequence_idx`, `split`, `segment_idx`, `label`, `snippet_path`, `start_frame`, `end_frame`, `duration_frames`, `start_s`, `end_s`, `duration_s`, `truncated`. Esse formato gera uma linha por trecho, facilitando auditoria e análises posteriores.

## Exemplos de uso
### Modo padrão (amostragem)
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

### Modo exaustivo (sem reposição)
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

Neste modo, cada fragmento é usado exatamente uma vez, os frames totais são divididos pelos splits segundo o orçamento 70/20/10, e novas sequências são abertas a cada ~30 s (em frames). Se você omitir `--max-sequence-duration`, o script produzirá uma única sequência por split com todos os frames atribuídos e registrará `pack_all_mode=True` no manifesto.

## Visualização
As ferramentas de visualização serão redesenhadas. Por enquanto, baseie-se nos manifestos gerados (`manifest_sequences.csv` e manifestos por split) para conferir a composição das fitas.
