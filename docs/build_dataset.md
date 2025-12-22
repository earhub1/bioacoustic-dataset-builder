# Guia do `build_dataset.py`

Este documento detalha o funcionamento do CLI responsável por montar sequências sintéticas a partir dos fragmentos já extraídos pelo `extract_fragments.py`. Ele explica as entradas esperadas, as principais flags, a lógica de montagem das sequências e o formato do manifesto gerado.

## Visão geral
O `build_dataset.py` lê um ou mais `manifest.csv` produzidos pelo extrator, carrega os arquivos `.npy` de features correspondentes e os concatena para formar sequências mais longas. O processo é reprodutível (via `--seed`), permite filtrar labels (por exemplo, excluir `NI`), e balancear a proporção de trechos `Nothing` em relação às demais classes com `--nothing-ratio`.

## Entradas
- **Fragmentos**: diretórios contendo subpastas por label e um `manifest.csv` com as colunas `snippet_path`, `label`, `source_filepath`, `onset_s`, `offset_s`, `duration_s`, `n_frames` e `index` (Nothing com `index=-1`). Por padrão, o script procura em `data/results/fragments`, mas você pode passar um ou mais caminhos com `--fragments-dir`.
- **Filtros de label**: use `--include-labels` para listar explicitamente quais rótulos usar ou `--exclude-labels` (padrão: `NI`) para ignorar classes. Os filtros são aplicados após concatenar todos os manifests encontrados.

## Parâmetros principais
- `--sequence-duration`: duração alvo (em segundos) de cada sequência gerada no modo padrão de amostragem. O script converte essa duração em número de frames usando `frame_length` e `hop_length` (padrões: 6400 cada, com `target_sr=64000`, equivalendo a ~0,1 s por frame).
- `--target-event-fragments`: quando definido, o builder seleciona N fragmentos de um label de evento e calcula o orçamento de frames como o dobro do total desses eventos, visando 50/50 entre eventos e `Nothing`. Esse modo substitui `--sequence-duration` e exige `--event-label` quando houver mais de um evento.
- `--target-event-fragments-train/val/test`: orçamento de fragmentos de evento por split (treino/validação/teste). Exige `--split-by-fragment` e permite garantir 50/50 por frames em cada conjunto.
- `--event-label`: rótulo do evento a ser usado com `--target-event-fragments` (ex.: `G01`). Se houver apenas um evento disponível (além de `Nothing`), o script pode inferir automaticamente.
- `--split-by-fragment`: divide os fragmentos em `train`/`val`/`test` **sem reposição** antes da montagem das sequências. A amostragem passa a ocorrer apenas dentro do pool de cada split e um `manifest_split.csv` é salvo em `--output-dir` para manter a divisão fixa.
- `--split-by-event-fragments`: divide primeiro os fragmentos do `event_label` (e os demais labels) em `train`/`val`/`test` sem reposição, calcula o orçamento de frames por split a partir dos eventos e completa com `Nothing` até 50/50 em frames. Também grava `manifest_split.csv`.
- `--pack-all-fragments`: ativa o modo exaustivo, que consome cada fragmento exatamente uma vez, sem reposição, e distribui os frames entre os splits (`train`/`val`/`test`) conforme o orçamento definido pelas razões de split. Nesse modo, `--sequence-duration` não é usado para limitar as fitas; em vez disso você pode opcionalmente definir `--max-sequence-duration`.
- `--max-sequence-duration`: (apenas com `--pack-all-fragments`) duração máxima de cada sequência gerada. Se omitido, o script cria **uma sequência por split** contendo todos os frames atribuídos àquele conjunto. Se definido, o builder abre novas sequências sempre que a atual atingiria o limite, mantendo todos os fragmentos (sem truncar) e marcando o manifesto como `pack_all_mode=True`.
- `--max-fragments-per-sequence`: limite opcional de quantos fragmentos podem ser concatenados. Se atingido, a sequência é finalizada mesmo que a duração alvo não tenha sido alcançada.
- `--allow-partial-fragments`: por padrão, fragmentos maiores que o orçamento restante são ignorados e um novo trecho é sorteado. Ative esta flag para permitir incluir fragmentos longos mesmo que excedam o alvo; eles serão cortados na etapa final de truncamento.
- `--num-sequences`: quantas sequências gerar.
- `--train-ratio`, `--val-ratio`, `--test-ratio`: proporções (padrão 0.7/0.15/0.15) usadas para direcionar cada sequência gerada para as pastas `train/`, `val/` ou `test` sob `--output-dir`. Os valores devem somar 1.0.
- `--max-consecutive-event-fragments`: máximo de fragmentos de evento consecutivos permitidos (padrão: 3) antes de forçar a inserção de `Nothing` (quando existir pool disponível).
- `--max-consecutive-event-frames`: limite opcional em frames para eventos consecutivos antes de forçar `Nothing`. Se omitido, o controle é feito apenas por número de fragmentos.
- `--min-nothing-after-event-frames`: quantidade mínima de frames de `Nothing` exigida logo após um evento antes que outro evento possa ser colocado (padrão: 20 frames). Se o fragmento de `Nothing` exceder esse valor, o excesso simplesmente reduz o orçamento restante.
- `--allow-partial-fragments`: por padrão, fragmentos maiores que o orçamento restante são ignorados e um novo trecho é sorteado. Ative esta flag para permitir incluir fragmentos longos mesmo que ultrapassem o limite; eles serão cortados na etapa final de truncamento.
- `--num-sequences`: quantas sequências gerar.
- `--train-ratio`, `--val-ratio`, `--test-ratio`: proporções (padrão 0.7/0.15/0.15) usadas para direcionar cada sequência gerada para as pastas `train/`, `val/` ou `test` sob `--output-dir`. Os valores devem somar 1.0.
- `--nothing-ratio`: controla a probabilidade relativa de amostrar fragmentos `Nothing` versus demais labels quando ambos estão disponíveis. Por exemplo, 1.0 tende a um equilíbrio 1:1 entre `Nothing` e eventos; valores menores reduzem a presença de `Nothing`.
- `--validate-composition`: gera **uma** sequência para inspeção sem salvar arquivos, logando a linha do tempo dos segmentos, métricas de runs e a forma do tensor resultante. Útil para testar combinações de `nothing_ratio` e limites de consecutivos antes de produzir o dataset completo.
- `--seed`: fixa o gerador pseudoaleatório para que a escolha de trechos e a ordem se repitam entre execuções.

## Lógica de montagem
1. **Carregamento e filtros**: todos os manifests encontrados em `--fragments-dir` são lidos e concatenados; cada linha recebe a coluna auxiliar `_manifest_dir` para resolver `snippet_path` relativo. Aplica-se então `--include-labels`/`--exclude-labels`.
   - `snippet_path` pode ser absoluto ou relativo. Se for relativo e o caminho já existir tal como está, ele é usado diretamente; caso contrário, é resolvido em relação à pasta do manifest para evitar duplicar prefixos como `data/results/fragments/...`.
2. **Pool por label**: o script agrupa os índices das linhas por `label`, mantendo listas de candidatos para amostragem.
3. **Modo padrão (amostragem com reposição)**:
   - **Seleção de label**: a cada iteração, escolhe-se um label via `nothing_ratio` (função `pick_label`), respeitando os limites de consecutivos e o gap obrigatório de `Nothing`:
     - se só houver eventos (nenhum `Nothing`), amostra-se entre os eventos;
     - se só houver `Nothing`, amostra-se dele;
     - se ambos existirem, sorteia-se `Nothing` com peso `nothing_ratio` e os demais labels com peso 1, **exceto** quando o limite de eventos consecutivos (`--max-consecutive-event-fragments` ou `--max-consecutive-event-frames`) foi atingido ou ainda restarem frames pendentes de `Nothing` exigidos por `--min-nothing-after-event-frames` (nesses casos `Nothing` é forçado, se disponível).
   - **Amostragem de fragmento**: seleciona-se aleatoriamente uma linha do pool do label escolhido e carrega-se o `.npy` correspondente. O script ignora fragmentos ausentes ou com `n_frames <= 0`.
   - **Concatenação temporal**: os fragmentos são empilhados na dimensão temporal (`axis=1`). O processo continua até atingir ou ultrapassar o número de frames alvo derivado de `--sequence-duration` (ou de `--target-event-fragments`, quando usado), respeitando `--max-fragments-per-sequence` (quando definido), os limites de consecutivos e o gap mínimo de `Nothing`, além de um limite de tentativas para evitar laços infinitos.
4. **Split sem reposição (`--split-by-fragment`)**:
   - Antes de montar as sequências, o builder embaralha os fragmentos e os distribui em `train`/`val`/`test` sem reposição, conforme `train_ratio/val_ratio/test_ratio`.
   - A amostragem passa a ocorrer **apenas** dentro do pool do split correspondente, evitando que um mesmo fragmento apareça em treino e validação/teste.
   - Um `manifest_split.csv` é gravado em `--output-dir` para manter a divisão fixa e auditável.
5. **Split por eventos (`--split-by-event-fragments`)**:
   - O builder divide os fragmentos do `event_label` (e os demais labels) em `train`/`val`/`test` sem reposição.
   - Para cada split, o orçamento de frames é calculado a partir do total de frames do `event_label` daquele split, e o `Nothing` completa até atingir 50/50.
   - A amostragem ocorre apenas dentro do split, evitando vazamento entre treino/val/teste.
   - **Tratamento de fragmentos longos**: por padrão, se um fragmento exceder o orçamento restante de frames, ele é ignorado e outro trecho é sorteado. Com `--allow-partial-fragments`, o fragmento pode ser usado mesmo que ultrapasse o limite; a sequência será truncada no ajuste final, marcando o segmento como truncado.
   - **Ajuste final**: se a sequência exceder os frames alvo, é truncada. Cada segmento recebe `start_frame`, `end_frame`, `start_s`, `end_s` e `truncated` (quando houve corte) calculados a partir de `frame_length`/`hop_length`/`target_sr`.
4. **Modo exaustivo (`--pack-all-fragments`)**:
   - Todos os fragments são embaralhados e alocados aos splits por **orçamento de frames** derivado de `train/val/test` (o último split recebe qualquer resíduo para cobrir 100% dos frames).
   - Dentro de cada split, os fragments são concatenados **sem reposição**, preservando cada entrada exatamente uma vez. Se `--max-sequence-duration` for definido, o builder abre novas fitas sempre que a sequência atual atingiria o limite; se não for definido, gera uma única fita por split com todos os frames atribuídos.
   - Os segmentos mantêm os metadados (`label`, `snippet_path`, `start_frame`, `end_frame`, `start_s`, `end_s`) e o manifesto marca `pack_all_mode=True` para essas fitas.

## Saídas
- **Sequências**: salvas como `.npy` em subpastas de split sob `--output-dir` (padrão `data/results/sequences/{train,val,test}`) com o padrão `sequence_<n>.npy`. Cada arquivo contém um tensor de features concatenadas (mesma dimensão de frequência dos fragmentos de entrada).
- **Manifestos**:
  - `manifest_sequences_summary.csv`: resumo por fita, salvo na raiz de `--output-dir` (e em cada subpasta de split). Colunas principais: `sequence_path`, `sequence_idx`, `split`, `total_frames`, `total_duration_s`, `n_segments`, `pack_all_mode`, `seed`, `skipped_too_long`, `fragment_limit_reached`, `truncated_segments`, `feature_type`, `mel_bins`, `db_ref`, `top_db`, `frames_by_label` (JSON), `frames_nothing`, `frames_events`, `pct_nothing`, `pct_events`, `max_event_run_frames`, `max_event_run_seconds`, `max_event_run_fragments`, `num_event_runs`.
  - `manifest_sequences.csv`: manifesto detalhado por **segmento**, salvo na raiz (e por split). Cada linha indica um trecho dentro de uma sequência com: `sequence_path`, `sequence_idx`, `split`, `segment_idx`, `label`, `snippet_path`, `start_frame`, `end_frame`, `duration_frames`, `start_s`, `end_s`, `duration_s`, `truncated`, `feature_type`, `mel_bins`. Esse formato gera uma linha por trecho, facilitando auditoria e análises posteriores.
  - `manifest_split.csv`: salvo apenas quando `--split-by-fragment` está ativo, contém os fragmentos originais com a coluna `split` atribuída (sem reposição) para garantir que não haja vazamento entre treino/val/teste.

## Exemplos de uso
### Modo padrão (amostragem)
```bash
python src/build_dataset.py \
  --fragments-dir data/results/fragments_combined \
  --exclude-labels NI \
  --sequence-duration 6.0 \
  --nothing-ratio 0.8 \
  --max-consecutive-event-fragments 3 \
  --min-nothing-after-event-frames 25 \
  --num-sequences 20 \
  --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 \
  --output-dir data/results/sequences \
  --seed 7
```

Este comando gera 20 sequências de aproximadamente 6 s cada, balanceando a seleção de `Nothing` e eventos com `nothing-ratio=0.8`, ignorando a label `NI`, e grava as sequências nas subpastas `train/`, `val/` e `test` de `data/results/sequences`, além do `manifest_sequences.csv` agregado (com coluna `split`).

Para inspecionar rapidamente a linha do tempo e as métricas de composição antes de salvar qualquer arquivo, rode o mesmo comando com `--validate-composition` e `--num-sequences 1` (o flag ignora a escrita em disco e loga a timeline e os percentuais/limites aplicados).

### Duração baseada em eventos (50/50 por frames)
```bash
python src/build_dataset.py \
  --fragments-dir data/results/fragments_combined \
  --exclude-labels NI \
  --target-event-fragments 120 \
  --event-label G01 \
  --nothing-ratio 1.0 \
  --split-by-fragment \
  --num-sequences 3 \
  --output-dir data/results/sequences_balanced \
  --seed 7
```

Neste modo, o builder soma os frames de 120 fragmentos do evento `G01`, duplica esse total para formar o orçamento global (eventos + `Nothing`) e gera as sequências respeitando esse limite. Se houver mais de um label de evento disponível, `--event-label` é obrigatório.

### Orçamento por split (70/20/10 com 50/50 por frames)
```bash
python src/build_dataset.py \
  --fragments-dir data/results/fragments_combined \
  --exclude-labels NI \
  --event-label G01 \
  --target-event-fragments-train 6241 \
  --target-event-fragments-val 1783 \
  --target-event-fragments-test 892 \
  --nothing-ratio 1.0 \
  --split-by-fragment \
  --num-sequences 3 \
  --train-ratio 0.34 --val-ratio 0.33 --test-ratio 0.33 \
  --output-dir data/results/sequences_balanced \
  --seed 7
```

Neste modo, o orçamento de frames é calculado separadamente por split a partir do número de fragmentos de evento. O `manifest_split.csv` garante que nenhum fragmento apareça em mais de um conjunto.

### Split por eventos (50/50 por split sem vazamento)
```bash
python src/build_dataset.py \
  --fragments-dir data/results/fragments_combined \
  --exclude-labels NI \
  --event-label G01 \
  --nothing-ratio 1.0 \
  --split-by-event-fragments \
  --num-sequences 3 \
  --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 \
  --output-dir data/results/sequences_balanced \
  --seed 7
```

Neste modo, os eventos são divididos primeiro entre os splits e o orçamento 50/50 é calculado separadamente para cada conjunto, mantendo a separação sem vazamento.

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
Para visualizar rapidamente o colormesh das sequências salvas, use `visualize_sequence_colormesh.py`, que lê o `manifest_sequences.csv` segmentado e plota apenas o tensor (freq x frames) com eixo de tempo em segundos (10 fps com os defaults `hop_length=6400`, `target_sr=64000`). É possível filtrar por `sequence_idx`, `segment_idx`, split e limitar a janela com `--max-plot-duration`. Consulte `docs/visualize_sequences.md` para o passo a passo.

## Como consumir as saídas no treinamento

### Formato das features e metadados
- Cada `sequence_*.npy` contém um tensor **log-mel em dB** com `ref=1.0` e `top_db=80`, shape `(n_mels, n_frames)` (padrão `n_mels=64`). Nenhuma normalização estatística adicional é aplicada na extração ou no builder.
- O manifesto (`manifest_sequences_summary.csv`) registra `feature_type=logmel_db`, `mel_bins`, `db_ref` e `top_db`, além da duração total em frames e segundos. O `manifest_sequences.csv` detalha os segmentos (intervalos `start_frame`/`end_frame`, label, se foi truncado etc.), permitindo reconstruir as janelas temporais.
- A taxa de quadros segue a configuração do extrator (`hop_length` e `target_sr`), ficando em ~10 fps com os defaults (6400/64000). Use esses valores para converter frame → segundo na pós-processamento ou no data loader.

### Carregando e preparando lotes
- Carregue o tensor diretamente com `np.load(sequence_path)` e obtenha os segmentos relevantes a partir do manifesto. O eixo 0 são as frequências (mel bins) e o eixo 1 é o tempo (frames).
- Para modelos que operam em janelas fixas, fatia-se `features[:, start_frame:end_frame]` usando os índices do manifesto. Se necessário, aplique padding temporal à direita para igualar comprimentos dentro do batch, mantendo o eixo de frequência intacto.
- Exemplo simples em PyTorch (janelas fixas):
  ```python
  import numpy as np
  import torch

  seq = np.load(sequence_path)  # (n_mels, T)
  # recorte de um segmento do manifesto
  window = seq[:, start_frame:end_frame]
  # padding opcional para alinhar comprimentos
  pad_right = target_frames - window.shape[1]
  if pad_right > 0:
      window = np.pad(window, ((0, 0), (0, pad_right)), mode="constant", constant_values=-80.0)
  x = torch.from_numpy(window).float()  # (n_mels, target_frames)
  ```

### Normalização recomendada (após o carregamento)
- **Não** aplique `ref=np.max` ou normalização por fragmento/segmento. Preserve a referência absoluta em dB para manter a comparabilidade de energia entre exemplos.
- Calcule estatísticas **apenas no split de treino** e reutilize nos demais splits. Opções comuns:
  - média e desvio padrão globais por frequência: `mean = train.mean(axis=(1,2))`, `std = train.std(axis=(1,2))`
  - ou apenas `mean` global por frequência, deixando a escala original do desvio.
- Aplique a normalização escolhida **depois** de carregar a janela de interesse (p. ex., `(window - mean[:, None]) / (std[:, None] + 1e-6)`).

### Boas práticas adicionais
- Se seu modelo aceitar comprimentos variáveis, evite padding e use máscaras de atenção derivadas do comprimento real em frames do segmento.
- Para tarefas frame a frame, o manifesto fornece `start_s`/`end_s` que podem ajudar a alinhar labels temporais ou métricas de atraso.
- Os valores em dB já estão limitados por `top_db=80`; caso você use padding, preencha com um valor baixo coerente (ex.: `-80.0`) para não distorcer a distribuição.
