# Visualizar sequências sintéticas (`visualize_sequences.py`)

Esta ferramenta gera um painel de 4 faixas para cada sequência criada pelo `build_dataset.py`, reconstruindo uma linha do tempo aproximada do áudio original e exibindo:
1) waveform resultante; 2) espectrograma em dB; 3) MFCC armazenado na sequência `.npy`; 4) máscara binária que diferencia Nothing (0) das demais classes (1).

## Entradas necessárias
- `manifest_sequences.csv` gerado pelo `build_dataset.py` (manifesto **por segmento**; aceita filtros por split/quantidade).
- Um ou mais diretórios de fragmentos contendo `manifest.csv` e os arquivos `.npy` originais (por padrão `data/results/fragments`). O script usa o `snippet_path` de cada segmento para buscar metadados como `source_filepath`, `onset_s` e `offset_s`, permitindo recarregar o áudio base.

## Principais parâmetros
- `--sequence-manifest`: caminho do manifesto de sequências por segmento (padrão: `data/results/sequences/manifest_sequences.csv`).
- `--fragments-dir`: diretório(s) com `manifest.csv` dos fragmentos (pode repetir a flag; padrão: `data/results/fragments`).
- Filtros: `--splits` (ex.: `train val test`) e `--max-sequences` para limitar a quantidade renderizada.
- Tempo e espectrograma: `--frame-length`, `--hop-length`, `--target-sr` alinham a conversão frame⇄tempo; `--n-fft` e `--spectrogram-hop-length` controlam a STFT da faixa de espectrograma.
- `--output-dir`: pasta onde os PNGs serão gravados (padrão: `data/results/sequence_viz`).

## Como funciona a reconstrução
1. Carrega o `manifest_sequences.csv` e, se informado, filtra por `split` e `max-sequences`.
2. Lê os `manifest.csv` dos fragmentos para mapear cada `snippet_path` para os metadados originais (`source_filepath`, `onset_s`, `offset_s`, `duration_s`, `n_frames`).
3. Para cada sequência:
   - Recarrega os trechos de áudio originais via `librosa.load(..., offset=onset_s, duration=offset_s-onset_s)` e os posiciona na linha do tempo segundo `start_frame/end_frame` usando `hop_length` e `target_sr`.
   - Calcula o espectrograma em dB da waveform reconstruída.
   - Lê o arquivo `.npy` da sequência para exibir os MFCCs armazenados.
   - Constrói uma máscara binária com base na lista de segmentos (`label == "Nothing"` → 0; demais classes → 1).
4. Salva um PNG por sequência com as quatro faixas empilhadas.

## Exemplo de uso
```bash
python src/visualize_sequences.py \
  --sequence-manifest data/results/sequences/manifest_sequences.csv \
  --fragments-dir data/results/fragments_combined \
  --splits train val \
  --max-sequences 3 \
  --output-dir data/results/sequence_viz
```

Saída: arquivos `sequence_<n>.png` em `data/results/sequence_viz/`, contendo waveform reconstruído, espectrograma (dB), MFCC da sequência e máscara binária Nothing vs. demais classes.
