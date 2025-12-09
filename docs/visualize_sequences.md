# Visualização de sequências (colormesh)

Este guia descreve o visualizador compacto `visualize_sequence_colormesh.py`, pensado para gerar rapidamente painéis com o
colormesh das sequências (`freq x frames`) salvas pelo `build_dataset.py`. Ele usa as informações do
`manifest_sequences.csv` por segmento e assume o mesmo passo temporal usado na montagem das fitas (padrão: 10 fps com
`hop_length=6400` e `target_sr=64000`).

## Entradas e parâmetros
- `--sequence-manifest`: caminho para o `manifest_sequences.csv` (formato segmentado gerado pelo builder).
- `--sequence-idx` / `--sequence-idx-range`: selecione explicitamente quais sequências serão plotadas. Se não for
  informado, todas as sequências disponíveis (após filtros de split) são consideradas.
- `--segment-idx-range`: recorta os frames ao menor intervalo que cobre os segmentos indicados (por índice dentro da
  sequência). Útil para destacar apenas parte de uma fita longa.
- `--splits`: opcional para filtrar por split (`train`, `val`, `test`).
- `--max-plot-duration`: limita a janela temporal plotada (em segundos), evitando espectrogramas muito grandes.
- `--hop-length` / `--target-sr`: usados para converter frames em segundos. Defaults preservam 10 fps e banda até 32 kHz.
- `--output-dir`: pasta onde os PNGs serão salvos.

Caminhos de sequência (`sequence_path`) podem ser absolutos ou relativos ao diretório do manifesto; o script preserva o caminho
quando já existe e, caso contrário, tenta resolvê-lo em relação ao diretório do manifest.

## Fluxo interno
1. Lê o manifest por segmentos, aplica filtros de split e de `sequence_idx`.
2. Para cada sequência, opcionalmente filtra os segmentos pelo intervalo de `segment_idx` e determina a janela de frames
   correspondente; `--max-plot-duration` pode encurtar ainda mais a janela.
3. Carrega o `.npy` da sequência, recorta a janela escolhida e cria o eixo de tempo em segundos
   (`frames * hop_length / target_sr`).
4. Plota dois painéis: (a) `pcolormesh` do tensor (freq x frames) e (b) uma máscara binária Nothing (0) vs. outras classes (1)
   derivada dos intervalos listados no manifesto.
5. Salva um PNG por sequência selecionada em `--output-dir`, nomeando pelo índice e pelo split.

## Exemplo de uso
```bash
python src/visualize_sequence_colormesh.py \
  --sequence-manifest data/results/sequences/manifest_sequences.csv \
  --splits train \
  --sequence-idx-range 0 2 \
  --segment-idx-range 50 80 \
  --max-plot-duration 30 \
  --output-dir data/results/sequence_viz_colormesh
```
Esse comando plota as sequências de índice 0 a 2 do split `train`, recortando os frames que cobrem os segmentos 50–80 (ou até 30 s,
se for menor) e salva os PNGs na pasta indicada.
