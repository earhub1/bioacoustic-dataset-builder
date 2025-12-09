# Guia rápido: MLP binário para classificação frame a frame

Este guia sugere um ponto de partida simples para treinar um classificador MLP
binário (Nothing vs. evento, ex.: G01) sobre o dataset de sequências gerado pelo
`build_dataset.py`. O modelo usa **27 entradas por frame**: 9 features do frame
atual, mais 9 do frame anterior e 9 do frame anterior do anterior.

## Premissas do dataset
- As sequências estão em `data/results/sequences/{train,val,test}` no formato
  `.npy`, com dimensão `(n_mels, n_frames)` (ex.: 9 × T).
- O manifesto `manifest_sequences.csv` traz uma linha por **segmento** com
  `label`, `split`, `start_frame`, `end_frame` e `duration_frames`.
- Para tarefas frame a frame, cada segmento pode ser expandido para rótulos por
  frame. Nothing costuma dominar (~85%) e G01 fica perto de 15%, portanto use
  pesos de perda ou balanceamento de amostragem.

## Pipeline recomendado
1. **Carregar manifestos**: leia `manifest_sequences.csv` para descobrir quais
   arquivos `.npy` e quais intervalos pertencem a cada split. Use
   `manifest_sequences_summary.csv` se precisar dos totais por split.
2. **Criar vetores de entrada com contexto (27 dims)**:
   - Para um frame de índice `f`, empilhe os frames `[f-2, f-1, f]` e
     concatene ao longo da dimensão de features, resultando em 27 valores.
   - **Início da sequência**: quando `f-1` ou `f-2` são negativos, escolha uma
     estratégia de preenchimento consistente:
     - `repeat`: repetir o primeiro frame disponível (mantém energia e escala).
     - `zero_pad`: preencher com zeros (destaca “ausência” de histórico).
     - `edge_pad`: repetir o frame corrente (equivalente a clamping em zero).
   - Use a mesma estratégia em treino/val/test para evitar discrepâncias.
3. **Gerar exemplos balanceados para o treino**:
   - Oversampling de G01 na seleção de frames ou `WeightedRandomSampler` (PyTorch)
     para reduzir o viés de Nothing.
   - Alternativa (ou complemento): pesar a função de perda (`pos_weight` em BCE)
     com base na fração de frames por label do split.
4. **Treinar o MLP binário**:
   - Arquitetura mínima: `nn.Sequential(Linear(27, 64), ReLU(), Dropout(0.2),
     Linear(64, 1))`, produzindo logits para BCEWithLogitsLoss.
   - Hiperparâmetros iniciais: Adam (lr≈1e-3), lote 512–2048 frames,
     embaralhamento a cada época. Monitore F1, AUROC e PR-AUC na validação.
5. **Pós-treino**: escolha o limiar de decisão usando a curva precision–recall
   na validação para equilibrar precisão e recall da classe G01.

## Exemplo de pré-processamento (PyTorch)
```python
import numpy as np
import torch
from torch.utils.data import Dataset

class FrameDataset(Dataset):
    def __init__(self, manifest_df, split='train', pad_mode='repeat'):
        self.rows = manifest_df[manifest_df.split == split].reset_index(drop=True)
        self.pad_mode = pad_mode

    def __len__(self):
        return len(self.rows)

    def _pad_index(self, t):
        if t >= 0:
            return t
        if self.pad_mode == 'zero_pad':
            return None  # tratado como zeros
        # repeat/edge: usa o índice zero
        return 0

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        feats = np.load(row.sequence_path)  # shape: (n_mels, T)
        start, end = int(row.start_frame), int(row.end_frame)
        label = 1 if row.label == 'G01' else 0
        # escolhe um frame aleatório dentro do segmento
        f = np.random.randint(start, end)
        frames = []
        for t in (f - 2, f - 1, f):
            idx_t = self._pad_index(t)
            if idx_t is None:
                frames.append(np.zeros(feats.shape[0], dtype=feats.dtype))
            else:
                frames.append(feats[:, idx_t])
        x = np.concatenate(frames, axis=0)  # shape: 27
        return torch.tensor(x, dtype=torch.float32), torch.tensor(label)
```

- Para validação/teste, use o mesmo `pad_mode` e percorra os frames de forma
  determinística (sem aleatoriedade) para métricas reproduzíveis.
- Se houver mais labels no futuro, troque o `label` para inteiro e use
  `CrossEntropyLoss` com saída de 2 unidades.

## Checklist rápido
- Confirmar as proporções por label em cada split usando `src/eda_sequences.py`.
- Fixar a estratégia de padding (`repeat`, `zero_pad` ou `edge_pad`) e mantê-la
  em todos os splits.
- Usar perda ponderada ou sampler balanceado para mitigar o domínio de Nothing.
- Monitorar F1, AUROC e PR-AUC; calibrar o limiar final na validação.

Este guia é apenas um pontapé inicial; acrescente normalização por feature,
regularização e tuning de arquitetura conforme necessário.
